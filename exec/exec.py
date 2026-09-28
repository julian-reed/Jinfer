'''
Everything about execution
'''

import sys
from api_server.request import Request, RequestStatus
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from model.jllama import JLlama
from model.util import load_hf_weights, resolve_torch_dtype
from random import randint
import torch
from model.constants import BATCH_SIZE, BLOCK_SIZE, MODEL_ID, EOS_TOKEN_ID, PAD_TOKEN_ID
from engine.kv_manager import PhysicalBlock

class Worker:
    def __init__(self, requests: list[Request] = [], device="mps", debug=0, skip_load=False):
        self.device = device
        self.debug = debug
        self.requests = requests
        self.config = AutoConfig.from_pretrained(MODEL_ID)
        self.dtype = resolve_torch_dtype(self.config.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        self.model = JLlama(self.config)
        if skip_load:
            self.model = self.model.to(dtype=self.dtype, device=device)
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    dtype=self.dtype,
            )
            self.model = load_hf_weights(self.model, hf_model)
            self.model = self.model.to(dtype=hf_model.dtype)
            self.model = self.model.to(device=device)
        self.model.eval()

        self.batch_tokens = None
        self.batch_cache_blocks = None
        self.req_to_bounds = {}
        self.physical_blocks = []
        self.batch_cache_lens = []
        self.alloc_physical_blocks()


    def get_num_physical_blocks(self) -> int:
        return len(self.physical_blocks)


    def warmup(self):
        seq_len = 25
        dummy_tokens = torch.arange(40, 40 + seq_len).to(device=self.device)
        dummy_cache_blocks = [[self.physical_blocks[i] for i in range((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)]]
        dummy_cache_lens = [0]
        dummy_req_to_bounds = {0: (0, seq_len)}
        with torch.no_grad():
            _ = self.model(
                dummy_tokens,
                dummy_cache_blocks,
                dummy_cache_lens,
                dummy_req_to_bounds,
            )
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()


    def execute(self, batch: list[Request], block_table: dict) -> tuple[list[str], list[int]]:
        self.set_batch(batch, block_table)

        with torch.no_grad():
            logits = self.model(
                    self.batch_tokens,
                    self.batch_cache_blocks,
                    self.batch_cache_lens,
                    self.req_to_bounds,
                )

        # outputs for this batch
        text_list = []
        token_ids = []

        for i in range(len(self.requests)):
            req = self.requests[i]
            _, end = self.req_to_bounds[i]
            probs = req.sampling_params.apply_params(logits[end - 1, ...])
            token_id = req.sampling_params.sample(probs).squeeze(-1).item()
            # get human readable version
            text = self.tokenizer.decode(token_id)
            text_list.append(text)
            req.append_token_id(int(token_id))
            req.append_text(text)

            # I don't want to pass EOS all the way through, it is too clutered
            # so hardcode it here, be wary of changing this for different models
            if token_id == EOS_TOKEN_ID:
                req.status = RequestStatus.FINISHED
                req.cache_blocks[-1].space_remaining -= 1
            elif req.status == RequestStatus.PREFILLING:
                # increment cache length to indicate we have cached the prompt
                req.cache_len = req.token_ids.shape[-1]
                req.status = RequestStatus.DECODING
                tokens_remaining = req.token_ids.shape[-1]
                for block in req.cache_blocks:
                      tokens_in_block = min(tokens_remaining, BLOCK_SIZE)
                      block.space_remaining = BLOCK_SIZE - tokens_in_block
                      tokens_remaining -= tokens_in_block
            elif req.status == RequestStatus.DECODING:
                req.cache_len += 1
                req.cache_blocks[-1].space_remaining -= 1

        # self.update_batch_params()

        if self.debug >= 1:
            print(text_list)

        return text_list, token_ids

    # def update_batch_params(self) -> None:
    #     # for now just update the cache length, only one that needs to
    #     # be changed at the end of execution
        # self.batch_cache_len = [request.cache_len for request in self.requests]

    def set_batch(self, batch: list[Request], block_table: dict) -> None:
        '''
        Populates the batch param by converting a list of requests into
        a tensor of token ids and a tensor of block ids for KV cache
        '''

        if len(batch) == 0:
            self.requests = []
            self.req_to_bounds = {}
            if self.debug >= 1:
                print("Warning: called set_batch with an empty batch")
            return

        self.requests = list(batch)

        # squeeze to 1D so that after embeddings, this is 2D (instead of having a floating
        # third dimension of size 1)
        tokens_per_batch = [request.get_tokens_for_batch().squeeze(0) for request in self.requests]
        self.batch_tokens = torch.cat(tuple(tokens_per_batch), dim=0)

        self.req_to_bounds = {}
        start = 0
        for i in range(len(tokens_per_batch)):
            end = start + tokens_per_batch[i].shape[-1]
            self.req_to_bounds[i] = (start, end)
            start = end

        # self.batch_cache_blocks = [request.cache_blocks for request in self.requests]
        # send physical tensors to the model, don't want to have to worry about conversion later
        self.batch_cache_blocks = []
        for req in self.requests:
            req_block_nums = [block_table[logical_block] for logical_block in req.cache_blocks]
            self.batch_cache_blocks.append(
                [self.physical_blocks[physical_block_num] for physical_block_num in req_block_nums]
            )

        self.batch_cache_lens = [req.cache_len for req in self.requests]

    # called on startup, determines how much space exists on this device for kv cache
    # physical pages. This is a real function on GPU but since everything is shared on
    # MPS, returning constant amount for now
    def alloc_physical_blocks(self) -> None:
        # first 2 for k/v, second 2 for bf16 data type
        bytes_per_block = self.config.num_hidden_layers * 2 * BLOCK_SIZE * self.config.num_key_value_heads * self.config.head_dim * 2
        total_memory = torch.mps.recommended_max_memory()
        allocated_memory = torch.mps.driver_allocated_memory()
        available_memory = total_memory - allocated_memory
        # allocate enough blocks to take up half of physical memory
        num_physical_blocks = (available_memory / 2) // bytes_per_block

        # JULIAN NOTE: try the above, but just use a constant while debugging
        num_physical_blocks = 5000

        self.physical_blocks = [
            PhysicalBlock(
                i,
                self.config.num_hidden_layers,
                self.config.num_key_value_heads,
                self.config.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            for i in range(num_physical_blocks)
        ]
