'''
Everything about execution
'''

import sys
from api_server.request import Request, RequestStatus
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from model.jllama import JLlama
from model.util import load_hf_weights
import torch
from model.constants import MODEL_ID, EOS_TOKEN_ID, BATCH_SIZE, PAD_TOKEN_ID

class Worker:
    def __init__(self, requests: list[Request] = [], device="mps", debug=0, skip_load=False):
        self.device = device
        self.debug = debug
        self.requests = requests
        self.config = AutoConfig.from_pretrained(MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        self.model = JLlama(self.config)
        if skip_load:
            self.model = self.model.to(dtype=self.config.dtype, device=device)
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    dtype=self.config.dtype,
            )
            self.model = load_hf_weights(self.model, hf_model)
            self.model = self.model.to(dtype=hf_model.dtype)
            self.model = self.model.to(device=device)
        self.model.eval()

        self.batch_tokens = None
        self.batch_cache_blocks = None
        self.batch_cache_len = None
        self.batch_prompt_lens = None
        self.kv_cache = None

    def warmup(self):
        seq_len = 20
        dummy_tokens = torch.arange(40, 40 + seq_len).unsqueeze(dim=0).to(device=self.device)
        dummy_cache_blocks = torch.zeros(1,1)
        dummy_cache_len = [0]
        dummy_sequence_lens = [0]
        dummy_kv = torch.zeros((
                self.config.num_hidden_layers,
                2,
                1,
                self.config.num_key_value_heads,
                seq_len,
                self.config.head_dim, 
                ), dtype=self.config.dtype).to(device=self.device)
        with torch.no_grad():
            _ = self.model(dummy_tokens, dummy_sequence_lens, dummy_cache_blocks, dummy_cache_len, dummy_kv)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()


    def execute(self) -> tuple[list[str], list[int]]:
        # a bit janky since this assumes all reqeusts in the batch have the 
        # same status as the first, which is true for static but not continuous batching
        if self.requests[0].status == RequestStatus.PREFILLING:
            # self.batch_tokens = self.requests[0].token_ids
            self.batch_tokens = torch.cat(tuple([request.token_ids for request in self.requests]), dim=0)
        else:
            # self.batch_tokens = self.requests[0].token_ids[:, -1:]
            self.batch_tokens = torch.cat(tuple([request.token_ids for request in self.requests]), dim=0)[:,-1:]

        # print(
        #       "Worker:",
        #       "batch_tokens=", self.batch_tokens.shape,
        #       "requests=", len(self.requests),
        #       "prompt_lens=", self.batch_prompt_lens,
        #       "prompt_lens_count=", len(self.batch_prompt_lens),
        #       "cache_len=", self.batch_cache_len,
        #       flush=True,
        #       file=sys.stderr,
        #   )

        with torch.no_grad():
            logits = self.model(
                    self.batch_tokens,
                    self.batch_prompt_lens,
                    self.batch_cache_blocks,
                    self.batch_cache_len,
                    self.kv_cache,
                )

        # we are doing inference only, so past this point, we only care about
        # the logits of the last tokens (this is only relevant in the prefill
        # case where the input is truly rank 3)
        logits = logits[:, -1, :]

        # outputs for this batch
        text_list = []
        token_ids = []

        for i in range(len(self.requests)):
            req = self.requests[i]
            probs = req.sampling_params.apply_params(logits[i, ...])
            token_id = req.sampling_params.sample(probs).squeeze(-1).item()
            # get human readable version
            text = self.tokenizer.decode(token_id)
            text_list.append(text)
            req.append_token_id(token_id)
            req.append_text(text)
            
            # I don't want to pass EOS all the way through, it is too clutered
            # so hardcode it here, be wary of changing this for different models
            if token_id == EOS_TOKEN_ID:
                req.status = RequestStatus.FINISHED
            elif req.status == RequestStatus.PREFILLING:
                # increment cache length to indicate we have cached the prompt
                req.cache_len = req.token_ids.shape[-1]
                req.status = RequestStatus.DECODING
            elif req.status == RequestStatus.DECODING:
                req.cache_len += 1

        self.update_batch_params()

        if self.debug >= 1:
            print(text_list)

        return text_list, token_ids

    def update_batch_params(self) -> None:
        # for now just update the cache length, only one that needs to be dynamic
        self.batch_cache_len = [request.cache_len for request in self.requests]

    def set_batch(self, batch: list[Request], new_status: RequestStatus = RequestStatus.PREFILLING) -> None:
        '''
        Populates the batch param by converting a list of requests into
        a tensor of token ids and a tensor of block ids for KV cache
        '''

        if len(batch) == 0:
            self.requests = []
            if self.debug >= 1:
                print("Warning: called set_batch with an empty batch")
            return

        self.requests = list(batch)

        for request in self.requests:
            request.set_status(new_status)

        self.batch_prompt_lens = [request.token_ids.shape[-1] for request in self.requests]

        # left padding
        target_len = max(self.batch_prompt_lens)
        for request in self.requests:
            num_padding_tokens = target_len - request.token_ids.shape[-1]
            if num_padding_tokens == 0:
                continue
            padding = torch.Tensor([PAD_TOKEN_ID] * num_padding_tokens).to(dtype=torch.long, device=self.device)
            request.token_ids = torch.cat((padding[None, :], request.token_ids), dim=-1)

        self.batch_cache_blocks = [request.cache_blocks for request in self.requests]

        # track how long the cache is, also wrap to get batch size dim
        # self.batch_cache_len = [self.requests[0].cache_len]
        self.batch_cache_len = [request.cache_len for request in self.requests]

        # kv cache includes batch size dim, so they all share same kv
        self.kv_cache = self.requests[0].kv_cache

    # def remove_request(self, req: Request) -> None:
    #     for i in range(len(self.requests)):
    #         if self.requests[i] == req:
    #             self.requests.pop(i)
    #             return
