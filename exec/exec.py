'''
Everything about execution
'''

from api_server.request import Request, RequestStatus
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from model.jllama import JLlama
from model.util import load_hf_weights
import torch
from model.constants import MODEL_ID, EOS_TOKEN_ID, BATCH_SIZE

class Worker:
    def __init__(self, requests: list[Request] = [], device="mps", debug=0):
        self.device = device
        self.debug = debug
        self.requests = requests
        self.config = AutoConfig.from_pretrained(MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        hf_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype=self.config.dtype,
        )
        self.model = JLlama(self.config)
        self.model = load_hf_weights(self.model, hf_model)
        self.model = self.model.to(dtype=hf_model.dtype)
        self.model = self.model.to(device=device)
        self.model.eval()

        self.batch_tokens = None
        self.batch_cache_blocks = None
        self.batch_cache_len = None
        self.kv_cache = None

    def warmup(self):
        seq_len = 20
        dummy_tokens = torch.arange(40, 40 + seq_len).unsqueeze(dim=0).to(device=self.device)
        dummy_cache_blocks = torch.zeros(1,1)
        dummy_cache_len = [0]
        dummy_kv = torch.zeros((
                self.config.num_hidden_layers,
                2,
                BATCH_SIZE,
                self.config.num_key_value_heads,
                seq_len,
                self.config.head_dim,
                ), dtype=self.config.dtype).to(device=self.device)
        with torch.no_grad():
            _ = self.model(dummy_tokens, dummy_cache_blocks, dummy_cache_len, dummy_kv)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()


    def execute(self) -> tuple[list[str], list[int]]:
        self.requests_to_batch()
        with torch.no_grad():
            logits = self.model(self.batch_tokens, self.batch_cache_blocks, self.batch_cache_len, self.kv_cache)

        # we are doing inference only, so past this point, we only care about
        # the logits of the last token (this is only relevant in the prefill
        # case where the input is truly rank 3)
        logits = logits[:, -1, :]

        probs = self.requests[0].sampling_params.apply_params(logits)
        # raw output from sample is [batch_size,1], so convert to
        # python list
        token_ids = self.requests[0].sampling_params.sample(probs).squeeze(-1).tolist()
        # get human readable version
        text = [self.tokenizer.decode(token_id) for token_id in token_ids]

        # update status for each request and append new token
        for i in range(len(self.requests)):
            request = self.requests[i]
            # I don't want to pass EOS all the way through, it is too clutered
            # so hardcode it here, be wary of changing this for different models
            if token_ids[i] == EOS_TOKEN_ID:
                request.status = RequestStatus.FINISHED
            elif request.status == RequestStatus.PREFILLING:
                # increment cache length to indicate we have cached the prompt
                request.cache_len = request.token_ids.shape[-1]
                request.status = RequestStatus.DECODING
            elif request.status == RequestStatus.DECODING:
                request.cache_len += 1
            request.append_token_id(token_ids[i])
            request.append_text(text[i])

        if self.debug >= 1:
            print(text)

        return text, token_ids

    def requests_to_batch(self) -> None:
        '''
        Populates the batch param by converting a list of requests into
        a tensor of token ids and a tensor of block ids for KV cache

        For batching: think about how padding comes into play here
        '''
        # TODO, for now since jsut one request just return tokens, dimensions
        # [batch_size, # tokens]
        if self.requests[0].status == RequestStatus.PREFILLING:
            self.batch_tokens = self.requests[0].token_ids
        else:
            self.batch_tokens = self.requests[0].token_ids[:, -1:]

        # wrap in brackets so that these also have shape [batch_size, # cache_blocks]
        self.batch_cache_blocks = torch.Tensor([self.requests[0].cache_blocks])

        # track how long the cache is, also wrap to get batch size dim
        self.batch_cache_len = [self.requests[0].cache_len]

        self.kv_cache = self.requests[0].kv_cache

    def add_batch(self, batch: list[Request]) -> None:
        self.requests = batch
        self.requests_to_batch()

    def remove_request(self, req: Request) -> None:
        for i in range(len(self.requests)):
            if self.requests[i] == req:
                self.requests.pop(i)
                return
