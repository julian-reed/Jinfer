import torch
import asyncio
from queue import Queue
from enum import Enum

END_OF_STREAM = object()

class SamplingParams:
    '''
    Object for the various sampling params used in model config.
    For now, only temperature, top_k and top_p are supported.
    '''
    
    def __init__(self, temperature, top_k, top_p, min_p) -> None:
        self.temperature = temperature if temperature != 1.0 else None
        self.top_k = top_k if top_k != -1 else None
        self.top_p = top_p if top_p != -1.0 else None
        self.min_p = min_p if min_p != -1.0 else None

    # applies all nonzero sampling parameters
    def apply_params(self, logits: torch.Tensor) -> torch.Tensor:
        # future step: apply repetition penalities
        if self.temperature:
            logits = logits / self.temperature 

        # top k could be applied before temp to save on computation
        # but for clarity purposes putting all of these in same place
        if self.top_k:
            top_k_vals, _ = torch.topk(logits, self.top_k, dim=-1)
            min_val = top_k_vals[..., -1:]
            logits = torch.where(logits < min_val, float('-inf'), logits)

        if self.top_p:
            sorted_logits, sorted_indicies = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs > self.top_p

            # shift to the right to include first elem that exceeds p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False

            logits = logits.masked_fill(mask, float('-inf'))

        if self.min_p:
            min_p_probs = torch.softmax(logits, dim=-1)
            max_vals, _ = torch.max(min_p_probs, dim=-1, keepdim=True)
            threshold = max_vals * self.min_p
            mask = logits < threshold
            logits = logits.masked_fill(mask, float('-inf'))

        probs = torch.softmax(logits, dim=-1)
        return probs

    # making this a function in case want to change up sampling algo,
    # using simple multinomial for now
    def sample(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.multinomial(probs, num_samples=1)

class RequestStatus(Enum):
    WAITING = 0
    PREFILLING = 1
    DECODING = 2
    FINISHED = 3

class RequestStreamer():
    '''
    Manages streaming the tokens for each request
    '''
    def __init__(self):
        self.token_stream = Queue()
        self.text = None
        self.token_ids = None
        self.error_msg = None

    def put(self, token_id: int) -> None:
        self.token_stream.put(token_id)

    def finish(self, result: tuple) -> None:
        self.text, self.token_ids = result
        self.token_stream.put((END_OF_STREAM, -1))

    def error(self, error_msg: str) -> None:
        self.error_msg = error_msg
        self.token_stream.put((END_OF_STREAM, -1))

    async def __aiter__(self):
        while True:
            new_token, generated_at = await asyncio.to_thread(self.token_stream.get)

            if new_token == END_OF_STREAM:
                if self.error_msg is not None:
                    raise Exception(self.error_msg)
                return

            yield (new_token, generated_at)

class Request:
    '''
    Object representing a request that the system has taken in from api server,
    passed to the engine for scheduling.

    Text represents just the new text generated while tokens includes the prompt
    tokenization
    '''
    def __init__(self, tokens: torch.Tensor, attn_mask: torch.Tensor, sampling_params: SamplingParams):
        self.token_ids = tokens
        self.attn_mask = attn_mask
        self.sampling_params = sampling_params
        self.text = []
        self.status = RequestStatus.WAITING

    def append_token_id(self, new_token: int) -> None:
        # todo: change this so it doesn't alloc new memory each time
        # use new_tensor to match data type and device
        new_tensor = self.token_ids.new_tensor([[new_token]])
        self.token_ids = torch.cat((self.token_ids, new_tensor), dim=-1)

    def append_text(self, new_token: str) -> None:
        self.text.append(new_token)

    def set_status(self, new_status: RequestStatus) -> None:
        self.status = new_status

    def get_text(self) -> str:
        return "".join(self.text)
