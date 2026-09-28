'''
Defines key functionlaity for the api server, which ultiamtely passes a
request object to the engine

Main entrypoint is by calling generate() with the single prompt
'''

from time import perf_counter
from typing import Any
from transformers import AutoTokenizer
from engine.engine import Engine
from api_server.request import Request, SamplingParams
import torch
from model.constants import MODEL_ID


class Server():
    def __init__(self, skip_load=False, debug=0, benchmark=0, device="mps"):
        self.tokenzier = AutoTokenizer.from_pretrained(MODEL_ID)
        self.debug = debug
        self.benchmark = benchmark

        self.device = "cpu"
        if device not in ["mps", "cuda", "cpu"]:
            raise Exception("unsupported device")
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        if device == "mps" and torch.mps.is_available():
            self.device = "mps"
        print(f"Using device {self.device}")

        self.engine = Engine(skip_load, debug, benchmark, self.device)
        self.requests_added = 0

    async def generate(self, prompt: str, temperature=1.0, top_k=-1, top_p=-1.0, min_p=-1.0) -> dict[str, Any]:
        tokenized = self.tokenize(prompt)
        params = SamplingParams(temperature, top_k, top_p, min_p)
        req = Request(
                tokens=tokenized["input_ids"], # shape [1, seq_len]
                attn_mask=tokenized["attention_mask"], # shape [1, seq_len]
                sampling_params=params,
            )
        streamer = self.engine.add_request(req)
        self.requests_added += 1
        req_num = self.requests_added

        last_generated = None
        tokens_processed = 0
        ttft = perf_counter()
        tbt_sum = 0.0
        # for benchmarking, use generated at time from executor so that
        # time to print logs here doesn't affect generation time
        async for token, generated_at in streamer:
            tokens_processed += 1
            if tokens_processed == 1:
                last_generated = perf_counter()
                ttft = last_generated - ttft
                if self.benchmark >= 1:
                    print(f"[Request {req_num}] token #{tokens_processed}: '{token}' generated in {ttft} sec")
            else: 

                tbt = generated_at - last_generated
                tbt_sum += tbt
                if self.benchmark >= 1:
                    print(f"[Request {req_num}] token #{tokens_processed}: '{token}' generated in {tbt} sec")
            last_generated = generated_at

        benchmarks = {'TTFT':ttft, 'TBT':tbt_sum / tokens_processed}
        out_dict = {'benchmarks':benchmarks, 'text':streamer.text, 'token_ids':streamer.token_ids}

        return out_dict


    def tokenize(self, prompt: str) -> torch.Tensor:
         messages = [
              {"role": "user", "content": prompt},
          ]

         return self.tokenzier.apply_chat_template(
                 messages,
                 tokenize=True,
                 add_generation_prompt=True,
                 return_tensors="pt",
                 return_dict=True,
                 ).to(self.device)



