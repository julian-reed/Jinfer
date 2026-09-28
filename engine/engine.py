from collections import deque
from api_server.request import Request, RequestStatus, RequestStreamer
from exec.exec import Worker
from queue import Queue
from threading import Thread
from time import perf_counter
from model.constants import BATCH_SIZE
from engine.kv_manager import KVManager
import torch


class Engine:
    '''
    Handles scheduling + kv cache management.
    Operates on Request objects, passed down from API server

    Requests handled in FCFS order
    '''
    def __init__(self, skip_load, debug, benchmark, device):
        self.debug = debug
        self.benchmark = benchmark
        self.device = device
        self.incoming_requests = Queue()
        self.req_to_streamer = {}
        self.kv_manager = KVManager(device)

        # requests are all requests that are pending, while batch
        # is a list of all requests in the next batch to be executed
        self.requests = deque()
        self.batch = []

        # spin up just one worker for now, this will be changed
        start = perf_counter()
        self.worker = Worker(device=device, skip_load=skip_load)
        end = perf_counter()
        if self.debug >= 1:
            print(f"done initializing worker, took {end-start} seconds")
        
        # do warmup pass to init kernels
        start = perf_counter()
        self.worker.warmup()
        end = perf_counter()
        if self.debug >= 1:
            print(f"done with warmup, took {end-start} seconds")

        Thread(target=self.scheduler, daemon=True).start()

    def add_request(self, req: Request) -> RequestStreamer:
        streamer = RequestStreamer()
        self.req_to_streamer[req] = streamer
        self.incoming_requests.put(req)
        return streamer

    def remove_from_batch(self, req: Request) -> None:
        for i in range(len(self.batch)):
            if self.batch[i] == req:
                self.batch.pop(i)
                return
        raise Exception(f"Tried to remove request {req} from batch but couldn't find it!")

    # returns T/F if we should process this batch,
    # this condition will be changed
    def batch_ready(self) -> bool:
        return len(self.batch) >= BATCH_SIZE

    # async runner to kick of execute batch
    def scheduler(self):
        while True:
            next_req = self.incoming_requests.get()

            # give this request a section of the KV cache (this should happen once we know it is batched)
            next_req.append_cache_block(self.kv_manager.get_block())
            next_req.set_kv(self.kv_manager.get_kv_cache())
            # for now, scheduling strategy is everything goes into batch
            # if a request didn't go into the batch it should be appended
            # to self.requests
            self.batch.append(next_req)

            if self.batch_ready():
                if self.debug >= 1:
                    print("executing batch!")
                self.execute_batch()
                # clear batch for now, in the future have execute_batch()
                # manage which requests stay in the batch vs which are removed
                self.batch = []

                # self.worker.execute_batch(self.batch)


    def execute_batch(self) -> None:
        self.worker.set_batch(self.batch)
        # static batching philosophy is all requests remain active until
        # completion, so continue autoregressively (we aren't at static
        # batching yet, but as a design choice here)
        # safe to take index 0 since for now batch size = 1
        while len(self.worker.requests) > 0:
            text, token_ids = self.worker.execute()
            torch.mps.synchronize()
            request_removed = False
            for i in range(len(self.worker.requests)):
                req = self.worker.requests[i]
                s = self.req_to_streamer[req]
                if req.status == RequestStatus.FINISHED:
                    request_removed = True
                    s.finish(("".join(req.text), req.token_ids))
                    self.remove_from_batch(req)
                else:
                    s.put((text[i], perf_counter()))
            if request_removed:
                self.worker.set_batch(self.batch, RequestStatus.DECODING)
        if self.debug >= 1:
            print("batch finished executing!")
