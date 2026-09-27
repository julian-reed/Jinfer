from collections import deque
from api_server.request import Request, RequestStatus, RequestStreamer
from exec.exec import Worker
from queue import Queue
from threading import Thread
from time import perf_counter
from model.constants import BATCH_SIZE


class Engine:
    '''
    Handles scheduling + kv cache management.
    Operates on Request objects, passed down from API server

    Requests handled in FCFS order
    '''
    def __init__(self, debug, benchmark, device):
        self.debug = debug
        self.benchmark = benchmark
        self.device = device
        self.incoming_requests = Queue()
        self.req_to_streamer = {}
        self.requests = deque()
        # spin up just one worker for now, this will be changed
        start = perf_counter()
        self.worker = Worker(device=device)
        end = perf_counter()
        if self.debug >= 1:
            print(f"done initializing worker, took {end-start} seconds")

        Thread(target=self.scheduler, daemon=True).start()

    def add_request(self, req: Request) -> RequestStreamer:
        streamer = RequestStreamer()
        self.req_to_streamer[req] = streamer
        self.incoming_requests.put(req)
        return streamer

    # returns T/F if we should process this batch,
    # this condition will be changed
    def batch_ready(self) -> bool:
        return len(self.requests) >= BATCH_SIZE

    # async runner to kick of execute batch
    def scheduler(self):
        # TODO: distinguish between requests and 
        # contents of the batch
        while True:
            self.requests.append(self.incoming_requests.get())

            if self.batch_ready():
                if self.debug >= 1:
                    print("executing batch!")
                self.execute_batch()


    def execute_batch(self) -> None:
        req = self.requests.popleft()
        self.worker.add_request(req)
        req.set_status(RequestStatus.PREFILLING)
        streamer = self.req_to_streamer[req]
        # static batching philosophy is all requests remain active until
        # completion, so continue autoregressively (we aren't at static
        # batching yet, but as a design choice here)
        # safe to take index 0 since for now batch size = 1
        while req.status != RequestStatus.FINISHED:
            text, token_ids = self.worker.execute()
            streamer.put((text[0], perf_counter()))

        # indicate that all requests are done
        for req in self.worker.requests:
            s = self.req_to_streamer[req]
            s.finish(("".join(req.text), req.token_ids))
        self.worker.remove_request(req)
