from api_server.request import Request, RequestStatus, RequestStreamer
from exec.exec import Worker
from queue import Queue, Empty
from threading import Thread
from time import perf_counter
from model.constants import BATCH_SIZE, PREFILL_ONLY_CUTOFF, DECODE_ONLY_CUTOFF, TIMEOUT
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
        self.requests = []
        self.batch = []
        self.includes_decode = False
        self.includes_prefill = False

        # spin up just one worker for now, this will be changed
        start = perf_counter()
        self.worker = Worker(device=device, skip_load=skip_load)
        end = perf_counter()
        if self.debug >= 1:
            print(f"done initializing worker, took {end-start} seconds")

        self.kv_manager.init_cache_for_worker(self.worker)
        
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

    def remove_from_requests(self, req: Request) -> None:
        for i in range(len(self.requests)):
            if self.requests[i] == req:
                self.requests.pop(i)
                return
        raise Exception(f"Tried to remove request {req} from requests but couldn't find it!")


    def should_add_to_batch(self, req: Request) -> bool:
        if req.status == RequestStatus.DECODING:
            return True
        elif ((not self.includes_decode) and len(self.batch) <= (PREFILL_ONLY_CUTOFF - 1)) or (self.includes_decode and not self.includes_prefill):
            return True
        else:
            return False

    # returns T/F if we should process this batch,
    # implements scheduling condition
    def batch_ready(self) -> bool:
        '''
        Scheduling decision: all requests that are in decode + one prefill,
        two prefills if there are no requests in decode or 4 decodes if there
        are no requests in prefill
        '''
        if self.includes_decode and self.includes_prefill:
            return True
        elif (not self.includes_decode) and len(self.batch) >= PREFILL_ONLY_CUTOFF:
            return True
        elif (not self.includes_prefill) and len(self.batch) >= DECODE_ONLY_CUTOFF:
            return True
        else:
            return False

    '''
    called once the current batch has finished executing, determines
    which contents of the current batch should remain or be added to
    the general requests pool, then handles general requests pool. 

    An easy solution is to evict everything to the request pool then 
    decide what goes into the batch based on the entire pool, but 
    with my choice of scheduling policy, all requests in the decode 
    stage (which is everything in self.batch since the requests that 
    have finished would already be removed) stay so self.batch actually 
    remains unchanged
    '''
    def fill_batch(self) -> None:
        # for now we don't need to actually edit the contents of the
        # batch, see above comment for reasoning
        self.includes_prefill = False
        self.includes_decode = len(self.batch) >= 1
        # what you might want to do in the general case:
        # self.requests.extend(self.batch), self.batch = []

        # important to make a copy since we could be deleting
        for req in list(self.requests):
            if self.should_add_to_batch(req):
                self.add_to_batch(req, True)

    # adds a request to the batch, and updates prefill/decode
    # flags accordingly
    def add_to_batch(self, req: Request, in_requests: bool = True) -> None:
        # this logic could be tighter, but should suffice
        if req.status == RequestStatus.WAITING:
            req.status = RequestStatus.PREFILLING
            self.includes_prefill = True
        else:
            self.includes_decode = True
        self.batch.append(req)

        if in_requests:
            self.remove_from_requests(req)
    
    # helper function for debugging, prints how
    # many requests in the current batch are of
    # what state
    def print_batch_config(self):
        prefill = 0
        decode = 0
        for req in self.batch:
            if req.status == RequestStatus.PREFILLING:
                prefill += 1
            elif req.status == RequestStatus.DECODING:
                decode += 1
            else:
                raise Exception(f"request in batch with unexpected state: {req.status}")
        print(f"batch config: prefill = {prefill}, decode = {decode}")

    # async runner to kick of execute batch
    def scheduler(self):
        force_execute = False
        while True:
            # make this a while loop so if fill_batch actually
            # fills the entire batch, we execute right away
            while self.batch_ready() or (force_execute and len(self.batch) > 0):
                if force_execute:
                    print("force executing")
                if self.debug >= 1:
                    print("executing batch!")
                    self.print_batch_config()
                self.execute_batch()
                self.fill_batch()
                if self.debug >= 1:
                    print("new batch populated!")
                force_execute = False


            try:
                if self.debug >= 2:
                    print("self.incoming_requests:")
                    print(self.incoming_requests)
                next_req = self.incoming_requests.get(timeout=TIMEOUT)
                if self.should_add_to_batch(next_req):
                    self.add_to_batch(next_req, False)
                else:
                    self.requests.append(next_req)

            except Empty:
                # if we don't get any new requests in TIMEOUT seoncds, just execute what we have
                force_execute = True

    # marks a request as finished by removing from worker batch
    # and writing finished to streamer
    def mark_as_finished(self, s: RequestStreamer, req: Request) -> None:
        s.finish(("".join(req.text), req.token_ids))
        self.kv_manager.reclaim_cache(req, self.worker)
        self.remove_from_batch(req)

    def execute_batch(self) -> None:
        for req in self.batch:
            self.kv_manager.set_cache(req, self.worker)
        table = self.kv_manager.get_table_for_worker(self.worker)
        text, token_ids = self.worker.execute(self.batch, table)
        torch.mps.synchronize()
        for i in range(len(self.worker.requests)):
            req = self.worker.requests[i]
            s = self.req_to_streamer[req]
            if req.status == RequestStatus.FINISHED:
                self.mark_as_finished(s, req)
            else:
                s.put((text[i], perf_counter()))

