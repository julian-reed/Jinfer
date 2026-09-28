from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exec.exec import Worker

from api_server.request import Request, RequestStatus
from model.constants import BATCH_SIZE, MODEL_ID, BLOCK_SIZE
from transformers import AutoConfig
from collections import deque
import torch

JR_TOKEN_LIMIT = 1024

class LogicalBlock:
    def __init__(self, block_num: int):
        self.block_num = block_num
        self.space_remaining = BLOCK_SIZE

class PhysicalBlock:
    def __init__(self, block_num: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype, device):
        self.block_num = block_num
        # include 2 for k and v, respectively
        self.tensor = torch.zeros(num_layers, 2, num_kv_heads, BLOCK_SIZE, head_dim, device=device, dtype=dtype)

class PoolEntry:
    def __init__(self, num_physical_blocks: int):
        self.block_table = {}
        # incremented to return a count for the logical block
        self.num_logical_blocks = 0
        self.physical_pool = deque(range(num_physical_blocks))

    def get_block(self) -> LogicalBlock:
        physical = self.physical_pool.popleft()
        logical = LogicalBlock(self.num_logical_blocks)
        self.num_logical_blocks += 1
        self.block_table[logical] = physical
        return logical

class KVManager:
    '''
    Manages the KV Cache.
    '''
    def __init__(self, device):
        # maps each worker to its associated kv state
        self.pool = {}
        # maps logical blocks to their underlying physical blocks
        self.config = AutoConfig.from_pretrained(MODEL_ID)
        self.device = device

    # creates the pool entry for the given worker
    def init_cache_for_worker(self, worker: Worker) -> None:
        self.pool[worker] = PoolEntry(worker.get_num_physical_blocks())

    # gets a new block from the pool of the given worker
    def get_block(self, worker: Worker) -> LogicalBlock:
        worker_pool = self.pool[worker]
        return worker_pool.get_block()
        

    # assigns a cache to this request, or updates it to have space
    # for one more token if it already exists
    def set_cache(self, req: Request, worker: Worker) -> None:
        if req.status == RequestStatus.PREFILLING:
            blocks_needed = ((req.token_ids.shape[-1] - 1) // BLOCK_SIZE) + 1
            req.cache_blocks = [self.get_block(worker) for _ in range(blocks_needed)]
        elif req.status == RequestStatus.DECODING:
            # invariant 1: for now I'm not removing the associated cache
            # from any request unless it has finished, so it should always
            # already have a cache if in decode stage
            # invariant 2: only the last block in cache_blocks can be non-full
            if req.cache_blocks[-1].space_remaining == 0:
                req.cache_blocks.append(self.get_block(worker))
        else:
            raise Exception(f"Calling set_cache on request {req} with unexpected status {req.status}")

    def get_table_for_worker(self, worker: Worker) -> dict:
        return self.pool[worker].block_table

    # reclaims physical blocks from a request that has finished
    def reclaim_cache(self, req: Request, worker: Worker) -> None:
        # by the same invariant explained in set_cache()
        if req.status != RequestStatus.FINISHED:
            raise Exception(f"Calling reclaim_cache on request {req} with unexpected status {req.status}")
        worker_pool = self.pool[worker]
        for logical in req.cache_blocks:
            worker_pool.physical_pool.append(worker_pool.block_table.pop(logical))
