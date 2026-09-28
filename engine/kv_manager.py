from model.constants import BATCH_SIZE, MODEL_ID
from transformers import AutoConfig
import torch

class KVManager:
    '''
    Manages the KV Cache

    For now takes in max_sequence_len since we are allocating entire tensors,
    this will be skipped when we allocate blockwise
    '''
    def __init__(self, device):
        self.block_table = None
        self.physical_block_to_tensor = {}
        # monotonically increasing count, doesn't enable reuse
        # but we aren't there yet so this is fine
        self.physical_block_count = 0
        self.config = AutoConfig.from_pretrained(MODEL_ID)
        self.device = device
        # pre paged attention, kv cache is a big contiguous tensor
        self.kv_cache = torch.zeros((
                self.config.num_hidden_layers,
                2,
                BATCH_SIZE,
                self.config.num_key_value_heads,
                self.config.max_position_embeddings,
                self.config.head_dim,
                ), dtype=self.config.dtype).to(device=device)

    # assigns a fresh logical block for the caller, currently blocks just
    # correspond to which index along batch dimension we want
    def get_block(self) -> int:
        # in the future will need to pass batch size once it's dynamic
        idx = self.physical_block_count
        self.physical_block_count += 1
        # self.physical_block_to_tensor[idx] = kv_tensor
        return idx
    
    # returns the actual kv cache tensor
    def get_kv_cache(self) -> torch.Tensor:
        return self.kv_cache

        
