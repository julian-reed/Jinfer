import torch
import sys
from torch import nn
import torch.nn.functional as F
from math import sqrt

class MLP(nn.Module):
    '''
    In Llama 3, MLP is FFN_SwiGLU.
    SwiGLU (no bias) has the equation: (Swish_1(xW)) * xV)W_2, where * is elementwise multiplication
    '''
    def __init__(self, hidden_size, intermediate_size, mlp_bias):
        super().__init__()

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        activated = F.silu(gate) * up
        return self.down_proj(activated)


def rope(
    x: torch.Tensor,
    config,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Apply Llama 3 RoPE to a tensor of shape [B, H, S, D].

    Assumes:
        - x is already split into attention heads
        - D == head_dim
        - position_ids has shape [S] or [B, S]
        - config contains Llama 3 rope parameters
    """
    # print(
    #   "ROPE:",
    #   "x=", x.shape,
    #   "position_ids=", position_ids.shape,
    #   flush=True,
    #   file=sys.stderr,
    # )


    batch_size, _, seq_len, head_dim = x.shape

    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    if position_ids.ndim == 1:
        if position_ids.shape[0] != seq_len:
            raise ValueError(
                f"position_ids must have shape [{seq_len}] or "
                f"[{batch_size}, {seq_len}], "
                f"got {tuple(position_ids.shape)}"
            )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    elif position_ids.shape != (batch_size, seq_len):
        raise ValueError(
            f"position_ids must have shape [{seq_len}] or "
            f"[{batch_size}, {seq_len}], "
            f"got {tuple(position_ids.shape)}"
        )

    position_ids = position_ids.to(
        device=x.device,
        dtype=torch.float32,
    )

    # Base inverse frequencies:
    # theta^(-2i / D), for i = 0 ... D/2 - 1
    inv_freq = 1.0 / (
        config["rope_theta"]
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=x.device,
                dtype=torch.float32,
            )
            / head_dim
        )
    )

    # Llama 3 frequency scaling
    factor = config["factor"]
    low_freq_factor = config["low_freq_factor"]
    high_freq_factor = config["high_freq_factor"]
    old_context_len = config["original_max_position_embeddings"]

    wavelen = 2 * torch.pi / inv_freq

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    scaled_inv_freq = inv_freq / factor

    smooth_factor = (
        old_context_len / wavelen - low_freq_factor
    ) / (
        high_freq_factor - low_freq_factor
    )

    smoothed_inv_freq = (
        (1.0 - smooth_factor) * scaled_inv_freq
        + smooth_factor * inv_freq
    )

    # Long wavelengths: fully scaled
    inv_freq = torch.where(
        wavelen > low_freq_wavelen,
        scaled_inv_freq,
        inv_freq,
    )

    # Medium wavelengths: interpolate
    is_medium_freq = (
        (wavelen <= low_freq_wavelen)
        & (wavelen >= high_freq_wavelen)
    )

    inv_freq = torch.where(
        is_medium_freq,
        smoothed_inv_freq,
        inv_freq,
    )

    # [B, S, D/2]
    freqs = position_ids[..., None] * inv_freq

    # Llama/HF convention duplicates frequencies across
    # the first and second halves of head_dim.
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos().to(dtype=x.dtype)
    sin = emb.sin().to(dtype=x.dtype)

    # [B, 1, S, D] so they broadcast over attention heads.
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]

    # Llama rotate_half convention
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    rotated = torch.cat(
        (-x2, x1),
        dim=-1,
    )

    return x * cos + rotated * sin


class GroupQueryAttention(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.head_dim * config.num_attention_heads, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.head_dim * config.num_key_value_heads, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.head_dim * config.num_key_value_heads, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor, prompt_lens: list[int], cache_blocks: list[list[int]], cache_len: list[int], kv_cache: torch.Tensor):
        # first, get projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # second, split into multi-head setup
        q = q.view(q.shape[0], q.shape[1], self.config.num_attention_heads, self.config.head_dim).transpose(1,2)
        k = k.view(k.shape[0], k.shape[1], self.config.num_key_value_heads, self.config.head_dim).transpose(1,2)
        v = v.view(v.shape[0], v.shape[1], self.config.num_key_value_heads, self.config.head_dim).transpose(1,2)

        num_tokens = x.shape[1]
        # this works because of left padding, all sequences in batch have the some
        # cache length and logit is rightmost (also assumes all requests are in same phase)
        start_idx = cache_len[0]
        end_idx = start_idx + num_tokens

        # Cache slots use a shared padded timeline, while RoPE uses each
        # sequence's logical position timeline.
        pad_lengths = torch.tensor(
            [(max(prompt_lens) - prompt_len) for prompt_len in prompt_lens],
            device=x.device,
            dtype=torch.long,
        )
        physical_positions = torch.arange(
            start_idx,
            end_idx,
            device=x.device,
        )
        position_ids = (
            physical_positions[None, :] - pad_lengths[:, None]
        ).clamp_min(0)
        # print(f"pad lengths: {pad_lengths.shape}")
        # print(f"physical_positions: {physical_positions.shape}")
        # print(f"position_ids: {position_ids.shape}")

        q_rope = rope(q, self.config.rope_parameters, position_ids)
        k_rope = rope(k, self.config.rope_parameters, position_ids)

        # update cache, this is pretty janky since no paged attention (block id is which 
        # index to use along batch dim), kv cache tensors have dim 
        # [layer_num, 2, batch size, num k/v heads, sequence len, hidden_size]

        # select just the requests (along batch size dimension) that are active right now, using
        # the fact that cache blocks is just a one element list
        batchwise_indicies = torch.Tensor([req_blocks[0] for req_blocks in cache_blocks]).to(device=kv_cache.device, dtype=torch.long)

        # choosing to store even the padded keys for simplicity, following code manages this
        kv_cache[self.layer_idx, 0, :, :, start_idx:end_idx, :].index_copy_(0, batchwise_indicies, k_rope)
        # kv_cache[self.layer_idx, 0, :, :, start_idx:end_idx, :] = k_rope
        kv_cache[self.layer_idx, 1, :, :, start_idx:end_idx, :].index_copy_(0, batchwise_indicies, v)
        # kv_cache[self.layer_idx, 1, :, :, start_idx:end_idx, :] = v

        # cached_k = kv_cache[self.layer_idx, 0, :, :, :end_idx, :]
        # cached_v = kv_cache[self.layer_idx, 1, :, :, :end_idx, :]
        cached_k = kv_cache[self.layer_idx, 0, :, :, :end_idx, :].index_select(0, batchwise_indicies)
        cached_v = kv_cache[self.layer_idx, 1, :, :, :end_idx, :].index_select(0, batchwise_indicies)

        # scale k and v to match correct dimensions
        scale_factor = self.config.num_attention_heads // self.config.num_key_value_heads
        # k_rope = torch.repeat_interleave(k_rope, scale_factor, dim=1)
        # v = torch.repeat_interleave(v, scale_factor, dim=1)
        k = torch.repeat_interleave(cached_k, scale_factor, dim=-3)
        v = torch.repeat_interleave(cached_v, scale_factor, dim=-3)

        # compute attention across each head
        # s = (q_rope @ k_rope.transpose(-1, -2)) / sqrt(self.config.head_dim)
        s = (q_rope @ k.transpose(-1, -2)) / sqrt(self.config.head_dim)

        all_positions = torch.arange(end_idx, device=s.device, dtype=torch.int32)
        valid_keys = all_positions[None, :] >= pad_lengths[:, None]
        valid_queries = valid_keys

        # we need more complicated masking for prefill
        if s.shape[-2] != 1:
            casual = all_positions[None, :] <= all_positions[:, None]
            # generate combined mask that is valid both casually and w.r.t. each seq len
            casual_and_positional = casual[None, :, :] & valid_keys[:, None, :]

            # at this point, the mask is correct. However, due to padded queries, some rows
            # in the score matrix will be entirely masked out, which would cause NaNs in the
            # softmax. To avoid this, allow padded queries to attend to valid keys, and 
            # remove the impact of padded queries after softmax has been caclculated
            # broadcasting: queries are rows in the score matmul and keys are columns, so bradcast
            # valid queries across all columns and broadcast keys across all rows
            valid_pos = casual_and_positional | (~valid_queries[:, :, None] & valid_keys[:, None, :])
            
            # unsqueeze to broadcast across head dimension
            s.masked_fill_(~valid_pos[:, None, :, :], float('-inf'))

            p = torch.softmax(s, dim=-1).to(v.dtype)
            # reverse the softmax patching we did before 
            p = p.masked_fill(~valid_queries[:, None, :, None], 0.0)

        else:
            # mask out invalid keys
            s.masked_fill_(~valid_keys[:, None, None, :], float('-inf'))
            p = torch.softmax(s, dim=-1).to(v.dtype)

        # rearrange output to [B, # heads, seq_len, head_dim] to [B, seq_len, hidden_dim]
        out = (p @ v).transpose(1, 2).flatten(-2, -1)
        return self.o_proj(out)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        self.attention = GroupQueryAttention(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config.hidden_size, config.intermediate_size, config.mlp_bias)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor, prompt_lens: list[int], cache_blocks: list[list[int]], cache_len: list[int], kv_cache: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(self.attn_norm(x), prompt_lens, cache_blocks, cache_len, kv_cache)
        x = x + attn_out
        mlp_out = self.mlp(self.mlp_norm(x))
        return x + mlp_out
        


class JLlama(nn.Module):
    '''
    Implements a Llama 3 style transformer. Implmented myself to allow
    fine grained kv-cache management, model weights loaded from hf
    '''
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)        
        # use ModuleList to track internal state correctly
        self.layers = nn.ModuleList([Block(config, i) for i  in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_seq_len = config.max_position_embeddings

    def forward(self, input_ids: torch.Tensor, prompt_lens: list[int], cache_blocks: list[list[int]], cache_len: list[int], kv_cache: torch.Tensor) -> torch.Tensor:
        # check that sequence length is valid
        if input_ids.shape[-1] >= self.max_seq_len:
            raise Exception("max sequence length exceeded")

        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, prompt_lens, cache_blocks, cache_len, kv_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
