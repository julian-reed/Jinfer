import torch
import sys
from torch import nn
import torch.nn.functional as F
from math import sqrt
from engine.kv_manager import PhysicalBlock
from model.constants import BLOCK_SIZE, PAD_TOKEN_ID

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


    if x.ndim == 3:
        x = x.unsqueeze(0)
        remove_batch_dimension = True
    elif x.ndim == 4:
        remove_batch_dimension = False
    else:
        raise ValueError(
            f"x must have shape [H, S, D] or [B, H, S, D], got {tuple(x.shape)}"
        )

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

    result = x * cos + rotated * sin

    if remove_batch_dimension:
        result = result.squeeze(0)

    return result


class GroupQueryAttention(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.head_dim * config.num_attention_heads, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.head_dim * config.num_key_value_heads, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.head_dim * config.num_key_value_heads, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        cache_blocks: list[list[PhysicalBlock]],
        cache_lens: list[int],
        req_to_bounds: dict,
    ):
        # for continuous batching/pagedattention, split x by request and reform into
        # stagewise batches.

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # second, split into multi-head setup
        q = q.view(q.shape[0], self.config.num_attention_heads, self.config.head_dim).transpose(0,1)
        k = k.view(k.shape[0], self.config.num_key_value_heads, self.config.head_dim).transpose(0,1)
        v = v.view(v.shape[0], self.config.num_key_value_heads, self.config.head_dim).transpose(0,1)

        position_ids = []
        # for (start, end), req_blocks in zip(req_to_bounds.values(), cache_blocks):
        for i in range(len(cache_blocks)):
            req_blocks = cache_blocks[i]
            seq_len = req_to_bounds[i][1] - req_to_bounds[i][0]
            cache_len = cache_lens[i]
            position_ids.append(
                torch.arange(
                    cache_len,
                    cache_len + seq_len,
                    device=x.device,
                )
            )
        position_ids = torch.cat(position_ids)

        q = rope(q, self.config.rope_parameters, position_ids)
        k = rope(k, self.config.rope_parameters, position_ids)

        # for prefill, we don't need to use the cache since it contains the same data as raw
        # qkv. The request bounds were computed before the forward pass because they are
        # also needed by the executor to decode the returned 2D logits tensor.
        prefill = {}
        # for decode we will actually use the cache so map its contents
        decode = {}
        decode_ranges = []
        max_prefill_len = 0
        for i, (ctr, end) in enumerate(req_to_bounds.values()):
            seq_len = end - ctr
            req_blocks = cache_blocks[i]
            if seq_len == 1:
                # idx_to_replace = BLOCK_SIZE - req_blocks[-1].space_remaining
                idx_to_replace = cache_lens[i] % BLOCK_SIZE
                req_blocks[-1].tensor[self.layer_idx, 0, :, idx_to_replace:idx_to_replace+1, :] = k[:, ctr:end, :]
                req_blocks[-1].tensor[self.layer_idx, 1, :, idx_to_replace:idx_to_replace+1, :] = v[:, ctr:end, :]
                req_q = q[:, ctr:end, :]
                req_k = []
                req_v = []
                for j in range(len(req_blocks)):
                    block = req_blocks[j]
                    # account for the fact that we just generated a new token
                    if j + 1 == len(req_blocks):
                        tokens_in_block = (cache_lens[i] % BLOCK_SIZE) + 1
                    else:
                        tokens_in_block = BLOCK_SIZE
                    req_k.append(block.tensor[self.layer_idx, 0, :, :tokens_in_block, :])
                    req_v.append(block.tensor[self.layer_idx, 1, :, :tokens_in_block, :])
                # only include valid tokens in each block, will add left padding later to balance as necessary
                # req_k = [block.tensor[self.layer_idx, 0, :, BLOCK_SIZE - block.space_remaining, :] for block in req_blocks]
                # req_v = [block.tensor[self.layer_idx, 1, :, BLOCK_SIZE - block.space_remaining, :] for block in req_blocks]
                decode[i] = (req_q, req_k, req_v)
                decode_ranges.append((i, ctr, end))
            else:
                for j in range(len(req_blocks)):
                    start = j * BLOCK_SIZE
                    p_end = min(start + BLOCK_SIZE, seq_len)
                    tokens_in_block = p_end - start
                    req_blocks[j].tensor[self.layer_idx, 0, :, :tokens_in_block, :] = k[:, ctr + start:ctr + p_end, :]
                    req_blocks[j].tensor[self.layer_idx, 1, :, :tokens_in_block, :] = v[:, ctr + start:ctr + p_end, :]
                prefill[i] = (ctr, end)
                max_prefill_len = max(max_prefill_len, end-ctr)

        # setup Q/K/V matricies for prefill
        p_q = []
        p_k = []
        p_v = []
        for start, end in prefill.values():
            padding = max_prefill_len - (end - start)
            req_q = F.pad(q[:, start:end, :], (0,0,padding,0))
            req_k = F.pad(k[:, start:end, :], (0,0,padding,0))
            req_v = F.pad(v[:, start:end, :], (0,0,padding,0))
            p_q.append(req_q)
            p_k.append(req_k)
            p_v.append(req_v)

        # stack q,k,v along batch size dimension
        if prefill:
            p_q = torch.stack(tuple(p_q))
            p_k = torch.stack(tuple(p_k))
            p_v = torch.stack(tuple(p_v))

        # set up Q/K/V matricies for decode
        d_q = []
        d_k = []
        d_v = []
        max_len = 0
        for decode_req in decode:
            vals = decode[decode_req]
            d_q.append(vals[0])
            k_concat = torch.cat(tuple(vals[1]), dim=-2)
            v_concat = torch.cat(tuple(vals[2]), dim=-2)
            d_k.append(k_concat)
            d_v.append(v_concat)
            max_len = max(max_len, k_concat.shape[-2])

        decode_lengths = [k_tensor.shape[-2] for k_tensor in d_k]

        if decode:
            # stack q,k,v along batch size dimension
            d_q = torch.stack(tuple(d_q))
            # add padding where necessary for k/v blocks
            for i in range(len(d_k)):
                padding = max_len - d_k[i].shape[-2]
                # F.pad arguments apply from last dimension, so (D_left, D_right, S_left, S_right)
                d_k[i] = F.pad(d_k[i], (0,0,padding,0))
                d_v[i] = F.pad(d_v[i], (0,0,padding,0))
            d_k = torch.stack(tuple(d_k))
            d_v = torch.stack(tuple(d_v))

        # scale k and v to match correct dimensions
        scale_factor = self.config.num_attention_heads // self.config.num_key_value_heads
        if decode:
            d_k = torch.repeat_interleave(d_k, scale_factor, dim=-3)
            d_v = torch.repeat_interleave(d_v, scale_factor, dim=-3)
            d_s = (d_q @ d_k.transpose(-1, -2)) / sqrt(self.config.head_dim)
        if prefill:
            p_k = torch.repeat_interleave(p_k, scale_factor, dim=-3)
            p_v = torch.repeat_interleave(p_v, scale_factor, dim=-3)
            p_s = (p_q @ p_k.transpose(-1, -2)) / sqrt(self.config.head_dim)

        # convert scores into probabilities via masking, then matmul with values
        output = torch.empty_like(x)
        if prefill:
            pad_lens = max_prefill_len - torch.tensor(
                [end - start for start, end in prefill.values()],
                device=p_s.device,
                dtype=torch.long,
            )
            positions = torch.arange(max_prefill_len, device=p_s.device)
            valid_queries = positions[None, :] >= pad_lens[:, None]
            valid_keys = valid_queries
            causal = positions[None, :] <= positions[:, None]
            valid_attention = valid_keys[:, None, :] & causal[None, :, :]

            # at this point, the mask is correct. However, due to padded queries, some rows
            # in the score matrix will be entirely masked out, which would cause NaNs in the
            # softmax. To avoid this, allow padded queries to attend to valid keys, and 
            # remove the impact of padded queries after softmax has been caclculated
            # broadcasting: queries are rows in the score matmul and keys are columns, so broadcast
            # valid queries across all columns and broadcast keys across all rows
            safe_attention = torch.where(
                valid_queries[:, :, None],
                valid_attention,
                valid_keys[:, None, :],
            )
            p_s = p_s.masked_fill(
                ~safe_attention[:, None, :, :],
                float("-inf"),
            )
            p_probs = torch.softmax(p_s, dim=-1).to(p_v.dtype)
            p_out = p_probs @ p_v
            p_out = p_out.masked_fill(
                ~valid_queries[:, None, :, None],
                0.0,
            )

            for batch_index, (start, end) in enumerate(prefill.values()):
                request_len = end - start
                # get last request_len tokens to avoid left padded ones
                output[start:end] = (
                    p_out[batch_index, :, -request_len:, :]
                    .transpose(0, 1)
                    .flatten(-2, -1)
                )

        if decode:
            pad_lens = max_len - torch.tensor(decode_lengths, device=d_s.device, dtype=torch.long)
            positions = torch.arange(max_len, device=d_s.device)
            valid_keys = positions[None, :] >= pad_lens[:, None]
            d_s = d_s.masked_fill(~valid_keys[:, None, None, :], float("-inf"))
            d_probs = torch.softmax(d_s, dim=-1).to(d_v.dtype)
            d_out = d_probs @ d_v

            for batch_index, (_, start, end) in enumerate(decode_ranges):
                output[start:end] = (
                    d_out[batch_index, :, 0, :]
                    .reshape(1, -1)
                )

        return self.o_proj(output)

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        self.attention = GroupQueryAttention(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config.hidden_size, config.intermediate_size, config.mlp_bias)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        cache_blocks: list[list[PhysicalBlock]],
        cache_lens: list[int],
        req_to_bounds: dict,
    ) -> torch.Tensor:
        attn_out = self.attention(self.attn_norm(x), cache_blocks, cache_lens, req_to_bounds)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_blocks: list[list[PhysicalBlock]],
        cache_lens: list[int],
        req_to_bounds: dict,
    ) -> torch.Tensor:
        # check that sequence length is valid
        if input_ids.shape[-1] >= self.max_seq_len:
            raise Exception("max sequence length exceeded")

        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, cache_blocks, cache_lens, req_to_bounds)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
