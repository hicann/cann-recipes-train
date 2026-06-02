# Copyright (c) 2026 SMULL_Group, Harbin Institute of Technology, Shenzhen.
# SPDX-License-Identifier: MIT

import torch
import torch_npu
import triton
import triton.language as tl

from tla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from tla.ops.utils.op import exp, exp2
from tla.utils import IS_NVIDIA_HOPPER, USE_CUDA_GRAPH, autotune_cache_kwargs, check_shared_mem

NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8, 16]


# =============================================================================
# Forward Kernel
# =============================================================================
@triton.autotune(
    configs=[
        triton.Config({'block_v': block_v})
        for block_v in [64]
    ],
    key=['num_heads', 'head_k_dim', 'head_v_dim', 'block_t', 'use_exp2'],
    use_cuda_graph=False,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['seq_len'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k, v, w, v_new,
    g, gk, h, h0, ht,
    cu_seqlens, chunk_offsets,
    seq_len,
    num_heads: tl.constexpr, head_k_dim: tl.constexpr, head_v_dim: tl.constexpr,
    block_t: tl.constexpr, block_v: tl.constexpr,
    use_g: tl.constexpr,
    use_gk: tl.constexpr,
    use_initial_state: tl.constexpr,
    store_final_state: tl.constexpr,
    save_new_value: tl.constexpr,
    use_exp2: tl.constexpr,
    is_varlen: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // num_heads, i_nh % num_heads
    if is_varlen:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        seq_len = eos - bos
        num_time_chunks = tl.cdiv(seq_len, block_t)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * seq_len, i_n * seq_len + seq_len
        num_time_chunks = tl.cdiv(seq_len, block_t)
        boh = i_n * num_time_chunks

    b_h1 = tl.zeros([64, block_v], dtype=tl.float32)
    if head_k_dim > 64: b_h2 = tl.zeros([64, block_v], dtype=tl.float32)
    if head_k_dim > 128: b_h3 = tl.zeros([64, block_v], dtype=tl.float32)
    if head_k_dim > 192: b_h4 = tl.zeros([64, block_v], dtype=tl.float32)

    # calculate offset
    h += (boh * num_heads + i_h).to(tl.int64) * head_k_dim*head_v_dim
    v += (bos * num_heads + i_h).to(tl.int64) * head_v_dim
    k += (bos * num_heads + i_h).to(tl.int64) * head_k_dim
    w += (bos * num_heads + i_h).to(tl.int64) * head_k_dim

    if save_new_value:
        v_new += (bos * num_heads + i_h).to(tl.int64) * head_v_dim

    if use_initial_state:
        h0 = h0 + i_nh * head_k_dim*head_v_dim
    if store_final_state:
        ht = ht + i_nh * head_k_dim*head_v_dim

    # load initial state
    if use_initial_state:
        p_h0_1 = tl.make_block_ptr(h0, (head_k_dim, head_v_dim), (head_v_dim, 1), (0, i_v * block_v), (64, block_v), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if head_k_dim > 64:
            p_h0_2 = tl.make_block_ptr(h0, (head_k_dim, head_v_dim), (head_v_dim, 1), (64, i_v * block_v), (64, block_v), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if head_k_dim > 128:
            p_h0_3 = tl.make_block_ptr(h0, (head_k_dim, head_v_dim), (head_v_dim, 1), (128, i_v * block_v), (64, block_v), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if head_k_dim > 192:
            p_h0_4 = tl.make_block_ptr(h0, (head_k_dim, head_v_dim), (head_v_dim, 1), (192, i_v * block_v), (64, block_v), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(num_time_chunks):
        p_h1 = tl.make_block_ptr(h + i_t * num_heads*head_k_dim*head_v_dim, (head_k_dim, head_v_dim), (head_v_dim, 1), (0, i_v * block_v), (64, block_v), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if head_k_dim > 64:
            p_h2 = tl.make_block_ptr(h + i_t * num_heads*head_k_dim*head_v_dim, (head_k_dim, head_v_dim), (head_v_dim, 1), (64, i_v * block_v), (64, block_v), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if head_k_dim > 128:
            p_h3 = tl.make_block_ptr(h + i_t * num_heads*head_k_dim*head_v_dim, (head_k_dim, head_v_dim), (head_v_dim, 1), (128, i_v * block_v), (64, block_v), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if head_k_dim > 192:
            p_h4 = tl.make_block_ptr(h + i_t * num_heads*head_k_dim*head_v_dim, (head_k_dim, head_v_dim), (head_v_dim, 1), (192, i_v * block_v), (64, block_v), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (seq_len, head_k_dim), (num_heads*head_k_dim, 1), (i_t * block_t, 0), (block_t, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))

        if head_k_dim > 64:
            p_w = tl.make_block_ptr(w, (seq_len, head_k_dim), (num_heads*head_k_dim, 1), (i_t * block_t, 64), (block_t, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if head_k_dim > 128:
            p_w = tl.make_block_ptr(w, (seq_len, head_k_dim), (num_heads*head_k_dim, 1), (i_t * block_t, 128), (block_t, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if head_k_dim > 192:
            p_w = tl.make_block_ptr(w, (seq_len, head_k_dim), (num_heads*head_k_dim, 1), (i_t * block_t, 192), (block_t, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (seq_len, head_v_dim), (num_heads*head_v_dim, 1), (i_t * block_t, i_v * block_v), (block_t, block_v), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if save_new_value:
            p_v = tl.make_block_ptr(v_new, (seq_len, head_v_dim), (num_heads*head_v_dim, 1), (i_t * block_t, i_v * block_v), (block_t, block_v), (1, 0))
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * block_t, seq_len) - 1
        if use_g:
            m_t = (i_t * block_t + tl.arange(0, block_t)) < seq_len
            b_g_last = tl.load(g + bos * num_heads + last_idx * num_heads + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(g + bos * num_heads + i_h, (seq_len,), (num_heads,), (i_t * block_t,), (block_t,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if use_exp2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if head_k_dim > 64: b_h2 *= b_g_last
            if head_k_dim > 128: b_h3 *= b_g_last
            if head_k_dim > 192: b_h4 *= b_g_last

        if use_gk:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * num_heads*head_k_dim + i_h * head_k_dim + o_k1, mask=(o_k1 < head_k_dim), other=0.).to(tl.float32)
            if use_exp2: b_h1 *= exp2(b_gk_last1)[:, None]
            else: b_h1 *= exp(b_gk_last1)[:, None]
            if head_k_dim > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * num_heads*head_k_dim + i_h * head_k_dim + o_k2, mask=(o_k2 < head_k_dim), other=0.).to(tl.float32)
                if use_exp2: b_h2 *= exp2(b_gk_last2)[:, None]
                else: b_h2 *= exp(b_gk_last2)[:, None]
            if head_k_dim > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * num_heads*head_k_dim + i_h * head_k_dim + o_k3, mask=(o_k3 < head_k_dim), other=0.).to(tl.float32)
                if use_exp2: b_h3 *= exp2(b_gk_last3)[:, None]
                else: b_h3 *= exp(b_gk_last3)[:, None]
            if head_k_dim > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * num_heads*head_k_dim + i_h * head_k_dim + o_k4, mask=(o_k4 < head_k_dim), other=0.).to(tl.float32)
                if use_exp2: b_h4 *= exp2(b_gk_last4)[:, None]
                else: b_h4 *= exp(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (head_k_dim, seq_len), (1, num_heads*head_k_dim), (0, i_t * block_t), (64, block_t), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        dot_h1 = tl.dot(b_k, b_v)
        b_h1 += tl.where(i_t >= 0, dot_h1, 0.0)

        if head_k_dim > 64:
            p_k = tl.make_block_ptr(k, (head_k_dim, seq_len), (1, num_heads*head_k_dim), (64, i_t * block_t), (64, block_t), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            dot_h2 = tl.dot(b_k, b_v)
            b_h2 += tl.where(i_t >= 0, dot_h2, 0.0)

        if head_k_dim > 128:
            p_k = tl.make_block_ptr(k, (head_k_dim, seq_len), (1, num_heads*head_k_dim), (128, i_t * block_t), (64, block_t), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            dot_h3 = tl.dot(b_k, b_v)
            b_h3 += tl.where(i_t >= 0, dot_h3, 0.0)

        if head_k_dim > 192:
            p_k = tl.make_block_ptr(k, (head_k_dim, seq_len), (1, num_heads*head_k_dim), (192, i_t * block_t), (64, block_t), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            dot_h4 = tl.dot(b_k, b_v)
            b_h4 += tl.where(i_t >= 0, dot_h4, 0.0)

    if store_final_state:
        p_ht = tl.make_block_ptr(ht, (head_k_dim, head_v_dim), (head_v_dim, 1), (0, i_v * block_v), (64, block_v), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if head_k_dim > 64:
            p_ht = tl.make_block_ptr(ht, (head_k_dim, head_v_dim), (head_v_dim, 1), (64, i_v * block_v), (64, block_v), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if head_k_dim > 128:
            p_ht = tl.make_block_ptr(ht, (head_k_dim, head_v_dim), (head_v_dim, 1), (128, i_v * block_v), (64, block_v), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if head_k_dim > 192:
            p_ht = tl.make_block_ptr(ht, (head_k_dim, head_v_dim), (head_v_dim, 1), (192, i_v * block_v), (64, block_v), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# =============================================================================
# Backward Kernel
# =============================================================================
@triton.autotune(
    configs=[
        triton.Config({'block_v': block_v})
        for block_v in [64, 32]
    ],
    key=['num_heads', 'head_k_dim', 'head_v_dim', 'block_t', 'block_v', 'use_g', 'use_exp2'],
    use_cuda_graph=False,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['seq_len'])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q, k, w,
    g, gk, dht, dh0,
    do, dh, dv, dv2,
    cu_seqlens, chunk_offsets,
    scale, seq_len,
    num_heads: tl.constexpr, head_k_dim: tl.constexpr, head_v_dim: tl.constexpr,
    block_t: tl.constexpr, block_v: tl.constexpr,
    use_g: tl.constexpr,
    use_gk: tl.constexpr,
    use_initial_state: tl.constexpr,
    use_final_state_gradient: tl.constexpr,
    use_exp2: tl.constexpr,
    is_varlen: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // num_heads, i_nh % num_heads
    if is_varlen:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        seq_len = eos - bos
        num_time_chunks = tl.cdiv(seq_len, block_t)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * seq_len, i_n * seq_len + seq_len
        num_time_chunks = tl.cdiv(seq_len, block_t)
        boh = i_n * num_time_chunks


    b_dh1 = tl.zeros([64, block_v], dtype=tl.float32)
    if head_k_dim > 64: b_dh2 = tl.zeros([64, block_v], dtype=tl.float32)
    if head_k_dim > 128: b_dh3 = tl.zeros([64, block_v], dtype=tl.float32)
    if head_k_dim > 192: b_dh4 = tl.zeros([64, block_v], dtype=tl.float32)

    # calculate offset
    q += (bos * num_heads + i_h).to(tl.int64) * head_k_dim
    k += (bos * num_heads + i_h).to(tl.int64) * head_k_dim
    w += (bos * num_heads + i_h).to(tl.int64) * head_k_dim
    do += (bos * num_heads + i_h).to(tl.int64) * head_v_dim
    dv += (bos * num_heads + i_h).to(tl.int64) * head_v_dim
    dv2 += (bos * num_heads + i_h).to(tl.int64) * head_v_dim
    dh += (boh * num_heads + i_h).to(tl.int64) * head_k_dim*head_v_dim

    if use_gk:
        gk += (bos * num_heads + i_h).to(tl.int64) * head_k_dim

    if use_initial_state:
        dh0 += i_nh * head_k_dim*head_v_dim
    if use_final_state_gradient:
        dht += i_nh * head_k_dim*head_v_dim

    if use_final_state_gradient:
        p_dht1 = tl.make_block_ptr(dht, (head_k_dim, head_v_dim), (head_v_dim, 1), (0, i_v * block_v), (64, block_v), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if head_k_dim > 64:
            p_dht2 = tl.make_block_ptr(dht, (head_k_dim, head_v_dim), (head_v_dim, 1), (64, i_v * block_v), (64, block_v), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if head_k_dim > 128:
            p_dht3 = tl.make_block_ptr(dht, (head_k_dim, head_v_dim), (head_v_dim, 1), (128, i_v * block_v), (64, block_v), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if head_k_dim > 192:
            p_dht4 = tl.make_block_ptr(dht, (head_k_dim, head_v_dim), (head_v_dim, 1), (192, i_v * block_v), (64, block_v), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(num_time_chunks - 1, -1, -1):
        p_dh1 = tl.make_block_ptr(dh + i_t*num_heads*head_k_dim*head_v_dim, (head_k_dim, head_v_dim), (head_v_dim, 1), (0, i_v * block_v), (64, block_v), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if head_k_dim > 64:
            p_dh2 = tl.make_block_ptr(dh + i_t*num_heads*head_k_dim*head_v_dim, (head_k_dim, head_v_dim), (head_v_dim, 1), (64, i_v * block_v), (64, block_v), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if head_k_dim > 128:
            p_dh3 = tl.make_block_ptr(dh + i_t*num_heads*head_k_dim*head_v_dim, (head_k_dim, head_v_dim), (head_v_dim, 1), (128, i_v * block_v), (64, block_v), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if head_k_dim > 192:
            p_dh4 = tl.make_block_ptr(dh + i_t*num_heads*head_k_dim*head_v_dim, (head_k_dim, head_v_dim), (head_v_dim, 1), (192, i_v * block_v), (64, block_v), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * block_t, seq_len) - 1
        if use_g:
            bg_last = tl.load(g + (bos + last_idx) * num_heads + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(g + bos * num_heads + i_h, (seq_len,), (num_heads,), (i_t * block_t,), (block_t,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if use_exp2:
                bg_last_exp = exp2(bg_last)
                b_g_exp = exp2(b_g)
            else:
                bg_last_exp = exp(bg_last)
                b_g_exp = exp(b_g)

        p_dv = tl.make_block_ptr(dv, (seq_len, head_v_dim), (num_heads*head_v_dim, 1), (i_t * block_t, i_v * block_v), (block_t, block_v), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (seq_len, head_v_dim), (num_heads*head_v_dim, 1), (i_t * block_t, i_v * block_v), (block_t, block_v), (1, 0))
        p_do = tl.make_block_ptr(do, (seq_len, head_v_dim), (num_heads*head_v_dim, 1), (i_t * block_t, i_v * block_v), (block_t, block_v), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))

        # Update dv
        p_k = tl.make_block_ptr(k, (seq_len, head_k_dim), (num_heads*head_k_dim, 1), (i_t * block_t, 0), (block_t, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if use_gk:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + last_idx * num_heads*head_k_dim + o_k1, mask=(o_k1 < head_k_dim), other=0.).to(tl.float32)
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if head_k_dim > 64:
            p_k = tl.make_block_ptr(k, (seq_len, head_k_dim), (num_heads*head_k_dim, 1), (i_t * block_t, 64), (block_t, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if use_gk:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + last_idx * num_heads*head_k_dim + o_k2, mask=(o_k2 < head_k_dim), other=0.).to(tl.float32)
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if head_k_dim > 128:
            p_k = tl.make_block_ptr(k, (seq_len, head_k_dim), (num_heads*head_k_dim, 1), (i_t * block_t, 128), (block_t, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if use_gk:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + last_idx * num_heads*head_k_dim + o_k3, mask=(o_k3 < head_k_dim), other=0.).to(tl.float32)
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if head_k_dim > 192:
            p_k = tl.make_block_ptr(k, (seq_len, head_k_dim), (num_heads*head_k_dim, 1), (i_t * block_t, 192), (block_t, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if use_gk:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + last_idx * num_heads*head_k_dim + o_k4, mask=(o_k4 < head_k_dim), other=0.).to(tl.float32)
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if use_g:
            m_t = (i_t * block_t + tl.arange(0, block_t)) < seq_len
            if use_exp2:
                b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
            else:
                b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        # Update dh
        p_w = tl.make_block_ptr(w, (head_k_dim, seq_len), (1, num_heads*head_k_dim), (0, i_t * block_t), (64, block_t), (0, 1))
        p_q = tl.make_block_ptr(q, (head_k_dim, seq_len), (1, num_heads*head_k_dim), (0, i_t * block_t), (64, block_t), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if use_g:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if use_gk:
            if use_exp2:
                b_dh1 *= exp2(b_gk_last1[:, None])
            else:
                b_dh1 *= exp(b_gk_last1[:, None])
        b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if head_k_dim > 64:
            p_q = tl.make_block_ptr(q, (head_k_dim, seq_len), (1, num_heads*head_k_dim), (64, i_t * block_t), (64, block_t), (0, 1))
            p_w = tl.make_block_ptr(w, (head_k_dim, seq_len), (1, num_heads*head_k_dim), (64, i_t * block_t), (64, block_t), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if use_g:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if use_gk:
                if use_exp2:
                    b_dh2 *= exp2(b_gk_last2[:, None])
                else:
                    b_dh2 *= exp(b_gk_last2[:, None])
            b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if head_k_dim > 128:
            p_q = tl.make_block_ptr(q, (head_k_dim, seq_len), (1, num_heads*head_k_dim), (128, i_t * block_t), (64, block_t), (0, 1))
            p_w = tl.make_block_ptr(w, (head_k_dim, seq_len), (1, num_heads*head_k_dim), (128, i_t * block_t), (64, block_t), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if use_g:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if use_gk:
                if use_exp2:
                    b_dh3 *= exp2(b_gk_last3[:, None])
                else:
                    b_dh3 *= exp(b_gk_last3[:, None])
            b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if head_k_dim > 192:
            p_q = tl.make_block_ptr(q, (head_k_dim, seq_len), (1, num_heads*head_k_dim), (192, i_t * block_t), (64, block_t), (0, 1))
            p_w = tl.make_block_ptr(w, (head_k_dim, seq_len), (1, num_heads*head_k_dim), (192, i_t * block_t), (64, block_t), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if use_g:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if use_gk:
                if use_exp2:
                    b_dh4 *= exp2(b_gk_last4[:, None])
                else:
                    b_dh4 *= exp(b_gk_last4[:, None])
            b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    if use_initial_state:
        p_dh0 = tl.make_block_ptr(dh0, (head_k_dim, head_v_dim), (head_v_dim, 1), (0, i_v * block_v), (64, block_v), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if head_k_dim > 64:
            p_dh1 = tl.make_block_ptr(dh0, (head_k_dim, head_v_dim), (head_v_dim, 1), (64, i_v * block_v), (64, block_v), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if head_k_dim > 128:
            p_dh2 = tl.make_block_ptr(dh0, (head_k_dim, head_v_dim), (head_v_dim, 1), (128, i_v * block_v), (64, block_v), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if head_k_dim > 192:
            p_dh3 = tl.make_block_ptr(dh0, (head_k_dim, head_v_dim), (head_v_dim, 1), (192, i_v * block_v), (64, block_v), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


# =============================================================================
# Forward Launcher (修复版：强制复用 Kernel)
# =============================================================================
def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, seq_len, num_heads, head_k_dim, head_v_dim = *k.shape, u.shape[-1]
    block_t = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, num_time_chunks, chunk_offsets = B, triton.cdiv(seq_len, block_t), None
    else:
        N, num_time_chunks, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, block_t)
    assert head_k_dim <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, num_time_chunks, num_heads, head_k_dim, head_v_dim)
    

    final_state = k.new_empty(N, num_heads, head_k_dim, head_v_dim, dtype=torch.float32)
    # 强制 Kernel 始终存储 final_state
    kernel_store_final_state = True 
        
    # 处理 g
    if g is None:
        g_ptr = k # 替身
        use_g_flag = False
    else:
        g_ptr = g
        use_g_flag = True
        
    # 处理 gk
    if gk is None:
        gk_ptr = k # 替身
        use_gk_flag = False
    else:
        gk_ptr = gk
        use_gk_flag = True
        
    # 处理 initial_state
    if initial_state is None:
        h0_ptr = k # 替身
        use_h0_flag = False
    else:
        h0_ptr = initial_state
        use_h0_flag = True
        
    # 处理 v_new
    if save_new_value:
        v_new_ptr = torch.empty_like(u)
    else:
        v_new_ptr = u # 替身
        
    # 处理 cu_seqlens
    if cu_seqlens is None:
        if chunk_indices is not None:
            cu_seqlens_ptr = chunk_indices
        else:
            cu_seqlens_ptr = torch.zeros(1, dtype=torch.int32, device=k.device)
        is_varlen_flag = False
    else:
        cu_seqlens_ptr = cu_seqlens
        is_varlen_flag = True
    
    if chunk_offsets is None:
        chunk_offsets_ptr = cu_seqlens_ptr
    else:
        chunk_offsets_ptr = chunk_offsets

    def grid(meta): return (triton.cdiv(head_v_dim, meta['block_v']), N*num_heads)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new_ptr,
        g=g_ptr,
        gk=gk_ptr,
        h=h,
        h0=h0_ptr,
        ht=final_state,
        cu_seqlens=cu_seqlens_ptr,
        chunk_offsets=chunk_offsets_ptr,
        seq_len=seq_len,
        num_heads=num_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        block_t=block_t,
        # 显式传递所有开关
        use_g=use_g_flag,
        use_gk=use_gk_flag,
        use_initial_state=use_h0_flag,
        store_final_state=kernel_store_final_state, # 强制为 True
        save_new_value=save_new_value,
        use_exp2=use_exp2,
        is_varlen=is_varlen_flag,
    )
    
    # 只有当用户真的需要时才返回 final_state，否则返回 None
    return h, (v_new_ptr if save_new_value else None), (final_state if output_final_state else None)


# =============================================================================
# Backward Launcher (修复版：处理所有 None 参数)
# =============================================================================
def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, seq_len, num_heads, head_k_dim, head_v_dim = *q.shape, do.shape[-1]
    block_t = 64
    assert head_k_dim <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, num_time_chunks, chunk_offsets = B, triton.cdiv(seq_len, block_t), None
    else:
        N, num_time_chunks, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, block_t)

    dh = q.new_empty(B, num_time_chunks, num_heads, head_k_dim, head_v_dim)
    
    # 处理 dh0
    if h0 is not None:
        dh0 = torch.empty_like(h0, dtype=torch.float32)
        h0_ptr = dh0 
        use_initial_state_flag = True
    else:
        dh0 = None
        h0_ptr = dh # 替身，使用 dh (float32)
        use_initial_state_flag = False

    dv2 = torch.empty_like(dv)
    
    # 处理 g
    if g is None:
        g_ptr = q # 替身
        use_g_flag = False
    else:
        g_ptr = g
        use_g_flag = True
        
    # 处理 gk
    if gk is None:
        gk_ptr = q # 替身
        use_gk_flag = False
    else:
        gk_ptr = gk
        use_gk_flag = True
        
    # 处理 dht (final state gradient)
    if dht is None:
        # 【修改】使用 dh (float32) 作为替身，而不是 q (bf16)
        # 这样 Backward Kernel 编译出的签名始终是 *fp32，避免类型推断问题
        dht_ptr = dh 
        use_final_state_grad_flag = False
    else:
        dht_ptr = dht
        use_final_state_grad_flag = True

    # 处理 cu_seqlens
    if cu_seqlens is None:
        if chunk_indices is not None:
            cu_seqlens_ptr = chunk_indices
        else:
            cu_seqlens_ptr = torch.zeros(1, dtype=torch.int32, device=q.device)
        is_varlen_flag = False
    else:
        cu_seqlens_ptr = cu_seqlens
        is_varlen_flag = True
        
    if chunk_offsets is None:
        chunk_offsets_ptr = cu_seqlens_ptr
    else:
        chunk_offsets_ptr = chunk_offsets

    def grid(meta): return (triton.cdiv(head_v_dim, meta['block_v']), N*num_heads)

    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g_ptr,
        gk=gk_ptr,
        dht=dht_ptr,
        dh0=h0_ptr,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens_ptr,
        chunk_offsets=chunk_offsets_ptr,
        scale=scale,
        seq_len=seq_len,
        num_heads=num_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        block_t=block_t,
        # 显式传递开关
        use_g=use_g_flag,
        use_gk=use_gk_flag,
        use_initial_state=use_initial_state_flag,
        use_final_state_gradient=use_final_state_grad_flag,
        use_exp2=use_exp2,
        is_varlen=is_varlen_flag,
    )
    return dh, dh0, dv2
