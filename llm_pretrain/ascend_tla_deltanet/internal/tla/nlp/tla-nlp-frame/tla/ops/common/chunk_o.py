# Copyright (c) 2026 SMULL_Group, Harbin Institute of Technology, Shenzhen.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import torch
import torch_npu
import triton
import triton.language as tl

from tla.ops.utils.index import prepare_chunk_indices
from tla.ops.utils.op import exp
from tla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs, check_shared_mem

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@dataclass(frozen=True, slots=True)
class _ChunkFwdKernelOLaunch:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    h: torch.Tensor
    g: torch.Tensor | None
    g_gamma: torch.Tensor | None
    o: torch.Tensor
    cu_seqlens: torch.Tensor | None
    chunk_indices: torch.Tensor | None
    scale: float
    t: int
    n_heads: int
    d_k: int
    d_v: int
    bt: int


# Triton @triton.jit entrypoints must keep a flat parameter list (tensor pointers,
# runtime scalars, tl.constexpr layout). Related launch arguments are grouped in
# _ChunkFwdKernelOLaunch and passed through _launch_chunk_fwd_kernel_o (G.FNM.03).


def _launch_chunk_fwd_kernel_o(grid, launch: _ChunkFwdKernelOLaunch) -> None:
    chunk_fwd_kernel_o[grid](
        q=launch.q,
        k=launch.k,
        v=launch.v,
        h=launch.h,
        g=launch.g,
        g_gamma=launch.g_gamma,
        o=launch.o,
        cu_seqlens=launch.cu_seqlens,
        chunk_indices=launch.chunk_indices,
        scale=launch.scale,
        t=launch.t,
        n_heads=launch.n_heads,
        d_k=launch.d_k,
        d_v=launch.d_v,
        bt=launch.bt,
    )


@triton.heuristics({
    'use_g': lambda args: args['g'] is not None,
    'use_g_gamma': lambda args: args['g_gamma'] is not None,
    'is_varlen': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'bk': 128, 'bv': 128}, num_warps=8, num_stages=3),
        triton.Config({'bk': 64, 'bv': 64}, num_warps=4, num_stages=3),
        triton.Config({'bk': 32, 'bv': 32}, num_warps=2, num_stages=3),
    ],
    key=['n_heads', 'd_k', 'd_v', 'bt'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['t'])
def chunk_fwd_kernel_o(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    t,
    n_heads: tl.constexpr,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    bt: tl.constexpr,
    bk: tl.constexpr,
    bv: tl.constexpr,
    use_g: tl.constexpr,
    use_g_gamma: tl.constexpr,
    is_varlen: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // n_heads, i_bh % n_heads

    if is_varlen:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        t = eos - bos
        nt = tl.cdiv(t, bt)
    else:
        nt = tl.cdiv(t, bt)
        i_tg = i_b * nt + i_t
        bos, eos = i_b * t, i_b * t + t

    # offset calculation
    q += (bos * n_heads + i_h) * d_k
    k += (bos * n_heads + i_h) * d_k
    v += (bos * n_heads + i_h) * d_v
    o += (bos * n_heads + i_h) * d_v
    h += (i_tg * n_heads + i_h).to(tl.int64) * d_k * d_v

    b_o = tl.zeros([bt, bv], dtype=tl.float32)
    b_a = tl.zeros([bt, bt], dtype=tl.float32)

    for i_k in range(tl.cdiv(d_k, bk)):
        p_q = tl.make_block_ptr(q, (t, d_k), (n_heads * d_k, 1), (i_t * bt, i_k * bk), (bt, bk), (1, 0))
        p_k = tl.make_block_ptr(k, (d_k, t), (1, n_heads * d_k), (i_k * bk, i_t * bt), (bk, bt), (0, 1))
        p_h = tl.make_block_ptr(h, (d_k, d_v), (d_v, 1), (i_k * bk, i_v * bv), (bk, bv), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h)
        b_a += tl.dot(b_q, b_k)

    if use_g:
        g += bos * n_heads + i_h
        p_g = tl.make_block_ptr(g, (t,), (n_heads,), (i_t * bt,), (bt,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_a = b_a * exp(b_g[:, None] - b_g[None, :])

    if use_g_gamma:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, bt) + 1)
        b_o = b_o * exp(b_g)[:, None]
        b_a = b_a * exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * bt + tl.arange(0, bt)
    m_t = o_t < t
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_a = tl.where(m_A, b_a, 0)

    p_v = tl.make_block_ptr(v, (t, d_v), (n_heads * d_v, 1), (i_t * bt, i_v * bv), (bt, bv), (1, 0))
    p_o = tl.make_block_ptr(o, (t, d_v), (n_heads * d_v, 1), (i_t * bt, i_v * bv), (bt, bv), (1, 0))

    b_v = tl.load(p_v, boundary_check=(0, 1))
    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_a.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'use_g': lambda args: args['g'] is not None,
    'use_g_gamma': lambda args: args['g_gamma'] is not None,
    'USE_DW': lambda args: args['dw'] is not None,
    'is_varlen': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['n_heads', 'd_k', 'd_v', 'bt', 'bk', 'bv', 'use_g', 'use_g_gamma', 'USE_DW'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['t'])
def chunk_bwd_kernel_dqkwg(
    q,
    k,
    v,
    g,
    g_gamma,
    h,
    do,
    dh,
    dq,
    dk,
    dw,
    dv,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    t,
    n_heads: tl.constexpr,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    bt: tl.constexpr,
    bk: tl.constexpr,
    bv: tl.constexpr,
    use_g: tl.constexpr,
    use_g_gamma: tl.constexpr,
    USE_DW: tl.constexpr,
    is_varlen: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // n_heads, i_bh % n_heads

    all = B * t
    if is_varlen:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        t = eos - bos
        nt = tl.cdiv(t, bt)
    else:
        nt = tl.cdiv(t, bt)
        i_tg = i_b * nt + i_t
        bos, eos = i_b * t, i_b * t + t

    # offset calculation
    v += (bos * n_heads + i_h) * d_v
    do += (bos * n_heads + i_h) * d_v
    h += (i_tg * n_heads + i_h).to(tl.int64) * d_k * d_v
    dh += (i_tg * n_heads + i_h).to(tl.int64) * d_k * d_v
    q += (bos * n_heads + i_h) * d_k
    k += (bos * n_heads + i_h) * d_k
    dq += (bos * n_heads + i_h) * d_k
    dk += (bos * n_heads + i_h) * d_k

    # for delta rule only
    if USE_DW:
        dw += (bos * n_heads + i_h) * d_k
        dv += (bos * n_heads + i_h) * d_v

    if use_g:
        dg += i_k * all * n_heads
        b_dg_last = tl.zeros([1], dtype=tl.float32) if use_g else None
    if use_g_gamma:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, bt) + 1)
        b_g_last = b_gamma * min(bt, t - i_t * bt)
    b_dq = tl.zeros([bt, bk], dtype=tl.float32)
    b_dk = tl.zeros([bt, bk], dtype=tl.float32)
    b_ds = tl.zeros([bt, bt], dtype=tl.float32)
    b_dw = tl.zeros([bt, bk], dtype=tl.float32) if USE_DW else None

    for i_v in range(tl.cdiv(d_v, bv)):
        p_v = tl.make_block_ptr(v, (t, d_v), (n_heads * d_v, 1), (i_t * bt, i_v * bv), (bt, bv), (1, 0))
        p_do = tl.make_block_ptr(do, (t, d_v), (n_heads * d_v, 1), (i_t * bt, i_v * bv), (bt, bv), (1, 0))
        p_h = tl.make_block_ptr(h, (d_v, d_k), (1, d_v), (i_v * bv, i_k * bk), (bv, bk), (0, 1))
        p_dh = tl.make_block_ptr(dh, (d_v, d_k), (1, d_v), (i_v * bv, i_k * bk), (bv, bk), (0, 1))
        
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        if use_g:
            b_dg_last += (tl.sum(b_h * b_dh))
        # [bt, bv] @ [bv, bt] -> [bt, bt]
        b_ds += tl.dot(b_do, tl.trans(b_v))
        # [bt, bv] @ [bv, bk] -> [bt, bk]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [bt, bv] @ [bv, bk] -> [bt, bk]
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        if USE_DW:
            p_dv = tl.make_block_ptr(dv, (t, d_v), (n_heads * d_v, 1), (i_t * bt, i_v * bv), (bt, bv), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

    if USE_DW:
        p_dw = tl.make_block_ptr(dw, (t, d_k), (n_heads * d_k, 1), (i_t * bt, i_k * bk), (bt, bk), (1, 0))
        tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    p_q = tl.make_block_ptr(q, (t, d_k), (n_heads * d_k, 1), (i_t * bt, i_k * bk), (bt, bk), (1, 0))
    p_k = tl.make_block_ptr(k, (t, d_k), (n_heads * d_k, 1), (i_t * bt, i_k * bk), (bt, bk), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    p_dq = tl.make_block_ptr(dq, (t, d_k), (n_heads * d_k, 1), (i_t * bt, i_k * bk), (bt, bk), (1, 0))
    p_dk = tl.make_block_ptr(dk, (t, d_k), (n_heads * d_k, 1), (i_t * bt, i_k * bk), (bt, bk), (1, 0))

    o_t = i_t * bt + tl.arange(0, bt)
    m_t = o_t < t
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    if use_g:
        b_dg = tl.zeros([bt], dtype=tl.float32)
        g += bos * n_heads + i_h
        dg += bos * n_heads + i_h
        p_g = tl.make_block_ptr(g, (t,), (n_heads,), (i_t * bt,), (bt,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_last = tl.load(g + (min(i_t * bt + bt, t) - 1) * n_heads)
        b_dg_last *= exp(b_g_last)

        b_dq = b_dq * exp(b_g)[:, None] * scale
        b_dg += tl.sum(b_dq * b_q, axis=1)

        b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
        b_dg -= tl.sum(b_k * b_dk, axis=1)
        b_dg_last += tl.sum(b_dk * b_k)

        b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0) * scale
        b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))
        b_dg += tl.sum(b_ds2, axis=1)
        b_dg -= tl.sum(b_ds2, axis=0)

        b_ds = b_ds.to(b_k.dtype)
        
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        p_dg = tl.make_block_ptr(dg, (t,), (n_heads,), (i_t * bt,), (bt,), (0,))
        # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
        # b_dg = tl.dot(tl.where(o_t[:, None] <= o_t[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
        b_dg = tl.where(o_t < min(i_t * bt + bt, t) - 1, b_dg, b_dg + b_dg_last)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

    elif use_g_gamma:
        b_dq = b_dq * exp(b_g)[:, None] * scale
        b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
        b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0) * scale
        b_ds = b_ds.to(b_k.dtype)
        
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    else:
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'use_g': lambda args: args['g'] is not None,
    'use_g_gamma': lambda args: args['g_gamma'] is not None,
    'USE_A': lambda args: args['A'] is not None,
    'is_varlen': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['n_heads', 'd_k', 'd_v', 'bt', 'bk', 'bv', 'use_g'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['t'])
def chunk_bwd_kernel_dv_local(
    q,
    k,
    g,
    g_gamma,
    A,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    t,
    n_heads: tl.constexpr,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    bt: tl.constexpr,
    bk: tl.constexpr,
    bv: tl.constexpr,
    use_g: tl.constexpr,
    use_g_gamma: tl.constexpr,
    USE_A: tl.constexpr,
    is_varlen: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // n_heads, i_bh % n_heads
    if is_varlen:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        t = eos - bos
    else:
        bos, eos = i_b * t, i_b * t + t

    # offset calculation
    q += (bos * n_heads + i_h) * d_k
    k += (bos * n_heads + i_h) * d_k
    do += (bos * n_heads + i_h) * d_v
    dv += (bos * n_heads + i_h) * d_v

    if USE_A:
        p_A = tl.make_block_ptr(A + (bos * n_heads + i_h) * bt, (bt, t), (1, n_heads * bt), (0, i_t * bt), (bt, bt), (0, 1))
        b_a = tl.load(p_A, boundary_check=(0, 1))
    else:
        if use_g:
            g += bos * n_heads + i_h
            p_g = tl.make_block_ptr(g, (t,), (n_heads,), (i_t * bt,), (bt,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
        if use_g_gamma:
            b_gamma = tl.load(g_gamma + i_h)
            b_g = b_gamma * (tl.arange(0, bt) + 1)

        b_a = tl.zeros([bt, bt], dtype=tl.float32)
        for i_k in range(tl.cdiv(d_k, bk)):
            p_k = tl.make_block_ptr(k, (t, d_k), (n_heads * d_k, 1), (i_t * bt, i_k * bk), (bt, bk), (1, 0))
            p_q = tl.make_block_ptr(q, (d_k, t), (1, n_heads * d_k), (i_k * bk, i_t * bt), (bk, bt), (0, 1))

            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_a += tl.dot(b_k, b_q) * scale
        if use_g or use_g_gamma:
            b_a *= exp(b_g[None, :] - b_g[:, None])

    o_t = i_t * bt + tl.arange(0, bt)
    m_t = o_t < t
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_a = tl.where(m_A, b_a, 0).to(do.dtype.element_ty)

    for i_v in range(tl.cdiv(d_v, bv)):
        p_do = tl.make_block_ptr(do, (t, d_v), (n_heads * d_v, 1), (i_t * bt, i_v * bv), (bt, bv), (1, 0))
        p_dv = tl.make_block_ptr(dv, (t, d_v), (n_heads * d_v, 1), (i_t * bt, i_v * bv), (bt, bv), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.dot(b_a.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, t, n_heads, d_k, d_v = *q.shape, v.shape[-1]
    bt = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, bt) if cu_seqlens is not None else None
    nt = triton.cdiv(t, bt) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    launch = _ChunkFwdKernelOLaunch(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        t=t,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        bt=bt,
    )

    def grid(meta): return (triton.cdiv(d_v, meta['bv']), nt, B * n_heads)
    _launch_chunk_fwd_kernel_o(grid, launch)
    return o


def chunk_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    A: torch.Tensor | None = None,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, t, n_heads, d_k, d_v = *k.shape, do.shape[-1]
    bt = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, bt)
    # H100 can have larger block size
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem:
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    bk = min(max(triton.next_power_of_2(d_k), 16), CONST_TILING)
    bv = min(max(triton.next_power_of_2(d_v), 16), CONST_TILING)
    nt = triton.cdiv(t, bt) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(do)
    grid = (nt, B * n_heads)
    chunk_bwd_kernel_dv_local[grid](
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        A=A,
        do=do,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        t=t,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        bt=bt,
        bk=bk,
        bv=bv,
    )
    return dv


def chunk_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, t, n_heads, d_k, d_v = *k.shape, v.shape[-1]
    bt = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, bt) if cu_seqlens is not None else None
    nt = triton.cdiv(t, bt) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING = 64 if check_shared_mem() else 32
    bk = min(max(triton.next_power_of_2(d_k), 16), CONST_TILING)
    bv = min(max(triton.next_power_of_2(d_v), 16), CONST_TILING)
    NK = triton.cdiv(d_k, bk)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
    dw = torch.empty_like(w) if w is not None else None

    grid = (NK, nt, B * n_heads)
    chunk_bwd_kernel_dqkwg[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        do=do,
        dh=dh,
        dw=dw,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        t=t,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        bt=bt,
        bk=bk,
        bv=bv,
    )

    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw, dg