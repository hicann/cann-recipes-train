# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch_npu
import torch.nn as nn
import triton
import triton.language as tl

from tla.utils import IS_AMD, autotune_cache_kwargs, input_guard

BLOCK_T_LIST = [8, 16, 32, 64, 128]
NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if IS_AMD else [1, 2, 4, 8, 16, 32]


@dataclass(frozen=True, slots=True)
class _L2NormKernel1LaunchFwd:
    x: torch.Tensor
    y: torch.Tensor
    rstd: torch.Tensor
    eps: float
    feat_dim: int
    block_d: int


@dataclass(frozen=True, slots=True)
class _L2NormKernel1LaunchBwd:
    y: torch.Tensor
    rstd: torch.Tensor
    dy: torch.Tensor
    dx: torch.Tensor
    eps: float
    feat_dim: int
    block_d: int


@dataclass(frozen=True, slots=True)
class _L2NormRowBlockLaunchFwd:
    x: torch.Tensor
    y: torch.Tensor
    rstd: torch.Tensor
    eps: float
    num_rows: int
    feat_dim: int
    block_d: int
    num_row_blocks: int


@dataclass(frozen=True, slots=True)
class _L2NormRowBlockLaunchBwd:
    y: torch.Tensor
    rstd: torch.Tensor
    dy: torch.Tensor
    dx: torch.Tensor
    eps: float
    num_rows: int
    feat_dim: int
    block_d: int
    num_row_blocks: int


# Triton @triton.jit entrypoints must keep a flat parameter list (tensor pointers,
# runtime scalars, tl.constexpr layout). Related launch arguments are grouped in
# _L2Norm*Launch* dataclasses and passed through the helpers below (G.FNM.03).


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['feat_dim'],
    **autotune_cache_kwargs,
)
@triton.jit
def l2norm_fwd_kernel1(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    x,
    y,
    rstd,
    eps,
    feat_dim,
    block_d: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * feat_dim
    y += i_t * feat_dim
    # Compute mean and variance
    cols = tl.arange(0, block_d)
    mask = cols < feat_dim

    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)
    tl.store(rstd + i_t, b_rstd)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['feat_dim'],
    **autotune_cache_kwargs,
)
@triton.jit
def l2norm_bwd_kernel1(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    y,
    rstd,
    dy,
    dx,
    eps,
    feat_dim,
    block_d: tl.constexpr,
):
    i_t = tl.program_id(0)
    y += i_t * feat_dim
    dx += i_t * feat_dim
    dy += i_t * feat_dim

    cols = tl.arange(0, block_d)
    mask = cols < feat_dim
    b_y = tl.load(y + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + i_t).to(tl.float32)
    b_dy = tl.load(dy + cols, mask=mask, other=0.0).to(tl.float32)
    b_dx = b_dy * b_rstd - tl.sum(b_dy * b_y) * b_y * b_rstd
    tl.store(dx + cols, b_dx, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'block_t': block_t}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for block_t in BLOCK_T_LIST
    ],
    key=['feat_dim', 'num_row_blocks'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['num_rows'])
def l2norm_fwd_kernel(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    x,
    y,
    rstd,
    eps,
    num_rows,
    feat_dim: tl.constexpr,
    block_d: tl.constexpr,
    num_row_blocks: tl.constexpr,
    block_t: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(
        x, (num_rows, feat_dim), (feat_dim, 1), (i_t * block_t, 0), (block_t, block_d), (1, 0)
    )
    p_y = tl.make_block_ptr(
        y, (num_rows, feat_dim), (feat_dim, 1), (i_t * block_t, 0), (block_t, block_d), (1, 0)
    )
    p_rstd = tl.make_block_ptr(rstd, (num_rows,), (1,), (i_t * block_t,), (block_t,), (0,))

    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({'block_t': block_t}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for block_t in BLOCK_T_LIST
    ],
    key=['feat_dim', 'num_row_blocks'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['num_rows'])
def l2norm_bwd_kernel(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    y,
    rstd,
    dy,
    dx,
    eps,
    num_rows,
    feat_dim: tl.constexpr,
    block_d: tl.constexpr,
    num_row_blocks: tl.constexpr,
    block_t: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_y = tl.make_block_ptr(
        y, (num_rows, feat_dim), (feat_dim, 1), (i_t * block_t, 0), (block_t, block_d), (1, 0)
    )
    p_rstd = tl.make_block_ptr(rstd, (num_rows,), (1,), (i_t * block_t,), (block_t,), (0,))
    p_dy = tl.make_block_ptr(
        dy, (num_rows, feat_dim), (feat_dim, 1), (i_t * block_t, 0), (block_t, block_d), (1, 0)
    )
    p_dx = tl.make_block_ptr(
        dx, (num_rows, feat_dim), (feat_dim, 1), (i_t * block_t, 0), (block_t, block_d), (1, 0)
    )

    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = tl.load(p_rstd, boundary_check=(0,)).to(tl.float32)
    b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
    b_dx = b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


def _run_l2norm_fwd_kernel1(launch: _L2NormKernel1LaunchFwd) -> None:
    num_rows = launch.x.shape[0]
    l2norm_fwd_kernel1[(num_rows,)](
        x=launch.x,
        y=launch.y,
        rstd=launch.rstd,
        eps=launch.eps,
        feat_dim=launch.feat_dim,
        block_d=launch.block_d,
    )


def _run_l2norm_bwd_kernel1(launch: _L2NormKernel1LaunchBwd) -> None:
    num_rows = launch.y.shape[0]
    l2norm_bwd_kernel1[(num_rows,)](
        y=launch.y,
        rstd=launch.rstd,
        dy=launch.dy,
        dx=launch.dx,
        eps=launch.eps,
        feat_dim=launch.feat_dim,
        block_d=launch.block_d,
    )


def _launch_l2norm_fwd_row_kernel(
    launch: _L2NormRowBlockLaunchFwd,
    grid: Callable[..., tuple[int, ...]],
) -> None:
    l2norm_fwd_kernel[grid](
        x=launch.x,
        y=launch.y,
        rstd=launch.rstd,
        eps=launch.eps,
        num_rows=launch.num_rows,
        feat_dim=launch.feat_dim,
        block_d=launch.block_d,
        num_row_blocks=launch.num_row_blocks,
    )


def _launch_l2norm_bwd_row_kernel(
    launch: _L2NormRowBlockLaunchBwd,
    grid: Callable[..., tuple[int, ...]],
) -> None:
    l2norm_bwd_kernel[grid](
        y=launch.y,
        rstd=launch.rstd,
        dy=launch.dy,
        dx=launch.dx,
        eps=launch.eps,
        num_rows=launch.num_rows,
        feat_dim=launch.feat_dim,
        block_d=launch.block_d,
        num_row_blocks=launch.num_row_blocks,
    )


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    num_rows, feat_dim = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    block_d = min(MAX_FUSED_SIZE, triton.next_power_of_2(feat_dim))
    if feat_dim > block_d:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    rstd = torch.empty((num_rows,), dtype=torch.float32, device=x.device)
    if feat_dim <= 512:
        num_row_blocks = triton.cdiv(num_rows, 2048)
        def grid(meta): return (triton.cdiv(num_rows, meta['block_t']), )
        _launch_l2norm_fwd_row_kernel(
            _L2NormRowBlockLaunchFwd(
                x=x,
                y=y,
                rstd=rstd,
                eps=eps,
                num_rows=num_rows,
                feat_dim=feat_dim,
                block_d=block_d,
                num_row_blocks=num_row_blocks,
            ),
            grid,
        )
    else:
        _run_l2norm_fwd_kernel1(
            _L2NormKernel1LaunchFwd(
                x=x,
                y=y,
                rstd=rstd,
                eps=eps,
                feat_dim=feat_dim,
                block_d=block_d,
            )
        )
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
):
    y_shape_og = y.shape
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    assert dy.shape == y.shape
    # allocate output
    dx = torch.empty_like(y)
    num_rows, feat_dim = y.shape[0], y.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // y.element_size()
    block_d = min(MAX_FUSED_SIZE, triton.next_power_of_2(feat_dim))
    if feat_dim > block_d:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if feat_dim <= 512:
        num_row_blocks = triton.cdiv(num_rows, 2048)
        def grid(meta): return (triton.cdiv(num_rows, meta['block_t']), )
        _launch_l2norm_bwd_row_kernel(
            _L2NormRowBlockLaunchBwd(
                y=y,
                rstd=rstd,
                dy=dy,
                dx=dx,
                eps=eps,
                num_rows=num_rows,
                feat_dim=feat_dim,
                block_d=block_d,
                num_row_blocks=num_row_blocks,
            ),
            grid,
        )
    else:
        _run_l2norm_bwd_kernel1(
            _L2NormKernel1LaunchBwd(
                y=y,
                rstd=rstd,
                dy=dy,
                dx=dx,
                eps=eps,
                feat_dim=feat_dim,
                block_d=block_d,
            )
        )

    return dx.view(y_shape_og)


class L2NormFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        eps=1e-6,
        output_dtype=None,
    ):
        y, rstd = l2norm_fwd(x, eps, output_dtype)
        ctx.eps = eps
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(y, rstd)
        return y

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        y, rstd = ctx.saved_tensors
        dx = l2norm_bwd(y, rstd, dy, ctx.eps)
        return dx, None, None


def l2norm(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)


l2_norm = l2norm


class L2Norm(nn.Module):

    def __init__(
        self,
        eps: float = 1e-6,
        output_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x, self.eps, self.output_dtype)