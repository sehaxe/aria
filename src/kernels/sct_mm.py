"""Triton kernel: fused SCT matmul x @ (U * s) @ V^T.
Fuses the two matmuls + scaling into one kernel — saves HBM bandwidth.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sct_fwd_kernel(
    x_ptr, u_ptr, s_ptr, v_ptr, out_ptr,
    M, K, N, R,
    stride_x_m, stride_x_k,
    stride_u_k, stride_u_r,
    stride_v_n, stride_v_r,
    stride_o_m, stride_o_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_offs = tl.arange(0, BLOCK_R)
    mask_m = m_offs < M
    mask_n = n_offs < N
    mask_r = r_offs < R

    s = tl.load(s_ptr + r_offs, mask=mask_r, other=0.0).to(tl.float32)
    h = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offs < K
        x_tile = tl.load(x_ptr + m_offs[:, None] * stride_x_m + k_offs[None, :] * stride_x_k,
                         mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        u_tile = tl.load(u_ptr + k_offs[:, None] * stride_u_k + r_offs[None, :] * stride_u_r,
                         mask=mask_k[:, None] & mask_r[None, :], other=0.0)
        h += tl.dot(x_tile, u_tile)
    h_scaled = h * s[None, :]
    # v must be fp32 to match the fp32 accumulator (model runs in bf16).
    v_tile = tl.load(v_ptr + n_offs[:, None] * stride_v_n + r_offs[None, :] * stride_v_r,
                     mask=mask_n[:, None] & mask_r[None, :], other=0.0).to(tl.float32)
    out_tile = tl.dot(h_scaled, tl.trans(v_tile))
    tl.store(out_ptr + m_offs[:, None] * stride_o_m + n_offs[None, :] * stride_o_n,
             out_tile.to(OUT_DTYPE), mask=mask_m[:, None] & mask_n[None, :])
