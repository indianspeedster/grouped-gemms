# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm MXFP8 grouped-GEMM kernels (gfx950+).

Use ``tl.dot_scaled`` to consume per-block e8m0 scales directly as a stand-in
for ``torch._scaled_grouped_mm``'s MXFP8 path until that ships on ROCm.

Public entry points:
  - ``triton_mxfp8_grouped_mm``: forward + dgrad grouped GEMM (A @ B^T per group)
  - ``triton_mxfp8_wgrad``: weight-gradient grouped GEMM (A^T @ B per group)
"""

from ._common import _rocm_mxfp8_available
from .backward import triton_mxfp8_wgrad
from .forward import triton_mxfp8_grouped_mm

__all__ = [
    "triton_mxfp8_grouped_mm",
    "triton_mxfp8_wgrad",
    "_rocm_mxfp8_available",
]
