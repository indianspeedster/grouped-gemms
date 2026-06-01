##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

"""ROCm microscaling grouped-GEMM kernels (gfx950+).

Use ``tl.dot_scaled`` to consume per-block e8m0 scales directly as a stand-in
for ``torch._scaled_grouped_mm``'s MX path until that ships on ROCm.

Subpackages:
  - ``kernels.mxfp8`` — MXFP8 (e4m3) forward+dgrad and weight-gradient kernels
  - ``kernels.mxfp4`` — MXFP4 (e2m1) forward+dgrad kernel

Public entry points (re-exported here for convenience):
  - ``triton_mxfp8_grouped_mm``: MXFP8 forward + dgrad (A @ B^T per group)
  - ``triton_mxfp8_wgrad``:      MXFP8 weight gradient (A^T @ B per group)
  - ``triton_mxfp4_grouped_mm``: MXFP4 forward + dgrad (A @ B^T per group)
"""

from ._common import _rocm_mxfp8_available
from .mxfp4 import triton_mxfp4_grouped_mm
from .mxfp8 import triton_mxfp8_grouped_mm, triton_mxfp8_wgrad

__all__ = [
    "triton_mxfp8_grouped_mm",
    "triton_mxfp8_wgrad",
    "triton_mxfp4_grouped_mm",
    "_rocm_mxfp8_available",
]
