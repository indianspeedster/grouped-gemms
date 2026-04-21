# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared availability probe for the ROCm MXFP8 kernels.

The forward and backward kernel modules gate their triton imports on this
flag so the package can be imported on non-ROCm / triton-less machines
without exploding.
"""

import torch


def _is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


try:
    import triton  # noqa: F401
    _triton_available = True
except ImportError:
    _triton_available = False

_rocm_mxfp8_available = _is_rocm() and _triton_available
