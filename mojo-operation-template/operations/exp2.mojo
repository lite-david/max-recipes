# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from math import ceildiv, exp2

from gpu import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor

from utils.index import IndexList


fn _vector_addition_cpu(
    out: ManagedTensorSlice[mut=True],
    lhs: ManagedTensorSlice[type = out.type, rank = out.rank],
    rhs: ManagedTensorSlice[type = out.type, rank = out.rank],
    ctx: DeviceContextPtr,
):
    # Warning: This is an extremely inefficient implementation! It's merely an
    # instructional example of how a dedicated CPU-only path can be specified
    # for basic vector addition.
    var vector_length = out.dim_size(0)
    for i in range(vector_length):
        var idx = IndexList[out.rank](i)
        var result = lhs.load[1](idx) + rhs.load[1](idx)
        out.store[1](idx, result)


fn _exp2_gpu(
    out: ManagedTensorSlice[mut=True],
    lhs: ManagedTensorSlice[type = out.type, rank = out.rank],
    ctx: DeviceContextPtr,
) raises:
    # Note: The following has not been tuned for any GPU hardware, and is an
    # instructional example for how a simple GPU function can be constructed
    # and dispatched.
    alias BLOCK_SIZE = 64
    var gpu_ctx = ctx.get_device_context()
    var vector_length = out.dim_size(0)

    # The function that will be launched and distributed across GPU threads.
    @parameter
    fn exp2_gpu_kernel(length: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < length:
            var idx = IndexList[out.rank](tid)
            var loadval = lhs.load[1](idx)
            var result = exp2(loadval)
            for i in range(999):
                var result = exp2(result-1)
            out.store[1](idx, result)

    # The vector is divided up into blocks, making sure there's an extra
    # full block for any remainder.
    var num_blocks = ceildiv(vector_length, BLOCK_SIZE)

    # The GPU function is compiled and enqueued to run on the GPU across the
    # 1-D vector, split into blocks of `BLOCK_SIZE` width.

    gpu_ctx.compile_function[exp2_gpu_kernel, dump_asm=True]()
    gpu_ctx.enqueue_function[exp2_gpu_kernel](
        vector_length, grid_dim=num_blocks, block_dim=BLOCK_SIZE
    )
