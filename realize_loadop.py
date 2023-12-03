from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps

t0 = Tensor([2.0], requires_grad=True, device="LLVM")
assert t0.lazydata.optype == LoadOps
assert t0.lazydata.op.op == LoadOps.FROM
# Also works without e.g. `device="GPU"` if launched with `GPU=1 python ...`
t0.lazydata.realize()
