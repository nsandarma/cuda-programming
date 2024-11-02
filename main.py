import ctypes
import cuda
from helpers import device_info,get_device_arch,check,prepare_kernel_params
from ptx_renderer import PTX
import numpy as np

cuda.cuInit(0)

arch = get_device_arch(cuda)

ptx = PTX("mul",arch,"+",).render()

# create context
context = cuda.CUcontext()
flags = ctypes.c_uint32(0)
status = cuda.cuCtxCreate_v2(ctypes.byref(context),flags,0)
check(cuda,status)

# load module (ptx)
module_data = ptx.ptx
module = cuda.CUmodule()
status = cuda.cuModuleLoadData(module,module_data)
check(cuda,status)

func_name = ptx.name.encode()
function = cuda.CUfunction()
status = cuda.cuModuleGetFunction(function,module,func_name)
check(cuda,status)


#initialization Data & allocation memory device
n = 10
np.random.seed(42)
a = np.random.randint(1,10,size = (n,)).astype(np.uint64)
b = np.random.randint(1,20,size = (n,)).astype(np.uint64)
c = np.zeros((n,)).astype(np.uint64)

def launchKernel(a,b,c,n):
  d_a = ctypes.c_uint64()
  d_b = ctypes.c_uint64()
  d_c = ctypes.c_uint64()


  cuda.cuMemAlloc_v2(ctypes.byref(d_a), a.nbytes)
  cuda.cuMemAlloc_v2(ctypes.byref(d_b), b.nbytes)
  cuda.cuMemAlloc_v2(ctypes.byref(d_c), c.nbytes)

  status = cuda.cuMemcpyHtoD_v2(d_a, a.ctypes.data, a.nbytes)
  check(cuda,status)

  status = cuda.cuMemcpyHtoD_v2(d_b, b.ctypes.data, b.nbytes)
  check(cuda,status)


  kernelParams = prepare_kernel_params(ctypes.byref(d_a),ctypes.byref(d_b),ctypes.byref(d_c))

  threads_per_block = 256
  blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

  grid_dim = (blocks_per_grid, 1, 1)
  block_dim = (threads_per_block, 1, 1)
  status = cuda.cuLaunchKernel(
    function,
    grid_dim[0], grid_dim[1], grid_dim[2],
    block_dim[0], block_dim[1], block_dim[2],
    0, None, 
    kernelParams,None  # Kernel parameter
  )
  check(cuda,status)

  check(cuda,cuda.cuMemcpyDtoH_v2(c.ctypes.data, d_c, c.nbytes))

  status = cuda.cuMemFree_v2(d_a)
  check(cuda,status)
  status = cuda.cuMemFree_v2(d_b)
  check(cuda,status)
  cuda.cuMemFree_v2(d_c)
  check(cuda,status)

launchKernel(a,b,c,n)
print(a)
print(b)
print(a + b)
print(c)
status = cuda.cuCtxDestroy_v2(context)
check(cuda,status)