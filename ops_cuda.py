import numpy as np
import cuda
from ptx_renderer import PTX
from helpers import check,prepare_kernel_params,get_grid_dim
from ctypes import *

class _Kernel:
  def __init__(self,ptx:PTX):
    check(cuda,cuda.cuInit(0))
    self.context = cuda.CUcontext()
    flags = c_uint32(0)
    check(cuda,cuda.cuCtxCreate_v2(byref(self.context),flags,0))

    module_data = ptx.ptx
    self.module = cuda.CUmodule()
    check(cuda,cuda.cuModuleLoadData(self.module,module_data))

    func_name = ptx.name.encode()
    self.function = cuda.CUfunction()
    check(cuda,cuda.cuModuleGetFunction(self.function,self.module,func_name))
  
  def alloc(self,a:np.ndarray,b:np.ndarray,c:np.ndarray):
    d_a = c_uint64()
    d_b = c_uint64()
    d_d = c_uint64()

    check(cuda,cuda.cuMemAlloc_v2(byref(d_a),a.nbytes))
    check(cuda,cuda.cuMemAlloc_v2(byref(d_b),b.nbytes))
    check(cuda,cuda.cuMemAlloc_v2(byref(d_d),c.nbytes))

    check(cuda,cuda.cuMemcpyHtoD_v2(d_a,a.ctypes.data,a.nbytes))
    check(cuda,cuda.cuMemcpyHtoD_v2(d_b,b.ctypes.data,b.nbytes))
    check(cuda,cuda.cuMemcpyHtoD_v2(d_d,c.ctypes.data,c.nbytes))

    return d_a,d_b,d_d
  
  def copyHtoD(self,h,d): check(cuda,cuda.cuMemcpyDtoH_v2(d,h.ctypes.data,h.nbytes))

  def copyDtoH(self,d,h): check(cuda,cuda.cuMemcpyDtoH_v2(h.ctypes.data,d,h.nbytes))

  def free(self,*args):
    for arg in args: check(cuda,cuda.cuMemFree_v2(arg))
  
  def launchKernel(self,d_a,d_b,d_d,grid_block):
    kernelParams = prepare_kernel_params(byref(d_a),byref(d_b),byref(d_d))
    grid,block = grid_block
    check(cuda,cuda.cuLaunchKernel(
      self.function,
      grid[0],grid[1],grid[2],
      block[0],block[1],block[2],
      0,None,kernelParams,None
    ))
    check(cuda,cuda.cuCtxSynchronize())
  
  def close(self,d_a,d_b,d_d):
    self.free(d_a,d_b,d_d)
    check(cuda,cuda.cuModuleUnload(self.module))
    check(cuda,cuda.cuCtxDestroy_v2(self.context))
    return 

class Ops:
  def __init__(self,a:np.ndarray,b:np.ndarray,ndim:int):
    self.a = a
    self.b = b
    if ndim == 1 :
      d = np.zeros_like(a)
      grid_block = get_grid_dim(d.shape[0],1)
    else: 
      d = np.zeros((a.shape[1],b.shape[0]),dtype=a.dtype)
      grid_block = get_grid_dim(d.shape[0],1)

    self.d = d
    self.grid_block = grid_block
    self.ndim = ndim


  def realize(self,op):
    ptx = PTX("func","compute_50",op,self.ndim).render()
    kernel = _Kernel(ptx)
    d_a,d_b,d_d = kernel.alloc(self.a,self.b,self.d)
    kernel.launchKernel(d_a,d_b,d_d,grid_block=self.grid_block)
    kernel.copyDtoH(d_d,self.d)
    kernel.close(d_a,d_b,d_d)
    return CUDA(self.d)
  


class CUDA:
  def __init__(self,data:np.ndarray,dtype=None):
    if dtype:
      data = data.astype(dtype)
    self.data = data
    self.device = "CUDA"

  def __str__(self):
    return str(self.data)

  def __repr__(self):
    return str(self)

  def __check(self,other,dim):
    assert type(other) == type(self), "type is not same !"
    if self.ndim == 1 :
      assert other.ndim == self.ndim, "ndim is not same !"
    elif self.ndim == 2:
      cols = self.shape[1]
      rows = other.shape[0]
      assert cols == rows, "A.shape[1] != B.shape[0]"
    else:
      raise ArgumentError("dim not support !")

  @property 
  def dtype(self): return self.data.dtype

  @property
  def ctype(self):
    return np.ctypeslib.as_ctypes_type(self.dtype)
  
  @property
  def ndim(self): return self.data.ndim

  @property
  def shape(self): return self.data.shape
  
  def numpy(self): return self.data

  
  # MATH OPS
  def add(self,other):
    self.__check(other,self.ndim)
    ops = Ops(self.data,other.data,ndim=self.ndim)
    return ops.realize("+")
  
  def sub(self,other):
    self.__check(other,self.ndim)
    ops = Ops(self.data,other.data,ndim=self.ndim)
    return ops.realize("-")
  
  def mul(self,other):
    self.__check(other,self.ndim)
    ops = Ops(self.data,other.data,ndim=self.ndim)
    return ops.realize("*")
  
  def matmul(self,other):
    self.__check(other,self.ndim)
    ops = Ops(self.data,other.data,ndim=self.ndim)
    return ops.realize("*")
  

  def __add__(self,other):return self.add(other)

  def __mul__(self,other): return self.mul(other)

  def __matmul__(self,other): return self.matmul(other)
  
  def __sub__(self,other): return self.sub(other)

  def __len__(self): return len(self.data)
  



if __name__ == "__main__":
  n = 1024
  a = np.random.randint(1,10,size=(n,)).astype(np.int32)
  b = np.random.randint(1,10,size=(n,)).astype(np.int32)
  a_cuda = CUDA(a)
  b_cuda = CUDA(b)

  # vector addition
  result = a_cuda + b_cuda

  print(result)



  

