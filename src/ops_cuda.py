import numpy as np
import src.cuda as cuda
from src.ptx_renderer import PTX
from src.helpers import check,prepare_kernel_params,grid_dim,ctype_to_c_string,ndtype_to_c_string
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
  
  def alloc_copy(self,a:np.ndarray,b:np.ndarray,c:np.ndarray):
    # Mengalokasikan dan menyalin memori ke device sekaligus
    d_a, d_b, d_d = c_uint64(), c_uint64(), c_uint64()

    check(cuda, cuda.cuMemAlloc_v2(byref(d_a), a.nbytes))
    check(cuda, cuda.cuMemAlloc_v2(byref(d_b), b.nbytes))
    check(cuda, cuda.cuMemAlloc_v2(byref(d_d), c.nbytes))

    check(cuda, cuda.cuMemcpyHtoD_v2(d_a, a.ctypes.data, a.nbytes))
    check(cuda, cuda.cuMemcpyHtoD_v2(d_b, b.ctypes.data, b.nbytes))
    check(cuda, cuda.cuMemcpyHtoD_v2(d_d, c.ctypes.data, c.nbytes))

    return d_a, d_b, d_d

  def copyHtoD(self,h,d): check(cuda,cuda.cuMemcpyDtoH_v2(d,h.ctypes.data,h.nbytes))

  def copyDtoH(self,d,h): check(cuda,cuda.cuMemcpyDtoH_v2(h.ctypes.data,d,h.nbytes))

  def free(self,*args):
    for arg in args: check(cuda,cuda.cuMemFree_v2(arg))
  
  def launchKernel(self,grid_block,*args):
    params = [byref(arg) for arg in args]
    kernelParams = prepare_kernel_params(*params)
    grid,block = grid_block
    check(cuda,cuda.cuLaunchKernel(
      self.function,
      grid[0],grid[1],grid[2],
      block[0],block[1],block[2],
      0,None,kernelParams,None
    ))
    check(cuda,cuda.cuCtxSynchronize())
  
  def cleanup(self, *device_ptrs):
    # Bebaskan resource perangkat
    for ptr in device_ptrs:
      check(cuda, cuda.cuMemFree_v2(ptr))
    check(cuda, cuda.cuModuleUnload(self.module))
    check(cuda, cuda.cuCtxDestroy_v2(self.context))


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

  def __check(self,other,op:str):
    assert type(other) == type(self), "type is not same !"
    if self.ndim == 1 :
      assert other.shape == self.shape, "shape is not same!"
    elif self.ndim == 2:
      if op !=  "@":
        assert other.shape == self.shape, "shape is not same!"
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

  def astype(self,dtype): return CUDA(self.data,dtype)


  def __realize(self,other,op,d,arch="compute_50"):
    dtypes = [ndtype_to_c_string(self.dtype),ndtype_to_c_string(other.dtype),ndtype_to_c_string(d.dtype)]
    ptx = PTX(op,dtypes=dtypes,dim=d.ndim,arch=arch).render()
    kernel = _Kernel(ptx)
    d_a,d_b,d_d = kernel.alloc_copy(self.data,other.data,d)

    if self.ndim == 1:
      extra = [c_int(len(self))]
    else:
      n_rows,n_cols = self.shape
      extra = [c_int(n_rows),c_int(n_cols),c_int(other.shape[1])] if op == "@" else [c_int(n_rows),c_int(n_cols)]

    grid_block = grid_dim(self.shape,other.shape)
    kernel.launchKernel(grid_block,d_a,d_b,d_d,*extra)
    kernel.copyDtoH(d_d,d)
    kernel.cleanup(d_a,d_b,d_d)
    
  
  # MATH OPS
  def add(self,other):
    self.__check(other,"+")
    d = np.zeros_like(self.data)
    self.__realize(other,"+",d)
    return d
  
  def sub(self,other):
    self.__check(other,"-")
    d = np.zeros_like(self.data)
    self.__realize(other,"-",d)
    return d
  
  def mul(self,other):
    self.__check(other,"*")
    d = np.zeros_like(self.data)
    self.__realize(other,"*",d)
    return d
  
  def matmul(self,other):
    self.__check(other,"@")
    d = np.zeros((self.shape[0],other.shape[1]),dtype=self.dtype)
    self.__realize(other,"@",d)
    return d
  
  def truediv(self,other):
    self.__check(other,"/")
    d = np.zeros_like(self.data).astype(np.float32)
    other = other.astype(np.float32)
    self.__realize(other,"/",d)
    return d

  def floordiv(self,other):
    self.__check(other,"/")
    d = np.zeros_like(self.data).astype(np.int32)
    self.__realize(other,"/",d)
    return d
  

  def __add__(self,other):return self.add(other)

  def __mul__(self,other): return self.mul(other)

  def __matmul__(self,other): return self.matmul(other)
  
  def __sub__(self,other): return self.sub(other)

  def __floordiv__(self,other): return self.floordiv(other)
  def __truediv__(self,other): return self.truediv(other)

  def __len__(self): return len(self.data)

