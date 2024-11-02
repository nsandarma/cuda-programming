import ctypes
import nvrtc
from helpers import check

def template1D(op:str,name:str) -> str:
  return f"""
  extern "C" __global__ 
    void {name}(int *a, int *b, int *c) {{
      int idx = threadIdx.x;
      c[idx] = a[idx] {op} b[idx];
  }}
  """


class PTX:
  def __init__(self,name,arch,op,dim=1):
    assert op in ["*","+","/","-"], "op is no support !"
    self.name = name
    self.arch = arch
    self.op = op
    self.dim = dim
    self.ptx = None
    self.is_rendered = False
  
  def __str__(self):
    if self.is_rendered:
      return self.ptx.value.decode()
    return None
  
  def render(self):
    self.is_rendered = True
    func_name = self.name.encode()
    prog = nvrtc.nvrtcProgram()
    cuda_code = template1D(self.op,self.name)
    nvrtc.nvrtcCreateProgram(ctypes.byref(prog),cuda_code.encode(),
                             func_name,0,None,None)
    arch  = f"--gpu-architecture={self.arch}"
    options = (ctypes.c_char_p * 1)(arch.encode())
    options = ctypes.cast(options, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))

    status = nvrtc.nvrtcCompileProgram(prog,1,options)
    check(nvrtc,status)

    ptx_size = ctypes.c_uint64()
    nvrtc.nvrtcGetPTXSize(prog,ctypes.byref(ptx_size))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    status = nvrtc.nvrtcGetPTX(prog,ptx)
    check(nvrtc,status)
    self.ptx = ptx
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    return self




    



    
