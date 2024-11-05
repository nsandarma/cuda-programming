import ctypes,os
import nvrtc
from helpers import check

def template1D(op:str,name:str) -> str:
  return f"""
  extern "C" __global__ 
    void {name}(int *a, int *b, int *c) {{
      int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
      c[idx] = a[idx] {op} b[idx];

  }}
  """
def _template1D(op:str,name:str,dtypes:list) -> str:
  assert len(dtypes) == 3, "error"
  return f"""
  extern "C" __global__ 
    void {name}({dtypes[0]} *a, {dtypes[1]} *b, {dtypes[2]} *c) {{
      int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
      c[idx] = a[idx] {op} b[idx];

  }}
  """

def template2D(op:str,name:str) -> str: 
  return ""

class PTX:
  def __init__(self,name,arch,op,dim=1):
    assert op in ["*","+","/","-"], "op is no support !"
    self.name = name
    self.arch = arch
    self.op = op
    self.dim = dim
    self.ptx = None
    self.is_rendered = False
    self.cuda_code = None
  
  def __str__(self):
    if self.is_rendered:
      return self.ptx.value.decode()
    return None
  
  def render(self):
    self.is_rendered = True
    func_name = self.name.encode()
    prog = nvrtc.nvrtcProgram()
    cuda_code = template1D(self.op,self.name)
    self.cuda_code = cuda_code
    status = nvrtc.nvrtcCreateProgram(ctypes.byref(prog),cuda_code.encode(),
                             func_name,0,None,None)
    check(nvrtc,status)

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

  def write(self,filename="examples"):
    assert self.is_rendered, "ptx is None"
    filename = f"{filename}.ptx" if "." not in filename else filename
    with open(filename,"w") as f:
      f.write(str(self))
    print("success!")
      
  
  



if __name__ == "__main__":
  ptx = PTX("vector_add","compute_50","*").render()
  ptx.write("vector_add.ptx")

