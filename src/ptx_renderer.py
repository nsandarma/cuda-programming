import ctypes,os
from typing import List
from src import nvrtc
from src.helpers import check


class PTX:
  prefix = 'extern "C" __global__ void '
  open_params = "( "
  close_params = ") "
  open_body = "{ "
  close_body = "} "

  gen_dim_1D = "int idx = (blockDim.x * blockIdx.x) + threadIdx.x; "
  gen_dim_2D = "int row = blockIdx.y * blockDim.y + threadIdx.y; int col = blockIdx.x * blockDim.x + threadIdx.x; "

  gen_ops_1D = lambda op,operand: f"if (idx < N){{ c[idx] = a[idx] {op} {operand}; }} "
  gen_ops_2D = lambda op,operand: f"if (row < rows && col < cols) {{ c[row * cols + col] = a[row * cols + col] {op} {operand}; }} "
  # gen_ops_2D = lambda op: f"if (row < rows && col < cols) {{ c[row * cols + col] = a[row * cols + col] {op} b }} "

  gen_ops_2D_matmul = "if (row < A_rows && col < B_cols) { float Cvalue = 0.0;\
                                                           for (int e = 0; e < A_cols; ++e) {\
                                                           Cvalue += a[row * A_cols + e] * b[e * B_cols + col]; } \
                                                           c[row * B_cols + col] = Cvalue; } "

  gen_args_1D = lambda dtypes :f"({dtypes[0]} a, {dtypes[1]} b, {dtypes[2]} c,int N)"
  gen_args_2D = lambda dtypes :f"({dtypes[0]} a, {dtypes[1]} b, {dtypes[2]} c, int rows, int cols)"
  gen_args_2D_matmul = lambda dtypes :f"({dtypes[0]} a, {dtypes[1]} b, {dtypes[2]} c, int A_rows, int A_cols, int B_cols)"


  def __init__(self,op,dtypes,dim:int,b_scalar = False,arch:str="compute_50"):
    assert op in ["*","+","/","-","@"], "op is not support !"
    assert len(dtypes) == 3, "dtypes < 3"
    name = "vector_func" if dim == 1 else "matrix_func"
    
    if dim == 1:
      operand = "*b" if b_scalar else "b[idx]"
      args = PTX.gen_args_1D(dtypes)
      dim = PTX.gen_dim_1D
      ops = PTX.gen_ops_1D(op,operand)
    elif dim == 2:
      dim = PTX.gen_dim_2D
      if op == "@":
        args = PTX.gen_args_2D_matmul(dtypes)
        ops = PTX.gen_ops_2D_matmul
      else:
        args = PTX.gen_args_2D(dtypes)
        operand = "*b" if b_scalar else "b[row * cols + col]"
        ops = PTX.gen_ops_2D(op,operand)

    else:
      raise ValueError("dim > 2 is not support !")
    codegen = f"{PTX.prefix}{name}{args}{PTX.open_body}{dim}{ops}{PTX.close_body}"
    self.args = args
    self.ops = ops
    self.dim = dim
    self.codegen = codegen

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
    func_name = self.name.encode()
    prog = nvrtc.nvrtcProgram()
    src_module = self.codegen.encode()
    status = nvrtc.nvrtcCreateProgram(ctypes.byref(prog),src_module,
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

    check(nvrtc,nvrtc.nvrtcDestroyProgram(ctypes.byref(prog)))
    self.is_rendered = True

    return self

  def write(self,filename="examples"):
    assert self.is_rendered, "ptx is None"
    filename = f"{filename}.ptx" if "." not in filename else filename
    with open(filename,"w") as f:
      f.write(str(self))
    print("success!")
      

if __name__ == "__main__":
  dtypes = ["float*","float*","float*"]
  ptx = PTX(op="+",dtypes=dtypes,dim=1,b_scalar=False).render()
  print(ptx.ops)
  print(ptx.args)
