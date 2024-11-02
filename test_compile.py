import ctypes
import nvrtc
from helpers import check

path = "./cuda-from-scratch/add.cu"
with open(path,"r",encoding="utf-8") as f:
  cuda_code = f.read().strip().encode()

func_name = b"add"
prog = nvrtc.nvrtcProgram()
nvrtc.nvrtcCreateProgram(ctypes.byref(prog),cuda_code,func_name,0,None,None)

options = (ctypes.c_char_p * 1)(b"--gpu-architecture=compute_50")
options_ptr = ctypes.cast(options, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))

res = nvrtc.nvrtcCompileProgram(prog,1,options_ptr)
print(res)
check(nvrtc,res)

ptx_size = ctypes.c_uint64()

nvrtc.nvrtcGetPTXSize(prog,ctypes.byref(ptx_size))
# Allocate memory for PTX code
ptx = ctypes.create_string_buffer(ptx_size.value)
nvrtc.nvrtcGetPTX(prog, ptx)
print(ptx.value.decode())

nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
