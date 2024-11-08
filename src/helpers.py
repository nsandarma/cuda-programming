import ctypes
import numpy as np

def check(call,status) -> None:
  err_enum = call.cudaError_enum__enumvalues if call.mod_name == "cuda" else call.nvrtcResult_enum
  assert status == 0, err_enum[status]
  return 

def device_info(cuda) -> None:
  count = cuda.CUdevice()
  status = cuda.cuDeviceGetCount(ctypes.byref(count))
  check(cuda,status)
  print(f"ID \t | NAME \t\t\t | MEMORY  \t | ARCH")
  for i in range(count.value):
    device = cuda.CUdevice(i)
    name = (ctypes.c_char * 128)()
    cuda.cuDeviceGetName(name,256,device)
    mem = ctypes.c_uint64(0)
    cuda.cuDeviceTotalMem_v2(mem,device)
    arch = get_device_arch(cuda,i)
    print(f"{i} \t | {name.value.decode()} \t | {mem.value} \t | {arch}")
  print(f"TOTAL DEVICE : {count.value}")

def get_device_arch(cuda,device=0):
  major = ctypes.c_int()
  minor = ctypes.c_int()
  status = cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, 0)  # 75 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
  check(cuda,status)
  status = cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, 0)  # 76 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
  check(cuda,status)
  return f"compute_{major.value}{minor.value}"


def grid_dim(a_shape:tuple,b_shape:tuple,threadsPerBlock=256):
  # assert len(a_shape) == len(b_shape), "a_shape != b_shape"
  if len(a_shape) == 2:
    threadsPerBlock = 16
    bd = [threadsPerBlock,threadsPerBlock,1]
    grid_dim_x = (b_shape[1] + bd[0] - 1) // bd[0]
    grid_dim_y = (a_shape[0] + bd[1] - 1) // bd[1]
    gd = [grid_dim_x,grid_dim_y,1]
  else:
    bd = [threadsPerBlock,1,1]
    num_blocks = (a_shape[0] + threadsPerBlock -1) // threadsPerBlock
    gd = [num_blocks,1,1]
  return gd,bd


def prepare_kernel_params(*args):
  params_array = (ctypes.c_void_p * len(args))\
  (*(ctypes.cast(arg, ctypes.c_void_p) for arg in args))
  return ctypes.cast(params_array, ctypes.POINTER(ctypes.POINTER(None)))


def ctype_to_c_string(ctype):
  type_name = ctype.__name__
  # Dictionary mapping ctypes to C type strings
  type_mapping = {
    'c_int': 'int',
    'c_float': 'float',
    'c_double': 'double',
    'c_char': 'char',
    'c_char_p': 'char*',
    'c_void_p': 'void*',
    'c_short': 'short',
    'c_long': 'long',
    'c_longlong': 'long long',
    'c_ushort': 'unsigned short',
    'c_uint': 'unsigned int',
    'c_ulong': 'unsigned long',
    'c_ulonglong': 'unsigned long long',
    'c_bool': 'bool',
  }
  return type_mapping.get(type_name, "unknown type")

def ndtype_to_c_string(dtype):
  dtype = np.dtype(dtype)
  type_mapping = {
    np.int8: 'char',
    np.int16: 'short',
    np.int32: 'int',
    np.int64: 'long',
    np.uint8: 'unsigned char',
    np.uint16: 'unsigned short',
    np.uint32: 'unsigned int',
    np.uint64: 'unsigned long',
    np.float32: 'float',
    np.float64: 'double',
    np.bool_: 'bool',
    np.str_: 'char*',
    np.void: 'void*'
  }
  return type_mapping.get(dtype.type, "unknown type")

def dtype_to_c_string(dtype):
  dtype = np.dtype(dtype)
  type_mapping = {
    np.int8: 'char',
    np.int16: 'short',
    np.int32: 'int',
    np.int64: 'long',
    np.uint8: 'unsigned char',
    np.uint16: 'unsigned short',
    np.uint32: 'unsigned int',
    np.uint64: 'unsigned long',
    np.float32: 'float',
    np.float64: 'double',
  }
  return type_mapping.get(dtype.type, "unknown type") + "*"



if __name__ == "__main__":
  import cuda
  cuda.cuInit(0)
  device_info(cuda)
