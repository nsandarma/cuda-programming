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
  # Mendapatkan compute capability (major dan minor) untuk GPU tersebut
  major = ctypes.c_int()
  minor = ctypes.c_int()
  status = cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, 0)  # 75 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
  status = cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, 0)  # 76 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
  return f"compute_{major.value}{minor.value}"


def get_grid_dim(n,dim= 1,threadsPerBlock=256):
  bd : list[int,int,int] = [1,1,1] # block dim
  td : list[int,int,int] = [threadsPerBlock,1,1] # thread dim
  numBlocks = int((n + threadsPerBlock - 1) / threadsPerBlock)
  if dim == 1:
    bd[0] = numBlocks
    return bd,td
  elif dim == 2:
    numBlocks = int(numBlocks ** 0.5)
    threadsPerBlock = int(threadsPerBlock ** 0.5)
    bd[0] = numBlocks
    bd[1] = numBlocks
    td[0] = threadsPerBlock
    td[1] = threadsPerBlock
    return bd,td
  else:
    numBlocks = int(numBlocks ** (1/3))
    threadsPerBlock = int(threadsPerBlock ** (1/3))
    bd = [numBlocks for i in range(3)]
    td = [threadsPerBlock for i in range(3)]
    return bd,td

def prepare_kernel_params(*args):
  # Mengonversi setiap parameter menjadi `c_void_p` dan menyimpannya dalam array
  params_array = (ctypes.c_void_p * len(args))\
  (*(ctypes.cast(arg, ctypes.c_void_p) for arg in args))
  # Mengembalikan sebagai `ctypes.POINTER(ctypes.POINTER(None))`
  return ctypes.cast(params_array, ctypes.POINTER(ctypes.POINTER(None)))

def view_kernel_params(params,n):
  # for i in range(n):
  value = ctypes.cast(params[3], ctypes.POINTER(ctypes.c_int))
  #   print(value)

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




if __name__ == "__main__":
  import cuda
  cuda.cuInit(0)
  device_info(cuda)




