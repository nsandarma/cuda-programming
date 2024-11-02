import ctypes

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


def prepare_kernel_params(d_a, d_b, d_c):
  # Membuat array dari pointer memori GPU
  params_array = (ctypes.c_void_p * 3)(
      ctypes.cast(d_a, ctypes.c_void_p),
      ctypes.cast(d_b, ctypes.c_void_p),
      ctypes.cast(d_c, ctypes.c_void_p)
  )

  # Mengkonversi array menjadi pointer ke void
  kernelParams = ctypes.cast(params_array, ctypes.POINTER(ctypes.c_void_p))
  return kernelParams
