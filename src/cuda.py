# reference : https://raw.githubusercontent.com/tinygrad/tinygrad/refs/heads/master/tinygrad/runtime/autogen/cuda.py
import ctypes,ctypes.util
from src.__struct import Structure

cudaError_enum__enumvalues = {
  0: 'CUDA_SUCCESS',
  1: 'CUDA_ERROR_INVALID_VALUE',
  2: 'CUDA_ERROR_OUT_OF_MEMORY',
  3: 'CUDA_ERROR_NOT_INITIALIZED',
  4: 'CUDA_ERROR_DEINITIALIZED',
  5: 'CUDA_ERROR_PROFILER_DISABLED',
  6: 'CUDA_ERROR_PROFILER_NOT_INITIALIZED',
  7: 'CUDA_ERROR_PROFILER_ALREADY_STARTED',
  8: 'CUDA_ERROR_PROFILER_ALREADY_STOPPED',
  34: 'CUDA_ERROR_STUB_LIBRARY',
  100: 'CUDA_ERROR_NO_DEVICE',
  101: 'CUDA_ERROR_INVALID_DEVICE',
  102: 'CUDA_ERROR_DEVICE_NOT_LICENSED',
  200: 'CUDA_ERROR_INVALID_IMAGE',
  201: 'CUDA_ERROR_INVALID_CONTEXT',
  202: 'CUDA_ERROR_CONTEXT_ALREADY_CURRENT',
  205: 'CUDA_ERROR_MAP_FAILED',
  206: 'CUDA_ERROR_UNMAP_FAILED',
  207: 'CUDA_ERROR_ARRAY_IS_MAPPED',
  208: 'CUDA_ERROR_ALREADY_MAPPED',
  209: 'CUDA_ERROR_NO_BINARY_FOR_GPU',
  210: 'CUDA_ERROR_ALREADY_ACQUIRED',
  211: 'CUDA_ERROR_NOT_MAPPED',
  212: 'CUDA_ERROR_NOT_MAPPED_AS_ARRAY',
  213: 'CUDA_ERROR_NOT_MAPPED_AS_POINTER',
  214: 'CUDA_ERROR_ECC_UNCORRECTABLE',
  215: 'CUDA_ERROR_UNSUPPORTED_LIMIT',
  216: 'CUDA_ERROR_CONTEXT_ALREADY_IN_USE',
  217: 'CUDA_ERROR_PEER_ACCESS_UNSUPPORTED',
  218: 'CUDA_ERROR_INVALID_PTX',
  219: 'CUDA_ERROR_INVALID_GRAPHICS_CONTEXT',
  220: 'CUDA_ERROR_NVLINK_UNCORRECTABLE',
  221: 'CUDA_ERROR_JIT_COMPILER_NOT_FOUND',
  222: 'CUDA_ERROR_UNSUPPORTED_PTX_VERSION',
  223: 'CUDA_ERROR_JIT_COMPILATION_DISABLED',
  224: 'CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY',
  300: 'CUDA_ERROR_INVALID_SOURCE',
  301: 'CUDA_ERROR_FILE_NOT_FOUND',
  302: 'CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND',
  303: 'CUDA_ERROR_SHARED_OBJECT_INIT_FAILED',
  304: 'CUDA_ERROR_OPERATING_SYSTEM',
  400: 'CUDA_ERROR_INVALID_HANDLE',
  401: 'CUDA_ERROR_ILLEGAL_STATE',
  500: 'CUDA_ERROR_NOT_FOUND',
  600: 'CUDA_ERROR_NOT_READY',
  700: 'CUDA_ERROR_ILLEGAL_ADDRESS',
  701: 'CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES',
  702: 'CUDA_ERROR_LAUNCH_TIMEOUT',
  703: 'CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING',
  704: 'CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED',
  705: 'CUDA_ERROR_PEER_ACCESS_NOT_ENABLED',
  708: 'CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE',
  709: 'CUDA_ERROR_CONTEXT_IS_DESTROYED',
  710: 'CUDA_ERROR_ASSERT',
  711: 'CUDA_ERROR_TOO_MANY_PEERS',
  712: 'CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED',
  713: 'CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED',
  714: 'CUDA_ERROR_HARDWARE_STACK_ERROR',
  715: 'CUDA_ERROR_ILLEGAL_INSTRUCTION',
  716: 'CUDA_ERROR_MISALIGNED_ADDRESS',
  717: 'CUDA_ERROR_INVALID_ADDRESS_SPACE',
  718: 'CUDA_ERROR_INVALID_PC',
  719: 'CUDA_ERROR_LAUNCH_FAILED',
  720: 'CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE',
  800: 'CUDA_ERROR_NOT_PERMITTED',
  801: 'CUDA_ERROR_NOT_SUPPORTED',
  802: 'CUDA_ERROR_SYSTEM_NOT_READY',
  803: 'CUDA_ERROR_SYSTEM_DRIVER_MISMATCH',
  804: 'CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE',
  805: 'CUDA_ERROR_MPS_CONNECTION_FAILED',
  806: 'CUDA_ERROR_MPS_RPC_FAILURE',
  807: 'CUDA_ERROR_MPS_SERVER_NOT_READY',
  808: 'CUDA_ERROR_MPS_MAX_CLIENTS_REACHED',
  809: 'CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED',
  900: 'CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED',
  901: 'CUDA_ERROR_STREAM_CAPTURE_INVALIDATED',
  902: 'CUDA_ERROR_STREAM_CAPTURE_MERGE',
  903: 'CUDA_ERROR_STREAM_CAPTURE_UNMATCHED',
  904: 'CUDA_ERROR_STREAM_CAPTURE_UNJOINED',
  905: 'CUDA_ERROR_STREAM_CAPTURE_ISOLATION',
  906: 'CUDA_ERROR_STREAM_CAPTURE_IMPLICIT',
  907: 'CUDA_ERROR_CAPTURED_EVENT',
  908: 'CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD',
  909: 'CUDA_ERROR_TIMEOUT',
  910: 'CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE',
  911: 'CUDA_ERROR_EXTERNAL_DEVICE',
  999: 'CUDA_ERROR_UNKNOWN',
}

_libraries = {}
_libraries['libcuda.so'] = ctypes.CDLL("libcuda.so")

CUresult = ctypes.c_uint32

CUdevice = ctypes.c_int32
CUdevice_attribute_enum = ctypes.c_uint32 # enum
CUdevice_attribute = CUdevice_attribute_enum

CUdeviceptr = ctypes.c_uint64

size_t = ctypes.c_uint64


class struct_CUstream_st(Structure): pass
CUstream = ctypes.POINTER(struct_CUstream_st)

class struct_CUctx_st(Structure):pass
CUcontext = ctypes.POINTER(struct_CUctx_st)

class struct_CUmod_st(Structure):pass
CUmodule = ctypes.POINTER(struct_CUmod_st)

class struct_CUfunc_st(Structure):pass
CUfunction = ctypes.POINTER(struct_CUfunc_st)

class struct_CUstream_st(Structure):pass
CUstream = ctypes.POINTER(struct_CUstream_st)


try:
  cuInit = _libraries['libcuda.so'].cuInit
  cuInit.restype = CUresult
  cuInit.argtypes = [ctypes.c_uint32]
except AttributeError: pass
  
try:
  cuInit = _libraries['libcuda.so'].cuInit
  cuInit.restype = CUresult
  cuInit.argtypes = [ctypes.c_uint32]
except AttributeError: pass

try:
  cuDeviceGet = _libraries['libcuda.so'].cuDeviceGet
  cuDeviceGet.restype = CUresult
  cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.c_int32]
except AttributeError:pass

try:
  cuDeviceGetCount = _libraries['libcuda.so'].cuDeviceGetCount
  cuDeviceGetCount.restype = CUresult
  cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:pass

try:
  cuDeviceGetAttribute = _libraries['libcuda.so'].cuDeviceGetAttribute
  cuDeviceGetAttribute.restype = CUresult
  cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int32), CUdevice_attribute, CUdevice]
except AttributeError: pass

try:
  cuDeviceGetName = _libraries['libcuda.so'].cuDeviceGetName
  cuDeviceGetName.restype = CUresult
  cuDeviceGetName.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, CUdevice]
except AttributeError: pass

try:
  cuDeviceTotalMem_v2 = _libraries['libcuda.so'].cuDeviceTotalMem_v2
  cuDeviceTotalMem_v2.restype = CUresult
  cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUdevice]
except AttributeError: pass

try:
  cuCtxCreate_v2 = _libraries['libcuda.so'].cuCtxCreate_v2
  cuCtxCreate_v2.restype = CUresult
  cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st)), ctypes.c_uint32, CUdevice]
except AttributeError: pass

try:
  cuCtxDestroy_v2 = _libraries['libcuda.so'].cuCtxDestroy_v2
  cuCtxDestroy_v2.restype = CUresult
  cuCtxDestroy_v2.argtypes = [CUcontext]
except AttributeError: pass

try:
  cuCtxSynchronize = _libraries['libcuda.so'].cuCtxSynchronize
  cuCtxSynchronize.restype = CUresult
  cuCtxSynchronize.argtypes = []
except AttributeError: pass

try:
  cuModuleLoadData = _libraries['libcuda.so'].cuModuleLoadData
  cuModuleLoadData.restype = CUresult
  cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmod_st)), ctypes.c_void_p]
except AttributeError: pass

try:
  cuModuleUnload = _libraries['libcuda.so'].cuModuleUnload
  cuModuleUnload.restype = CUresult
  cuModuleUnload.argtypes = [CUmodule]
except AttributeError: pass

try:
  cuModuleGetFunction = _libraries['libcuda.so'].cuModuleGetFunction
  cuModuleGetFunction.restype = CUresult
  cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUfunc_st)), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try:
  cuMemGetInfo_v2 = _libraries['libcuda.so'].cuMemGetInfo_v2
  cuMemGetInfo_v2.restype = CUresult
  cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError: pass

try:
  cuMemAlloc_v2 = _libraries['libcuda.so'].cuMemAlloc_v2
  cuMemAlloc_v2.restype = CUresult
  cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t]
except AttributeError: pass

try:
  cuMemFree_v2 = _libraries['libcuda.so'].cuMemFree_v2
  cuMemFree_v2.restype = CUresult
  cuMemFree_v2.argtypes = [CUdeviceptr]
except AttributeError: pass

try:
  cuMemcpyHtoD_v2 = _libraries['libcuda.so'].cuMemcpyHtoD_v2
  cuMemcpyHtoD_v2.restype = CUresult
  cuMemcpyHtoD_v2.argtypes = [CUdeviceptr, ctypes.c_void_p, size_t]
except AttributeError: pass

try:
  cuMemcpyDtoH_v2 = _libraries['libcuda.so'].cuMemcpyDtoH_v2
  cuMemcpyDtoH_v2.restype = CUresult
  cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, CUdeviceptr, size_t]
except AttributeError: pass

try:
  cuLaunchKernel = _libraries['libcuda.so'].cuLaunchKernel
  cuLaunchKernel.restype = CUresult
  cuLaunchKernel.argtypes = [CUfunction, 
                             ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, # gridDimX, gridDimY, gridDimz
                             ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, # blockDimX, blockDimY, blockDimZ
                             ctypes.c_uint32, # sharedMemBytes
                             CUstream, 
                             ctypes.POINTER(ctypes.POINTER(None)), # kernelParams -> LP_c_void_p
                             ctypes.POINTER(ctypes.POINTER(None))] # Extra
except AttributeError: pass

try:
  cuStreamCreate = _libraries['libcuda.so'].cuStreamCreate
  cuStreamCreate.restype = CUresult
  cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUstream_st)),
                            ctypes.c_uint32]
except AttributeError: pass

try:
    cuStreamDestroy_v2 = _libraries['libcuda.so'].cuStreamDestroy_v2
    cuStreamDestroy_v2.restype = CUresult
    cuStreamDestroy_v2.argtypes = [CUstream]
except AttributeError: pass


mod_name = "cuda"


__all__ =\
[
  "cudaError_enum__enumvalues","CUresult","CUdevice",
  "CUdeviceptr","size_t","CUcontext","CUmodule","CUfunction",
  "CUstream","cuInit","cuDeviceGet","cuDeviceGetCount","cuDeviceGetAttribute",
  "cuDeviceGetName","cuDeviceTotalMem_v2","cuCtxCreate_v2","cuCtxDestroy_v2",
  "cuCtxSynchronize","cuModuleLoadData","cuModuleUnload","cuModuleGetFunction","cuMemGetInfo_v2",
  "cuMemAlloc_v2","cuMemFree_v2","cuMemcpyHtoD_v2","cuMemcpyDtoH_v2","mod_name",
  "cuLaunchKernel","cuStreamCreate","cuStreamDestroy_v2"

]
