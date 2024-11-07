# reference : https://raw.githubusercontent.com/tinygrad/tinygrad/refs/heads/master/tinygrad/runtime/autogen/nvrtc.py

import ctypes,ctypes.util
from src.__struct import Structure

c__EA_nvrtcResult__enumvalues = {
  0: 'NVRTC_SUCCESS',
  1: 'NVRTC_ERROR_OUT_OF_MEMORY',
  2: 'NVRTC_ERROR_PROGRAM_CREATION_FAILURE',
  3: 'NVRTC_ERROR_INVALID_INPUT',
  4: 'NVRTC_ERROR_INVALID_PROGRAM',
  5: 'NVRTC_ERROR_INVALID_OPTION',
  6: 'NVRTC_ERROR_COMPILATION',
  7: 'NVRTC_ERROR_BUILTIN_OPERATION_FAILURE',
  8: 'NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION',
  9: 'NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION',
  10: 'NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID',
  11: 'NVRTC_ERROR_INTERNAL_ERROR',
  12: 'NVRTC_ERROR_TIME_FILE_WRITE_FAILED',
}
nvrtcResult_enum = c__EA_nvrtcResult__enumvalues
c__EA_nvrtcResult = ctypes.c_uint32 # enum
nvrtcResult = c__EA_nvrtcResult

_libraries = {}
_libraries["libnvrtc.so"] = ctypes.CDLL(ctypes.util.find_library('nvrtc'))
try:
  nvrtcVersion = _libraries['libnvrtc.so'].nvrtcVersion
  nvrtcVersion.restype = nvrtcResult
  nvrtcVersion.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError: pass

class struct__nvrtcProgram(Structure): pass

nvrtcProgram = ctypes.POINTER(struct__nvrtcProgram)
try:
  nvrtcCreateProgram = _libraries['libnvrtc.so'].nvrtcCreateProgram
  nvrtcCreateProgram.restype = nvrtcResult
  nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.POINTER(struct__nvrtcProgram)), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try:
  nvrtcDestroyProgram = _libraries['libnvrtc.so'].nvrtcDestroyProgram
  nvrtcDestroyProgram.restype = nvrtcResult
  nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.POINTER(struct__nvrtcProgram))]
except AttributeError: pass

try:
  nvrtcCompileProgram = _libraries['libnvrtc.so'].nvrtcCompileProgram
  nvrtcCompileProgram.restype = nvrtcResult
  nvrtcCompileProgram.argtypes = [nvrtcProgram, ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try:
  nvrtcGetPTXSize = _libraries['libnvrtc.so'].nvrtcGetPTXSize
  nvrtcGetPTXSize.restype = nvrtcResult
  nvrtcGetPTXSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError: pass

try:
  nvrtcGetPTX = _libraries['libnvrtc.so'].nvrtcGetPTX
  nvrtcGetPTX.restype = nvrtcResult
  nvrtcGetPTX.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try:
  nvrtcGetProgramLogSize = _libraries['libnvrtc.so'].nvrtcGetProgramLogSize
  nvrtcGetProgramLogSize.restype = nvrtcResult
  nvrtcGetProgramLogSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError: pass

try:
  nvrtcGetProgramLog = _libraries['libnvrtc.so'].nvrtcGetProgramLog
  nvrtcGetProgramLog.restype = nvrtcResult
  nvrtcGetProgramLog.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass


mod_name = "nvrtc"

__all__ = [ 
  "nvrtcResult_enum","nvrtcResult","nvrtcVersion",
  "nvrtcProgram","nvrtcCreateProgram", "nvrtcDestroyProgram",
  "nvrtcCompileProgram","nvrtcGetPTXSize","nvrtcGetPTX",
  "nvrtcProgram","nvrtcGetProgramLogSize","nvrtcGetProgramLog",
  "mod_name"
]
