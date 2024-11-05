from ops_cuda import CUDA
import numpy as np

n = 1024
a = np.random.randint(1,10,size=(n,)).astype(np.int32)
b = np.random.randint(1,10,size=(n,)).astype(np.int32)
a_cuda = CUDA(a)
b_cuda = CUDA(b)

# vector addition
result = a_cuda + b_cuda

print(result)