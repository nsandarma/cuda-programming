from src import CUDA
import numpy as np
import time

n = 2042
a = np.random.randint(1,10,size=(n,n))
b = np.random.randint(1,10,size=(n,n))
a_cuda = CUDA(a)
b_cuda = CUDA(b)

# Matrix Multiplication

start = time.monotonic()
result = a @ b
end = time.monotonic()
times = round(end-start,4)
print("times [numpy] : ",times)

start = time.monotonic()
result = a_cuda @ b_cuda
end = time.monotonic()
times = round(end-start,4)
print("times [cuda] : ",times)
