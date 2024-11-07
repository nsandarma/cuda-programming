# Requirements
  - [CUDA Toolkit](https://docs.nvidia.com/cuda/index.html) & [Numpy](https://pypi.org/project/numpy/)

# Examples

```python
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
```

Output :
```
times [numpy] :  21.3533
times [cuda] :  0.8445
```

Get Devices:
```python
import cuda
from helpers import device_info

cuda.cuInit(0)
device_info(cuda)
```
Output :
```
ID       | NAME                          | MEMORY        | ARCH
0        | NVIDIA GeForce MX130          | 2092892160    | compute_50
TOTAL DEVICE : 1
```
