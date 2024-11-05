# Requirements
  - CUDA Toolkit & Numpy

# Examples

```python
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
```

Output :
```
[15  3 12 ... 11 11  5]
```

Get List Devices:
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