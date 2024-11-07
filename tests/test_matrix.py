import unittest
from src import CUDA
import numpy as np

class TestMatrix(unittest.TestCase):

  def setUp(self):
    N = np.random.randint(100,1000)
    x = np.random.randn(N,N)
    y = np.random.randn(N,N)

    self.x = x
    self.y = y
    self.xc = CUDA(x)
    self.yc = CUDA(y)

  def test_addition(self):
    r1 = self.xc + self.yc
    r2 = self.x + self.y
    np.testing.assert_allclose(r1,r2)

  def test_subtraction(self):
    r1 = self.xc - self.yc
    r2 = self.x - self.y
    np.testing.assert_allclose(r1,r2)

  def test_division(self):
    # task: reduce tolerance value
    r1 = self.xc / self.yc
    r2 = self.x / self.y
    np.testing.assert_allclose(r1,r2,rtol=1.15918535e-07) 

  def test_floordivision(self):
    # task: reduce tolerance value
    r1 = self.xc // self.yc
    r2 = self.x // self.y
    np.testing.assert_allclose(r1,r2,rtol=1)

  def test_multiplication_element_wise(self):
    r1 = self.xc * self.yc
    r2 = self.x * self.y
    np.testing.assert_allclose(r1,r2)

  def test_multiplication_dot_product(self):
    # task: reduce tolerance value
    r1 = self.xc @ self.yc
    r2 = self.x @ self.y
    np.testing.assert_allclose(r1,r2,rtol=1)
    
  def test_multiplication_dot_product_int(self):
    n = 500
    x = np.random.randint(10,100,size=(n,n))
    y = np.random.randint(10,100,size=(n,n))
    xc = CUDA(x)
    yc = CUDA(y)

    r1 = xc @ yc
    r2 = x @ y
    np.testing.assert_array_equal(r1,r2)

if __name__ == "__main__":
  unittest.main()

