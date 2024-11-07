import unittest
import numpy as np
from src import CUDA

class TestVectors(unittest.TestCase):

  def setUp(self):
    N = np.random.randint(100,1000)
    x = np.random.randn(N) + 100
    y = np.random.randn(N) + 100
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
    r1 = self.xc / self.yc
    r2 = self.x / self.y
    np.testing.assert_allclose(r1,r2)

  def test_floordivision(self):
    r1 = self.xc // self.yc
    r2 = self.x // self.y
    np.testing.assert_allclose(r1,r2)

  def test_multiplication(self):
    r1 = self.xc * self.yc
    r2 = self.x * self.y
    np.testing.assert_allclose(r1,r2)

  def test_dot_product(self):
    r1 = self.xc @ self.yc
    r2 = self.x @ self.y
    np.testing.assert_allclose(r1,r2)

if __name__ == "__main__":
  unittest.main()


  

