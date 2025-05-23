import pytest
import numpy as np
from dct import compute_dct_matrix, dct2_naive, dct2_scipy

class TestDCT:
  TEST_ROW = [231, 32, 233, 161, 24, 71, 140, 245]
  EXPECTED_ROW = [4.01e2, 6.60e0, 1.09e2, -1.12e2, 6.54e1, 1.21e2, 1.16e2, 2.88e1]
  TEXT_MATRIX = [
    TEST_ROW,
    [247, 40, 248, 245, 124, 204, 36, 107],
    [234, 202, 245, 167, 9, 217, 239, 173],
    [193, 190, 100, 167, 43, 180, 8, 70],
    [11, 24, 210, 177, 81, 243, 8, 112],
    [97, 195, 203, 47, 125, 114, 165, 181],
    [193, 70, 174, 167, 41, 30, 127, 245],
    [87, 149, 57, 192, 65, 129, 178, 228]
  ]
  EXPECTED_MATRIX = [
    [1.11e3, 4.40e1, 7.59e1, -1.38e2, 3.50e0, 1.22e2, 1.95e2, -1.01e2],
    [7.71e1, 1.14e2, -2.18e1, 4.13e1, 8.77e0, 9.90e1, 1.38e2, 1.09e1],
    [4.48e1, -6.27e1, 1.11e2, -7.63e1, 1.24e2, 9.55e1, -3.98e1, 5.85e1],
    [-6.99e1, -4.02e1, -2.34e1, -7.67e1, 2.66e1, -3.68e1, 6.61e1, 1.25e2],
    [-1.09e2, -4.33e1, -5.55e1, 8.17e0, 3.02e1, -2.86e1, 2.44e0, -9.41e1],
    [-5.38e0, 5.66e1, 1.73e2, -3.54e1, 3.23e1, 3.34e1, -5.81e1, 1.90e1],
    [7.88e1, -6.45e1, 1.18e2, -1.50e1, -1.37e2, -3.06e1, -1.05e2, 3.98e1],
    [1.97e1, -7.81e1, 9.72e-1, -7.23e1, -2.15e1, 8.13e1, 6.37e1, 5.90e0]
  ]

  def test_scaling_naive(self) -> None:
    # 1D check
    row = np.array(TestDCT.TEST_ROW, dtype=float)
    expected_row = np.array(TestDCT.EXPECTED_ROW)
    got_row = compute_dct_matrix(8) @ row
    assert np.allclose(got_row, expected_row, rtol=1e-2, atol=1e-1), "1D Naive DCT2 check failed!"
    # 2D check
    block = np.array(TestDCT.TEXT_MATRIX, dtype=float)
    expected_block = np.array(TestDCT.EXPECTED_MATRIX)
    got_block = dct2_naive(block)
    assert np.allclose(got_block, expected_block, rtol=1e-2, atol=1e-1), "2D Naive DCT2 check failed!"

  def test_scaling_scipy(self) -> None:
    # 1D check
    row = np.array(TestDCT.TEST_ROW, dtype=float)
    expected_row = np.array(TestDCT.EXPECTED_ROW)
    got_row = dct2_scipy(row)
    assert np.allclose(got_row, expected_row, rtol=1e-2, atol=1e-1), "1D SciPy's DCT2 check failed!"
    # 2D check
    block = np.array(TestDCT.TEXT_MATRIX, dtype=float)
    expected_block = np.array(TestDCT.EXPECTED_MATRIX)
    got_block = dct2_scipy(block)
    assert np.allclose(got_block, expected_block, rtol=1e-2, atol=1e-1), "2D SciPy's DCT2 check failed!"

if __name__ == "__main__":
  pytest.main()
