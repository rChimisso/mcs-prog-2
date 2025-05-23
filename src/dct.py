import math
from typing import Any
import numpy as np
from scipy.fft import dctn
from timeit import repeat
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_dct_matrix(N: int) -> np.typing.NDArray[Any]:
	"""
	Return the N×N orthonormal DCT2 matrix D such that y = D @ x.

	:param N: Data size.
	:type N: int
	:return: N×N orthonormal DCT2 matrix D.
	:rtype: np.typing.NDArray[Any]
	"""
	alpha = np.full(N, math.sqrt(2 / N))
	alpha[0] = 1 / math.sqrt(N)
	k = np.arange(N).reshape(-1, 1) # Column vector.
	i = np.arange(N).reshape(1, -1) # Row vector.
	return alpha[:, None] * np.cos(k * math.pi * (2 * i + 1) / (2 * N))

def dct2_naive(data: np.typing.NDArray[Any]) -> np.typing.NDArray[Any]:
	"""
	Naive O(N³) 2D DCT2 using explicit matrix multiplication.

	:param data: Data matrix.
	:type data: np.typing.NDArray[Any]
	:return: Data matrix with DCT2 applied.
	:rtype: np.typing.NDArray[Any]
	"""
	D = compute_dct_matrix(data.shape[0])
	return D @ data @ D.T

def dct2_scipy(data: np.typing.NDArray[Any]) -> np.typing.NDArray[Any]:
	"""
	Fast O(N² log N) DCT2 using SciPy's implementation.

	:param data: Data.
	:type data: np.typing.NDArray[Any]
	:return: Data with DCT2 applied.
	:rtype: np.typing.NDArray[Any]
	"""
	return dctn(data, type=2, norm="ortho") # type: ignore

def benchmark(sizes: list[int] = [2**i for i in range(3, 13)]) -> pd.DataFrame:
	"""
	Compares Naive 2D DCT2 and SciPy's 2D DCT2 implementations.

	:param sizes: Matrix sizes, defaults to [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096].
	:type sizes: list[int], optional
	:return: DataFrame with execution times.
	:rtype: pd.DataFrame
	"""
	rng = np.random.default_rng(42)
	rows: list[dict[str, float]] = []
	for N in tqdm(sizes, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', ncols=80):
		x = rng.integers(0, 256, size=(N, N)).astype(float)
		t_naive = min(repeat(lambda: dct2_naive(x), repeat=3, number=5))
		t_fast = min(repeat(lambda: dct2_scipy(x), repeat=3, number=5))
		rows.append(dict(N=N, naive=t_naive, fast=t_fast))
	return pd.DataFrame(rows).set_index("N")

def plot(df: pd.DataFrame) -> Figure:
	"""
	Plots execution times (semi-logarithmic y-axis).

	:param df: DataFrame with data for naive and fast 2D DCT2 executions.
	:type df: pd.DataFrame
	:return: Plot with execution times as matrix size increases.
	:rtype: Figure
	"""
	figure, axis = plt.subplots()
	# Reference complexity lines
	N = np.array(df.index)
	n3_ref = (N ** 3) / (N ** 3).max() * df["naive"].max() # Normalize to naive.
	n2logn_ref = (N ** 2 * np.log2(N)) / (N ** 2 * np.log2(N)).max() * df["fast"].max() # Normalize to fast.
	axis.semilogy(N, n3_ref, linestyle="--", label="Ideal $O(N^3)$")
	axis.semilogy(N, n2logn_ref, linestyle="--", label="Ideal $O(N^2\\log N)$")
	# Plot actual execution times.
	axis.semilogy(df.index, df["naive"], marker="o", label="Naive $O(N^3)$")
	axis.semilogy(df.index, df["fast"], marker="o", label="Fast $O(N^2\\log N)$")
	# Labels and formatting.
	axis.set_xlabel("Matrix size $N$")
	axis.set_ylabel("Execution time [s]")
	axis.set_title("2D DCT execution time vs size")
	axis.grid(True, which="both", linestyle="--", linewidth=0.5)
	axis.legend()
	figure.tight_layout()
	return figure
