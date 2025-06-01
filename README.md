# GrayDCT2

##### *2025, Metodi del Calcolo Scientifico, Riccardo Chimisso 866009 & Mauro Zorzin 866001*

## Description

A simple project made for our university course *Metodi del Calcolo Scientifico* that compares naive and fast DCT2 implementations as well as providing a simple application to apply JPEG-like compression to gray-scale `.bmp` images.

This project includes:

- **Documentation**: Extensive [Readme](/README.md), [Changelog](/CHANGELOG.md), and [Sphinx-generated codebase docs](https://rchimisso.github.io/mcs-prog-2/).
- **Report**: A [short report](/REPORT.md) with results and analysis.
- **Test data**: A few `.bmp` [example images](/data/).
- **CI**: Automatic code analysis and deployment.
- **Releases**: Prebuilt [executables](https://github.com/rChimisso/mcs-prog-2/releases) for Linux and Windows.

## Setup

Setting up the environment is pretty easy:

1. Set up **Python 3.12.9** (you can use any environment manager or none).
2. Install the dependencies from the file [`requirements.txt`](/requirements.txt).

The suggested IDE is [Visual Studio Code](https://code.visualstudio.com/), and settings for it are included.

## Documentation

The source code is fully documented with Docstrings in [reST](https://docutils.sourceforge.io/rst.html).  
Documentation for the latest release is already live at [rChimisso.github.io/mcs-prog-2](https://rchimisso.github.io/mcs-prog-21/).  

The structured documentation can be generated with [Sphinx](https://www.sphinx-doc.org/en/master/).  
To build the documentation yourself, simply run the following command under the `docs/` directory:
```powershell
make html
```
To view it, simply open the file `docs/build/html/index.html` with a browser.

## Usage

Available engine commands:

- `info`: Displays the identifier string of the engine.
- `help [command]`: Displays the list of available commands. If a command is specified, displays the help for that command.
- `dct`: Compares a naive implementation of the DCT2 to SciPy's implementation.
- `bmp`: Launches the application window to select and compress a .bmp image with JPEG compression type.
- `exit`: Exits the engine.

You can either use the prebuilt [executables](https://github.com/rChimisso/mcs-prog-2/releases) for your platform, or build it yourself.

To build the `EngineDCT2` executable yourself, simply run the following command in the project root:
```powershell
pyinstaller ./src/engine.py --name EngineDCT2 --onefile
```
This will create an executable for your platform.

## Background

Technical background notions behind the algorithms used in this project.

### Fourier Transform

The Fourier transform rests on the idea that any $\text{T-periodic}$ signal $f(t)$ can be rebuilt from harmonically related sines and cosines

$$
f(t)=a_{0}+\sum_{k=1}^{\infty}\Bigl[a_{k}\cos\!\Bigl(\tfrac{2\pi k}{T}\,t\Bigr)+b_{k}\sin\!\Bigl(\tfrac{2\pi k}{T}\,t\Bigr)\Bigr].
$$

Orthogonality of the trigonometric basis over one period makes the coefficients easy to compute:

$$
a_{0}= \frac1T\int_{0}^{T}f(t)\,dt,\qquad
a_{k}= \frac{2}{T}\int_{0}^{T}f(t)\cos\!\Bigl(\tfrac{2\pi k}{T}\,t\Bigr)dt,\qquad
b_{k}= \frac{2}{T}\int_{0}^{T}f(t)\sin\!\Bigl(\tfrac{2\pi k}{T}\,t\Bigr)dt.
$$

These formulas follow directly from the orthogonality relations
$\int_{0}^{T}\cos(\tfrac{2\pi k}{T}t)\cos(\tfrac{2\pi \ell}{T}t)dt=\tfrac{T}{2}\delta_{k\ell}$ and analogous ones for sine–cosine and sine–sine pairs.

### 1D Discrete Cosine Transform type 2 (DCT2)

In the discrete world a length-*N* signal is seen as a vector $x=(x_0,\dots,x_{N-1})$. The cosine basis vectors

$$
w_k(i)=\cos\!\Bigl(\frac{\pi k\,(2i+1)}{2N}\Bigr),\qquad
k=0,\dots,N-1,\; i=0,\dots,N-1
$$

form an orthogonal basis of $\mathbb R^{N}$.  
Projecting *x* onto this basis gives the DCT coefficients

$$
c_k=\begin{cases}
\displaystyle\frac1N\sum_{i=0}^{N-1}x_i\,w_k(i), & k=0\\
\displaystyle\frac{2}{N}\sum_{i=0}^{N-1}x_i\,w_k(i), & k\ge 1
\end{cases}
$$

**Key properties**

- Real orthogonal basis → no complex arithmetic;
- Energy compaction: most natural signals concentrate energy in the low-*k* coefficients, enabling compression.

### 2D DCT2

For an $N\times M$ block $A=[a_{ij}]$ the 2-D DCT is separable:

$$
\alpha_{sr}= \frac{2}{N}\frac{2}{M}
\sum_{i=0}^{N-1}\sum_{j=0}^{M-1}
a_{ij}\,
\cos\!\Bigl(\frac{\pi(2i+1)s}{2N}\Bigr)
\cos\!\Bigl(\frac{\pi(2j+1)r}{2M}\Bigr).
$$

Computationally this is just "DCT rows, then DCT columns" and costs $O(NM(N+M))$ without fast algorithms.  
$O(NM(N+M)) = O(N^3) \text{ if } N=M$; with FFT $O(N^2 \log N)$.

### Inverse DCT2 (IDCT2)

- **1D** - recovering samples from DCT coefficients:

  $$
  x_j=\sum_{k=0}^{N-1}\gamma_k\,c_k\,
  \cos\!\Bigl(\frac{\pi k\,(2j+1)}{2N}\Bigr),\qquad 
  \gamma_0=\tfrac12,\;\gamma_{k\ge1}=1.
  $$

- **2D** - apply the 1D IDCT to columns, then to rows (or vice-versa).

### JPEG-style Compression (custom F, d)

Standard JPEG uses $8 \times 8$ blocks, a quantisation matrix, zig-zag scanning and entropy coding.  
In our simpler variant we replace the quantisation matrix by two tunable parameters:

| Symbol | Meaning                                                                                   |
| ------ | ----------------------------------------------------------------------------------------- |
| $F$    | Block size (e.g. $8 \times 8$, $16 \times 16$, $...$)                                     |
| $d$    | Index threshold $d_{\text{thr}}$. Coefficients with row + col index $\ge d$ are discarded |

We apply the algorithm only to gray-scale images, but it could be used for RGB images too by applying it in parallel for each color channel.

#### Encoding algorithm

1. **Blocking** - Split the image into non-overlapping $F\times F$ tiles (discard any excess).

2. **Level shift** - Subtract $128$ from each pixel so that the average block value is near zero.

3. **2D DCT2** - Apply the DCT2 to every block.

4. **Thresholding** - For each block set

   $$
   \alpha_{ij}=0\qquad\text{if } i+j\ge d_{\text{thr}}.
   $$

   Low-frequency information (small $i + j$) is preserved; higher frequencies are thrown away, giving compression and suppressing Gibbs artefacts.

5. **Store** the surviving coefficients together with $F$ and $d$ (or simply keep them in memory for the project).

#### Decoding algorithm

1. Read $F$, $d$ and the surviving $\alpha_{ij}$.
2. Re-insert zeros where $i+j\ge d_{\text{thr}}$.
3. Apply the **2D IDCT2** to each block.
4. Add $128$ to every reconstructed sample, then round and clip to $[0,255]$.

Because the most perceptually important information sits in the lowest-frequency corner, even aggressive thresholds give visually pleasing results; increasing $d$ yields smaller files at the cost of blur and blockiness.
