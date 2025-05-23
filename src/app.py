import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Any
from scipy.fft import dctn, idctn
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def to_disp(data: np.typing.NDArray[Any]) -> np.typing.NDArray[Any]:
  """
  Converts a data matrix into gray-scale for visualization.

  :param data: Data matrix.
  :type data: np.typing.NDArray[Any]
  :return: Gray-scale data matrix.
  :rtype: np.typing.NDArray[Any]
  """
  disp = np.log10(1 + data)
  disp *= 255 / disp.max() if disp.max() > 0 else 1
  return disp.astype(np.uint8)

def dct_pipeline_steps(img: np.typing.NDArray[Any], F: int, d_thr: int) -> tuple[list[np.typing.NDArray[Any] | tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]], list[str]]:
  """
  Builds the JPEG compression steps.

  :param img: Image to compress.
  :type img: np.typing.NDArray[Any]
  :param F: Block size.
  :type F: int
  :param d_thr: Threshold.
  :type d_thr: int
  :raises ValueError: _description_
  :return: JPEG compression steps and titles.
  :rtype: tuple[list[np.typing.NDArray[Any] | tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]], list[str]]
  """
  if img.ndim != 2:
    raise ValueError("Image must be gray-scale!")

  h, w = img.shape
  h2, w2 = h - h % F, w - w % F
  orig_crop = img[:h2, :w2]

  # Mask for k+ℓ < d_thr
  k_idx, l_idx = np.meshgrid(np.arange(F), np.arange(F), indexing="ij")
  block_mask = (k_idx + l_idx) < d_thr

  coeff_mag = np.zeros_like(orig_crop, dtype=float)
  coeff_masked_mag = np.zeros_like(orig_crop, dtype=float)
  idct_float = np.empty_like(orig_crop, dtype=float)

  for y in range(0, h2, F):
    for x in range(0, w2, F):
      patch = orig_crop[y : y + F, x : x + F].astype(float)
      # DCT
      c: np.typing.NDArray[Any] = dctn(patch, type=2, norm="ortho") # type: ignore
      coeff_mag[y : y + F, x : x + F] = np.abs(c)
      # Mask
      c_mask = c * block_mask
      coeff_masked_mag[y : y + F, x : x + F] = np.abs(c_mask)
      # IDCT
      rec_patch: np.typing.NDArray[Any] = idctn(c_mask, type=2, norm="ortho") # type: ignore
      idct_float[y : y + F, x : x + F] = np.clip(np.round(rec_patch), 0, 255).astype(np.uint8)

  images: list[np.typing.NDArray[Any] | tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]] = [
    img,
    orig_crop,
    to_disp(coeff_mag),
    to_disp(coeff_masked_mag),
    idct_float,
    (orig_crop, idct_float)
  ]

  titles = [
    "Original image",
    "Cropped image",
    "|DCT| (log₁₀)",
    f"Mask k+ℓ ≥ {d_thr}",
    "IDCT (round & clip 0-255)",
    "Original vs Compressed images"
  ]

  return images, titles

class DCT2App(tk.Tk):
  """
  Application for compressing .bmp (gray-scale) images with a JPEG-like compression.
  """
  def __init__(self) -> None:
    super().__init__()
    # Initialize TKinter stuff.
    self.title("JPEG-like compression step-by-step")
    self.geometry("1080x720")
    self.minsize(720, 480)
    self.protocol("WM_DELETE_WINDOW", self.on_close)
    # Initialize data stuff.
    self.image_path: Path | None = None
    self.img_orig: np.typing.NDArray[Any] | None = None
    self.step_imgs: list[np.typing.NDArray[Any] | tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]] = []
    self.step_titles: list[str] = []
    self.step_idx: int = 0
    # Build widgets.
    self._build_widgets()

  def _build_widgets(self) -> None:
    ctrl = ttk.Frame(self)
    ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    ttk.Button(ctrl, text="Load image…", command=self.load_image).pack(side=tk.LEFT)

    ttk.Label(ctrl, text="Block F:").pack(side=tk.LEFT, padx=(15, 2))
    self.var_block = tk.IntVar(value=8)
    ttk.Spinbox(ctrl, from_=2, to=128, increment=1, width=5, textvariable=self.var_block).pack(side=tk.LEFT)

    ttk.Label(ctrl, text="Threshold d:").pack(side=tk.LEFT, padx=(15, 2))
    self.var_d = tk.IntVar(value=10)
    ttk.Spinbox(ctrl, from_=1, to=256, increment=1, width=5, textvariable=self.var_d).pack(side=tk.LEFT)

    ttk.Button(ctrl, text="Compress", command=self.compress_and_show).pack(side=tk.LEFT, padx=15)

    self.prev_btn = ttk.Button(ctrl, text="◀", command=self.prev_step, state=tk.DISABLED)
    self.prev_btn.pack(side=tk.LEFT)
    self.next_btn = ttk.Button(ctrl, text="▶", command=self.next_step, state=tk.DISABLED)
    self.next_btn.pack(side=tk.LEFT)
    self.step_label = ttk.Label(ctrl, text="")
    self.step_label.pack(side=tk.LEFT, padx=5)

    # figure matplotlib
    self.fig, self.ax = plt.subplots()
    self.ax.axis("off")
    self.canvas = FigureCanvasTkAgg(self.fig, master=self)
    self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

  def load_image(self) -> None:
    filename = filedialog.askopenfilename(
      title="Select an image (BMP/PNG/JPG)",
      filetypes=[("Images", "*.bmp;*.png;*.jpg;*.jpeg"), ("All files", "*.*")],
    )
    if filename:
      self.image_path = Path(filename)
      img = Image.open(filename).convert("L")
      self.img_orig = np.array(img)
      self._reset_steps([self.img_orig], ["Original image"])

  def compress_and_show(self) -> None:
    if self.img_orig is None:
      messagebox.showwarning("No image", "You must first load an image.")
      return

    F = self.var_block.get()
    dthr = self.var_d.get()
    if F < 2:
      messagebox.showerror("Block not valid", "F must be ≥ 2.")
      return
    if dthr < 1:
      messagebox.showerror("Parameter d not valid", "d must ≥ 1.")
      return

    try:
      imgs, titles = dct_pipeline_steps(self.img_orig, F, dthr)
    except Exception as exc:
      messagebox.showerror("Compression error", str(exc))
      return

    self._reset_steps(imgs, titles)

  def _reset_steps(self, imgs: list[np.typing.NDArray[Any] | tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]], titles: list[str]) -> None:
    self.step_imgs = imgs
    self.step_titles = titles
    self.step_idx = 0
    self._update_nav_buttons()
    self._show_current_step()

  def prev_step(self) -> None:
    if self.step_idx > 0:
      self.step_idx -= 1
      self._update_nav_buttons()
      self._show_current_step()

  def next_step(self) -> None:
    if self.step_idx < len(self.step_imgs) - 1:
      self.step_idx += 1
      self._update_nav_buttons()
      self._show_current_step()

  def _show_current_step(self) -> None:
    self.fig.clf()
    img_obj = self.step_imgs[self.step_idx]
    title = self.step_titles[self.step_idx]

    if isinstance(img_obj, tuple):
      ax1 = self.fig.add_subplot(1, 2, 1)
      ax2 = self.fig.add_subplot(1, 2, 2)
      ax1.imshow(img_obj[0], cmap="gray", vmin=0, vmax=255)
      ax2.imshow(img_obj[1], cmap="gray", vmin=0, vmax=255)
      ax1.set_title("Original")
      ax2.set_title("Compressed")
      for ax in (ax1, ax2):
        ax.axis("off")
    else:
      ax = self.fig.add_subplot(1, 1, 1)
      ax.imshow(img_obj, cmap="gray", vmin=0, vmax=255)
      ax.axis("off")
      ax.set_title(title)

    self.fig.tight_layout()
    self.canvas.draw()
    self.step_label.config(text=f"Step {self.step_idx} / {len(self.step_imgs) - 1}")

  def _update_nav_buttons(self) -> None:
    if not self.step_imgs:
      self.prev_btn.config(state=tk.DISABLED)
      self.next_btn.config(state=tk.DISABLED)
      return
    self.prev_btn.config(state=tk.NORMAL if self.step_idx > 0 else tk.DISABLED)
    self.next_btn.config(state=tk.NORMAL if self.step_idx < len(self.step_imgs) - 1 else tk.DISABLED)

  def on_close(self) -> None:
    """
    Performs closing operations to quit the application properly.
    """
    plt.close("all")
    self.destroy()
