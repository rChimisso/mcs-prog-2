import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Any, Optional
from scipy.fft import dctn, idctn
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def crop(img: np.typing.NDArray[Any], F: int) -> tuple[np.typing.NDArray[Any], int, int]:
  """
  Crops an image to be perfectly divisible in blocks of side length F.

  :param img: Image to crop.
  :type img: np.typing.NDArray[Any]
  :param F: Block side length.
  :type F: int
  :return: Cropped image along with its new width and height.
  :rtype: tuple[np.typing.NDArray[Any], int, int]
  """
  h, w = img.shape
  height, width = h - h % F, w - w % F
  cropped = img[:height, :width]
  return cropped, width, height

def to_visual(data: np.typing.NDArray[Any]) -> np.typing.NDArray[Any]:
  """
  Converts a data matrix into gray-scale for visualization.

  :param data: Data matrix.
  :type data: np.typing.NDArray[Any]
  :return: Gray-scale data matrix.
  :rtype: np.typing.NDArray[Any]
  """
  disp = np.log10(1 + np.abs(data)) # Apply absolute to avoid negatives and + 1 to avoid 0s and make original 0s become 0s when scaled (as log(1) is 0).
  disp *= 255 / disp.max() if disp.max() > 0 else 1
  return disp.astype(np.uint8)

def jpeg_pipeline_steps(img: np.typing.NDArray[Any], F: int, d_thr: int) -> tuple[list[np.typing.NDArray[Any] | tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]], list[str]]:
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

  cropped, width, height = crop(img, F)

  # Mask for k+ℓ < d_thr
  k_idx, l_idx = np.meshgrid(np.arange(F), np.arange(F), indexing="ij")
  block_mask = (k_idx + l_idx) < d_thr

  coeff_mag = np.zeros_like(cropped, dtype=float)
  coeff_masked_mag = np.zeros_like(cropped, dtype=float)
  idct_float = np.empty_like(cropped, dtype=float)

  for y in range(0, height, F):
    for x in range(0, width, F):
      patch = cropped[y : y + F, x : x + F].astype(float) # noqa: E203 - False positive, see https://github.com/PyCQA/pycodestyle/issues/373
      # DCT
      c: np.typing.NDArray[Any] = dctn(patch - 128, norm="ortho") # type: ignore
      coeff_mag[y : y + F, x : x + F] = c # noqa: E203 - False positive, see https://github.com/PyCQA/pycodestyle/issues/373
      # Mask
      c_mask = c * block_mask
      coeff_masked_mag[y : y + F, x : x + F] = c_mask # noqa: E203 - False positive, see https://github.com/PyCQA/pycodestyle/issues/373
      # IDCT
      rec_patch: np.typing.NDArray[Any] = idctn(c_mask, norm="ortho") # type: ignore
      idct_float[y : y + F, x : x + F] = np.clip(np.round(rec_patch + 128), 0, 255).astype(np.uint8) # noqa: E203 - False positive, see https://github.com/PyCQA/pycodestyle/issues/373

  images: list[np.typing.NDArray[Any] | tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]] = [
    img,
    cropped,
    to_visual(coeff_mag),
    to_visual(coeff_masked_mag),
    idct_float,
    (cropped, idct_float)
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
  Application for compressing gray-scale images with a JPEG-like compression.
  """
  def __init__(self) -> None:
    super().__init__()
    # Initialize TKinter stuff.
    self.title("JPEG-like compression step-by-step")
    self.geometry("1080x720")
    self.minsize(720, 480)
    self.protocol("WM_DELETE_WINDOW", self._on_close)
    # Initialize data stuff.
    self.image_path: Path | None = None
    self.img_orig: np.typing.NDArray[Any] | None = None
    self.step_imgs: list[np.typing.NDArray[Any] | tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]] = []
    self.step_titles: list[str] = []
    self.step_idx: int = 0
    self.filename: Optional[str] = None
    # Build widgets.
    self._build_widgets()

  def load_image(self) -> None:
    """
    Loads an image selected by the user.
    """
    filename = filedialog.askopenfilename(title="Select an image (BMP/PNG/JPG)", filetypes=[("Images", "*.bmp;*.png;*.jpg;*.jpeg"), ("All files", "*.*")])
    if filename:
      self.filename = os.path.basename(filename)
      self.image_path = Path(filename)
      img = Image.open(filename).convert("L")
      self.img_orig = np.array(img)
      self._reset_steps([self.img_orig], ["Original image"])
      self.download_btn.config(state=tk.DISABLED)

  def compress_and_show(self) -> None:
    """
    Applies the JPEG-like compression pipeline to the selected image with the chosen parameters.
    """
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
      imgs, titles = jpeg_pipeline_steps(self.img_orig, F, dthr)
    except Exception as exc:
      messagebox.showerror("Compression error", str(exc))
      return

    self._reset_steps(imgs, titles)
    self.download_btn.config(state=tk.NORMAL)

  def prev_step(self) -> None:
    """
    Goes to the previous step in the pipeline.
    """
    if self.step_idx > 0:
      self.step_idx -= 1
      self._update_nav_buttons()
      self._show_current_step()

  def next_step(self) -> None:
    """
    Goes to the next step in the pipeline.
    """
    if self.step_idx < len(self.step_imgs) - 1:
      self.step_idx += 1
      self._update_nav_buttons()
      self._show_current_step()

  def download_steps(self) -> None:
    """
    Downloads all step images except for the original into a chosen directory.
    """
    if len(self.step_imgs) <= 1:
      messagebox.showwarning("Nothing to download", "Compress an image first.")
      return

    directory = filedialog.askdirectory(title="Select folder to save the step images")
    if not directory:
      return

    save_dir = Path(directory)
    count = 0
    F = self.var_block.get()
    dthr = self.var_d.get()
    for idx, img_obj in enumerate(self.step_imgs[1:5], start=1):
      out_path = save_dir / f"{os.path.splitext(str(self.filename))[0]}_step_{idx}_{F}_{dthr}.bmp"
      Image.fromarray(img_obj.astype(np.uint8)).save(out_path) # type: ignore
      count += 1

    messagebox.showinfo("Download complete", f"Saved {count} images to {save_dir}")

  def _build_widgets(self) -> None:
    """
    Builds the application widgets.
    """
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

    self.download_btn = ttk.Button(ctrl, text="Download steps…", command=self.download_steps, state=tk.DISABLED)
    self.download_btn.pack(side=tk.LEFT, padx=(15, 0))

    self.fig, self.ax = plt.subplots()
    self.ax.axis("off")
    self.canvas = FigureCanvasTkAgg(self.fig, master=self)
    self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

  def _reset_steps(self, imgs: list[np.typing.NDArray[Any] | tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]], titles: list[str]) -> None:
    """
    Resets the pipeline steps.

    :param imgs: Images to display for each step.
    :type imgs: list[np.typing.NDArray[Any] | tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]]
    :param titles: Titles for each step.
    :type titles: list[str]
    """
    self.step_imgs = imgs
    self.step_titles = titles
    self.step_idx = 0
    self._update_nav_buttons()
    self._show_current_step()

  def _show_current_step(self) -> None:
    """
    Displays the current step image(s).
    """
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
    """
    Updates the state of the navigation buttons.
    """
    if not self.step_imgs:
      self.prev_btn.config(state=tk.DISABLED)
      self.next_btn.config(state=tk.DISABLED)
      return
    self.prev_btn.config(state=tk.NORMAL if self.step_idx > 0 else tk.DISABLED)
    self.next_btn.config(state=tk.NORMAL if self.step_idx < len(self.step_imgs) - 1 else tk.DISABLED)

  def _on_close(self) -> None:
    """
    Performs closing operations to quit the application properly.
    """
    plt.close("all")
    self.destroy()
    print("app quit")
