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

### 1D DCT

TODO

### 2D DCT

TODO

### IDCT

TODO

### JPEG

TODO
