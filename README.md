# metadetect-driver

Executable scripts to run metadetection on Roman Space Telescope images.

The basic functionality of the driver is to take PyIMCOM images, and more specifically PyIMCOM object that know how to read the images, and run Metadetection on those images to produce catalogs with photometry and shape information. This driver allows you to do exactly that, and the `Example_driver.ipynb` in the `notebooks` directory walks you through how to do use the different functionalities enabled. In short, the code can process single PyIMCOM Blocks (which hold single coadded images) or Mosaics (a collection of Blocks). The class in PyIMCOM that handles Blocks is called OutImage. The functions allow you to pass multi-bands Blocks or Mosaics as lists.

## Installation

```bash
pip install git+https://github.com/Roman-HLIS-Cosmology-PIT/metadetect-driver.git

```
For development, you might want to run this instead:
```bash
git clone https://github.com/Roman-HLIS-Cosmology-PIT/metadetect-driver.git
cd metadetect-driver
pip install -e .

```

