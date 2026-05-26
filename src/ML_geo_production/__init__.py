# Load GDAL before fastai/Pillow so libtiff/jpeg symbols resolve correctly in conda envs.
from osgeo import gdal  # noqa: F401
