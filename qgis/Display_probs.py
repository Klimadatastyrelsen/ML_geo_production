
import os
from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsColorRampShader,
    QgsSingleBandPseudoColorRenderer,
    QgsStyle
)
from qgis.PyQt.QtWidgets import QFileDialog

# Settings
MIN_VALUE = 0.0
MAX_VALUE = 1.0
NUM_COLORS = 256

# Get Turbo color ramp from QGIS
style = QgsStyle.defaultStyle()


# Select input rasters
files, _ = QFileDialog.getOpenFileNames(
    iface.mainWindow(),
    "Select probability rasters",
    "",
    "GeoTIFF (*.tif *.tiff)"
)

project = QgsProject.instance()
root = project.layerTreeRoot()


# Apply Turbo styling to a specific raster band
def style_probability(layer, band):

    turbo_ramp = style.colorRamp("Turbo")

    color_shader = QgsColorRampShader(MIN_VALUE, MAX_VALUE)
    color_shader.setColorRampType(QgsColorRampShader.Interpolated)
    color_shader.setSourceColorRamp(turbo_ramp)
    color_shader.classifyColorRamp(NUM_COLORS)

    raster_shader = QgsRasterShader(MIN_VALUE, MAX_VALUE)
    raster_shader.setRasterShaderFunction(color_shader)

    renderer = QgsSingleBandPseudoColorRenderer(
        layer.dataProvider(),
        band,
        raster_shader
    )

    renderer.setClassificationMin(MIN_VALUE)
    renderer.setClassificationMax(MAX_VALUE)

    layer.setRenderer(renderer)
    layer.triggerRepaint()


# Load rasters and create separate layers for each channel
for path in files:
    
    # name = os.path.splitext(os.path.basename(path))[0]
    name = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
    check = QgsRasterLayer(path, name, "gdal")

    if not check.isValid() or check.bandCount() < 2:
        print(f"Skipped {name}: invalid raster or fewer than 2 bands")
        continue

    group = root.addGroup(name)

    class_settings = [
        (2, "Background - Probability"),
        (3, "Building - Probability")
    ]

    for band, label in class_settings:

        layer = QgsRasterLayer(path, label, "gdal")
        style_probability(layer, band)

        project.addMapLayer(layer, False)
        node = group.addLayer(layer)

        # Show Band 1 by default
        node.setItemVisibilityChecked(band == 1)

    print(f"Added: {name}")

# Refresh map
iface.mapCanvas().refresh()
print("Finished loading probability rasters.")
