# -*- coding: utf-8 -*-

import arcpy
from arcpy.sa import Raster

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Spectral_indices"
        self.alias = "specindic"

        # List of tool classes associated with this toolbox
        self.tools = [Calculate_NDVI]

class Calculate_NDVI(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "NDVI"
        self.description = "Calculate normalized difference vegetation index (NDVI) from raster layers."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        RED = arcpy.Parameter(
            displayName="Red layer",
            name="red",
            datatype="DERasterBand",
            parameterType="Required",
            direction="Input")
        NIR = arcpy.Parameter(
            displayName="Near infrared (NIR) layer",
            name="nir",
            datatype="DERasterBand",
            parameterType="Required",
            direction="Input")
        OUT_DIR = arcpy.Parameter(
            displayName="Output directory",
            name="dir",
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Output")
        parameters = [RED, NIR, OUT_DIR]
        return parameters

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        RED = parameters[0].valueAsText
        NIR = parameters[1].valueAsText
        NDVI = (Raster(NIR) - Raster(RED)) / (Raster(NIR) + Raster(RED))
        OUT_DIR = parameters[2].valueAsText
        if OUT_DIR is None:
            arcpy.env.workspace = "memory"
            memory = arcpy.CreateUniqueName("NDVI")
            OUT_DIR = memory
        NDVI.save(OUT_DIR)
        return
