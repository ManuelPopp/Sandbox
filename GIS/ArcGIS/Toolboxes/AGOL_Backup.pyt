#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:12:37 2023
"""
__author__ = "Manuel"
__date__ = "Thu Feb  9 12:12:37 2023"
__license__ = "Unlicense"
__version__ = "1.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Beta"

import os, shutil, math
import arcpy as ap
import arcgis
from arcgis.gis import GIS

global AGOL_TYPES
AGOL_TYPES = ["CSV",
    "Dashboard",
    "Feature Collection",
    "Feature Service",
    "File Geodatabase",
    "Form",
    "KML",
    "KML Collection",
    "Service Definition",
    "Shapefile",
    "Web Map"]

def get_value(inp):
    keep_types = (int, str, bool, type(None))
    val = inp.value if isinstance(inp.value, keep_types) else inp.valueAsText
    return val

class Toolbox(object):
    def __init__(self):
        self.label = "AGOL Backup"
        self.alias = "Backup"
        self.tools = [BackupLocal]

class BackupLocal(object):
    def __init__(self):
        self.label = "Backup"
        self.description = "Create local backup of ArcGis Online items."
        self.canTunInBackground = False
    
    def getParameterInfo(self):
        global AGOL_TYPES
        USR = ap.Parameter(
            displayName = "AGOL username",
            name = "usr",
            datatype = "GPString",
            parameterType = "Required",
            direction = "Input")
        USR.value = "manuel.popp_KIT"
        PW = ap.Parameter(
            displayName = "Password",
            name = "psswrd",
            datatype = "GPEncryptedString",
            parameterType = "Optional",
            direction = "Input")
        PW.value = None
        COMPLETE = ap.Parameter(
            displayName = "Complete backup",
            name = "complete",
            datatype = "GPBoolean",
            parameterType = "Required",
            direction = "Input")
        COMPLETE.value = True
        OWNER = ap.Parameter(
            displayName = "Owner (if different from user)",
            name = "owner",
            datatype = "GPString",
            parameterType = "Optional",
            direction = "Input")
        OWNER.value = None
        DTYPES = ap.Parameter(
            displayName = "Data types",
            name = "dtypes",
            datatype = "GPString",
            parameterType = "Optional",
            direction = "Input",
            multiValue = True)
        DTYPES.filter.list = AGOL_TYPES
        DTYPES.value = None
        TAGS = ap.Parameter(
            displayName = "Tags",
            name = "tags",
            datatype = "GPString",
            parameterType = "Optional",
            direction = "Input",
            multiValue = True)
        OVERWRITE = ap.Parameter(
            displayName = "Overwrite existing files",
            name = "overwrite",
            datatype = "GPBoolean",
            parameterType = "Required",
            direction = "Input")
        OVERWRITE.value = False
        OUT_DIR = ap.Parameter(
            displayName = "Backup directory",
            name = "out_dir",
            datatype = "DEFolder",
            parameterType = "Required",
            direction = "Output")
        
        parameters = [USR, PW, COMPLETE, OWNER, DTYPES, TAGS, OVERWRITE, \
                      OUT_DIR]
        return parameters

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        [USR, PW, COMPLETE, OWNER, DTYPES, TAGS, OVERWRITE, OUT_DIR] = \
            parameters
        if COMPLETE.value:
            DTYPES.enabled = TAGS.enabled = False
        else:
            DTYPES.enabled = TAGS.enabled = True
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        global AGOL_TYPES
        [USR, PW, COMPLETE, OWNER, DTYPES, TAGS, OVERWRITE, OUT_DIR] = \
            [get_value(p) for p in parameters]
        
        if not os.path.isdir(OUT_DIR):
            os.makedirs(OUT_DIR, exist_ok = True)
        
        if PW is None:
            gis = GIS("pro")
        else:
            gis = GIS(username = USR, password = PW)
        
        account_name = USR if OWNER is None else OWNER
        QUERY_STRING = "owner:{0}".format(account_name)
        my_content = gis.content.search(QUERY_STRING,
                                  max_items = 999)
        
        mssg = "Found {0} items for user {1} on AGOL."
        n_items = len(my_content)
        ap.AddMessage(mssg.format(n_items, account_name))
        allowed_types = str(DTYPES).split(";") if not COMPLETE else \
            AGOL_TYPES
        
        allowed_tags = set(str(TAGS).split(";")) if TAGS is not None else set()
        
        failed_downloads = []
        
        p = int(math.log10(n_items))
        
        if not p:
            p = 1
        
        increment = int(math.pow(10, p - 1))
        
        ap.SetProgressor("step", "Downloading items from AGOL...", 0, n_items,
                         increment)
        
        for i, item in enumerate(my_content):
            mssg = "Checking conditions for item {0} of {1}"
            ap.SetProgressorLabel(mssg.format(i, n_items))
            
            tagmatch = allowed_tags == set() or allowed_tags.intersection(
                set(item.tags))
            typematch = item.type in allowed_types
            mssg = "Item {0}. Type: {1}\nMatches item types: {2}"
            ap.AddMessage(mssg.format(i, item.type, typematch))
            
            if tagmatch and typematch:
                try:
                    item_name = item.title if item.title is not None else \
                        item.name
                    path = os.path.join(OUT_DIR, item_name)
                    
                    if os.path.exists(path) and not OVERWRITE:
                        continue
                    
                    mssg = "Attempting to download {0} from ArcGIS Online."
                    
                    if item.type == "Feature Service":
                        ap.SetProgressorLabel(mssg.format(item.title))
                        ap.AddMessage(mssg.format(item.title))
                        result = item.export(item.name, "Shapefile")
                        result.download(path)
                        result.delete()
                    else:
                        ap.SetProgressorLabel(mssg.format(item.name))
                        ap.AddMessage(mssg.format(item.name))
                        tmp = item.get_data()
                        mssg = "Saved as temporary file at " + tmp
                        ap.AddMessage(mssg)
                        mssg = "Attempting to move temporary file to " + \
                            OUT_DIR
                        ap.AddMessage(mssg)
                        shutil.move(tmp, path)
                    
                    if os.path.getsize(path) > 0.:
                        ap.AddMessage("Download successful.")
                except:
                    failed_downloads.append(item)
                    mssg = "Failed to download {0}."
                    ap.AddWarning(mssg.format(item_name))
            ap.SetProgressorPosition(i)
        return
