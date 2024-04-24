# -*- coding: utf-8 -*-
"""
@author: Klemet

Script that reads the table results for the Manawan simulations produced by
pythonAnalysis_Manawan_v0.2.py to make figures showing the volution of the 4
variables for each family zone in the manawan region

Works with the factors used for simulation batch v0.3, that has 3 factors :
Climate (Baseline, RCP 4.5, RCP 8.5), biomass harvested (50%, 100% and 200% of BAU),
prescription regime (normal, ClearCutsPlus, PartialCutsPlus)

This script is under the format of a Streamlit app.
"""

#%% IMPORTING MODULES

import sys, os, glob, re, io
import pandas as pd
# from osgeo import gdal
import numpy as np
import math
import pickle
import streamlit as st
import altair as alt
import hmac
from cryptography.fernet import Fernet
import tifffile
# import gdown
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from webdav3.client import Client
from rasterio.plot import show
from rasterio.mask import mask
import rasterio
from scipy.ndimage import zoom
# import skimage
import pydeck as pdk

matplotlib.use('agg')
#%% CHECK PASSWORD

# Restricts the usage of the app to people without password.
# See https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password_Streamlit_app"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Main Streamlit app starts here

#%% STREAMLIT PARAMETERS

# matplotlib.use('SVG')

# DEBUG

# pathToDebugRaster = r"D:\OneDrive - UQAM\1 - Projets\Thèse - Simulations Manawan projet DIVERSE\3 - Résultats\simulation_batch_v0.22_Manawan_WithMagicHarvest_Narval\Moose_HQI\Average and SD Rasters\Average\BAU-Baseline_Average_HQI_Moose_DUSSAULT_Timestep_30.tif"


#%% PREPARING CLIENT TO ACCESS REMOTE FILES ON NEXTCLOUD

# Problem : Moose HQI maps data is too big for github and for streamlit, even pickled.
# Solution : put everything on a nextcloud install. Files can be read through webdav
# into a buffer, and loaded into a numpy array without creating local files (see loadNumpyArrayFromNextcloudTIFFile)
# It's a bit slower than I'd want, but it works.

webDavClientOptions = {
 'webdav_hostname': st.secrets["nextcloudAdress"],
 'webdav_login':    st.secrets["nextcloudUser"],
 'webdav_password': st.secrets["nextcloudPasswordApp"]
}

# webDavClient = Client(options)
# client.list()

# client.list("Data - StreamlitApps/appmanawanresultsanalysisbatch03.streamlit.app/Moose_HQI/Average and SD Rasters/Average/")

# res1 = client.resource("Data - StreamlitApps/appmanawanresultsanalysisbatch03.streamlit.app/Moose_HQI/Average and SD Rasters/Average/BAU50%-ClearCutsPlus-Baseline_Average_HQI_Moose_KOITZSCH_Timestep_50.tif")
# buffer = io.BytesIO()
# res1.write_to(buffer)
# tiff_buffer = io.BytesIO(buffer.getbuffer())
# image = tifffile.imread(tiff_buffer)

# DEBUG : Tests the connection by displayng a raster.
# img_array = loadNumpyArrayFromNextcloudTIFFile(webDavClientOptions,
#                                                 "Data - StreamlitApps/appmanawanresultsanalysisbatch03.streamlit.app/Moose_HQI/Average and SD Rasters/Average/",
#                                                 "BAU100%-NormalCuts-Baseline_Average_HQI_Moose_KOITZSCH_Timestep_100.tif")


# remoteFolder = "Data - StreamlitApps/appmanawanresultsanalysisbatch03.streamlit.app/Moose_HQI/Average and SD Rasters/Average/"
# remoteFileName = "BAU100%-NormalCuts-RCP45_Average_HQI_Moose_KOITZSCH_Timestep_100.tif"

# webDavClient = Client(webDavClientOptions)
# res1 = webDavClient.resource(remoteFolder + remoteFileName)
# buffer = io.BytesIO()
# res1.write_to(buffer)

# # NOT faster at all
# # webDavClient.download_sync(remote_path=remoteFolder + remoteFileName,
# #                             local_path="file1.tif")

# tiff_buffer = io.BytesIO(buffer.getbuffer())
# image = tifffile.imread(tiff_buffer)
# print(np.any(np.isnan(image)))
# np.argwhere(np.isnan(image))
# image = np.nan_to_num(image, nan=0.0)
# image = skimage.transform.resize_local_mean(image, (int(image.shape[0]/2), int(image.shape[1]/2)))
# image = bilinear_resize(image, (int(image.shape[0]/2), int(image.shape[1]/2)))

# matplotlib.use('qtagg')
# fig = plt.figure(figsize=(6, 6))
# plt.imshow(image)
# st.pyplot(fig)

# testArray = tifffile.imread("file1.tif")
# np.save('my_array.npy', testArray)
# with open('my_array.pkl', 'wb') as f:
#     pickle.dump(testArray, f)

#%% DEFINING FUNCTIONS

# def getRasterData(path):
#     raster = gdal.Open(path)
#     rasterData = raster.GetRasterBand(1)
#     rasterData = rasterData.ReadAsArray()
#     rasterData - rasterData.astype('float64')
#     # rasterData = zoom(rasterData, (0.5, 0.5), order=0, prefilter=False)
#     return(np.array(rasterData))

# def getRasterDataAsList(path):
#     return(getRasterData(path).tolist())

# def writeNewRasterData(rasterDataArray, pathOfTemplateRaster, pathOfOutput):
#     # Saves a raster in int16 with a nodata value of 0
#     # Inspired from https://gis.stackexchange.com/questions/164853/reading-modifying-and-writing-a-geotiff-with-gdal-in-python
#     # Loading template raster
#     template = gdal.Open(pathOfTemplateRaster)
#     driver = gdal.GetDriverByName("GTiff")
#     [rows, cols] = template.GetRasterBand(1).ReadAsArray().shape
#     outputRaster = driver.Create(pathOfOutput, cols, rows, 1, gdal.GDT_Int16)
#     outputRaster.SetGeoTransform(template.GetGeoTransform())##sets same geotransform as input
#     outputRaster.SetProjection(template.GetProjection())##sets same projection as input
#     outputRaster.GetRasterBand(1).WriteArray(rasterDataArray)
#     outputRaster.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
#     outputRaster.FlushCache() ##saves to disk!!
#     outputRaster = None
    
# def writeNewRasterDataFloat32(rasterDataArray, pathOfTemplateRaster, pathOfOutput):
#     # Saves a raster in Float32 with a nodata value of 0.0
#     # Inspired from https://gis.stackexchange.com/questions/164853/reading-modifying-and-writing-a-geotiff-with-gdal-in-python
#     # Loading template raster
#     template = gdal.Open(pathOfTemplateRaster)
#     driver = gdal.GetDriverByName("GTiff")
#     [rows, cols] = template.GetRasterBand(1).ReadAsArray().shape
#     outputRaster = driver.Create(pathOfOutput, cols, rows, 1, gdal.GDT_Float32)
#     outputRaster.SetGeoTransform(template.GetGeoTransform())##sets same geotransform as input
#     outputRaster.SetProjection(template.GetProjection())##sets same projection as input
#     outputRaster.GetRasterBand(1).WriteArray(rasterDataArray)
#     outputRaster.GetRasterBand(1).SetNoDataValue(-1)##if you want these values transparent
#     outputRaster.FlushCache() ##saves to disk!!
#     outputRaster = None

# def writeExistingRasterData(rasterDataArray, pathOfRasterToEdit):
#     # Edits the data of an existing raster
#     rasterToEdit = gdal.Open(pathOfRasterToEdit, gdal.GF_Write)
#     rasterToEdit.GetRasterBand(1).WriteArray(rasterDataArray)
#     rasterToEdit.FlushCache() ##saves to disk!!
#     rasterToEdit = None

# From https://pynative.com/python-write-list-to-file/
# write list to binary file
def write_list(a_list, filePath):
    # store list in binary file so 'wb' mode
    with open(filePath, 'wb') as fp:
        pickle.dump(a_list, fp)
        # print('List saved at path :' + str(filePath))

# Read list to memory
def read_list(filePath):
    # for reading also binary mode is important
    with open(filePath, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

def weightedStandardDeviation(listOfValues, listOfWeights):
    """Taken from https://stackoverflow.com/a/65434084, and validated
    against results from  DescrStatsW(values, weights=weights).std (see https://stackoverflow.com/a/36464881)"""
    return math.sqrt(np.average((listOfValues - np.average(listOfValues, weights=listOfWeights))**2, weights=listOfWeights))

def createAxisForRaster(listOfCoordinatesRelativeToFigure, figure, disableAxis = True):
    axis = figure.add_axes(listOfCoordinatesRelativeToFigure, transform=figure.dpi_scale_trans)
    if disableAxis:
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axis.spines['right'].set_color('#3b4252')
    axis.spines['left'].set_color('#3b4252')
    axis.spines['bottom'].set_color('#3b4252')
    axis.spines['top'].set_color('#3b4252')
    return(axis)

# We define two function that will allow us to sort 
# the name of the raster files properly on Linux.
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

# We define several functions used to read the files with results
def CSVFilesDetection(path, restrictions):
    """Detects and return all of the .csv files in the path
    that do not contain on of the words in the "restriction" list."""
    listOfFiles = list()
    # We look at the files with the .csv extension in the path;
    # If these do not contain one of the restricted words, then we
    # add them to the list of files to return
    for file in glob.glob(path + "*.csv"):
        if len(restrictions) > 0:
            restrictionsValidated = False
            for restriction in restrictions:
                if restriction in file:
                    restrictionsValidated = True
            if restrictionsValidated:
                # print(file)
                listOfFiles.append(file)
        else:
            # print(file)
            listOfFiles.append(file)
    
    return(listOfFiles)

def CSVFilesToDataFrames(listOfFiles):
    """Takes a list of paths for the .csv result files and returns
    a dictionnary containing list of data frames with the results, organised
    in lists for replicates, each list being accessible by a key representing 
    each scenario."""
    # We initialized the objects necessary
    listOfCheckedScenarios = list()
    dictionnaryOfResultFiles = dict()
    # We sort the list of files according to the natural keys defined above
    listOfFiles = sorted(listOfFiles, key=natural_keys)
    # For each .csv files, we check the name of the scenario and create a new
    # entry for it if it doesn't already exist.
    # If the scenario doesn't already exist, we find all of the files for the
    # same scenario to put them ina list of replicates
    for file in listOfFiles:
        nameOfScenario = os.path.basename(file).split("-Rep")[0]
        if nameOfScenario not in listOfCheckedScenarios:
            listOfCheckedScenarios.append(nameOfScenario)
            for fileToLookAt in listOfFiles:
                if os.path.basename(fileToLookAt).split("-Rep")[0] == nameOfScenario:
                    if nameOfScenario not in dictionnaryOfResultFiles.keys():
                        df = pd.read_csv(fileToLookAt)
                        dictionnaryOfResultFiles[nameOfScenario] = [df]
                    else:
                        df = pd.read_csv(fileToLookAt)
                        dictionnaryOfResultFiles[nameOfScenario].append(df)

            # print("Detected scenario :" + nameOfScenario)
            # print("Found " + str(len(dictionnaryOfResultFiles[nameOfScenario])) + " replicates.")
    
    # print(sorted(dictionnaryOfResultFiles.keys()))
    return(dictionnaryOfResultFiles)


# def CreateDictionnaryOfMeanAndStandardDeviation(timesteps,
#                                                 variables,
#                                                 FactorLevels,
#                                                 dictionnaryOfBasicResults):
#     """
#     Takes a dictionnary containing one dataframe of results by key entries,
#     and produces a nested dictionnary categorized by variables in thoses results,
#     then two factors that changes between scenarios (e.g. Climate, management).
#     Hence, for each variable, for each level of the first factor, and for each
#     level of the second factor, for each level of the third factor,
#     we get the mean and standard deviation of all of
#     the results that have the levels of the first and second and thirdfactors.
#     There is no averaging on the third factors here.
#     In this script, we are dealing with results that have not been averaged on
#     all of the stands inside a simulation; therefore, at the first averaging during this
#     process, we will not use the meanPropagationOfError function to propagate the variance
#     from average to average.
    
#     This one takes 3 factors into account, given as a list of 3 lists with the
#     factor levels.
#     """
    
#     dictOfValuesForBasicMeasures = dict()
#     variables = list(variables)
    
#     for variable in variables:
#         dictOfValuesForBasicMeasures[variable] = dict()
#         for factor1Level in FactorLevels[0]:
#             dictOfValuesForBasicMeasures[variable][factor1Level] = dict()
#             for factor2Level in FactorLevels[1]:
#                 dictOfValuesForBasicMeasures[variable][factor1Level][factor2Level] = dict()
#                 for factor3Level in FactorLevels[2]:
#                     dictOfValuesForBasicMeasures[variable][factor1Level][factor2Level][factor3Level] = dict()
#                     dictOfValuesForBasicMeasures[variable][factor1Level][factor2Level][factor3Level]["Mean"] = list()
#                     dictOfValuesForBasicMeasures[variable][factor1Level][factor2Level][factor3Level]["SD"] = list()

#     # Recupérer les bons dataframe avec first et second factor level avec des for;
#     # Récuperer la bonne variable au bon timestep dans les dataframes
#     # faire la moyenne, la mettre dans une liste
#     # Mettre la liste dans le dictionnaire groupé
#     for variable in variables:
#         for factor1Level in FactorLevels[0]:
#             for factor2Level in FactorLevels[1]:
#                 for factor3Level in FactorLevels[2]:
#                     # We detect the replicates
#                     for scenario in dictionnaryOfBasicResults.keys():
#                         # factorsInScenarioName = scenario.replace('_', '-').split("-")
#                         # print("Searching " + factorLevel)
#                         if all(word in scenario for word in [factor1Level, factor2Level, factor3Level]):
#                             # print("Found " +  str(scenario) + " !")
#                             dataFrameWithFactorLevels = scenario
#                     listOfMeanValuesForReplicates = list()
#                     listOfStandardDeviationForReplicates = list()
#                     for timestep in range(0, len(timesteps)):
#                         # print(timestep)
#                         totalValuesForTimestep = list()
#                         for replicateNumber in range(0, len(dictionnaryOfBasicResults[dataFrameWithFactorLevels])):
#                             # print(str(firstFactorLevel) + " " + str(secondFactorLevel))
#                             totalValuesForTimestep.append(dictionnaryOfBasicResults[dataFrameWithFactorLevels][replicateNumber][variable][timestep])
        
#                         # First mean has no propagation of variance
#                         meanForTimestepForAllReplicates = np.mean(totalValuesForTimestep)
#                         SDForTimestepForAllReplicates = np.std(totalValuesForTimestep)
#                         listOfMeanValuesForReplicates.append(meanForTimestepForAllReplicates)
#                         listOfStandardDeviationForReplicates.append(SDForTimestepForAllReplicates)
                        
#                     dictOfValuesForBasicMeasures[variable][factor1Level][factor2Level][factor3Level]["Mean"] = listOfMeanValuesForReplicates
#                     dictOfValuesForBasicMeasures[variable][factor1Level][factor2Level][factor3Level]["SD"] = listOfStandardDeviationForReplicates
                

#     return dictOfValuesForBasicMeasures

def readRasterTifffile(path):
    """
    Allows the reading of a .tif image data into a numpy array,
    without using GDAL !
    """
    return(tifffile.imread(path))

def loadRessourceWebDav(webDavClientOptions, remoteFolder, remoteFileName):
    webDavClient = Client(webDavClientOptions)
    res1 = webDavClient.resource(remoteFolder + remoteFileName)
    buffer = io.BytesIO()
    res1.write_to(buffer)
    file_buffer = io.BytesIO(buffer.getbuffer())
    # We close and delete objects, just in case, for RAM usage.
    buffer.close()
    del(webDavClient)
    del(buffer)
    return(file_buffer)

def loadNumpyArrayFromNextcloudTIFFile(webDavClientOptions, remoteFolder, remoteFileName, resize2X = True):
    """
    Done after a lot of tweaking around with perplexity.
    The function does not need to save a local file, and downloads things relatively
    quickly.
    Can be used with the nextcloud of ComputeCanada.
    Reduces the resoltuion by dividing by 2 to reduce memory usage from strealit; can be overriden.
    """
    tiff_buffer = loadRessourceWebDav(webDavClientOptions, remoteFolder, remoteFileName)
    image = tifffile.imread(tiff_buffer)
    # We remove NaN; if there is a single one, the resized map becomes all NaNs
    # because of skimage.transform.resize_local_mean.
    image = np.nan_to_num(image, nan=0.0)
    if resize2X:
        # Uses bilinear interpolation (local mean). Should be better.
        # image = skimage.transform.resize_local_mean(image, (int(image.shape[0]/2), int(image.shape[1]/2)))
        # Uses nearest-neighbor interpolation. Not great for continuous variables as we're doing here.
        image = zoom(image, (0.5, 0.5), order=1, prefilter=False)
    return(image)

def displayPydeckMp(geoDataFrame_areas_Manawan, familyAreaName):
    # We load the raster dataset with rasterio
    
    # raster_FamilyZone = rasterio.open(rasterFilePath)
    # raster_FamilyZone_data = raster_FamilyZone.read()
    
    # # We transform the raster into a polygon readable by pydeck
    # shapes = rasterio.features.shapes(raster_FamilyZone.read(1), transform=raster_FamilyZone.transform)
    
    # polygons_list = []
    # for geom, value in shapes:
    #     if value > 0:
    #         polygons_list.append(geom)
    #         # st.text(str(geom))
            
    # polygons_latlon = [rasterio.warp.transform_geom(raster_FamilyZone.crs, 'EPSG:4326', polygon) for polygon in polygons_list]
            
        
    # polygon_final_list = list()
    # for polygon in polygons_latlon:
    #     # st.text(str(polygon))
    #     polygon_final_list.append(polygon["coordinates"])
        
    # st.text(str(polygon_final_list))
    
    # We select the polygons to display
    selected_polygons = geoDataFrame_areas_Manawan[geoDataFrame_areas_Manawan['FAMILLE'] == familyAreaName]
    # st.text(str(selected_polygons))
    
    # We put it in a list format that pydeck can read by extracting the coordinates
    polygon_final_list = list()
    for polygon in selected_polygons.geometry.apply(lambda g: list(g.exterior.coords)):
        # st.text(str(polygon))
        polygon_final_list.append(polygon)
        # st.text(str(polygon))
    
    # st.text(str(polygon_final_list))
    
    # Create the Deck object
    # Initial view is centered on manawan
    tooltip = {"html": "<b>Zone considered in the results</b>"}
    
    r = pdk.Deck(
        layers=[
            pdk.Layer(
                'PolygonLayer',
                polygon_final_list,
                stroked=False,
                pickable=True,
                extruded=True,
                # auto_highlight=True,
                get_polygon='-',
                get_fill_color=[191, 97, 106, 100]
            )],
        initial_view_state = pdk.ViewState(
            latitude=47.207744,
            longitude=-74.374665,
            zoom=7,
            pitch=45,
            bearing=0
        ),
        map_style='mapbox://styles/mapbox/outdoors-v12',
        tooltip=tooltip
    )

    # We display the object with a title
    # r.to_html('raster_chart.html')
    
    # Display in streamlit
    st.markdown("<h3 style='text-align: center;'>" + "Zone considered for the results - " + str(familyArea) + "</h3>", unsafe_allow_html=True)
    st.pydeck_chart(pydeck_obj=r, use_container_width=False)

#%% PLOTTING FUNCTIONS

def CreateAltairChartWithMeanAndSD(listOfTimesteps,
                                   listOfDataSeriesNames,
                                   listOfMeanDataSeries,
                                   listOfSDSeries,
                                   listOfColors,
                                   listOfScenarioNames,
                                   variableName):
    """Given a list of mean data series + list of SD,
    returns an altair chart layer superposing all of the curves +
    standard deviation areas around them.
    ListOfColors must same length as listOfMeanDataSeries and listOfSDSeries."""
    
    # We create the dataframe containing all of the data,
    # in the format that altair needs
    listOfMeanValues = list()
    listOfTimestepValues = list()
    listofYUppwer = list()
    listOfYDowner = list()
    listOfScenarios = list()
    for i in range(0, len(listOfMeanDataSeries)):
        listOfMeanValues.extend(listOfMeanDataSeries[i])
        listOfTimestepValues.extend(listOfTimesteps)
        listOfScenarios.extend([listOfScenarioNames[i]] * len(listOfTimesteps))
        listofYUppwer.extend([x + y for x, y in zip(listOfMeanDataSeries[i], listOfSDSeries[i])])
        listOfYDowner.extend([x - y for x, y in zip(listOfMeanDataSeries[i], listOfSDSeries[i])])
    dataFrameCurves = dataFrameForestTypesStack = pd.DataFrame(listOfTimestepValues, columns=(["y"]))
    dataFrameCurves["y"] = listOfMeanValues
    dataFrameCurves["Climate Scenario"] = listOfScenarios
    # dataFrameCurves["Variability"] = listOfScenarios
    dataFrameCurves["x"] = listOfTimestepValues
    dataFrameCurves["y_upper"] = listofYUppwer
    dataFrameCurves["y_downer"] = listOfYDowner
    
    # selector = alt.selection_single(
    # fields=['Climate Scenario'], 
    # empty='all',
    # bind='legend')
    
    # We create the curve and confidence interval charts
    # that we are going to combine
    curve = alt.Chart(dataFrameCurves).mark_line().encode(
        x=alt.X("x", axis=alt.Axis(title="Time step")),
        y=alt.Y("y", axis=alt.Axis(title=variableName)),
        color=alt.Color('Climate Scenario:N',
                    scale=alt.Scale(domain=listOfScenarioNames,
                      range=listOfColors))
    )

    confidence_interval = alt.Chart(dataFrameCurves).mark_area(opacity = 0.4).encode(
        x=alt.X("x", axis=alt.Axis(title="Time step")),
        y='y_downer',
        y2='y_upper',
        color=alt.Color('Climate Scenario:N',
                    scale=alt.Scale(domain=listOfScenarioNames,
                      range=listOfColors), legend = None) # Legend none to avoid double label in legend
    )
    
    # We combine the layers and return the chart
    comboOfChart = curve + confidence_interval
    comboOfChart.configure_range(
        category=alt.RangeScheme(listOfColors))
    # comboOfChart.add_selection(
    # selector).transform_filter(
    # selector)
    return(comboOfChart.resolve_scale(color='independent').configure_axis(grid=False))


def createFigureOfMooseHQI(biomassHarvest, cutRegime, indexType):
    
    # img_array = loadNumpyArrayFromNextcloudTIFFile(client,
    #                                                "Data - StreamlitApps/appmanawanresultsanalysisbatch03.streamlit.app/Moose_HQI/Average and SD Rasters/Average/",
    #                                                "BAU50%-ClearCutsPlus-Baseline_Average_HQI_Moose_KOITZSCH_Timestep_50.tif")
    # fig = plt.figure(figsize=(6, 6))
    # biomassHarvest = "50% of BAU"
    # cutRegime = "More clearcuts"
    # indexType = "DUSSAULT"
    
    
    loading_indicator = st.empty()
    progressIndicator = 0
    loading_indicator.write("⚙ Downloading additional data to generate figure : " + str(progressIndicator) + "%")
    
    pathOfMooseHQIRasters = "Data - StreamlitApps/appmanawanresultsanalysisbatch03.streamlit.app/Moose_HQI/Average and SD Rasters/"
    
    dictT0Raster = {"KOITZSCH":"Average/Initial_HQI_Moose_KOITZSCH.tif",
                    "DUSSAULT":"Average/Initial_HQI_Moose_DUSSAULT.tif"}
    
    dictTitle = {"KOITZSCH":"Koitzch (2002)",
                  "DUSSAULT":"Dussault et al. (2006)"}
    
    dictYlim = {"KOITZSCH":1,
                  "DUSSAULT":1.2}
    
    dictTicksAverage = {"KOITZSCH":[0, 0.5, 1],
                        "DUSSAULT":[0, 0.6, 1.2]}
    
    dictTicksSD = {"KOITZSCH":[0, 0.5],
                    "DUSSAULT":[0, 0.6]}
    
    ecoregion_mask_raster = "ecoregions_Mask.tif"
    
    # Gotta use a dict that is a little bit different here because of differences
    # in the data labels.
    dictTransformCutRegim2 = {"Normal (cuts as in BAU)":"NormalCuts",
                             "More clearcuts":"ClearCutsPlus",
                             "More partial cuts":"PartialCutsPlus"}
    bioHarvestedHQI = dictTransformBioHarvest[biomassHarvest]
    cutRegimeHQI = dictTransformCutRegim2[cutRegime]
    listClimateScenarios = ["Baseline", "RCP45", "RCP85"]
    
    # We create the big figure
    widthFigureInches = 14
    heightFigureInches = 4*3
    bigFigure = plt.figure(figsize = [widthFigureInches, heightFigureInches])
    bigFigure.set_facecolor("#eceff4")
    bigFigure.show()
    
    # We create the axis that we will fill in the big figure, and put them
    # in a dictionnary to call them clearly with a name
    dictAxis = dict()
    
    # We define the "Mask" for the raster, to avoid displaying 0 values/pixels
    # This mask is simply the timestep 0 raster.
    if 'maskRasterMooseHQI' not in st.session_state:
        maskRasterMooseHQI = loadNumpyArrayFromNextcloudTIFFile(webDavClientOptions,
                                                                pathOfMooseHQIRasters,
                                                                ecoregion_mask_raster)
        # maskRasterMooseHQI = getRasterData(pathToDebugRaster)
    maskRasterMooseHQI = np.where(maskRasterMooseHQI > 0, 1, 0)
    
    progressIndicator = 5
    loading_indicator.write("⚙ Downloading additional data to generate figure : " + str(progressIndicator) + "%")
    # We make the cmap for the mask
    # Color map for fire raster
    levels = [0, 1, 999]
    clrs = ['#FFFFFF', '#FFFFFF00'] 
    cmapMooseMask, norm = matplotlib.colors.from_levels_and_colors(levels, clrs)
    
    # MOOVING THE RASTERS AXIS
    # Used for the creation of axis bellow
    bottomAxisModifier = 0.05
    rightAxisModifier = 0.03
    
    # We create the "Example" on the side
    # We display the average raster for t = 30 for a scenario
    dictAxis["exampleRaster"] = createAxisForRaster([0.4-rightAxisModifier, 0.8-bottomAxisModifier, 0.198, 0.2], bigFigure, disableAxis = True)
    dictAxis["exampleRaster"].zorder = 3 # We prepare to pu the legend below
    if 'exampleMeanRaster' not in st.session_state:
        exampleMeanRaster = loadNumpyArrayFromNextcloudTIFFile(webDavClientOptions,
                                                                pathOfMooseHQIRasters,
                                                                "Average/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[0]) + "_Average_HQI_Moose_" + indexType +  "_Timestep_30.tif")
        # exampleMeanRaster = getRasterData(pathToDebugRaster)
    exeampleMeanRasterShow = show(exampleMeanRaster, ax=dictAxis["exampleRaster"],
                                  alpha=1, cmap = 'viridis')
    exeampleMeanRasterShow = exeampleMeanRasterShow.get_images()[0]
    show(maskRasterMooseHQI, ax=dictAxis["exampleRaster"], alpha=1, cmap = cmapMooseMask)
    progressIndicator = 10
    loading_indicator.write("⚙ Downloading additional data to generate figure : " + str(progressIndicator) + "%")
    
    
    # We display a legend on the side for the average raster
    # To do that without altering the raster, we create a new hidden axis "below"
    # the one of the raster, and plot the colorbar based on the raster.
    # It's ugly, but it works.
    exeampleMeanRasterShow.set_clim([0, dictYlim[indexType]])
    dictAxis["exampleRasterLegend"] = createAxisForRaster([0.35-rightAxisModifier, 0.82-bottomAxisModifier, 0.220, 0.16], bigFigure, disableAxis = True)
    bigFigure.colorbar(exeampleMeanRasterShow, ax = dictAxis["exampleRasterLegend"],
                        location = "left", ticks = dictTicksAverage[indexType])
    
    # We display the SD raster in small next to it
    dictAxis["exampleRasterSD"] = createAxisForRaster([0.60-rightAxisModifier, 0.90-bottomAxisModifier, 0.100, 0.087], bigFigure, disableAxis = True)
    dictAxis["exampleRasterSD"].zorder = 3 # We prepare to pu the legend below
    if 'exampleSDRaster' not in st.session_state:
        exampleSDRaster = loadNumpyArrayFromNextcloudTIFFile(webDavClientOptions,
                                                                pathOfMooseHQIRasters,
                                                                "SD/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[0]) + "_SD_HQI_Moose_" + indexType +  "_Timestep_30.tif")
        # exampleSDRaster = getRasterData(pathToDebugRaster)
    exampleSDRasterShow = show(exampleSDRaster, ax=dictAxis["exampleRasterSD"],
                                alpha=1, cmap = 'magma')
    exampleSDRasterShow = exampleSDRasterShow.get_images()[0]
    show(maskRasterMooseHQI, ax=dictAxis["exampleRasterSD"], alpha=1, cmap = cmapMooseMask)
    progressIndicator = 15
    loading_indicator.write("⚙ Downloading additional data to generate figure : " + str(progressIndicator) + "%")
    
    # We display a legend for the variability raster
    exampleSDRasterShow.set_clim([0, dictYlim[indexType]/2])
    dictAxis["exampleRasterSDLegend"] = createAxisForRaster([0.60-rightAxisModifier, 0.90-bottomAxisModifier, 0.125, 0.087], bigFigure, disableAxis = True)
    bigFigure.colorbar(exampleSDRasterShow, ax = dictAxis["exampleRasterSDLegend"],
                        location = "right", ticks = dictTicksSD[indexType])
    
    # We display some text to explain
    titleOfFigure = 'Habitat Quality Map for the Moose - Index of ' + dictTitle[indexType]
    bigFigure.text(0.32-rightAxisModifier - len(titleOfFigure) * 0.0019, 1.01-bottomAxisModifier, 
                    titleOfFigure, 
                    fontsize = 22,
                    fontweight = "bold",
                    color = "#2e3440")
    bigFigure.text(0.222-rightAxisModifier, 0.885-bottomAxisModifier, 
                    "Average between\nreplicates", 
                    fontsize = "medium",
                    color = "#2e3440",
                    weight = "medium",
                    horizontalalignment = "center")
    bigFigure.text(0.85-rightAxisModifier, 0.93-bottomAxisModifier, 
                    "Variability between\nreplicates", 
                    fontsize = "medium",
                    color = "#2e3440",
                    weight = "medium",
                    horizontalalignment = "center")
    bigFigure.add_artist(Line2D((0.27-rightAxisModifier,
                                  0.34-rightAxisModifier),
                                (0.90-bottomAxisModifier,
                                  0.90-bottomAxisModifier),
                                color='#2e3440', transform=bigFigure.transFigure,
                                zorder = 0,
                                ls  = "--"))
    bigFigure.add_artist(Line2D((0.80-rightAxisModifier,
                                  0.73-rightAxisModifier),
                                (0.945-bottomAxisModifier,
                                  0.945-bottomAxisModifier),
                                color='#2e3440', transform=bigFigure.transFigure,
                                zorder = 0,
                                ls  = "--"))
    
    # We add the axis for the rasters
    # We define their size
    meanRasterWidth = 0.150
    meanRasterHeight = meanRasterWidth * 1.0101010101
    SDrasterWidth = meanRasterWidth * 0.5050505051
    SDrasterHeight = meanRasterWidth * 0.5050505051 * 1.16
    # We define the space between them
    horizontalSpaceBetweenMeanAndSDRaster = 0.60 - 0.598
    horizontalSpaceBetweenBlobsOfRaster = 0.05 
    verticalSpaceBetweenBlobsOfRaster = 0.02
    for x in range(0, 3):
        for y in range (0, 4):
            rasterBlobOrigin = (0.15 + x * (horizontalSpaceBetweenBlobsOfRaster +
                                            meanRasterWidth +
                                            horizontalSpaceBetweenMeanAndSDRaster +  SDrasterWidth),
                                0.01 + y * (verticalSpaceBetweenBlobsOfRaster +
                                            meanRasterHeight))
            dictAxis["Mean-" + str((x, y))] = createAxisForRaster([rasterBlobOrigin[0], rasterBlobOrigin[1],
                                                              meanRasterWidth, meanRasterHeight],
                                                            bigFigure, disableAxis = True)
            
            dictAxis["SD-" + str((x, y))] = createAxisForRaster([rasterBlobOrigin[0] + meanRasterWidth + horizontalSpaceBetweenMeanAndSDRaster,
                                                                rasterBlobOrigin[1] + meanRasterHeight - SDrasterHeight,
                                                                SDrasterWidth, SDrasterHeight], bigFigure, disableAxis = True)
    
    
    # We indicate words to refer to what the axis mean
    bioHarvestedHQI = dictTransformBioHarvest[biomassHarvest]
    cutRegimeHQI = dictTransformCutRegim2[cutRegime]
    listOfScenarios = [str(bioHarvestedHQI) + " - " + str(cutRegimeHQI) + "\nBaseline",
                       str(bioHarvestedHQI) + " - " + str(cutRegimeHQI) + "\nRCP 4.5",
                       str(bioHarvestedHQI) + " - " + str(cutRegimeHQI) + "\nRCP 8.5"]
    for x in range(0, 3):
        bigFigure.text(0.27 + x * (horizontalSpaceBetweenBlobsOfRaster +
                                        meanRasterWidth +
                                        horizontalSpaceBetweenMeanAndSDRaster +  SDrasterWidth),
                        0.68, 
                        listOfScenarios[x], 
                        fontsize = 17,
                        fontname = "Arial",
                        fontweight = "bold",
                        color = "#2e3440",
                        horizontalalignment='center')
    listOfTimeStep = ["t = 0 (2023)", "t = 30 (2053)", "t = 50 (2073)", "t = 100 (2123)"]
    for y in range(0, 4):
        bigFigure.text(0.02,
                        0.59 - y * (verticalSpaceBetweenBlobsOfRaster +
                                  meanRasterHeight), 
                        listOfTimeStep[y], 
                        fontsize = 17,
                        fontweight = "bold",
                        color = "#2e3440")
    
    # We fill the axis
    # WARNING : Be careful about the order here !
    # from bottom to top, then left to right
    listOfMeanRasters = ["Average/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[0]) + "_Average_HQI_Moose_" + indexType +  "_Timestep_100.tif",
                          "Average/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[0]) + "_Average_HQI_Moose_" + indexType +  "_Timestep_50.tif",
                          "Average/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[0]) + "_Average_HQI_Moose_" + indexType +  "_Timestep_30.tif",
                          dictT0Raster[indexType],
                          "Average/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[1]) + "_Average_HQI_Moose_" + indexType +  "_Timestep_100.tif",
                          "Average/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[1]) + "_Average_HQI_Moose_" + indexType +  "_Timestep_50.tif",
                          "Average/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[1]) + "_Average_HQI_Moose_" + indexType +  "_Timestep_30.tif",
                          dictT0Raster[indexType],
                          "Average/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[2]) + "_Average_HQI_Moose_" + indexType +  "_Timestep_100.tif",
                          "Average/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[2]) + "_Average_HQI_Moose_" + indexType +  "_Timestep_50.tif",
                          "Average/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[2]) + "_Average_HQI_Moose_" + indexType +  "_Timestep_30.tif",
                          dictT0Raster[indexType]]
    
    listOfSDRasters = ["SD/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[0]) + "_SD_HQI_Moose_" + indexType +  "_Timestep_100.tif",
                          "SD/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[0]) + "_SD_HQI_Moose_" + indexType +  "_Timestep_50.tif",
                          "SD/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[0]) + "_SD_HQI_Moose_" + indexType +  "_Timestep_30.tif",
                          "",
                          "SD/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[1]) + "_SD_HQI_Moose_" + indexType +  "_Timestep_100.tif",
                          "SD/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[1]) + "_SD_HQI_Moose_" + indexType +  "_Timestep_50.tif",
                          "SD/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[1]) + "_SD_HQI_Moose_" + indexType +  "_Timestep_30.tif",
                          "",
                          "SD/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[2]) + "_SD_HQI_Moose_" + indexType +  "_Timestep_100.tif",
                          "SD/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[2]) + "_SD_HQI_Moose_" + indexType +  "_Timestep_50.tif",
                          "SD/" + str(bioHarvestedHQI) + "-" + str(cutRegimeHQI) + "-" + str(listClimateScenarios[2]) + "_SD_HQI_Moose_" + indexType +  "_Timestep_30.tif"]
    
    progressIndicator = 20
    loading_indicator.write("⚙ Downloading additional data to generate figure : " + str(progressIndicator) + "%")
    
    testingScript = False
    
    for x in range(0, 3):
        for y in range (0, 4):
            
            loading_indicator.write("⚙ Downloading additional data to generate figure : " + str(round(progressIndicator, 2)) + "%")
            if testingScript:
                meanRaster = exampleMeanRaster
            else:
                meanRaster = loadNumpyArrayFromNextcloudTIFFile(webDavClientOptions,
                                                                pathOfMooseHQIRasters,
                                                                listOfMeanRasters[y + 4 * x])
                # meanRaster = getRasterData(pathToDebugRaster)
            meanRasterShow = show(meanRaster, ax=dictAxis["Mean-" + str((x, y))],
                                          alpha=1, cmap = 'viridis')
            meanRasterShow = meanRasterShow.get_images()[0]
            meanRasterShow.set_clim([0, dictYlim[indexType]])
            show(maskRasterMooseHQI, ax=dictAxis["Mean-" + str((x, y))], alpha=1, cmap = cmapMooseMask)
    
            if y < 3: # No variability if t = 0 raster
                if testingScript:
                    SDRaster = exampleSDRaster
                else:
                    SDRaster = loadNumpyArrayFromNextcloudTIFFile(webDavClientOptions,
                                                                  pathOfMooseHQIRasters,
                                                                  listOfSDRasters[y + 4 * x])
                    # SDRaster = getRasterData(pathToDebugRaster)
                SDRasterShow = show(SDRaster, ax=dictAxis["SD-" + str((x, y))],
                                            alpha=1, cmap = 'magma')
                SDRasterShow = SDRasterShow.get_images()[0]
                SDRasterShow.set_clim([0, dictYlim[indexType]/2])
                show(maskRasterMooseHQI, ax=dictAxis["SD-" + str((x, y))], alpha=1, cmap = cmapMooseMask)
            if y == 3:
                SDRaster = np.zeros(SDRaster.shape)
                SDRasterShow = show(SDRaster, ax=dictAxis["SD-" + str((x, y))],
                                            alpha=1, cmap = 'magma')
                SDRasterShow = SDRasterShow.get_images()[0]
                SDRasterShow.set_clim([0, dictYlim[indexType]/2])
                show(maskRasterMooseHQI, ax=dictAxis["SD-" + str((x, y))], alpha=1, cmap = cmapMooseMask)
                
            progressIndicator += 6.6
    
    loading_indicator.write("")
    return(bigFigure)

#%% RETRIEVING ENCRYPTED DATA FOR BATCH 0.3

# Encrypted with createEcryptedPickleResults_Batch_0.3
# Password is in secrets

# This is for debug in spyder :
# os.chdir(r"D:\OneDrive - UQAM\1 - Projets\Thèse - Simulations Manawan projet DIVERSE\3 - Résultats\Streamlit_Results_Apps\batch0.3_results_analysis_GITHUB\streamlit_ManawanResultsAnalysis_batch0.3")

# We only load the dictionnary if not already loaded
if 'dictOfValuesForBasicMeasures' not in st.session_state:
    with open("./data/basicResults/encrypted_Batch_0.3_dictOfValuesForBasicMeasures.txt", "rb") as f:
        # Decrypt the data from memory
        file_contents = f.read()
        decrypted_data = Fernet(st.secrets["data_batch0_3_password"].encode()).decrypt(file_contents)
        retrieved_dict = pickle.loads(decrypted_data)
    
    # print(retrieved_dict)
    
    # We put the variable in st.session; this way, no need to re-do this part if already loaded.
    st.session_state.dictOfValuesForBasicMeasures = retrieved_dict

# We also retrieve the encrypted geodataframe
if 'geoDataFrame_areas_Manawan' not in st.session_state:
    with open("./data/geoDataFrame_areas_Manawan.txt", "rb") as f:
        # Decrypt the data from memory
        file_contents = f.read()
        decrypted_data = Fernet(st.secrets["data_batch0_3_password"].encode()).decrypt(file_contents)
        geoDataFrame_areas_Manawan = pickle.loads(decrypted_data)
        
    st.session_state.geoDataFrame_areas_Manawan = geoDataFrame_areas_Manawan
    
#%% DISPLAYING TITLE AND BANNER

st.markdown("<h1 style='text-align: center;'>" + "📊 LANDIS-II Manawan Results Visualisator" + "</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align: left;'>" + "Welcome ! Here, you can visualize results from the LANDIS-II simulation for the manawan area and the family area it contains." + "</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left;'>" + "Just select the variable you want to display, the area you want to look at, and the harvesting scenario. The results will display the values for 3 different climate scenarios : Baseline (no climate change), RCP 4.5 (climate change) and RCP 8.5 (intense climate change)." + "</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left;'>" + "Current version of the simulations : v0.3" + "</p>", unsafe_allow_html=True)
st.markdown("</br></br>", unsafe_allow_html=True)


#%% ASKING FOR VARIABLES TO DISPLAY

if 'variableList' not in st.session_state or "variableUnit" not in st.session_state:
    # We make the list of unique variables
    variableList = st.session_state.dictOfValuesForBasicMeasures.keys()
    char_to_remove = ' - '
    variableList = [s.split(char_to_remove)[0] for s in variableList]
    # We add the "HQI moose map" variable
    variableList.append("Moose Habitat Quality Index Maps")
    variableList.append("Area of all forest types")
    variableList = sorted(list(set(variableList)))
    
    # We make the list of familly territories
    famillyAreasNames = st.session_state.dictOfValuesForBasicMeasures.keys()
    char_to_remove = ' - '
    famillyAreasNames = [s.split(char_to_remove)[1] for s in famillyAreasNames]
    famillyAreasNames = sorted(list(set(famillyAreasNames)))

    # Dictionnary of variables units
    variableUnit = dict()
    for variable in variableList:
        if "Biomass" in variable:
            variableUnit[variable] = "(Mg)"
        elif "Max Age" in variable:
            variableUnit[variable] = "(Years)"
        elif "Forest" in variable or "Surface" in variable:
            variableUnit[variable] = "(Hectares)"
        else:
            variableUnit[variable] = ""
            
    # Dictionnaries to transform things back to select things
    dictTransformBioHarvest = {"50% of BAU":"BAU50%",
                               "100% of BAU":"BAU100%",
                               "200% of BAU":"BAU200%"}
    
    dictTransformCutRegim = {"Normal (cuts as in BAU)":"Normal",
                             "More clearcuts":"ClearCutsPlus",
                             "More partial cuts":"PartialCutsPlus"}
    
    dictTransformClimateScenario = {"Baseline":"Baseline",
                                    "RCP 4.5":"RCP45",
                                    "RCP 8.5":"RCP85"}

variable = st.selectbox("Choose the variable to display : ", variableList, list(variableList).index('Total Biomass'))
if variable != "Moose Habitat Quality Index Maps":
    familyArea = st.selectbox("Choose the family area to display : ", famillyAreasNames, list(famillyAreasNames).index('Territoire Manawan Entier'))
# otherVariable = st.selectbox("Choose the variable to display : ", st.session_state.dictOfValuesForBasicMeasures.keys())
biomassHarvest = st.selectbox("Choose the intensity of harvesting : ", ["50% of BAU", "100% of BAU", "200% of BAU"], 1)
cutRegime = st.selectbox("Choose the cutting regime : ", ["Normal (cuts as in BAU)",
                                                          "More clearcuts",
                                                          "More partial cuts"], 0)
if variable == "Moose Habitat Quality Index Maps":
    indexType = st.selectbox("Select a Moose HQI index type : ",
                                   ["DUSSAULT", "KOITZSCH"],
                                   0)
# climateScenario = st.selectbox("Choose the climate scenario : ", ["Baseline", "RCP 4.5", "RCP 8.5"])

#%% DISPLAYING GRAPHS OF BASIC MEASURES

if variable != "Moose Habitat Quality Index Maps" and variable != "Area of all forest types":
    
    colorDictionnary = [""]
    
    variableFinal = (variable + " - " + familyArea)
    
    # dataTest = st.session_state.dictOfValuesForBasicMeasures[variableFinal][dictTransformBioHarvest[biomassHarvest]][dictTransformCutRegim[cutRegime]][dictTransformClimateScenario[climateScenario]]["Mean"]
    
    
    # dataFrameTest = pd.DataFrame(st.session_state.dictOfValuesForBasicMeasures[variableFinal][dictTransformBioHarvest[biomassHarvest]][dictTransformCutRegim[cutRegime]][list(dictTransformClimateScenario.values())[0]]["Mean"],
    #                               columns=[list(dictTransformClimateScenario.keys())[0]])
    # dataFrameTest[list(dictTransformClimateScenario.keys())[1]] = st.session_state.dictOfValuesForBasicMeasures[variableFinal][dictTransformBioHarvest[biomassHarvest]][dictTransformCutRegim[cutRegime]][list(dictTransformClimateScenario.values())[1]]["Mean"]
    # dataFrameTest[list(dictTransformClimateScenario.keys())[2]] = st.session_state.dictOfValuesForBasicMeasures[variableFinal][dictTransformBioHarvest[biomassHarvest]][dictTransformCutRegim[cutRegime]][list(dictTransformClimateScenario.values())[2]]["Mean"]
    
    
    # st.line_chart(data=dataFrameTest,
    #               color=["#5e81ac", "#DCA22E", "#bf616a"], width=0, height=0,
    #               use_container_width=True)
    
    
    listOfTimesteps = range(0, 110, 10)
    listOfMeanDataSeries = [st.session_state.dictOfValuesForBasicMeasures[variableFinal][dictTransformBioHarvest[biomassHarvest]][dictTransformCutRegim[cutRegime]][list(dictTransformClimateScenario.values())[0]]["Mean"],
                            st.session_state.dictOfValuesForBasicMeasures[variableFinal][dictTransformBioHarvest[biomassHarvest]][dictTransformCutRegim[cutRegime]][list(dictTransformClimateScenario.values())[1]]["Mean"],
                            st.session_state.dictOfValuesForBasicMeasures[variableFinal][dictTransformBioHarvest[biomassHarvest]][dictTransformCutRegim[cutRegime]][list(dictTransformClimateScenario.values())[2]]["Mean"]]
    listOfSDSeries = [st.session_state.dictOfValuesForBasicMeasures[variableFinal][dictTransformBioHarvest[biomassHarvest]][dictTransformCutRegim[cutRegime]][list(dictTransformClimateScenario.values())[0]]["SD"],
                      st.session_state.dictOfValuesForBasicMeasures[variableFinal][dictTransformBioHarvest[biomassHarvest]][dictTransformCutRegim[cutRegime]][list(dictTransformClimateScenario.values())[1]]["SD"],
                      st.session_state.dictOfValuesForBasicMeasures[variableFinal][dictTransformBioHarvest[biomassHarvest]][dictTransformCutRegim[cutRegime]][list(dictTransformClimateScenario.values())[2]]["SD"]]
    listOfColors = ["#5e81ac", "#DCA22E", "#bf616a"]
    listOfDataSeriesNames = ["Baseline", "RCP 4.5", "RCP 8.5"]
    
    chartsCurvesAndConfidence = CreateAltairChartWithMeanAndSD(listOfTimesteps,
                                                               listOfDataSeriesNames,
                                                               listOfMeanDataSeries,
                                                               listOfSDSeries,
                                                               listOfColors,
                                                               ["Baseline", "RCP 4.5", "RCP 8.5"],
                                                               variable + " " + variableUnit[variable])
    
    # We display map with family area of interest
    displayPydeckMp(st.session_state.geoDataFrame_areas_Manawan, familyArea)
    
    # We display a title
    st.markdown("<h4 style='text-align: center;'>" +
                "Variation of " + str(variable) +
                # " in area of " + str(familyArea) +
                " in harvest scenario " + str(biomassHarvest) + " and " + str(cutRegime) +
                " accross the 3 climate scenarios" + "</h4>", unsafe_allow_html=True)

    # We display the chart
    st.altair_chart(chartsCurvesAndConfidence, use_container_width=True)


#%% DISPLAYING MAPS OF MOOSE HQI 


if variable == "Moose Habitat Quality Index Maps":
    
    figureMapMooseHQI = createFigureOfMooseHQI(biomassHarvest,
                                               cutRegime,
                                               indexType)
    st.pyplot(figureMapMooseHQI)

#%% STACKS OF FOREST AREA

# From https://altair-viz.github.io/user_guide/marks/area.html#normalized-stacked-area-chart

if variable == "Area of all forest types":
    
    # We make the data set to adapt to the Altair functions
    dictDataFrames = dict()
    
    for climateScenario in list(dictTransformClimateScenario.keys()):
    
        forestTypes = ["Young Maple Grove",
                       "Old Maple Grove",
                       "Young Deciduous Forest",
                       "Old Deciduous Forest",
                       "Young Coniferous Forest",
                       "Old Coniferous Forest",
                       "Young Mixed Forest",
                       "Old Mixed Forest"]
        
        timesteps = list(range(0, 110, 10))
        dataFrameForestTypesStack = pd.DataFrame([item for item in forestTypes for _ in range(len(timesteps))], columns=(["Variable"]))
        dataFrameForestTypesStack["Timestep"] = timesteps * len(forestTypes)
        dataFrameForestTypesStack["OrderInChart"] = [item for item in list(range(0, len(forestTypes))) for _ in range(len(timesteps))]
        
        listOfValues = list()
        for forestType in forestTypes:
            variableName = forestType + " - " + familyArea
            listOfValues.extend(st.session_state.dictOfValuesForBasicMeasures[variableName][dictTransformBioHarvest[biomassHarvest]][dictTransformCutRegim[cutRegime]][dictTransformClimateScenario[climateScenario]]["Mean"])

        dataFrameForestTypesStack["Values"] = listOfValues
        
        dictDataFrames[climateScenario] = dataFrameForestTypesStack.copy(deep=True)

    
    
    # We display the graphs
    
    colorList = ["#bf616a",
                "#742F36",
                "#ebcb8b",
                "#DCA22E",
                "#a3be8c",
                "#6E9051",
                "#8fbcbb",
                "#548C8B"]
    
    displayPydeckMp(st.session_state.geoDataFrame_areas_Manawan, familyArea)
    
    for climateScenario in list(dictTransformClimateScenario.keys()):
        # We display a title for the climate scenario of each graph
        st.markdown("<h2 style='text-align: center;'>" + climateScenario + "</h2>", unsafe_allow_html=True)
        stackChart = alt.Chart(dictDataFrames[climateScenario]).mark_area().encode(
                    alt.X("Timestep").axis(domain=False, tickSize=0),
                    alt.Y("Values:Q").stack("normalize"),
                    alt.Color("Variable:N").sort(forestTypes).scale(alt.Scale(range=colorList)),
                    alt.Order("OrderInChart"))
        
        st.altair_chart(stackChart, use_container_width=True)

#%% DEBUG : DISPLAY LOCAL VARIABLES AND SIZE ?

# def sizeof_fmt(num, suffix='B'):
#     for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
#         if abs(num) < 1024.0:
#             return "%3.1f%s%s" % (num, unit, suffix)
#         num /= 1024.0
#     return "%.1f%s%s" % (num, 'Yi', suffix)

# local_vars = locals()
# env_vars = [(key, sys.getsizeof(value)) for key, value in local_vars.items()]
# env_vars.sort(key=lambda x: x[1], reverse=True)

# for key, size in env_vars:
#     st.text(f"{key}: {sizeof_fmt(size)}")
#     print(f"{key}: {sizeof_fmt(size)}")

#%% UP NEXT

# Display map with visual zone along with result curves (put the family area raster
# safely in Nextcloud, change password)

# CHARTS SIDE BY SIDE FOR COMPARING

# https://altair-viz.github.io/user_guide/compound_charts.html
# See vertical/horizontal concatenation

# CHARTS WITH INTERACTIVE LEGEND TO ISOLATE LINES BY OPACITY ?
# https://github.com/altair-viz/altair/issues/984#issuecomment-591978609

# MAKE THINGS PRETTY
# Customise banner, etc.