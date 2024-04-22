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
from scipy.ndimage import zoom
import skimage

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
    
def readRasterTifffile(path):
    """
    Allows the reading of a .tif image data into a numpy array,
    without using GDAL !
    """
    return(tifffile.imread(path))

def bilinear_resize(arr, new_shape):
    """
    Resize a 2D NumPy array using bilinear interpolation.
    
    Args:
        arr (np.ndarray): The input 2D array to be resized.
        new_shape (tuple): The new shape (height, width) of the resized array.
    
    Returns:
        np.ndarray: The resized 2D array.
    """
    height, width = arr.shape
    new_height, new_width = new_shape

    # Create meshgrid of original and new coordinates
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    new_x = np.linspace(0, width - 1, new_width)
    new_y = np.linspace(0, height - 1, new_height)
    
    xx, yy = np.meshgrid(new_x, new_y)

    # Perform bilinear interpolation
    resized_arr = np.zeros((new_height, new_width))
    for i in range(new_height):
        for j in range(new_width):
            x1 = int(xx[i, j])
            y1 = int(yy[i, j])
            x2 = min(x1 + 1, width - 1)
            y2 = min(y1 + 1, height - 1)
            
            # Bilinear interpolation formula
            q11 = arr[y1, x1]
            q12 = arr[y2, x1]
            q21 = arr[y1, x2]
            q22 = arr[y2, x2]
            
            dx = xx[i, j] - x1
            dy = yy[i, j] - y1
            
            resized_arr[i, j] = (1 - dx) * (1 - dy) * q11 + \
                               (1 - dx) * dy * q12 + \
                               dx * (1 - dy) * q21 + \
                               dx * dy * q22
    
    return resized_arr

def loadNumpyArrayFromNextcloudTIFFile(webDavClientOptions, remoteFolder, remoteFileName, resize2X = True):
    """
    Done after a lot of tweaking around with perplexity.
    The function does not need to save a local file, and downloads things relatively
    quickly.
    Can be used with the nextcloud of ComputeCanada.
    Reduces the resoltuion by dividing by 2 to reduce memory usage from strealit; can be overriden.
    """
    webDavClient = Client(webDavClientOptions)
    res1 = webDavClient.resource(remoteFolder + remoteFileName)
    buffer = io.BytesIO()
    res1.write_to(buffer)
    tiff_buffer = io.BytesIO(buffer.getbuffer())
    image = tifffile.imread(tiff_buffer)
    # We remove NaN; if there is a single one, the resized map becomes all NaNs
    # because of skimage.transform.resize_local_mean.
    # REMOVED : Numpy function bilinear_resize given by perplexity works even better
    # and handles NaN with no problems.
    # image = np.nan_to_num(image, nan=0.0)
    if resize2X:
        # Uses bilinear interpolation (local mean). Should be better.
        # image = skimage.transform.resize_local_mean(image, (int(image.shape[0]/2), int(image.shape[1]/2)))
        # Uses nearest-neighbor interpolation. Not great for continuous variables as we're doing here.
        # image = zoom(image, (0.5, 0.5), order=0, prefilter=False)
        # Bilinear with custom numpy function
        image = bilinear_resize(image, (int(image.shape[0]/2), int(image.shape[1]/2)))
    # We close and delete objects, just in case, for RAM usage.
    buffer.close()
    tiff_buffer.close()
    del(webDavClient)
    del(buffer)
    del(tiff_buffer)
    return(image)


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

# webDavClientOptions = {
#   'webdav_hostname': "https://nextcloud.computecanada.ca/remote.php/dav/files/chardy2/",
#   'webdav_login':    "chardy2",
#   'webdav_password': "HCXko-MoKW8-xSQ7A-XGtxk-5DHJJ"
# }

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


def CreateDictionnaryOfMeanAndStandardDeviation(timesteps,
                                                variables,
                                                FactorLevels,
                                                dictionnaryOfBasicResults):
    """
    Takes a dictionnary containing one dataframe of results by key entries,
    and produces a nested dictionnary categorized by variables in thoses results,
    then two factors that changes between scenarios (e.g. Climate, management).
    Hence, for each variable, for each level of the first factor, and for each
    level of the second factor, for each level of the third factor,
    we get the mean and standard deviation of all of
    the results that have the levels of the first and second and thirdfactors.
    There is no averaging on the third factors here.
    In this script, we are dealing with results that have not been averaged on
    all of the stands inside a simulation; therefore, at the first averaging during this
    process, we will not use the meanPropagationOfError function to propagate the variance
    from average to average.
    
    This one takes 3 factors into account, given as a list of 3 lists with the
    factor levels.
    """
    
    dictOfValuesForBasicMeasures = dict()
    variables = list(variables)
    
    for variable in variables:
        dictOfValuesForBasicMeasures[variable] = dict()
        for factor1Level in FactorLevels[0]:
            dictOfValuesForBasicMeasures[variable][factor1Level] = dict()
            for factor2Level in FactorLevels[1]:
                dictOfValuesForBasicMeasures[variable][factor1Level][factor2Level] = dict()
                for factor3Level in FactorLevels[2]:
                    dictOfValuesForBasicMeasures[variable][factor1Level][factor2Level][factor3Level] = dict()
                    dictOfValuesForBasicMeasures[variable][factor1Level][factor2Level][factor3Level]["Mean"] = list()
                    dictOfValuesForBasicMeasures[variable][factor1Level][factor2Level][factor3Level]["SD"] = list()

    # Recupérer les bons dataframe avec first et second factor level avec des for;
    # Récuperer la bonne variable au bon timestep dans les dataframes
    # faire la moyenne, la mettre dans une liste
    # Mettre la liste dans le dictionnaire groupé
    for variable in variables:
        for factor1Level in FactorLevels[0]:
            for factor2Level in FactorLevels[1]:
                for factor3Level in FactorLevels[2]:
                    # We detect the replicates
                    for scenario in dictionnaryOfBasicResults.keys():
                        # factorsInScenarioName = scenario.replace('_', '-').split("-")
                        # print("Searching " + factorLevel)
                        if all(word in scenario for word in [factor1Level, factor2Level, factor3Level]):
                            # print("Found " +  str(scenario) + " !")
                            dataFrameWithFactorLevels = scenario
                    listOfMeanValuesForReplicates = list()
                    listOfStandardDeviationForReplicates = list()
                    for timestep in range(0, len(timesteps)):
                        # print(timestep)
                        totalValuesForTimestep = list()
                        for replicateNumber in range(0, len(dictionnaryOfBasicResults[dataFrameWithFactorLevels])):
                            # print(str(firstFactorLevel) + " " + str(secondFactorLevel))
                            totalValuesForTimestep.append(dictionnaryOfBasicResults[dataFrameWithFactorLevels][replicateNumber][variable][timestep])
        
                        # First mean has no propagation of variance
                        meanForTimestepForAllReplicates = np.mean(totalValuesForTimestep)
                        SDForTimestepForAllReplicates = np.std(totalValuesForTimestep)
                        listOfMeanValuesForReplicates.append(meanForTimestepForAllReplicates)
                        listOfStandardDeviationForReplicates.append(SDForTimestepForAllReplicates)
                        
                    dictOfValuesForBasicMeasures[variable][factor1Level][factor2Level][factor3Level]["Mean"] = listOfMeanValuesForReplicates
                    dictOfValuesForBasicMeasures[variable][factor1Level][factor2Level][factor3Level]["SD"] = listOfStandardDeviationForReplicates
                

    return dictOfValuesForBasicMeasures

#%% PLOTTING FUNCTIONS

def CreateAltairChartWithMeanAndSD(listOfTimesteps,
                                   listOfDataSeriesNames,
                                   listOfMeanDataSeries,
                                   listOfSDSeries,
                                   listOfColors,
                                   variableName):
    """Given a list of mean data series + list of SD,
    returns an altair chart layer superposing all of the curves +
    standard deviation areas around them.
    ListOfColors must same length as listOfMeanDataSeries and listOfSDSeries."""
    # We put everything as data frames
    listOfDataFrames = list()
    for i in range(0, len(listOfMeanDataSeries)):
        dataFrame = pd.DataFrame({'x': listOfTimesteps,
                                  'y': listOfMeanDataSeries[i],
                                  'y_upper': [x + y for x, y in zip(listOfMeanDataSeries[i], listOfSDSeries[i])],
                                  'y_downer': [x - y for x, y in zip(listOfMeanDataSeries[i], listOfSDSeries[i])]})
        listOfDataFrames.append(dataFrame)
    
    # Next, we create the charts
    listOfCurves = list()
    listOfSDArea = list()
    for i in range(0, len(listOfMeanDataSeries)):
        colorToUse = listOfColors[i]
        
        curve = alt.Chart(listOfDataFrames[i]).mark_line(color=colorToUse).encode(
            x=alt.X("x", axis=alt.Axis(title="Time step")),
            y=alt.Y("y", axis=alt.Axis(title=variableName)),
        ).properties(title=listOfDataSeriesNames[i])
        
        confidence_interval = alt.Chart(listOfDataFrames[i]).mark_area(opacity = 0.4, fill=colorToUse).encode(
            x=alt.X("x", axis=alt.Axis(title="Time step")),
            y='y_downer',
            y2='y_upper',
        ).properties(title=listOfDataSeriesNames[i])
        
        listOfCurves.append(curve)
        listOfSDArea.append(confidence_interval)
        
    # Then, we sum everything and return
    listOfSDArea.extend(listOfCurves)
    sumOfCharts = listOfSDArea[0]
    for i in range(1, len(listOfSDArea)):
        sumOfCharts += listOfSDArea[i]
    return(sumOfCharts.resolve_scale(color='independent').configure_axis(grid=False))


#%% RETRIEVING ENCRYPTED DATA FOR BATCH 0.3

# Encrypted with createEcryptedPickleResults_Batch_0.3
# Password is in secrets

# This is for debug in spyder :
# os.chdir(r"D:\OneDrive - UQAM\1 - Projets\Thèse - Simulations Manawan projet DIVERSE\3 - Résultats\Streamlit_Results_Apps\batch0.3_results_analysis")

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
                                                               variableFinal + " " + variableUnit[variable])
    
    st.altair_chart(chartsCurvesAndConfidence, use_container_width=True)

#%% DISPLAYING AREA CHARTS FOR FOREST TYPES




#%% DISPLAYING MAPS OF MOOSE HQI 

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
    loading_indicator.write("⚙ Generating figure : " + str(progressIndicator) + "%")
    
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
    loading_indicator.write("Generating figure : " + str(progressIndicator) + "%")
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
    loading_indicator.write("⚙ Generating figure : " + str(progressIndicator) + "%")
    
    
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
    loading_indicator.write("⚙ Generating figure : " + str(progressIndicator) + "%")
    
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
    loading_indicator.write("⚙ Generating figure : " + str(progressIndicator) + "%")
    
    testingScript = False
    
    for x in range(0, 3):
        for y in range (0, 4):
            
            loading_indicator.write("⚙ Generating figure : " + str(round(progressIndicator, 2)) + "%")
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

if variable == "Moose Habitat Quality Index Maps":
    
    figureMapMooseHQI = createFigureOfMooseHQI(biomassHarvest,
                                               cutRegime,
                                               indexType)
    st.pyplot(figureMapMooseHQI)


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

# AREA CHART FOR FOREST TYPES :
# https://altair-viz.github.io/user_guide/marks/area.html


# CHARTS SIDE BY SIDE FOR COMPARING

# https://altair-viz.github.io/user_guide/compound_charts.html
# See vertical/horizontal concatenation


# CHARTS WITH INTERACTIVE LEGEND TO ISOLATE LINES BY OPACITY
# https://github.com/altair-viz/altair/issues/984#issuecomment-591978609


# CREATE COMPLEX MAPS OF MOOSE HQI
# Make a dictionnary with the np.arrays of all the maps..and encrypt it ?
# Maybe no need to encrypt 

# MAKE THINGS PRETTY
# Customise banner, etc.

#%% DRAFT

# Test : Generating graphs with confidence intervals

# # Generate some sample data
# np.random.seed(0)
# x = np.linspace(0, 10, 100)
# y = np.sin(x) + np.random.normal(0, 0.1, 100)

# # Create a DataFrame with the data
# df = pd.DataFrame({'x': x, 'y': y})
# df["y_upper"] = df["y"] + 0.2
# df["y_downer"] = df["y"] - 0.2

# # Define the curve and the confidence interval
# curve = alt.Chart(df).mark_line().encode(
#     x='x',
#     y='y'
# )
# confidence_interval = alt.Chart(df).mark_area(opacity = 0.4).encode(
#     x='x',
#     y='y_downer',
#     y2='y_upper'
# )

# # Combine the curve and the confidence interval into a single chart
# chart = alt.layer(curve, confidence_interval)

# # Display the chart using Streamlit
# chartSum = curve
# for chart in [curve, confidence_interval]:
#     chartSum += chart
# st.altair_chart(chartSum, use_container_width=True)


# ACCESSING FILES FROM PRIVATE GITHUB REPO

# from bs4 import BeautifulSoup
# import re

# # define parameters for a request
# token = REMOVED
# owner = 'Klemet'
# repo = 'ManawanResults'
# folderWithCsv = 'batch_0.3/Basic Results csv/'

# # First, we get all of the csv files names

# # send a request
# r = requests.get(
#     'https://api.github.com/repos/{owner}/{repo}/contents/{path}'.format(
#     owner=owner, repo=repo, path=folderWithCsv),
#     headers={
#         'accept': 'application/vnd.github.v3.raw',
#         'authorization': 'token {}'.format(token)
#             }
#     )

# # Parsing the response : Request returns a complex text.
# # The file names are sandwished between two string of characters we can identify.
# # This makes for easy parsing with a regex expression.
# start_string = "https://api.github.com/repos/Klemet/ManawanResults/contents/batch_0.3/Basic%20Results%20csv/"
# end_string = "\?ref=main"
# long_string = str(r.text)
# substrings = re.findall(r"(?<=" + start_string + ").*?(?=" + end_string + ")", long_string)
# csvFilesList = list(set(substrings))

# # Now, we gather all of the files




# substrings = re.search(start_string + '(.*)' + end_string, long_string)
# print(substrings.group(1))

# # convert string to StringIO object
# string_io_obj = StringIO(r.text)

# # Load data to df
# df = pd.read_csv(string_io_obj, sep=",", index_col=0)


#%% PREVIOUS CODE

# pathToSaveFigures = glob.glob(r"D:\OneDrive - UQAM\1 - Projets\Thèse - Simulations Manawan projet DIVERSE\3 - Résultats\figures_simulations_batch_v0.2_Manawan\\")

# saveFigure = False

# # for familyRaster in listOfFamilyRasters[0:1]:
    
# familyRaster = listOfFamilyRasters[0]
    
# # We create the big figure
# widthFigureInches = 14
# heightFigureInches = 4
# bigFigure = plt.figure(figsize = [widthFigureInches, heightFigureInches])
# bigFigure.set_facecolor("#eceff4")
# # bigFigure.show()

# # We create the rectangles for the big figure
# widthFigure = 1
# heightFigure = 1
# normalSpacing = heightFigure * (1/2) * (1/9) * (1/3)
# widthOfRectangles = widthFigure - normalSpacing*2
# normalSpaceRelative = (1/2) * (1/9)
# # First, the rectangle that will contain the zone raster
# heighRectangleZoneRaster = heightFigure - normalSpacing * 2
# bigFigure.patches.extend([plt.Rectangle((normalSpacing,heightFigure - normalSpacing - heighRectangleZoneRaster),
#                                         (1/4)- normalSpacing*2 , heighRectangleZoneRaster,
#                                         fill=False, color='#eceff4', alpha=1,
#                                         transform=bigFigure.transFigure, figure=bigFigure)])

# # Then, the rectangle that will contain the graphs for the variables
# heighRectangleVariables = heightFigure*(2/3) - normalSpacing*2
# bigFigure.patches.extend([plt.Rectangle((normalSpacing * 2 + ((1/4)- normalSpacing*2),heightFigure - normalSpacing - heighRectangleZoneRaster),
#                                         (3/4)- normalSpacing , heighRectangleZoneRaster,
#                                         fill=False, color='#eceff4', alpha=1,
#                                         transform=bigFigure.transFigure, figure=bigFigure)])

# # We add lines to connect the rectangles of the big figure
# bigFigure.add_artist(Line2D((normalSpacing + ((1/4)- normalSpacing*2),normalSpacing + ((1/4)- normalSpacing*2) + normalSpacing),
#                             (normalSpacing + heighRectangleZoneRaster * 0.5,
#                              normalSpacing + heighRectangleZoneRaster * 0.5),
#                             color='#eceff4', transform=bigFigure.transFigure))
# bigFigure.show()    

# # We add the text titles for the big figure
# fontSizeForTextTitles = "xx-large"
# familyZoneName = familyRaster.split("/")[-1].split("\\")[-1][0:-4]
# familyZoneNameWrapped = '\n'.join(textwrap.wrap(familyZoneName, 30))

# bigFigure.text(1.4, 3.6, 
#                'Zone', 
#                fontsize = fontSizeForTextTitles,
#                fontname = "Arial",
#                fontweight = "bold",
#                color = "#2e3440",
#                transform=bigFigure.dpi_scale_trans)

# bigFigure.text(0.45, 2.9, 
#                familyZoneNameWrapped, 
#                fontsize = "large",
#                fontname = "Arial",
#                color = "#2e3440",
#                weight = "semibold",
#                transform=bigFigure.dpi_scale_trans)

# bigFigure.text(8.3, 3.6, 
#                'Variables', 
#                fontsize = fontSizeForTextTitles,
#                fontname = "Arial",
#                fontweight = "bold",
#                color = "#2e3440",
#                transform=bigFigure.dpi_scale_trans)

# # We create the axis that we will fill in the big figure, and put them
# # in a dictionnary to call them clearly with a name
# dictAxis = dict()

# # Axis for the zone raster
# dictAxis["ZoneRaster"] = createAxisForRaster([0.027, 0.045, 0.198, 0.6], bigFigure)

# # Axis for the 4 variables
# variablesList = ["Age Moyen", "Biomasse récoltée", "Biomasse Totale", "Surface Brulée"]
# horizontalSpacingBetweenAxis = 0.08
# verticalSpacingBetweenAxis = 0.07
# startBottomOfGrid = 0.175
# startLeftOfGrid = 0.3
# widthOfMaps = 0.21/1.95
# heightOfMaps = 0.85/1.68
# for i in range(0, 2):
#     # factor = "Baseline"
#     for j in range(0, 2):
#         dictAxis[str(variablesList[(i*2) + (j)])] = createAxisForRaster([startLeftOfGrid +(((i*2) + (j)) * (horizontalSpacingBetweenAxis + widthOfMaps)),
#                                                                                          startBottomOfGrid,
#                                                                                          widthOfMaps,
#                                                                                          heightOfMaps], bigFigure, False)
#         dictAxis[str(variablesList[(i*2) + (j)])].set_ylabel(str(variablesList[(i*2) + (j)]), fontsize = "medium")
#         dictAxis[str(variablesList[(i*2) + (j)])].set_xlim(0, 100)
#         dictAxis[str(variablesList[(i*2) + (j)])].tick_params(axis='both', which='major', labelsize="x-small")


# # Now, we fill everything with rasters and data

# # First, the zone raster
# # We open the rasters and the polygons of the zones
# backgroundManawanRaster = rasterio.open(directoryRastersImages[0] + "Territoire Familiaux/Background Manawan.tif")
# familyZonesShapefile = geopandas.read_file(directoryRastersImages[0] + "Territoire Familiaux/territoire_familiaux_manawan_2020_QuebecLambert.shp")
# # We select the polygon to display
# shapefileToPlot = familyZonesShapefile[familyZonesShapefile["FAMILLE"] == familyZoneName]   
# # We display the polygon
# shapefileToPlot.plot(ax = dictAxis["ZoneRaster"], edgecolor='#2e3440', linewidth = 2, facecolor="none")
# # We get the raster data to mask the background raster
# rasterZone = rasterio.open(familyRaster)
# rasterZoneData = getRasterData(familyRaster)
# # We mask the background raster to make a nice effect of contrasting the inside with the outside of the area
# backgroundInPolygon, transform = mask(backgroundManawanRaster, shapefileToPlot.geometry, invert=False)
# backgroundoutOfPolygon, transform = mask(backgroundManawanRaster, shapefileToPlot.geometry, invert=True)
# # We display everything
# show(backgroundInPolygon, ax=dictAxis["ZoneRaster"], alpha=0.9, transform=transform)
# show(backgroundoutOfPolygon, ax=dictAxis["ZoneRaster"], alpha=0.3, transform=transform)
# # We add the text indicating the area of the zone and the amount of forest in it
# ecoregionRasterData = getRasterData(directoryOfAllParameterFiles + "Core/raster_ecoregions_manawan_v1.0.tif")
# totalArea = np.sum(rasterZoneData > 0)
# percentageOfForest = (np.sum(ecoregionRasterData[np.where(rasterZoneData > 0)] > 0) / totalArea) * 100
# bigFigure.text(0.45, 2.60, 
#                '{:,}'.format(totalArea) + " hectares | " + str(round(percentageOfForest, 0)) + "% de forêts", 
#                fontsize = "large",
#                fontname = "Arial",
#                color = "#2e3440",
#                transform=bigFigure.dpi_scale_trans)
# # We remove the bars around the raster
# dictAxis["ZoneRaster"].spines['top'].set_visible(False)
# dictAxis["ZoneRaster"].spines['bottom'].set_visible(False)
# dictAxis["ZoneRaster"].spines['left'].set_visible(False)
# dictAxis["ZoneRaster"].spines['right'].set_visible(False)

# # We plot on each of the 4 variables
# # At each time, we choose the name of the right variable to go pick in the dictionnary of results
# variablesList = ["Age Moyen", "Biomasse récoltée", "Biomasse Totale", "Surface Brulée"]
# for i in range(0, 2):
#     # factor = "Baseline"
#     for j in range(0, 2):
#         variable = str(variablesList[(i*2) + (j)])
        
#         trueVariableName = ""
#         # We translate the variable name to read in the datatable; should have used the same names !
#         if variable == "Age Moyen":
#             trueVariableName = "Mean Max Age"
#             labelVariableName = "Age Moyen (années)"
#         elif variable == "Biomasse récoltée":
#             trueVariableName = "Biomass Harvested"
#             labelVariableName = "Biomasse récoltée (Mg)"
#         elif variable == "Biomasse Totale":
#             trueVariableName = "Total Biomass"
#             labelVariableName = "Biomasse totale (Mg)"
#         elif variable == "Surface Brulée":
#             trueVariableName = "Surface Burned"
#             labelVariableName = "Surface brulée (Ha)"
            
#         trueVariableName = trueVariableName + " - " + familyZoneName
        
#         # colorPaletteFactors = {"BAU-Baseline":"#5e81ac", "BAU-RCP45":"#DCA22E", "BAU-RCP85":"#bf616a"}
#         # We define the variation in the styles of the curves
#         # Colors = factor 3
#         # lines styles = factor 1
#         # marker style = factor 2
#         colors = itertools.cycle(("#5e81ac", "#DCA22E", "#bf616a"))
#         lineStyles = itertools.cycle(("dotted", "dashed", "solid"))
#         markerStyle = itertools.cycle((".", "x", "+"))
        
#         labelList = list()
#         labelColors = dict()
#         labelMarkers = dict()
#         labelStyles = dict()

#         # We put the image on which we will plot
#         # for factor1Level in Factor1:
#         factor1Level = Factor1[1]
#         # styleOfCurve = next(lineStyles)
#         styleOfCurve = "solid"
#         for factor2Level in Factor2:
#             markerOfCurve = next(markerStyle)
#             for factor3Level in Factor3:
#                 colorOfCurve = next(colors)
                
#                 # We get the values
#                 meanResults = dictOfValuesForBasicMeasures[trueVariableName][factor1Level][factor2Level][factor3Level]["Mean"]
#                 SDResults = dictOfValuesForBasicMeasures[trueVariableName][factor1Level][factor2Level][factor3Level]["SD"]
#                 sequenceOfYears = range(0, len(timesteps)*10, 10)
#                 # We get the label for the curve, which is used to make the legend
#                 labelForCurve = (str(factor1Level) + "-" + str(factor2Level) + "-" + str(factor3Level))
#                 labelList.append(labelForCurve)
#                 labelColors[labelForCurve] = colorOfCurve
#                 labelMarkers[labelForCurve] = markerOfCurve
#                 labelStyles[labelForCurve] = styleOfCurve
#                 f = dictAxis[variable].fill_between(sequenceOfYears, list(np.array(meanResults) + np.array(SDResults)),
#                          list(np.array(meanResults) - np.array(SDResults)), color=colorOfCurve,
#                          alpha=0.12, label = labelForCurve, zorder = 3)
#                 f2 = dictAxis[variable].fill_between(sequenceOfYears, list(np.array(meanResults) + np.array(SDResults)),
#                          list(np.array(meanResults) - np.array(SDResults)), color=colorOfCurve,
#                          facecolor="none", edgecolor=(0,0,0,0.06), linewidth=0.0, zorder = 3)
#                 # We plot the curve
#                 p = dictAxis[variable].plot(sequenceOfYears, meanResults,
#                          label = labelForCurve, color = colorOfCurve, linewidth = 0.7, zorder = 4,
#                          linestyle = styleOfCurve, marker=markerOfCurve, markersize=7)
        
#         # We finish by making the graph prettier
#         # We put the right ticks for the timesteps
#         dictAxis[variable].set_xticks([0,50,100],
#                                       ["0","50","100"])
#         # We put the right label
#         dictAxis[variable].set_ylabel(labelVariableName)
#         # We make the axis begin at 0
#         dictAxis[variable].set_ylim(0, dictAxis[variable].get_ylim()[1])
#         # We force the scientific notation
#         dictAxis[variable].ticklabel_format(axis="y", style="sci", scilimits=(0,3))   
#         # We add the image in the background
#         # We get the extent of the axis and plot it
#         extentOfAxis = dictAxis[variable].get_xlim() + dictAxis[variable].get_ylim()
#         img = plt.imread(directoryRastersImages[0] + "imagesVariables/" + str(variablesList[(i*2) + (j)]) + '.jpg')
#         dictAxis[variable].imshow(img, extent=extentOfAxis, aspect='auto', alpha=0.15, zorder = 0)
#         # We remove the spines that are too numerous
#         dictAxis[variable].spines['right'].set_visible(False)
#         dictAxis[variable].spines['top'].set_visible(False)
        
        
# # We add a "time" label for the y axis
# bigFigure.text(8.2, 0.20, 
#                'Temps (Années)', 
#                fontsize = "large",
#                fontweight = "medium",
#                color = "#2e3440",
#                transform=bigFigure.dpi_scale_trans)
# # When done, we create a custom legend for the variables section
# columnsNumber = 3
# # Because the labels are complex in this figure, we make the legend manually
# # See https://stackoverflow.com/questions/57340415/matplotlib-bar-plot-add-legend-from-categories-dataframe-column
# # labels = list(colorPaletteFactors.keys())
# # handles = [plt.Rectangle((0,0),1,1, color=labelColors[label]) for label in labelList]
# # dictAxis[str(variablesList[0])].legend(handles, labelList, prop={'size': 8}, loc = "upper center",
# #                                        bbox_to_anchor=(3.025, 1.30), ncol=columnsNumber, frameon=False)
   
# st.pyplot(bigFigure)
# We save the figure
# if saveFigure:
#     bigFigure.savefig(pathToSaveFigures[0] + "FigureZone-" + str(familyZoneName) + ".png", format='png', dpi=300, transparent=False)
#     plt.close()
    
#%% MAKING THE FIGURES : TYPES OF FORESTS IN THE AREA

# pathToSaveFigures = glob.glob(r"D:\OneDrive - UQAM\1 - Projets\Thèse - Simulations Manawan projet DIVERSE\3 - Résultats\figures_simulations_batch_v0.2_Manawan\\")

# saveFigure = False

# for familyRaster in listOfFamilyRasters:
    
#     # We create the big figure
#     widthFigureInches = 14
#     heightFigureInches = 4
#     bigFigure = plt.figure(figsize = [widthFigureInches, heightFigureInches])
#     bigFigure.set_facecolor("#eceff4")
#     bigFigure.show()
    
#     # We create the rectangles for the big figure
#     widthFigure = 1
#     heightFigure = 1
#     normalSpacing = heightFigure * (1/2) * (1/9) * (1/3)
#     widthOfRectangles = widthFigure - normalSpacing*2
#     normalSpaceRelative = (1/2) * (1/9)
#     # First, the rectangle that will contain the zone raster
#     heighRectangleZoneRaster = heightFigure - normalSpacing * 2
#     bigFigure.patches.extend([plt.Rectangle((normalSpacing,heightFigure - normalSpacing - heighRectangleZoneRaster),
#                                             (1/4)- normalSpacing*2 , heighRectangleZoneRaster,
#                                             fill=False, color='#eceff4', alpha=1,
#                                             transform=bigFigure.transFigure, figure=bigFigure)])
    
#     # Then, the rectangle that will contain the graphs for the variables
#     heighRectangleVariables = heightFigure*(2/3) - normalSpacing*2
#     bigFigure.patches.extend([plt.Rectangle((normalSpacing * 2 + ((1/4)- normalSpacing*2),heightFigure - normalSpacing - heighRectangleZoneRaster),
#                                             (3/4)- normalSpacing , heighRectangleZoneRaster,
#                                             fill=False, color='#eceff4', alpha=1,
#                                             transform=bigFigure.transFigure, figure=bigFigure)])
    
#     # We add lines to connect the rectangles of the big figure
#     bigFigure.add_artist(Line2D((normalSpacing + ((1/4)- normalSpacing*2),normalSpacing + ((1/4)- normalSpacing*2) + normalSpacing),
#                                 (normalSpacing + heighRectangleZoneRaster * 0.5,
#                                  normalSpacing + heighRectangleZoneRaster * 0.5),
#                                 color='#eceff4', transform=bigFigure.transFigure))
#     bigFigure.show()    

#     # We add the text titles for the big figure
#     fontSizeForTextTitles = "xx-large"
#     familyZoneName = familyRaster.split("/")[-1].split("\\")[-1][0:-4]
#     familyZoneNameWrapped = '\n'.join(textwrap.wrap(familyZoneName, 30))
    
#     bigFigure.text(1.4, 3.6, 
#                    'Zone', 
#                    fontsize = fontSizeForTextTitles,
#                    fontname = "Arial",
#                    fontweight = "bold",
#                    color = "#2e3440",
#                    transform=bigFigure.dpi_scale_trans)
    
#     bigFigure.text(0.45, 2.9, 
#                    familyZoneNameWrapped, 
#                    fontsize = "large",
#                    fontname = "Arial",
#                    color = "#2e3440",
#                    weight = "semibold",
#                    transform=bigFigure.dpi_scale_trans)
    
#     bigFigure.text(8.3, 3.6, 
#                    'Variables', 
#                    fontsize = fontSizeForTextTitles,
#                    fontname = "Arial",
#                    fontweight = "bold",
#                    color = "#2e3440",
#                    transform=bigFigure.dpi_scale_trans)
    
#     # We create the axis that we will fill in the big figure, and put them
#     # in a dictionnary to call them clearly with a name
#     dictAxis = dict()
    
#     # Axis for the zone raster
#     dictAxis["ZoneRaster"] = createAxisForRaster([0.027, 0.045, 0.198, 0.6], bigFigure)
    
#     #### CREATING MEAN GRAPHS ####
#     # Axis for the 4 variables
#     factorList = ["BAU-Baseline", "BAU-RCP45", "BAU-RCP85"]
#     horizontalSpacingBetweenAxis = 0.03
#     verticalSpacingBetweenAxis = 0.07
#     startBottomOfGrid = 0.34
#     startLeftOfGrid = 0.48
#     widthOfMaps = 0.28/1.95
#     heightOfMaps = 0.72/1.68
#     for i in range(0, 3):
#         dictAxis[str(factorList[i])] = createAxisForRaster([startLeftOfGrid +((i) * (horizontalSpacingBetweenAxis + widthOfMaps)),
#                                                                                          startBottomOfGrid,
#                                                                                          widthOfMaps,
#                                                                                          heightOfMaps], bigFigure, False)
#         # dictAxis[str(factorList[i])].set_ylabel("Surface (Ha)", fontsize = "medium")
#         dictAxis[str(factorList[i])].set_xlabel("Temps (années)", fontsize = "medium")
#         dictAxis[str(factorList[i])].set_xlim(0, 100)
#         dictAxis[str(factorList[i])].tick_params(axis='both', which='major', labelsize="x-small")
    
#     # Now, we fill everything with rasters and data
    
#     # First, the zone raster
#     # We open the rasters and the polygons of the zones
#     backgroundManawanRaster = rasterio.open(directoryRastersImages[0] + "Territoire Familiaux/Background Manawan.tif")
#     familyZonesShapefile = geopandas.read_file(directoryRastersImages[0] + "Territoire Familiaux/territoire_familiaux_manawan_2020_QuebecLambert.shp")
#     # We select the polygon to display
#     shapefileToPlot = familyZonesShapefile[familyZonesShapefile["FAMILLE"] == familyZoneName]   
#     # We display the polygon
#     shapefileToPlot.plot(ax = dictAxis["ZoneRaster"], edgecolor='#2e3440', linewidth = 2, facecolor="none")
#     # We get the raster data to mask the background raster
#     rasterZone = rasterio.open(familyRaster)
#     rasterZoneData = getRasterData(familyRaster)
#     # We mask the background raster to make a nice effect of contrasting the inside with the outside of the area
#     backgroundInPolygon, transform = mask(backgroundManawanRaster, shapefileToPlot.geometry, invert=False)
#     backgroundoutOfPolygon, transform = mask(backgroundManawanRaster, shapefileToPlot.geometry, invert=True)
#     # We display everything
#     show(backgroundInPolygon, ax=dictAxis["ZoneRaster"], alpha=0.9, transform=transform)
#     show(backgroundoutOfPolygon, ax=dictAxis["ZoneRaster"], alpha=0.3, transform=transform)
#     # We add the text indicating the area of the zone and the amount of forest in it
#     ecoregionRasterData = getRasterData(directoryOfAllParameterFiles + "Core/raster_ecoregions_manawan_v1.0.tif")
#     totalArea = np.sum(rasterZoneData > 0)
#     percentageOfForest = (np.sum(ecoregionRasterData[np.where(rasterZoneData > 0)] > 0) / totalArea) * 100
#     bigFigure.text(0.45, 2.60, 
#                    '{:,}'.format(totalArea) + " hectares | " + str(round(percentageOfForest, 0)) + "% de forêts", 
#                    fontsize = "large",
#                    fontname = "Arial",
#                    color = "#2e3440",
#                    transform=bigFigure.dpi_scale_trans)
#     # We remove the bars around the raster
#     dictAxis["ZoneRaster"].spines['top'].set_visible(False)
#     dictAxis["ZoneRaster"].spines['bottom'].set_visible(False)
#     dictAxis["ZoneRaster"].spines['left'].set_visible(False)
#     dictAxis["ZoneRaster"].spines['right'].set_visible(False)
    
#     # We plot on each of the 4 variables
#     # At each time, we choose the name of the right variable to go pick in the dictionnary of results
#     #### FILLING MEAN GRAPHS ####
#     factorList = ["BAU-Baseline", "BAU-RCP45", "BAU-RCP85"]
#     for i in range(0, 3):
        
#         sequenceOfYears = range(0, len(timesteps)*10, 10)
        
#         # We define the color vector
#         colorVectorStackPlotAreas = ["#bf616a",
#                                     "#742F36",
#                                     "#ebcb8b",
#                                     "#DCA22E",
#                                     "#a3be8c",
#                                     "#6E9051",
#                                     "#8fbcbb",
#                                     "#548C8B"]
  
#         dictAxis[str(factorList[i])].stackplot(sequenceOfYears,
#                                               dictOfValuesForBasicMeasures["Young Maple Grove - " + familyZoneName][factorList[i]]["Mean"],
#                                               dictOfValuesForBasicMeasures["Old Maple Grove - " + familyZoneName][factorList[i]]["Mean"],
#                                               dictOfValuesForBasicMeasures["Young Deciduous Forest - " + familyZoneName][factorList[i]]["Mean"],
#                                               dictOfValuesForBasicMeasures["Old Deciduous Forest - " + familyZoneName][factorList[i]]["Mean"],
#                                               dictOfValuesForBasicMeasures["Young Coniferous Forest - " + familyZoneName][factorList[i]]["Mean"],
#                                               dictOfValuesForBasicMeasures["Old Coniferous Forest - " + familyZoneName][factorList[i]]["Mean"],
#                                               dictOfValuesForBasicMeasures["Young Mixed Forest - " + familyZoneName][factorList[i]]["Mean"],
#                                               dictOfValuesForBasicMeasures["Old Mixed Forest - " + familyZoneName][factorList[i]]["Mean"],
#                                               colors = colorVectorStackPlotAreas)
        
#         # We finish by making the graph prettier
#         # We put the right ticks for the timesteps
#         dictAxis[str(factorList[i])].set_xticks([0,50,100],
#                                       ["0","50","100"])
#         # We make the axis begin at 0
#         # Don't know why, but matplotlib stackplot leaves a hugly border on top. Gotta readjust y limit.
#         # But for that, gotta get the maximum
#         ylimMax = (dictOfValuesForBasicMeasures["Young Maple Grove - " + familyZoneName][factorList[i]]["Mean"][0] +
#                         dictOfValuesForBasicMeasures["Old Maple Grove - " + familyZoneName][factorList[i]]["Mean"][0] +
#                         dictOfValuesForBasicMeasures["Young Deciduous Forest - " + familyZoneName][factorList[i]]["Mean"][0] +
#                         dictOfValuesForBasicMeasures["Old Deciduous Forest - " + familyZoneName][factorList[i]]["Mean"][0] +
#                         dictOfValuesForBasicMeasures["Young Coniferous Forest - " + familyZoneName][factorList[i]]["Mean"][0] +
#                         dictOfValuesForBasicMeasures["Old Coniferous Forest - " + familyZoneName][factorList[i]]["Mean"][0] +
#                         dictOfValuesForBasicMeasures["Young Mixed Forest - " + familyZoneName][factorList[i]]["Mean"][0] +
#                         dictOfValuesForBasicMeasures["Old Mixed Forest - " + familyZoneName][factorList[i]]["Mean"][0])
#         dictAxis[str(factorList[i])].set_ylim(0, ylimMax)
#         # We force the scientific notation
#         dictAxis[str(factorList[i])].ticklabel_format(axis="y", style="sci", scilimits=(0,3))  
#         # We add title above plot
#         # dictAxis[str(factorList[i])].set_title(str(factorList[i]), y = 1.05)
#         dictAxis[str(factorList[i])].set_title(str(factorList[i]), y = 1.05, fontname = "Arial")
#         # We limit the amount of ticks
#         dictAxis[str(factorList[i])].locator_params(nbins=4)
#         # We remove the spines that are too numerous
#         dictAxis[str(factorList[i])].spines['right'].set_visible(False)
#         dictAxis[str(factorList[i])].spines['top'].set_visible(False)
#         # We remove the x axis
#         dictAxis[str(factorList[i])].get_xaxis().set_visible(False)
#         dictAxis[str(factorList[i])].xaxis.set_ticks_position('none')
#         dictAxis[str(factorList[i])].spines['bottom'].set_visible(False)
#         # We remove the y axis + label if we are not the first graph
#         if (i > 0):
#             plt.setp(dictAxis[str(factorList[i])].get_yticklabels(), visible=False)
#             dictAxis[str(factorList[i])].spines['left'].set_visible(False)
#             dictAxis[str(factorList[i])].get_yaxis().set_visible(False)
#             dictAxis[str(factorList[i])].yaxis.set_ticks_position('none')
            
#     # When done, we create a custom legend for the variables section
#     # Because the labels are complex in this figure, we make the legend manually
#     # See https://stackoverflow.com/questions/57340415/matplotlib-bar-plot-add-legend-from-categories-dataframe-column
#     legendLabels = ["Forêt mixte\nmature",
#                     "Forêt mixte\njeune",
#                     "Forêt de\nconifères mature",
#                     "Forêt de\nconifères jeune",
#                     "Forêt feuillue\nmature",
#                     "Forêt feuillue\njeune",
#                     "Érablière\nmature",
#                     "Érablière\njeune",]
#     colorsLegend = dict(zip(legendLabels,list(reversed(colorVectorStackPlotAreas))))
  
#     labels = list(colorsLegend.keys())
#     handles = [plt.Rectangle((0,0),1,1, color=colorsLegend[label]) for label in labels]
#     dictAxis[str(factorList[0])].legend(handles, colorsLegend, prop={'size': 8, 'family':"Arial"},
#                                         loc = "upper center", bbox_to_anchor=(-1.10, 1.10), ncol=1, labelspacing = 1, frameon=False)
            
#     #### VARIABILITY GRAPH ####
#     # And now, the cherry on the cake : we add a variance graph above all of this !
    
#     # We start by adding some flavor text to explain
#     bigFigure.add_artist(Line2D((0.37,
#                                   0.46),
#                                 (0.16,
#                                   0.16),
#                                 color='#2e3440', transform=bigFigure.transFigure,
#                                 zorder = 0,
#                                 ls  = "--"))
#     bigFigure.add_artist(Line2D((0.37,
#                                   0.46),
#                                 (0.52,
#                                   0.52),
#                                 color='#2e3440', transform=bigFigure.transFigure,
#                                 zorder = 0,
#                                 ls  = "--"))
#     bigFigure.add_artist(Line2D((0.37,
#                                   0.37),
#                                 (0.08,
#                                   0.79),
#                                 color='#2e3440', transform=bigFigure.transFigure,
#                                 zorder = 0,
#                                 ls  = "-"))
#     bigFigure.text(5.85, 2.19, 
#                     "Moyenne des\nsurfaces entre\nréplicats (Ha)", 
#                     fontsize = "medium",
#                     fontname = "Arial",
#                     color = "#2e3440",
#                     transform=bigFigure.dpi_scale_trans,
#                     weight = "medium",
#                     horizontalalignment = "center")
#     bigFigure.text(5.85, 0.74, 
#                     "Variabilité des\nsurfaces entre\nréplicats (Ha)", 
#                     fontsize = "medium",
#                     fontname = "Arial",
#                     color = "#2e3440",
#                     transform=bigFigure.dpi_scale_trans,
#                     weight = "medium",
#                     horizontalalignment = "center")
    
#     # Now, we plot the variability graphs
#     factorList = ["BAU-Baseline", "BAU-RCP45", "BAU-RCP85"]
#     horizontalSpacingBetweenAxis = 0.03
#     verticalSpacingBetweenAxis = 0.07
#     startBottomOfGrid = 0.12
#     startLeftOfGrid = 0.48
#     widthOfMaps = 0.28/1.95
#     heightOfMaps = 0.25/1.68
#     for i in range(0, 3):
#         dictAxis[str(factorList[i]) + "-SD"] = createAxisForRaster([startLeftOfGrid +((i) * (horizontalSpacingBetweenAxis + widthOfMaps)),
#                                                                                          startBottomOfGrid,
#                                                                                          widthOfMaps,
#                                                                                          heightOfMaps], bigFigure, False)
#         dictAxis[str(factorList[i]) + "-SD"].set_xlim(0, 100)
#         dictAxis[str(factorList[i]) + "-SD"].tick_params(axis='both', which='major', labelsize="x-small")
#         dictAxis[str(factorList[i]) + "-SD"].spines['right'].set_visible(False)
#         dictAxis[str(factorList[i]) + "-SD"].spines['top'].set_visible(False)
#         # We remove the y axis + label if we are not the first graph
#         if (i > 0):
#             plt.setp(dictAxis[str(factorList[i]) + "-SD"].get_yticklabels(), visible=False)
#             dictAxis[str(factorList[i]) + "-SD"].spines['left'].set_visible(False)
#             dictAxis[str(factorList[i]) + "-SD"].get_yaxis().set_visible(False)
#             dictAxis[str(factorList[i]) + "-SD"].yaxis.set_ticks_position('none')
#         # We define the x axis
#         dictAxis[str(factorList[i]) + "-SD"].set_xlabel("Temps (années)", fontsize = "medium")
#         # We remove the "time" mention if it's not the middle graph
#         if (i != 1):
#             dictAxis[str(factorList[i]) + "-SD"].set_xlabel("", fontsize = "medium")
#         # We limit the amount of ticks
#         dictAxis[str(factorList[i]) + "-SD"].locator_params(nbins=2)
#         # We force the scientific notation
#         dictAxis[str(factorList[i]) + "-SD"].ticklabel_format(axis="y", style="sci", scilimits=(0,3)) 
#         # We set the background as the rest of the figure
#         dictAxis[str(factorList[i]) + "-SD"].set_facecolor('#eceff4')
        
#         # Now, we compute the sum of SDs to display it
#         sumOfSd = (np.array(dictOfValuesForBasicMeasures["Young Maple Grove - " + familyZoneName][factorList[i]]["SD"]) +
#                    np.array(dictOfValuesForBasicMeasures["Old Maple Grove - " + familyZoneName][factorList[i]]["SD"]) +
#                    np.array(dictOfValuesForBasicMeasures["Young Deciduous Forest - " + familyZoneName][factorList[i]]["SD"]) +
#                    np.array(dictOfValuesForBasicMeasures["Old Deciduous Forest - " + familyZoneName][factorList[i]]["SD"]) +
#                    np.array(dictOfValuesForBasicMeasures["Young Coniferous Forest - " + familyZoneName][factorList[i]]["SD"]) +
#                    np.array(dictOfValuesForBasicMeasures["Old Coniferous Forest - " + familyZoneName][factorList[i]]["SD"]) +
#                    np.array(dictOfValuesForBasicMeasures["Young Mixed Forest - " + familyZoneName][factorList[i]]["SD"]) +
#                    np.array(dictOfValuesForBasicMeasures["Old Mixed Forest - " + familyZoneName][factorList[i]]["SD"]))
        
#         dictAxis[str(factorList[i]) + "-SD"].stackplot(sequenceOfYears,
#                                               dictOfValuesForBasicMeasures["Young Maple Grove - " + familyZoneName][factorList[i]]["SD"],
#                                               dictOfValuesForBasicMeasures["Old Maple Grove - " + familyZoneName][factorList[i]]["SD"],
#                                               dictOfValuesForBasicMeasures["Young Deciduous Forest - " + familyZoneName][factorList[i]]["SD"],
#                                               dictOfValuesForBasicMeasures["Old Deciduous Forest - " + familyZoneName][factorList[i]]["SD"],
#                                               dictOfValuesForBasicMeasures["Young Coniferous Forest - " + familyZoneName][factorList[i]]["SD"],
#                                               dictOfValuesForBasicMeasures["Old Coniferous Forest - " + familyZoneName][factorList[i]]["SD"],
#                                               dictOfValuesForBasicMeasures["Young Mixed Forest - " + familyZoneName][factorList[i]]["SD"],
#                                               dictOfValuesForBasicMeasures["Old Mixed Forest - " + familyZoneName][factorList[i]]["SD"],
#                                               colors = colorVectorStackPlotAreas)

#     # We force the limits to be the same as the main graph
#     # Actually, we set them to be 33% of the max ylim, as these graphs are 33% smaller.
#     # This way, the scale is exactly the same as the graphe above.
#     # Except if this is not enough to display the curves.
#     ylimForVarGraph = 0
#     for i in range(0, 3):
#         ylimForVarGraph = max(ylimForVarGraph, dictAxis[str(factorList[i]) + "-SD"].get_ylim()[1], dictAxis[str(factorList[i])].get_ylim()[1] * 0.33)
#     for i in range(0, 3):
#         dictAxis[str(factorList[i]) + "-SD"].set_ylim(0, ylimForVarGraph) 
    
#     # We save the figure
#     if saveFigure:
#         bigFigure.savefig(pathToSaveFigures[0] + "FigureZone-" + str(familyZoneName) + ".png", format='png', dpi=300, transparent=False)
#         plt.close()
        
        
# #%% MAKING THE FIGURES : MOOSE KOITZSCH AND DUSSAULT HQI MAPS

# pathToSaveFigures = glob.glob(r"D:\OneDrive - UQAM\1 - Projets\Thèse - Simulations Manawan projet DIVERSE\3 - Résultats\figures_simulations_batch_v0.2_Manawan\\")

# pathOfMooseHQIRasters = r"D:\OneDrive - UQAM\1 - Projets\Thèse - Simulations Manawan projet DIVERSE\3 - Résultats\simulation_batch_v0.2_Manawan_Narval_NOHARVEST\Moose_HQI\\"

# saveFigure = False

# dictT0Raster = {"KOITZSCH":"Koitzch_HQI_Moose_Timestep_0_batch_v0.2.tif",
#                 "DUSSAULT":"Dussault_HQI_Moose_Timestep_0_batch_v0.2.tif"}

# dictTitle = {"KOITZSCH":"Koitzch (2002)",
#              "DUSSAULT":"Dussault et al. (2006)"}

# dictYlim = {"KOITZSCH":1,
#              "DUSSAULT":1.2}

# dictTicksAverage = {"KOITZSCH":[0, 0.5, 1],
#                     "DUSSAULT":[0, 0.6, 1.2]}

# dictTicksSD = {"KOITZSCH":[0, 0.5],
#                "DUSSAULT":[0, 0.6]}

# for indexType in ["KOITZSCH", "DUSSAULT"]:

#     # We create the big figure
#     widthFigureInches = 14
#     heightFigureInches = 4*3
#     bigFigure = plt.figure(figsize = [widthFigureInches, heightFigureInches])
#     bigFigure.set_facecolor("#eceff4")
#     bigFigure.show()
    
#     # We create the axis that we will fill in the big figure, and put them
#     # in a dictionnary to call them clearly with a name
#     dictAxis = dict()
    
#     # We define the "Mask" for the raster, to avoid displaying 0 values/pixels
#     # This mask is simply the timestep 0 raster.
#     maskRasterMooseHQI = getRasterData(pathOfMooseHQIRasters + dictT0Raster[indexType])
#     maskRasterMooseHQI = np.where(maskRasterMooseHQI > 0, 1, 0)
#     # We make the cmap for the mask
#     # Color map for fire raster
#     levels = [0, 1, 999]
#     clrs = ['#FFFFFF', '#FFFFFF00'] 
#     cmapMooseMask, norm = matplotlib.colors.from_levels_and_colors(levels, clrs)
    
#     # MOOVING THE RASTERS AXIS
#     # Used for the creation of axis bellow
#     bottomAxisModifier = 0.05
#     rightAxisModifier = 0.03
    
#     # We create the "Example" on the side
#     # We display the average raster for t = 30 for a scenario
#     dictAxis["exampleRaster"] = createAxisForRaster([0.4-rightAxisModifier, 0.8-bottomAxisModifier, 0.198, 0.2], bigFigure, disableAxis = True)
#     dictAxis["exampleRaster"].zorder = 3 # We prepare to pu the legend below
#     exampleMeanRaster = getRasterData(pathOfMooseHQIRasters + "Average and SD Rasters/Average/BAU-Baseline_Average_HQI_Moose_" + indexType + "_Timestep_30.tif")
#     exeampleMeanRasterShow = show(exampleMeanRaster, ax=dictAxis["exampleRaster"],
#                                   alpha=1, cmap = 'viridis')
#     exeampleMeanRasterShow = exeampleMeanRasterShow.get_images()[0]
#     show(maskRasterMooseHQI, ax=dictAxis["exampleRaster"], alpha=1, cmap = cmapMooseMask)
    
#     # We display a legend on the side for the average raster
#     # To do that without altering the raster, we create a new hidden axis "below"
#     # the one of the raster, and plot the colorbar based on the raster.
#     # It's ugly, but it works.
#     exeampleMeanRasterShow.set_clim([0, dictYlim[indexType]])
#     dictAxis["exampleRasterLegend"] = createAxisForRaster([0.35-rightAxisModifier, 0.82-bottomAxisModifier, 0.220, 0.16], bigFigure, disableAxis = True)
#     bigFigure.colorbar(exeampleMeanRasterShow, ax = dictAxis["exampleRasterLegend"],
#                        location = "left", ticks = dictTicksAverage[indexType])
    
#     # We display the SD raster in small next to it
#     dictAxis["exampleRasterSD"] = createAxisForRaster([0.60-rightAxisModifier, 0.90-bottomAxisModifier, 0.100, 0.087], bigFigure, disableAxis = True)
#     dictAxis["exampleRasterSD"].zorder = 3 # We prepare to pu the legend below
#     exampleSDRaster = getRasterData(pathOfMooseHQIRasters + "Average and SD Rasters/SD/BAU-Baseline_SD_HQI_Moose_" + indexType + "_Timestep_30.tif")
#     exampleSDRasterShow = show(exampleSDRaster, ax=dictAxis["exampleRasterSD"],
#                                alpha=1, cmap = 'magma')
#     exampleSDRasterShow = exampleSDRasterShow.get_images()[0]
#     show(maskRasterMooseHQI, ax=dictAxis["exampleRasterSD"], alpha=1, cmap = cmapMooseMask)
    
#     # We display a legend for the variability raster
#     exampleSDRasterShow.set_clim([0, dictYlim[indexType]/2])
#     dictAxis["exampleRasterSDLegend"] = createAxisForRaster([0.60-rightAxisModifier, 0.90-bottomAxisModifier, 0.125, 0.087], bigFigure, disableAxis = True)
#     bigFigure.colorbar(exampleSDRasterShow, ax = dictAxis["exampleRasterSDLegend"],
#                        location = "right", ticks = dictTicksSD[indexType])
    
#     # We display some text to explain
#     titleOfFigure = 'Carte de Qualité d\'Habitat pour l\'Orignal - Indice de ' + dictTitle[indexType]
#     bigFigure.text(0.32-rightAxisModifier - len(titleOfFigure) * 0.0019, 1.01-bottomAxisModifier, 
#                    titleOfFigure, 
#                    fontsize = 22,
#                    fontname = "Arial",
#                    fontweight = "bold",
#                    color = "#2e3440")
#     bigFigure.text(0.23-rightAxisModifier, 0.885-bottomAxisModifier, 
#                     "Moyenne entre\nréplicats", 
#                     fontsize = "medium",
#                     fontname = "Arial",
#                     color = "#2e3440",
#                     weight = "medium",
#                     horizontalalignment = "center")
#     bigFigure.text(0.85-rightAxisModifier, 0.93-bottomAxisModifier, 
#                     "Variabilité entre\nréplicats", 
#                     fontsize = "medium",
#                     fontname = "Arial",
#                     color = "#2e3440",
#                     weight = "medium",
#                     horizontalalignment = "center")
#     bigFigure.add_artist(Line2D((0.27-rightAxisModifier,
#                                   0.34-rightAxisModifier),
#                                 (0.90-bottomAxisModifier,
#                                   0.90-bottomAxisModifier),
#                                 color='#2e3440', transform=bigFigure.transFigure,
#                                 zorder = 0,
#                                 ls  = "--"))
#     bigFigure.add_artist(Line2D((0.80-rightAxisModifier,
#                                   0.73-rightAxisModifier),
#                                 (0.945-bottomAxisModifier,
#                                   0.945-bottomAxisModifier),
#                                 color='#2e3440', transform=bigFigure.transFigure,
#                                 zorder = 0,
#                                 ls  = "--"))
    
#     # We add the axis for the rasters
#     # We define their size
#     meanRasterWidth = 0.150
#     meanRasterHeight = meanRasterWidth * 1.0101010101
#     SDrasterWidth = meanRasterWidth * 0.5050505051
#     SDrasterHeight = meanRasterWidth * 0.5050505051 * 1.16
#     # We define the space between them
#     horizontalSpaceBetweenMeanAndSDRaster = 0.60 - 0.598
#     horizontalSpaceBetweenBlobsOfRaster = 0.05 
#     verticalSpaceBetweenBlobsOfRaster = 0.02
#     for x in range(0, 3):
#         for y in range (0, 4):
#             rasterBlobOrigin = (0.15 + x * (horizontalSpaceBetweenBlobsOfRaster +
#                                            meanRasterWidth +
#                                            horizontalSpaceBetweenMeanAndSDRaster +  SDrasterWidth),
#                                 0.01 + y * (verticalSpaceBetweenBlobsOfRaster +
#                                            meanRasterHeight))
#             dictAxis["Mean-" + str((x, y))] = createAxisForRaster([rasterBlobOrigin[0], rasterBlobOrigin[1],
#                                                              meanRasterWidth, meanRasterHeight],
#                                                             bigFigure, disableAxis = True)
            
#             dictAxis["SD-" + str((x, y))] = createAxisForRaster([rasterBlobOrigin[0] + meanRasterWidth + horizontalSpaceBetweenMeanAndSDRaster,
#                                                                rasterBlobOrigin[1] + meanRasterHeight - SDrasterHeight,
#                                                                SDrasterWidth, SDrasterHeight], bigFigure, disableAxis = True)
    
    
#     # We indicate words to refer to what the axis mean
#     listOfScenarios = ["BAU - Baseline", "BAU - RCP 4.5", "BAU - RCP 8.5"]
#     for x in range(0, 3):
#         bigFigure.text(0.21 + x * (horizontalSpaceBetweenBlobsOfRaster +
#                                        meanRasterWidth +
#                                        horizontalSpaceBetweenMeanAndSDRaster +  SDrasterWidth),
#                        0.7, 
#                        listOfScenarios[x], 
#                        fontsize = 17,
#                        fontname = "Arial",
#                        fontweight = "bold",
#                        color = "#2e3440")
#     listOfTimeStep = ["t = 0 (2023)", "t = 30 (2053)", "t = 50 (2073)", "t = 100 (2123)"]
#     for y in range(0, 4):
#         bigFigure.text(0.02,
#                        0.59 - y * (verticalSpaceBetweenBlobsOfRaster +
#                                   meanRasterHeight), 
#                        listOfTimeStep[y], 
#                        fontsize = 17,
#                        fontname = "Arial",
#                        fontweight = "bold",
#                        color = "#2e3440")
    
#     # We fill the axis
#     # WARNING : Be careful about the order here !
#     # from bottom to top, then left to right
#     listOfMeanRasters = [pathOfMooseHQIRasters + "Average and SD Rasters/Average/BAU-Baseline_Average_HQI_Moose_" + indexType +  "_Timestep_100.tif",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/Average/BAU-Baseline_Average_HQI_Moose_" + indexType +  "_Timestep_50.tif",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/Average/BAU-Baseline_Average_HQI_Moose_" + indexType +  "_Timestep_30.tif",
#                          pathOfMooseHQIRasters + dictT0Raster[indexType],
#                          pathOfMooseHQIRasters + "Average and SD Rasters/Average/BAU-RCP45_Average_HQI_Moose_" + indexType +  "_Timestep_100.tif",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/Average/BAU-RCP45_Average_HQI_Moose_" + indexType +  "_Timestep_50.tif",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/Average/BAU-RCP45_Average_HQI_Moose_" + indexType +  "_Timestep_30.tif",
#                          pathOfMooseHQIRasters + dictT0Raster[indexType],
#                          pathOfMooseHQIRasters + "Average and SD Rasters/Average/BAU-RCP85_Average_HQI_Moose_" + indexType +  "_Timestep_100.tif",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/Average/BAU-RCP85_Average_HQI_Moose_" + indexType +  "_Timestep_50.tif",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/Average/BAU-RCP85_Average_HQI_Moose_" + indexType +  "_Timestep_30.tif",
#                          pathOfMooseHQIRasters + dictT0Raster[indexType]]
    
#     listOfSDRasters = [pathOfMooseHQIRasters + "Average and SD Rasters/SD/BAU-Baseline_SD_HQI_Moose_" + indexType +  "_Timestep_100.tif",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/SD/BAU-Baseline_SD_HQI_Moose_" + indexType +  "_Timestep_50.tif",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/SD/BAU-Baseline_SD_HQI_Moose_" + indexType +  "_Timestep_30.tif",
#                          "",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/SD/BAU-RCP45_SD_HQI_Moose_" + indexType +  "_Timestep_100.tif",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/SD/BAU-RCP45_SD_HQI_Moose_" + indexType +  "_Timestep_50.tif",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/SD/BAU-RCP45_SD_HQI_Moose_" + indexType +  "_Timestep_30.tif",
#                          "",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/SD/BAU-RCP85_SD_HQI_Moose_" + indexType +  "_Timestep_100.tif",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/SD/BAU-RCP85_SD_HQI_Moose_" + indexType +  "_Timestep_50.tif",
#                          pathOfMooseHQIRasters + "Average and SD Rasters/SD/BAU-RCP85_SD_HQI_Moose_" + indexType +  "_Timestep_30.tif"]
    
#     for x in range(0, 3):
#         for y in range (0, 4):
            
#             meanRaster = getRasterData(listOfMeanRasters[y + 4 * x])
#             meanRasterShow = show(meanRaster, ax=dictAxis["Mean-" + str((x, y))],
#                                           alpha=1, cmap = 'viridis')
#             meanRasterShow = meanRasterShow.get_images()[0]
#             meanRasterShow.set_clim([0, dictYlim[indexType]])
#             show(maskRasterMooseHQI, ax=dictAxis["Mean-" + str((x, y))], alpha=1, cmap = cmapMooseMask)
    
#             if y < 3: # No variability if t = 0 raster
#                 SDRaster = getRasterData(listOfSDRasters[y + 4 * x])
#                 SDRasterShow = show(SDRaster, ax=dictAxis["SD-" + str((x, y))],
#                                            alpha=1, cmap = 'magma')
#                 SDRasterShow = SDRasterShow.get_images()[0]
#                 SDRasterShow.set_clim([0, dictYlim[indexType]/2])
#                 show(maskRasterMooseHQI, ax=dictAxis["SD-" + str((x, y))], alpha=1, cmap = cmapMooseMask)
#             if y == 3:
#                 SDRaster = np.zeros(getRasterData(listOfSDRasters[0]).shape)
#                 SDRasterShow = show(SDRaster, ax=dictAxis["SD-" + str((x, y))],
#                                            alpha=1, cmap = 'magma')
#                 SDRasterShow = SDRasterShow.get_images()[0]
#                 SDRasterShow.set_clim([0, dictYlim[indexType]/2])
#                 show(maskRasterMooseHQI, ax=dictAxis["SD-" + str((x, y))], alpha=1, cmap = cmapMooseMask)
