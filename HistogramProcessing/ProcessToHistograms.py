from itertools import count
import json
from pathlib import Path
import numpy as np
import IPython.display as disp
from osgeo import gdal
import pandas as pd



#29 images per year for 13 years
def split_images(array,total_bands,yearcount):
    """
        - total_bands = number of images per year * number of bands
    """
    split_dict = {}
    start = 0
    for i in range(1,yearcount + 1):
        #year : Total bands for the year
        split_dict[i] = array[:,:,start : start + total_bands]
        start = start + total_bands
    return split_dict

def split_to_year_bands(timestep_dict:dict,yearcount:int,bands:int,imagecountperyear:int):
    """
     - year_dict[year][band_no] = list()
    """
    year_dict = dict()
    for i in range(1,yearcount + 1):
        year_dict[i] = dict()
        for j in range(bands):
            year_dict[i][j] = list()
        #dim = [:,:,189]
        #Get the images of all bands for the year
        year_images =  timestep_dict[i]
        for k in range(imagecountperyear):
            #get the first set of bands for the particular image for the given year (1 to 27 for Surface and 1 to 2 for Temp)
            image_bands = year_images[:,:,k*bands:(k + 1)*bands]
            for l in range(bands):
                year_dict[i][l].append(image_bands[:,:,l])
    return year_dict


def filter_cropland(landcover_data):
    for i in range(landcover_data.shape[2]):
        is12 = np.where(landcover_data[:,:,i] == np.uint16(12))
        not12 = np.where(landcover_data[:,:,i] != np.uint16(12))
        landcover_data[:,:,i][is12] = 1
        landcover_data[:,:,i][not12] = 0
    return landcover_data

#Surface reflectance imagery is for 13 years while the landcover images are only for 12 years. You et al. extends thee landcover by conctenating the last image to match the same number of years as the surface reflextance
def extend_landcover(img,extend_years):
    for i in range(0,extend_years):
        img = np.concatenate((img,img[:,:,-2:-1]),axis = 2)
    return img

def divide_to_years(landcover_data):
    timestep_dict = dict()
    for i in range(landcover_data.shape[2]):
        timestep_dict[i] = landcover_data[:,:,i]
    return timestep_dict


def mask_image(mask_data,surface_data,temp_data):
    for year in mask_data:
        landcover = mask_data[year]
        #iterate through the bands in the particular year for the surface data
        for band in surface_data[year + 1]:
            for i in range(len(surface_data[year + 1][band])):
                #range is between -100 and 16000
                band_data_surface = surface_data[year + 1][band][i]
                band_data_surface[np.where(band_data_surface < 0)] = 0
                band_data_surface[np.where(band_data_surface > 16000)] = 16000
                surface_data[year + 1][band][i] =  band_data_surface* landcover
        for band in temp_data[year + 1]:
            for i in range(len(temp_data[year + 1][band])):
                #range is between 0 and 16331 for temp data
                band_data = temp_data[year + 1][band][i]
                band_data[np.where(band_data < 0)] = 0
                band_data[np.where(band_data > 16000)] = 16000
                temp_data[year + 1][band][i] = band_data * landcover
    return surface_data,temp_data,mask_data
    
def merge_bands(surface_data,temp_data):
    merged_data = surface_data.copy()
    for year in merged_data:
        merged_data[year][7] = temp_data[year][0]
        merged_data[year][8] = temp_data[year][1]
    return merged_data


def create_histograms(merged_data):
    merged_hist_data = dict()
    for year in merged_data:
        #will be 9
        bands = len(merged_data[year].keys())
        #will be 27
        timesteps = len(merged_data[year][0])
        #will be 64
        bins = 64
        hist = np.zeros(shape=(bins,timesteps,bands))
        for band in merged_data[year]:
            for i in range(len(merged_data[year][band])):
                #65 points allow for 64 bins
                density,_ = np.histogram(merged_data[year][band][i],np.linspace(1,15999,65),density = False)
                hist[:,i,band] = density/float(density.sum())
        #process nan values
        if np.isnan(hist).sum() > 0:
            hist[np.where(np.isnan(hist))] = 0
        merged_hist_data[year] = hist
    return merged_hist_data


if __name__ == "__main__":
    #,"Ethiopia","Malawi"
    for country in ['Kenya']:
        #Process surface data
        filepath_surface = [i for i in list(Path(Path.cwd()/"TIFF"/country).resolve().iterdir()) if "Surface" in str(i)][0]
        surface_data = gdal.Open(str(filepath_surface)).ReadAsArray()
        surface_data = np.transpose(surface_data,axes = (1,2,0))
        timestep_dict_surface = split_images(array=surface_data,total_bands=29*7,yearcount=13)
        band_dict_surface = split_to_year_bands(timestep_dict_surface,yearcount = 13,bands=7,imagecountperyear=29)
        #Process landcover data
        filepath_landcover = [i for i in list(Path(Path.cwd()/"TIFF"/country).resolve().iterdir()) if "LandCover" in str(i)][0]
        landcover_data = np.array(gdal.Open(str(filepath_landcover)).ReadAsArray(),dtype = 'uint16')
        landcover_data = np.transpose(landcover_data,axes = (1,2,0))
        landcover_data = filter_cropland(landcover_data)
        landcover_data = extend_landcover(landcover_data,1)
        landcover_band_dict = divide_to_years(landcover_data)
        #Process Temperature
        filepath_temp = [i for i in list(Path(Path.cwd()/"TIFF"/country).resolve().iterdir()) if "Temperature" in str(i)][0]
        temp_data = np.array(gdal.Open(str(filepath_temp)).ReadAsArray(),dtype = 'uint16')
        temp_data = np.transpose(temp_data,axes = (1,2,0))
        timestep_dict_temp = split_images(temp_data,29*2,13)
        band_dict_temp = split_to_year_bands(timestep_dict_temp,yearcount = 13,bands=2,imagecountperyear=29)
        #masking using landcover
        band_dict_surface_mask,band_dict_temp_mask,landcover_band_dict = mask_image(landcover_band_dict,band_dict_surface,band_dict_temp)
        merged_data = merge_bands(band_dict_surface_mask,band_dict_temp_mask)
        merged_hist_data = create_histograms(merged_data=merged_data)
        #start year = 2009
        for i in merged_hist_data:
            np.save(str(Path(Path.cwd(),"ProcessedHistograms",f"{country}_{2009 + i - 1}.npy").resolve()),merged_hist_data[i])