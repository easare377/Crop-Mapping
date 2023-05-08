from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr

import numpy as np
import pandas as pd
import image_utils as iu
import tensorflow as tf
import os
from PIL import Image, ImageEnhance
from PIL import ImageFilter
from keras.models import load_model
from shutil import copy2
import matplotlib.pyplot as plt


def get_rgb_bands(tiff, color_depth='uint8'):
    red = tiff.GetRasterBand(1).ReadAsArray()
    green = tiff.GetRasterBand(2).ReadAsArray()
    blue = tiff.GetRasterBand(3).ReadAsArray()
    # rgbOutput = source.ReadAsArray() #Easier method
    rgbOutput = np.zeros((tiff.RasterYSize, tiff.RasterXSize, 3), color_depth)
    rgbOutput[..., 0] = red
    rgbOutput[..., 1] = green
    rgbOutput[..., 2] = blue
    # Clear so file isn't locked
    source = None
    return rgbOutput


def get_l_band(tiff, band, color_depth='uint8'):
    L = tiff.GetRasterBand(band).ReadAsArray()
    # rgbOutput = source.ReadAsArray() #Easier method
    rgbOutput = np.zeros((tiff.RasterYSize, tiff.RasterXSize, 1), color_depth)
    rgbOutput[..., 0] = L
    # Clear so file isn't locked
    source = None
    return rgbOutput


def save_image_as_geotiff(mask, destination_path, dataset):
    mask = mask.resize((dataset.RasterXSize, dataset.RasterYSize))
    array = np.asarray(mask)
    nrows, ncols = array.shape[0], array.shape[1]
    depth = array.shape[2]
    # geotransform = (0, 1024, 0, 0, 0, 512)
    # That's (top left x, w-e pixel resolution, rotation (0 if North is up),
    #         top left y, rotation (0 if North is up), n-s pixel resolution)
    # I don't know why rotation is in twice???
    output_raster = gdal.GetDriverByName('GTiff').Create(
        destination_path, ncols, nrows, depth, gdal.GDT_Byte)  # Open the file
    # print(output_raster)
    output_raster.SetGeoTransform(
        dataset.GetGeoTransform())  # Specify its coordinates
    output_raster.SetProjection(dataset.GetProjection())
    output_raster.SetMetadata(output_raster.GetMetadata())
   # to the file
    for x in range(depth):
        output_raster.GetRasterBand(x + 1).WriteArray(
            array[:, :, x])   # Writes my array to the raster
    output_raster.FlushCache()


def save_np_array_as_geoTiff(np_array, destination_path, output_dtype=gdal.GDT_UInt16, geoTransform=None, projection=None, metadata=None):
    # np_array = np_array.resize((dataset.RasterXSize, dataset.RasterYSize))
    array = np_array
    height, width = array.shape[0], array.shape[1]
    depth = array.shape[2]
    # geotransform = (0, 1024, 0, 0, 0, 512)
    # That's (top left x, w-e pixel resolution, rotation (0 if North is up),
    #         top left y, rotation (0 if North is up), n-s pixel resolution)
    # I don't know why rotation is in twice???
    output_raster = gdal.GetDriverByName('GTiff').Create(
        destination_path, width, height, depth, output_dtype)  # Open the file
    # print(output_raster)
    if geoTransform:
        output_raster.SetGeoTransform(geoTransform)  # Specify its coordinates
    if projection:
        output_raster.SetProjection(projection)
    if metadata:
        output_raster.SetMetadata(metadata)
    for x in range(depth):
        output_raster.GetRasterBand(x + 1).WriteArray(
            array[:, :, x])   # Writes my array to the raster
    output_raster.FlushCache()
    return output_raster


def read_geoTiff_bands(geoTiff, bands=None, dtype='uint16'):
    width = geoTiff.RasterXSize
    height = geoTiff.RasterYSize
    band_list = []
    if bands != None:
        for val in bands:
            band = geoTiff.GetRasterBand(val).ReadAsArray()
            band_list.append(band)
    else:
        for x in range(geoTiff.RasterCount):
            band = geoTiff.GetRasterBand(x + 1).ReadAsArray()
            band_list.append(band)
    output = np.zeros(
        (int(height), int(width), len(band_list)), dtype)
    for x in range(len(band_list)):
        output[..., x] = band_list[x]
    return output


def display_np_geoTiff(np_geoTiff, rgb_bands):
    r, g, b = rgb_bands
    red_band = np_geoTiff[:, :, r]
    green_band = np_geoTiff[:, :, g]
    blue_band = np_geoTiff[:, :, b]
    # max = np_geoTiff.max()
    # np_rgb_image = np_geoTiff[:, :, 0:3][:, :, ::-1]
    rgbOutput = np.zeros((tiff.RasterYSize, tiff.RasterXSize, 3), color_depth)
    rgbOutput[..., 0] = red_band
    rgbOutput[..., 1] = green_band
    rgbOutput[..., 2] = blue_band
    rgbOutput = (rgbOutput / rgbOutput.max()) * 255
    im = Image.fromarray(np.array(rgbOutput, dtype=np.uint8))
    return im


def display_np_geoTiff_band(band_geoTiff, band_number):
    np_gray_image = band_geoTiff[:, :, band_number]
    np_n_im = (np_gray_image / np_gray_image.max()) * 255
    im = Image.fromarray(np.array(np_n_im, dtype=np.uint8))
    return im


def crop_geoTiff(geoTiff, left, top, right, bottom, dtype='uint16'):
    gt_width, gt_height = geoTiff.RasterXSize, geoTiff.RasterYSize
    if right > gt_width:
        right = gt_width
    if bottom > gt_height:
        bottom = gt_height
    width = abs(right - left)
    height = abs(top - bottom)
    bands = []
    for x in range(geoTiff.RasterCount):
        band = geoTiff.GetRasterBand(x + 1).ReadAsArray(left, top,
                                                        int(width), int(height))
        bands.append(band)
    output = np.zeros(
        (int(height), int(width), geoTiff.RasterCount), dtype)
    for x in range(len(bands)):
        output[..., x] = bands[x]
    return output

# Splits image into specified rows and columns


def split_geoTiff_image(geoTiff, row_count, col_count):
    parts = []
    width, height = geoTiff.RasterXSize, geoTiff.RasterYSize
    left = 0
    top = 0
    right = width / col_count
    bottom = height / row_count
    # bands = read_geoTiff_bands(geoTiff)
    for r in range(row_count):
        top = int(r * (height / row_count))
        bottom = int(top + (height / row_count))
        for c in range(col_count):
            left = int(c * (width / col_count))
            right = int(left + (width / col_count))
            part = crop_geoTiff(geoTiff, left, top, right, bottom)
            parts.append(part)
    return parts
