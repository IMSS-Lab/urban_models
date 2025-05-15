import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import netCDF4
from PIL import Image
import torchvision.transforms.functional as F
import os

def find_names(start_file_name):
    """
    Generate file paths with incrementing years.
    
    Args:
        start_file_name: Initial file path with year 1992
        
    Returns:
        list of file paths from 1992 to 2016
    """
    arr = []
    curr_name = start_file_name
    curr_yr = 1992
    max_yr = 2016
    ran = max_yr - curr_yr
    
    for i in range(ran + 1):
        arr.append(curr_name)
        curr_name = curr_name.replace(str(curr_yr), str(curr_yr + 1))
        curr_yr += 1
    
    return np.array(arr)

def calc_bounds(files, var_type):
    """
    Calculate global min and max values across netCDF files.
    
    Args:
        files: List of netCDF files
        var_type: Variable type to extract ('tas' for temperature, 'pr' for precipitation)
        
    Returns:
        tuple of (min_value, max_value)
    """
    global_var_min = float('inf')
    global_var_max = float('-inf')
    
    for file_path in files:
        nc = netCDF4.Dataset(file_path)
        time = nc.variables['time'][:]
        
        for time_index in range(len(time)):
            var = nc.variables[var_type][:]
            global_var_min = min(global_var_min, var.min())
            global_var_max = max(global_var_max, var.max())
    
    return global_var_min, global_var_max

def calc_bounds_crop(files):
    """
    Calculate global min and max values across crop yield NetCDF files.
    
    Args:
        files: List of crop yield netCDF files
        
    Returns:
        tuple of (min_value, max_value)
    """
    global_var_min = float('inf')
    global_var_max = float('-inf')
    
    for file_path in files:
        nc = netCDF4.Dataset(file_path)
        yield_at_time = nc.variables['var'][:]
        global_var_min = min(global_var_min, yield_at_time.min())
        global_var_max = max(global_var_max, yield_at_time.max())
    
    return global_var_min, global_var_max

def resize_image(img, new_size):
    """
    Resize an image to new_size.
    
    Args:
        img: Input image (as numpy array)
        new_size: Target size as (height, width)
        
    Returns:
        Resized image
    """
    pil_img = Image.fromarray(img.astype(np.uint8))
    resized_img = pil_img.resize(new_size)
    return np.array(resized_img)

def preprocess_data(precipitation_array, urbanization_array, crop_yield_array, temperature_array, new_size=(32, 32)):
    """
    Resize input arrays to the specified size.
    
    Args:
        precipitation_array: Array of precipitation data
        urbanization_array: Array of urbanization data 
        crop_yield_array: Array of crop yield data
        temperature_array: Array of temperature data
        new_size: Target size for images (height, width)
        
    Returns:
        Tuple of resized arrays
    """
    def resize_array_images(arr):
        shape = arr.shape
        resized = np.zeros((shape[0], shape[1], shape[2], new_size[0], new_size[1], 3))
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    resized[i, j, k] = resize_image(arr[i, j, k], new_size)
        return resized
    
    precipitation_array = resize_array_images(precipitation_array)
    urbanization_array = resize_array_images(urbanization_array)
    crop_yield_array = resize_array_images(crop_yield_array)
    temperature_array = resize_array_images(temperature_array)
    
    return precipitation_array, urbanization_array, crop_yield_array, temperature_array

def load_and_preprocess_agriculture(data_dir, output_dir=None):
    """
    Load and preprocess agriculture (crop yield) data.
    
    Args:
        data_dir: Directory containing the data
        output_dir: Directory to save visualizations (if None, no visualizations are saved)
        
    Returns:
        Array of agriculture data
    """
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    
    agriculture = []
    files = find_names(os.path.join(data_dir, "crop_yield/wheat/yield_1992.nc4"))
    vmin_found, vmax_found = calc_bounds_crop(files)
    
    for i in range(1992, 2017):
        nc = netCDF4.Dataset(os.path.join(data_dir, f"crop_yield/wheat/yield_{i}.nc4"))
        lon1 = nc.variables['lon'][:]
        lat1 = nc.variables['lat'][:]
        yield_at_time1 = nc.variables['var'][:]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(yield_at_time1, origin='lower', extent=[lat1.min(), lat1.max(), lon1.min(), lon1.max()], 
                  aspect='auto', cmap='coolwarm', vmin=vmin_found, vmax=vmax_found)
        plt.title(f"Yield in year {i}")
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.colorbar(label='Yield')
        
        agriculture.append(plt.gcf())
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"agriculture_{i}.png"))
        
        plt.close()
    
    return np.array(agriculture)

def load_and_preprocess_precipitation(data_dir, output_dir=None):
    """
    Load and preprocess precipitation data.
    
    Args:
        data_dir: Directory containing the data
        output_dir: Directory to save visualizations (if None, no visualizations are saved)
        
    Returns:
        Array of precipitation data
    """
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    
    precipitation = []
    files = find_names(os.path.join(data_dir, "temp_and_precip/CRU_total_precipitation_mon_0.5x0.5_global_1992_v4.03.nc"))
    vmin_found, vmax_found = calc_bounds(files, 'pr')
    
    for i in range(1992, 2017):
        nc = netCDF4.Dataset(os.path.join(data_dir, f"temp_and_precip/CRU_total_precipitation_mon_0.5x0.5_global_{i}_v4.03.nc"))
        time = nc.variables['time'][:]
        lon = nc.variables['lon'][:]
        lat = nc.variables['lat'][:]
        var_precip = nc.variables['pr'][:]
        pr_at_time = var_precip[0, :, :]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(pr_at_time, origin='lower', extent=[lat.min(), lat.max(), lon.min(), lon.max()], 
                  aspect='auto', cmap='coolwarm', vmin=vmin_found, vmax=vmax_found)
        plt.title(f"Precipitation in year {i}")
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.colorbar(label='Precipitation')
        
        precipitation.append(plt.gcf())
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"precipitation_{i}.png"))
        
        plt.close()
    
    return np.array(precipitation)

def load_and_preprocess_temperature(data_dir, output_dir=None):
    """
    Load and preprocess temperature data.
    
    Args:
        data_dir: Directory containing the data
        output_dir: Directory to save visualizations (if None, no visualizations are saved)
        
    Returns:
        Array of temperature data
    """
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    
    temperature = []
    files = find_names(os.path.join(data_dir, "temp_and_precip/CRU_mean_temperature_mon_0.5x0.5_global_1992_v4.03.nc"))
    vmin_found, vmax_found = calc_bounds(files, 'tas')
    
    for i in range(1992, 2017):
        nc = netCDF4.Dataset(os.path.join(data_dir, f"temp_and_precip/CRU_mean_temperature_mon_0.5x0.5_global_{i}_v4.03.nc"))
        time = nc.variables['time'][:]
        lon = nc.variables['lon'][:]
        lat = nc.variables['lat'][:]
        var_temp = nc.variables['tas'][:]
        tas_at_time = var_temp[0, :, :]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(tas_at_time, origin='lower', extent=[lat.min(), lat.max(), lon.min(), lon.max()], 
                  aspect='auto', cmap='coolwarm', vmin=vmin_found, vmax=vmax_found)
        plt.title(f"Temperature in year {i}")
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.colorbar(label='Temperature')
        
        temperature.append(plt.gcf())
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"temperature_{i}.png"))
        
        plt.close()
    
    return np.array(temperature)

def load_and_preprocess_urbanization(data_dir, output_dir=None):
    """
    Load and preprocess urbanization data.
    
    Args:
        data_dir: Directory containing the data
        output_dir: Directory to save visualizations (if None, no visualizations are saved)
        
    Returns:
        Array of urbanization data
    """
    try:
        import georasters as gr
    except ImportError:
        raise ImportError("Please install georasters: pip install georasters")
    
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    
    activity = []
    
    for k in range(1992, 2017):
        urbanization = gr.from_file(os.path.join(data_dir, f"urban/annual_urbanMap_global_{k}.tif"))
        
        # Convert to pandas first for processing
        urbanization_df = urbanization.to_pandas()
        urbanization_df = urbanization_df[urbanization_df['value'] != 0.0]
        urbanization_df = urbanization_df.sort_values(['x', 'y'])
        
        # Convert to polars for faster operations
        urb_pl = pl.DataFrame(urbanization_df)
        
        # Process the data
        urb_pl = urb_pl.select(
            [
                (pl.col('x') / 0.25).round() * 0.25,
                (pl.col('y') / 0.25).round() * 0.25,
                pl.col('value')
            ]
        )
        
        # Filter and drop duplicates
        urb_pl = urb_pl.filter((pl.col('x') % 0.5 != 0.0) & (pl.col('y') % 0.5 != 0.0))
        urb_pl = urb_pl.unique(subset=['x', 'y'])
        urb_pl = urb_pl.sort(['x', 'y'])
        
        # Convert back to pandas for compatibility with the plotting code
        urbanization = urb_pl.to_pandas()
        
        finals = ""
        finals2 = []
        iter = 0
        
        for i in range(-17975, 17976, 50):
            finals = ""
            for j in range(-8975, 8976, 50):
                if (iter < urbanization.shape[0] and 
                    urbanization['x'].iloc[iter] == i/100 and 
                    urbanization['y'].iloc[iter] == j/100):
                    finals += "1"
                    iter += 1
                else:
                    finals += "0"
            finals2.append(list(finals))
        
        finals2 = np.array(finals2)
        finals2 = np.rot90(finals2, k=3, axes=(0, 1))
        finals2 = np.fliplr(finals2)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(finals2.astype(float), origin='lower', extent=[6.5, 38.5, 66.5, 100], 
                  aspect='auto', cmap='coolwarm', vmin=finals2.astype(float).min(), vmax=finals2.astype(float).max())
        plt.title(f"Urbanization in year {k}")
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.colorbar(label='Urbanization')
        
        activity.append(plt.gcf())
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"urbanization_{k}.png"))
        
        plt.close()
    
    return np.array(activity)

def prepare_model_data(precipitation_array, urbanization_array, crop_yield_array, temperature_array, out_dir=None):
    """
    Prepare data for model training, combining the arrays into appropriate input/output format.
    
    Args:
        precipitation_array: Array of precipitation data
        urbanization_array: Array of urbanization data
        crop_yield_array: Array of crop yield data
        temperature_array: Array of temperature data
        out_dir: Directory to save processed arrays
        
    Returns:
        X_data, y_data arrays ready for model training
    """
    years, lat_partitions, lon_partitions, height, width, channels = precipitation_array.shape
    
    # Create years array (input features) and crops array (target)
    X_data = []
    y_data = []
    
    # Needs at least 5 years of history
    for year in range(5, years):
        for lat in range(lat_partitions):
            for lon in range(lon_partitions):
                x_sample = []
                
                # Last 5 years of all data
                for past_year in range(year - 5, year):
                    x_sample.append(crop_yield_array[past_year, lat, lon])
                    x_sample.append(temperature_array[past_year, lat, lon])
                    x_sample.append(precipitation_array[past_year, lat, lon])
                    x_sample.append(urbanization_array[past_year, lat, lon])
                
                # Current year's temp, precip, urban
                x_sample.append(temperature_array[year, lat, lon])
                x_sample.append(precipitation_array[year, lat, lon])
                x_sample.append(urbanization_array[year, lat, lon])
                
                X_data.append(np.array(x_sample))
                y_data.append(crop_yield_array[year, lat, lon])
    
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'years_array_32_segmented_prevUrb.npy'), X_data)
        np.save(os.path.join(out_dir, 'crops_array_32_segmented_prevUrb.npy'), y_data)
    
    return X_data, y_data