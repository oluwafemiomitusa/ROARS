#!/usr/bin/env python3
import xarray as xr
import pyart
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import os
import glob
import subprocess
import re
from datetime import datetime

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 14 
matplotlib.rcParams['xtick.labelsize'] = 12 
matplotlib.rcParams['ytick.labelsize'] = 12 
matplotlib.rcParams['savefig.transparent'] = False

vmin, vmax = -20, 60
cmap = 'NWSRef'

def parse_roars_time(filename):
    """Extract datetime from ROARS filename: S20250731T000618"""
    match = re.search(r'S(\d{8})T(\d{6})', filename)
    if match:
        date_str, time_str = match.groups()
        return datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
    return None

def parse_nexrad_time(filename):
    """Extract datetime from NEXRAD filename: 20250731_000209"""
    match = re.search(r'(\d{8})_(\d{6})', filename)
    if match:
        date_str, time_str = match.groups()
        return datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
    return None

def find_closest_nexrad(roars_time, nexrad_files):
    """Find NEXRAD file closest in time to ROARS file using internal timestamps"""
    best_file = None
    min_diff = float('inf')
    
    for nfile in nexrad_files:
        try:
            # Use internal timestamp instead of filename
            ds_temp = xr.open_dataset(nfile)
            ntime = pd.to_datetime(ds_temp['time'].values[0])
            ds_temp.close()
            
            diff = abs((roars_time - ntime).total_seconds())
            if diff < min_diff:
                min_diff = diff
                best_file = nfile
        except:
            # Fallback to filename parsing if internal time fails
            ntime = parse_nexrad_time(nfile)
            if ntime:
                diff = abs((roars_time - ntime).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    best_file = nfile
    
    return best_file, min_diff

def calculate_correlation_plot(ax, roars_data, khtx_data, roars_ranges, roars_azimuths, ds_nexrad):
    """Calculate and plot correlation between ROARS and KHTX data"""
    from scipy.interpolate import griddata
    from scipy.stats import pearsonr
    
    # Convert ROARS polar to Cartesian
    roars_ranges_km = roars_ranges / 1000  # Convert to km
    roars_azimuth_rad = np.radians(roars_azimuths)
    roars_x = np.outer(roars_ranges_km, np.sin(roars_azimuth_rad)).flatten()
    roars_y = np.outer(roars_ranges_km, np.cos(roars_azimuth_rad)).flatten()
    roars_z = roars_data.flatten()
    
    # Create common grid (10km x 10km to match SP range)
    grid_x, grid_y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    
    # Interpolate ROARS to grid
    roars_grid = griddata((roars_x, roars_y), roars_z, (grid_x, grid_y), method='linear')
    
    # Get KHTX data and interpolate to same grid
    grid_size_km = 20
    num_pixels = ds_nexrad.dims['x']
    nexrad_x = np.linspace(-grid_size_km / 2, grid_size_km / 2, num_pixels)
    nexrad_y = np.linspace(-grid_size_km / 2, grid_size_km / 2, num_pixels)
    khtx_xx, khtx_yy = np.meshgrid(nexrad_x, nexrad_y)
    khtx_grid = griddata((khtx_xx.flatten(), khtx_yy.flatten()), khtx_data.flatten(), 
                         (grid_x, grid_y), method='linear')
    
    # Calculate correlation for valid data points
    valid_mask = ~(np.isnan(roars_grid) | np.isnan(khtx_grid))
    if np.sum(valid_mask) > 10:  # Need at least 10 valid points
        roars_valid = roars_grid[valid_mask]
        khtx_valid = khtx_grid[valid_mask]
        correlation, p_value = pearsonr(roars_valid, khtx_valid)
        
        # Create scatter plot
        ax.scatter(roars_valid, khtx_valid, alpha=0.6, s=1, c='blue')
        ax.set_xlabel('ROARS SP Reflectivity (dBZ)')
        ax.set_ylabel('KHTX Reflectivity (dBZ)')
        ax.set_title(f'ROARS SP vs KHTX Correlation\nr = {correlation:.3f}, p = {p_value:.3f}\nn = {len(roars_valid)}', 
                      fontweight='bold', pad=15)
        ax.grid(True, alpha=0.7)
        
        # Add 1:1 line
        min_val = max(np.min(roars_valid), np.min(khtx_valid))
        max_val = min(np.max(roars_valid), np.max(khtx_valid))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1 line')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Insufficient overlapping data\nfor correlation analysis', 
                 ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('ROARS SP vs KHTX Correlation', fontweight='bold', pad=15)

def create_radar_object(ds, pulse_type):
    return pyart.core.Radar(
        time={'data': ds['cpi_timestamp'].values, 'units': 'seconds since 1970-01-01T00:00:00Z'},
        _range={'data': ds[f'range_{pulse_type}'].values, 'units': 'meters'},
        fields={
            'reflectivity': {
                'data': ds[f'{pulse_type}_reflectivity_dbz'].values,
                'units': 'dBZ',
                'long_name': 'Reflectivity'
            }
        },
        metadata={'instrument_name': 'ROARS'},
        scan_type='ppi',
        latitude={'data': np.array([ds[f'{pulse_type}_lat_deg'].values.mean()])},
        longitude={'data': np.array([ds[f'{pulse_type}_lon_deg'].values.mean()])},
        altitude={'data': np.array([ds[f'{pulse_type}_alt'].values.mean()])},
        sweep_number={'data': np.array([0])},
        sweep_mode={'data': np.array(['azimuth_surveillance'])},
        fixed_angle={'data': np.array([ds['elevation_deg'].values.mean()])},
        sweep_start_ray_index={'data': np.array([0])},
        sweep_end_ray_index={'data': np.array([len(ds['beam']) - 1])},
        azimuth={'data': ds['azimuth_deg'].values, 'units': 'degrees'},
        elevation={'data': ds['elevation_deg'].values, 'units': 'degrees'},
        instrument_parameters={}
    )

def plot_single_file(ds, ds_nexrad, output_file):
    """Plot a single PPI file with NEXRAD comparison"""
    # Extract timestamp and metadata
    timestamp = ds['cpi_timestamp'].values[0]  # Level2 files have shape (121,)
    dt = datetime.utcfromtimestamp(timestamp)
    date_str = dt.strftime('%Y%m%d')
    time_str = dt.strftime('%H%M%S')
    
    azimuth = ds['azimuth_deg'].values[0]  # Level2 files have shape (121,)
    elevation = ds['elevation_deg'].values[0]  # Level2 files have shape (121,)
    
    # NEXRAD metadata
    nexrad_time = ds_nexrad['time'].values[0]
    nexrad_dt = pd.to_datetime(nexrad_time)
    nexrad_date_str = nexrad_dt.strftime('%Y%m%d')
    nexrad_time_str = nexrad_dt.strftime('%H%M%S')
    nexrad_elevation = ds_nexrad['HGT'].values[0]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    radar_sp = create_radar_object(ds, 'sp')
    radar_lp = create_radar_object(ds, 'lp')
    
    display_sp = pyart.graph.RadarDisplay(radar_sp)
    display_lp = pyart.graph.RadarDisplay(radar_lp)
    
    display_sp.plot('reflectivity', 0, ax=ax1, vmin=vmin, vmax=vmax, cmap=cmap)
    ax1.set_title(f'ROARS SP {date_str} {time_str} Z [dBZ] \n az: {azimuth:.1f}°, el: {elevation:.1f}°', fontweight='bold', pad=15)
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Distance from Radar (km)')
    ax1.set_ylabel('Distance from Radar (km)')
    
    # NEXRAD plot
    nexrad_ref = ds_nexrad['REF'].values
    grid_size_km = 20
    num_pixels = ds_nexrad.dims['x']
    nexrad_x = np.linspace(-grid_size_km / 2, grid_size_km / 2, num_pixels)
    nexrad_y = np.linspace(-grid_size_km / 2, grid_size_km / 2, num_pixels)
    
    im = ax2.imshow(nexrad_ref, extent=[-100, 100, -100, 100], vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    ax2.set_title(f'KHTX NEXRAD {nexrad_date_str} {nexrad_time_str} Z [dBZ] \n CAPPI {nexrad_elevation:.1f} km', fontweight='bold', pad=15)
    ax2.set_xlim(-100, 100)
    ax2.set_ylim(-100, 100)
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Distance from Radar (km)')
    ax2.set_ylabel('Distance from Radar (km)')
    cbar2 = plt.colorbar(im, ax=ax2)
    cbar2.set_label('Reflectivity (dBZ)', labelpad=15)
    
    display_lp.plot('reflectivity', 0, ax=ax3, vmin=vmin, vmax=vmax, cmap=cmap)
    ax3.set_title(f'ROARS LP {date_str} {time_str} Z [dBZ] \n az: {azimuth:.1f}°, el: {elevation:.1f}°', fontweight='bold', pad=15)
    ax3.set_xlim(-120, 120)
    ax3.set_ylim(-120, 120)
    ax3.set_aspect('equal')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_xlabel('Distance from Radar (km)')
    ax3.set_ylabel('Distance from Radar (km)')
    
    # Calculate and plot correlation between ROARS SP and KHTX
    calculate_correlation_plot(ax4, ds['sp_reflectivity_dbz'].values, nexrad_ref, 
                              ds['range_sp'].values, ds['azimuth_deg'].values, ds_nexrad)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def load_nexrad_timestamps(nexrad_files):
    """Pre-load all NEXRAD timestamps once for fast matching"""
    print(f"Pre-loading timestamps from {len(nexrad_files)} NEXRAD files...")
    nexrad_times = {}
    
    for i, nfile in enumerate(nexrad_files):
        try:
            ds_temp = xr.open_dataset(nfile)
            ntime = pd.to_datetime(ds_temp['time'].values[0])
            ds_temp.close()
            nexrad_times[nfile] = ntime
        except:
            continue
        
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(nexrad_files)} timestamps...")
    
    print(f"✓ Pre-loaded {len(nexrad_times)} NEXRAD timestamps")
    return nexrad_times

def find_closest_nexrad_fast(roars_time, nexrad_times):
    """Fast NEXRAD matching using pre-loaded timestamps"""
    best_file = None
    min_diff = float('inf')
    
    for nfile, ntime in nexrad_times.items():
        diff = abs((roars_time - ntime).total_seconds())
        if diff < min_diff:
            min_diff = diff
            best_file = nfile
    
    return best_file, min_diff

def process_all_ppi_files(max_files=None):
    """Process PPI files with time-matched NEXRAD data (optimized)"""
    print("ROARS Fast Processing System")
    print("============================")
    
    # Find files
    roars_files = sorted(glob.glob('/Users/oomitusa/Documents/Research/ROARS/roars/20250731/ROARS_Level2*.nc'))
    nexrad_files = sorted(glob.glob('/Users/oomitusa/Documents/Research/ROARS/nexrad_cappi/20250731/KHTX/*.nc'))
    
    # Limit files for testing
    if max_files:
        roars_files = roars_files[:max_files]
        print(f"Limited to first {max_files} files for testing")
    
    print(f"Found {len(roars_files)} ROARS files")
    print(f"Found {len(nexrad_files)} NEXRAD files")
    
    if not roars_files or not nexrad_files:
        print("Error: No files found")
        return []
    
    # PRE-LOAD ALL NEXRAD TIMESTAMPS (huge speed boost!)
    nexrad_times = load_nexrad_timestamps(nexrad_files)
    
    # Create output directory
    output_dir = 'roars_output'
    os.makedirs(output_dir, exist_ok=True)
    
    processed_files = []
    
    for i, roars_file in enumerate(roars_files):
        print(f"Processing {i+1}/{len(roars_files)}: {os.path.basename(roars_file)}")
        
        try:
            # Find closest NEXRAD file in time
            roars_time = parse_roars_time(roars_file)
            if not roars_time:
                print(f"✗ Could not parse time from {roars_file}")
                continue
                
            closest_nexrad, time_diff = find_closest_nexrad_fast(roars_time, nexrad_times)
            if not closest_nexrad:
                print(f"✗ No matching NEXRAD file found")
                continue
                
            print(f"  -> {os.path.basename(closest_nexrad)} (Δt={time_diff:.0f}s)")
            
            # Load both datasets
            ds_roars = xr.open_dataset(roars_file)
            ds_nexrad = xr.open_dataset(closest_nexrad)
            
            # Extract timestamp for filename
            time_str = roars_time.strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(output_dir, f'roars_plot_{time_str}.png')
            
            # Plot comparison
            plot_single_file(ds_roars, ds_nexrad, output_file)
            
            processed_files.append(output_file)
            ds_roars.close()
            ds_nexrad.close()
            print(f"✓ Saved: {output_file}")
            
        except Exception as e:
            print(f"✗ Error processing {roars_file}: {e}")
            continue
    
    print(f"\n=== Batch Processing Complete! ===")
    print(f"✓ Processed {len(processed_files)} files")
    print(f"✓ Output directory: {output_dir}")
    
    # Create animation
    if len(processed_files) > 1:
        create_animation(output_dir)
    
    return processed_files

def create_animation(output_dir):
    """Create animation from individual plots using ffmpeg"""
    print(f"\n=== Creating Animation ===")
    
    animation_file = os.path.join(output_dir, 'roars_animation.mp4')
    
    # ffmpeg command to create animation
    cmd = [
        'ffmpeg', '-y',
        '-framerate', '2',
        '-pattern_type', 'glob',
        '-i', os.path.join(output_dir, 'roars_plot_*.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        animation_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Animation created: {animation_file}")
        else:
            print(f"✗ ffmpeg error: {result.stderr}")
    except FileNotFoundError:
        print("✗ ffmpeg not found. Install ffmpeg to create animations.")
    except Exception as e:
        print(f"✗ Animation creation failed: {e}")

if __name__ == "__main__":
    process_all_ppi_files()

