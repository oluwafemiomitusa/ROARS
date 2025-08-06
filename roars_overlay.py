#!/usr/bin/env python3
"""
ROARS Echo Overlay Script
Overlays ROARS echo outlines on NEXRAD reflectivity data
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import subprocess
import re
from datetime import datetime
import pyart

matplotlib = plt.matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 14 
matplotlib.rcParams['xtick.labelsize'] = 12 
matplotlib.rcParams['ytick.labelsize'] = 12 
matplotlib.rcParams['savefig.transparent'] = False

vmin, vmax = -20, 60
cmap = 'NWSRef'

def parse_roars_time(filename):
    """Extract datetime from ROARS filename"""
    match = re.search(r'S(\d{8})T(\d{6})', filename)
    if match:
        date_str, time_str = match.groups()
        return datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
    return None

def load_nexrad_timestamps(nexrad_files):
    """Pre-load all NEXRAD timestamps for fast matching"""
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

def polar_to_cartesian(ranges, azimuths, data):
    """Convert ROARS polar coordinates to Cartesian"""
    ranges_km = ranges / 1000  # Convert to km
    az_rad = np.radians(azimuths)
    
    # Create coordinate grids
    r_mesh, az_mesh = np.meshgrid(ranges_km, az_rad)
    x = r_mesh * np.sin(az_mesh)
    y = r_mesh * np.cos(az_mesh)
    
    return x, y, data

def plot_overlay(ds_roars, ds_nexrad, output_file):
    """Create three-panel plot: ROARS LP, NEXRAD, and Overlay"""
    # Extract metadata
    roars_time = ds_roars['cpi_timestamp'].values[0]
    dt = datetime.utcfromtimestamp(roars_time)
    nexrad_time = ds_nexrad['time'].values[0]
    nexrad_dt = pd.to_datetime(nexrad_time)
    
    # Prepare data
    nexrad_ref = ds_nexrad['REF'].values
    if nexrad_ref.ndim == 3:
        nexrad_ref = nexrad_ref[0]
    
    lp_ref = ds_roars['lp_reflectivity_dbz'].values
    lp_ranges = ds_roars['range_lp'].values
    azimuths = ds_roars['azimuth_deg'].values
    x_lp, y_lp, z_lp = polar_to_cartesian(lp_ranges, azimuths, lp_ref)

    extent = 120  # ±120 km for 240x240 km view
    
    # Extract azimuth and elevation for titles
    azimuth = ds_roars['azimuth_deg'].values[0]
    elevation = ds_roars['elevation_deg'].values[0]
    nexrad_elevation = ds_nexrad['HGT'].values[0]

    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: ROARS LP only
    im1 = ax1.pcolormesh(x_lp, y_lp, z_lp, vmin=-20, vmax=60, cmap='NWSRef', shading='nearest')
    ax1.set_xlim(-extent, extent)
    ax1.set_ylim(-extent, extent)
    ax1.set_aspect('equal')
    ax1.grid(ls='--', alpha=0.3)
    ax1.set_title(f'ROARS LP Reflectivity\naz: {azimuth:.1f}°, el: {elevation:.1f}°')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Distance (km)')
    plt.colorbar(im1, ax=ax1, shrink=0.75, label='dBZ')
    
    # Plot 2: NEXRAD only
    im2 = ax2.imshow(nexrad_ref, extent=[-extent, extent, -extent, extent], 
                     vmin=-20, vmax=60, cmap='NWSRef', origin='lower')
    ax2.set_aspect('equal')
    ax2.grid(ls='--', alpha=0.3)
    ax2.set_title(f'NEXRAD Reflectivity\nCAPPI {nexrad_elevation:.1f} km')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Distance (km)')
    plt.colorbar(im2, ax=ax2, shrink=0.75, label='dBZ')
    
    # Plot 3: Overlay
    im3 = ax3.imshow(nexrad_ref, extent=[-extent, extent, -extent, extent], 
                     vmin=-20, vmax=60, cmap='NWSRef', origin='lower')
    
    # ROARS contours in deepskyblue for contrast
    lp_valid = ~np.isnan(z_lp)
    lp_max = z_lp[lp_valid].max() if np.any(lp_valid) else -999
    print(f"    LP max: {lp_max:.1f} dBZ")
    
    if lp_max > 20:
        contours = ax3.contour(x_lp, y_lp, z_lp, levels=[20, 30, 40, 50],
                              colors='deepskyblue', linewidths=2, alpha=1.0)
        print(f"    Drew LP contours")
    
    ax3.set_xlim(-extent, extent)
    ax3.set_ylim(-extent, extent)
    ax3.set_aspect('equal')
    ax3.grid(ls='--', alpha=0.3)
    ax3.set_title('NEXRAD + ROARS LP Overlay')
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel('Distance (km)')
    plt.colorbar(im3, ax=ax3, shrink=0.75, label='NEXRAD dBZ')
    
    # Add legend to overlay plot
    from matplotlib.lines import Line2D
    ax3.legend([Line2D([0], [0], color='deepskyblue', lw=2)], ['ROARS LP'], loc='upper right')
    
    # Overall title
    fig.suptitle(f'NEXRAD: {nexrad_dt.strftime("%Y%m%d %H%M%S")}Z \n ROARS: {dt.strftime("%Y%m%d %H%M%S")}Z', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def process_overlay_files(max_files=None):
    """Process all files to create overlay plots"""
    print("ROARS Echo Overlay System")
    print("=========================")
    
    # Find files
    roars_files = sorted(glob.glob('/Volumes/My Book/ROARS/roars/20250731/ROARS_Level2*.nc'))
    nexrad_files = sorted(glob.glob('/Volumes/My Book/ROARS/nexrad_cappi/20250731/KHTX/*.nc'))

    if max_files:
        roars_files = roars_files[:max_files]
        print(f"Limited to first {max_files} files for testing")
    
    print(f"Found {len(roars_files)} ROARS files")
    print(f"Found {len(nexrad_files)} NEXRAD files")
    
    if not roars_files or not nexrad_files:
        return []
    
    # Pre-load NEXRAD timestamps
    nexrad_times = load_nexrad_timestamps(nexrad_files)
    
    # Create output directory
    output_dir = 'overlay_output'
    os.makedirs(output_dir, exist_ok=True)
    
    processed_files = []
    
    for i, roars_file in enumerate(roars_files):
        print(f"Processing {i+1}/{len(roars_files)}: {os.path.basename(roars_file)}")
        
        try:
            # Find closest NEXRAD file
            roars_time = parse_roars_time(roars_file)
            if not roars_time:
                continue
                
            closest_nexrad, time_diff = find_closest_nexrad_fast(roars_time, nexrad_times)
            if not closest_nexrad:
                continue
                
            print(f"  -> {os.path.basename(closest_nexrad)} (Δt={time_diff:.0f}s)")
            
            # Load datasets
            ds_roars = xr.open_dataset(roars_file)
            ds_nexrad = xr.open_dataset(closest_nexrad)
            
            # Create overlay plot
            time_str = roars_time.strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(output_dir, f'overlay_{time_str}.png')
            
            plot_overlay(ds_roars, ds_nexrad, output_file)
            
            processed_files.append(output_file)
            ds_roars.close()
            ds_nexrad.close()
            print(f"✓ Saved: {output_file}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    print(f"\\n=== Overlay Processing Complete! ===")
    print(f"✓ Processed {len(processed_files)} files")
    print(f"✓ Output directory: {output_dir}")
    
    # Create animation if multiple files
    if len(processed_files) > 1:
        create_animation(output_dir)
    
    return processed_files

def create_animation(output_dir):
    """Create animation from overlay plots"""
    print(f"\\n=== Creating Animation ===")
    
    animation_file = os.path.join(output_dir, 'overlay_animation.mp4')
    cmd = [
        'ffmpeg', '-y',
        '-framerate', '5',
        '-pattern_type', 'glob', 
        '-i', os.path.join(output_dir, 'overlay_*.png'),
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
    process_overlay_files()