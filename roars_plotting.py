#!/usr/bin/env python3
import xarray as xr
import pyart
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['axes.labelsize'] = 14 #fontsize in pts
matplotlib.rcParams['axes.titlesize'] = 14 
matplotlib.rcParams['xtick.labelsize'] = 12 
matplotlib.rcParams['ytick.labelsize'] = 12 
matplotlib.rcParams['savefig.transparent'] = False

ds = xr.open_dataset('ROARS_Level2_PPI_CpiMode1_S20250731T001911_E20250731T001922.nc')
ds_nexrad = xr.open_dataset('KHTX20250731_001817_V06.nc')

# After loading ds_nexrad:
grid_size_km = 20  # total grid from -10 to +10 km (adjust if needed)
num_pixels = ds_nexrad.dims['x']  # number of pixels, e.g. 500

nexrad_x = np.linspace(-grid_size_km / 2, grid_size_km / 2, num_pixels)
nexrad_y = np.linspace(-grid_size_km / 2, grid_size_km / 2, num_pixels)

from datetime import datetime
timestamp = ds['cpi_timestamp'].values[0]
dt = datetime.utcfromtimestamp(timestamp)
date_str = dt.strftime('%Y%m%d')
time_str = dt.strftime('%H%M%S')

nexrad_time = ds_nexrad['time'].values[0]
nexrad_dt = pd.to_datetime(nexrad_time)
nexrad_date_str = nexrad_dt.strftime('%Y%m%d')
nexrad_time_str = nexrad_dt.strftime('%H%M%S')
nexrad_elevation = ds_nexrad['HGT'].values[0]

azimuth = ds['azimuth_deg'].values[0]
elevation = ds['elevation_deg'].values[0]

vmin, vmax = -20, 60
cmap = 'NWSRef'

def calculate_correlation_plot(ax, roars_data, khtx_data, roars_ranges, roars_azimuths):
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

def create_radar_object(pulse_type):
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

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

radar_sp = create_radar_object('sp')
radar_lp = create_radar_object('lp')

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

nexrad_ref = ds_nexrad['REF'].values
im = ax2.imshow(nexrad_ref, extent=[-10, 10, -10, 10], vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
ax2.set_title(f'KHTX NEXRAD {nexrad_date_str} {nexrad_time_str} Z [dBZ] \n CAPPI {nexrad_elevation:.1f} km', fontweight='bold', pad=15)
ax2.set_xlim(-10, 10)
ax2.set_ylim(-10, 10)
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
                          ds['range_sp'].values, ds['azimuth_deg'].values)

plt.tight_layout()
plt.savefig('roars_2x2_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved as: roars_2x2_plot.png")
plt.show()

