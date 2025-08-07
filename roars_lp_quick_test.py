#!/usr/bin/env python3
"""
Quick LP vs SP Comparison Test
"""
import xarray as xr
import numpy as np
import cv2
from scipy.ndimage import rotate
from scipy.interpolate import griddata

# Test with one file
roars_file = 'ROARS_Level2_PPI_CpiMode1_S20250731T001911_E20250731T001922.nc'
nexrad_file = 'KHTX20250731_001817_V06.nc'

def quick_rotation_test(roars_data, nexrad_data):
    """Quick rotation detection test"""
    # Normalize data
    roars_clean = np.nan_to_num(roars_data, nan=-20)
    nexrad_clean = np.nan_to_num(nexrad_data, nan=-20)
    
    roars_norm = np.clip((roars_clean + 20) / 80 * 255, 0, 255).astype(np.uint8)
    nexrad_norm = np.clip((nexrad_clean + 20) / 80 * 255, 0, 255).astype(np.uint8)
    
    best_angle = 0
    best_correlation = -1
    
    for angle in np.arange(-10, 11, 1):
        rotated = rotate(roars_norm, angle, reshape=False, order=1)
        correlation = cv2.matchTemplate(nexrad_norm, rotated, cv2.TM_CCOEFF_NORMED)
        max_corr = np.max(correlation)
        
        if max_corr > best_correlation:
            best_correlation = max_corr
            best_angle = angle
    
    return best_angle, best_correlation

try:
    # Load data
    ds_roars = xr.open_dataset(roars_file)
    ds_nexrad = xr.open_dataset(nexrad_file)
    
    # Test SP data
    sp_ref = ds_roars['sp_reflectivity_dbz'].values
    sp_ranges = ds_roars['range_sp'].values / 1000
    azimuths = ds_roars['azimuth_deg'].values
    
    # Test LP data  
    lp_ref = ds_roars['lp_reflectivity_dbz'].values
    lp_ranges = ds_roars['range_lp'].values / 1000
    
    # NEXRAD data
    nexrad_ref = ds_nexrad['REF'].values
    if nexrad_ref.ndim == 3:
        nexrad_ref = nexrad_ref[0]
    
    # Create grids
    extent = 50
    grid_size = 100
    x_grid = np.linspace(-extent, extent, grid_size)
    y_grid = np.linspace(-extent, extent, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Convert to cartesian
    az_rad = np.radians(azimuths)
    
    # SP grid
    r_mesh_sp, az_mesh = np.meshgrid(sp_ranges, az_rad)
    x_sp = r_mesh_sp * np.sin(az_mesh)
    y_sp = r_mesh_sp * np.cos(az_mesh)
    sp_grid = griddata((x_sp.flatten(), y_sp.flatten()), sp_ref.flatten(), (xx, yy), method='linear')
    
    # LP grid
    r_mesh_lp, az_mesh = np.meshgrid(lp_ranges, az_rad)
    x_lp = r_mesh_lp * np.sin(az_mesh)
    y_lp = r_mesh_lp * np.cos(az_mesh)
    lp_grid = griddata((x_lp.flatten(), y_lp.flatten()), lp_ref.flatten(), (xx, yy), method='linear')
    
    # Resize NEXRAD
    nexrad_resized = cv2.resize(nexrad_ref, (grid_size, grid_size))
    
    print("ROARS LP vs SP Quick Comparison Test")
    print("===================================")
    
    # Test SP
    sp_angle, sp_corr = quick_rotation_test(sp_grid, nexrad_resized)
    print(f"SP (Short Pulse) Result: {sp_angle:+.1f}° rotation error (confidence: {sp_corr:.3f})")
    
    # Test LP
    lp_angle, lp_corr = quick_rotation_test(lp_grid, nexrad_resized)
    print(f"LP (Long Pulse) Result:  {lp_angle:+.1f}° rotation error (confidence: {lp_corr:.3f})")
    
    print(f"\nConclusion:")
    print(f"SP data shows: {sp_angle:+.1f}° systematic error")
    print(f"LP data shows: {lp_angle:+.1f}° systematic error")
    
    if abs(sp_angle - lp_angle) < 1:
        print("✓ SP and LP data show consistent results")
        print(f"✓ Recommended correction: {-sp_angle:+.1f}°")
    else:
        print("⚠ SP and LP data show different results")
        print("⚠ More analysis needed to determine which is more accurate")
        
    ds_roars.close()
    ds_nexrad.close()
    
except Exception as e:
    print(f"Error: {e}")