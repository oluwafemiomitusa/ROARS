#!/usr/bin/env python3
"""
Debug ROARS LP Data Issues
Investigate why LP data isn't showing in the alignment plots
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2

# Import functions from our analysis script
from roars_plotting import get_coordinate_offset, get_roars_radar_location

def debug_lp_data():
    """Debug ROARS LP data to find why it's not displaying"""
    
    # Load the 023019 file specifically
    roars_file = None
    import glob
    roars_files = sorted(glob.glob('/Users/oomitusa/Documents/Research/ROARS/roars/20250731/ROARS_Level2*PPI*.nc'))
    
    for rfile in roars_files:
        if '023019' in rfile:
            roars_file = rfile
            break
    
    if not roars_file:
        # Use any available file for debugging
        roars_file = 'ROARS_Level2_PPI_CpiMode1_S20250731T001911_E20250731T001922.nc'
    
    nexrad_file = 'KHTX20250731_001817_V06.nc'
    
    print(f"=== ROARS LP DATA DEBUGGING ===")
    print(f"ROARS file: {roars_file}")
    print(f"NEXRAD file: {nexrad_file}")
    
    try:
        # Load datasets
        ds_roars = xr.open_dataset(roars_file)
        ds_nexrad = xr.open_dataset(nexrad_file)
        
        print(f"\n1. RAW DATA ANALYSIS:")
        print(f"   ROARS dimensions: {dict(ds_roars.dims)}")
        print(f"   NEXRAD dimensions: {dict(ds_nexrad.dims)}")
        
        # Check LP data specifically
        lp_ref = ds_roars['lp_reflectivity_dbz'].values
        lp_ranges = ds_roars['range_lp'].values / 1000  # km
        azimuths = ds_roars['azimuth_deg'].values
        
        print(f"\n2. LP REFLECTIVITY DATA:")
        print(f"   Shape: {lp_ref.shape}")
        print(f"   Min: {np.nanmin(lp_ref):.2f} dBZ")
        print(f"   Max: {np.nanmax(lp_ref):.2f} dBZ")
        print(f"   Mean: {np.nanmean(lp_ref):.2f} dBZ")
        print(f"   NaN count: {np.sum(np.isnan(lp_ref))}/{lp_ref.size} ({100*np.sum(np.isnan(lp_ref))/lp_ref.size:.1f}%)")
        print(f"   Values > 0: {np.sum(lp_ref > 0)}")
        print(f"   Values > 10: {np.sum(lp_ref > 10)}")
        print(f"   Values > 20: {np.sum(lp_ref > 20)}")
        
        print(f"\n3. LP RANGES:")
        print(f"   Min range: {np.min(lp_ranges):.2f} km")
        print(f"   Max range: {np.max(lp_ranges):.2f} km")
        print(f"   Range bins: {len(lp_ranges)}")
        
        print(f"\n4. COORDINATE OFFSET:")
        offset_x, offset_y = get_coordinate_offset(ds_roars, ds_nexrad)
        print(f"   Offset X (East): {offset_x:.2f} km")
        print(f"   Offset Y (North): {offset_y:.2f} km")
        print(f"   Total offset: {np.sqrt(offset_x**2 + offset_y**2):.2f} km")
        
        # Check if offset is pushing data outside grid
        extent = 50  # Original grid extent
        extent_large = 130  # New larger grid extent
        print(f"   Original grid extent: ±{extent} km")
        print(f"   New grid extent: ±{extent_large} km")
        print(f"   After offset, ROARS data spans:")
        print(f"     X: {np.min(lp_ranges) + offset_x:.1f} to {np.max(lp_ranges) + offset_x:.1f} km")
        print(f"     Y: {np.min(lp_ranges) + offset_y:.1f} to {np.max(lp_ranges) + offset_y:.1f} km")
        
        # Check if data falls outside original vs new grid
        if (np.max(lp_ranges) + abs(offset_x)) > extent or (np.max(lp_ranges) + abs(offset_y)) > extent:
            print(f"   ⚠️  Data would be outside original ±{extent} km grid")
        if (np.max(lp_ranges) + abs(offset_x)) > extent_large or (np.max(lp_ranges) + abs(offset_y)) > extent_large:
            print(f"   ❌ Data still outside new ±{extent_large} km grid!")
        else:
            print(f"   ✅ Data should fit in new ±{extent_large} km grid")
        
        print(f"\n5. GRID INTERPOLATION TEST:")
        # Convert to Cartesian
        az_rad = np.radians(azimuths)
        r_mesh, az_mesh = np.meshgrid(lp_ranges, az_rad)
        x_roars = r_mesh * np.sin(az_mesh)
        y_roars = r_mesh * np.cos(az_mesh)
        
        # Create grid with larger extent to accommodate offset
        extent_large = 130  # Increased to handle 98 km offset
        grid_size = 200
        x_grid = np.linspace(-extent_large, extent_large, grid_size)
        y_grid = np.linspace(-extent_large, extent_large, grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        # Test interpolation with and without offset
        print("   Testing without offset:")
        roars_grid_no_offset = griddata(
            (x_roars.flatten(), y_roars.flatten()),
            lp_ref.flatten(),
            (xx, yy),
            method='linear',
            fill_value=np.nan
        )
        valid_no_offset = ~np.isnan(roars_grid_no_offset)
        print(f"     Valid points: {np.sum(valid_no_offset)}/{roars_grid_no_offset.size} ({100*np.sum(valid_no_offset)/roars_grid_no_offset.size:.2f}%)")
        if np.sum(valid_no_offset) > 0:
            print(f"     Data range: {np.nanmin(roars_grid_no_offset):.2f} to {np.nanmax(roars_grid_no_offset):.2f} dBZ")
        
        print("   Testing with offset:")
        roars_grid_with_offset = griddata(
            (x_roars.flatten() + offset_x, y_roars.flatten() + offset_y),
            lp_ref.flatten(),
            (xx, yy),
            method='linear',
            fill_value=np.nan
        )
        valid_with_offset = ~np.isnan(roars_grid_with_offset)
        print(f"     Valid points: {np.sum(valid_with_offset)}/{roars_grid_with_offset.size} ({100*np.sum(valid_with_offset)/roars_grid_with_offset.size:.2f}%)")
        if np.sum(valid_with_offset) > 0:
            print(f"     Data range: {np.nanmin(roars_grid_with_offset):.2f} to {np.nanmax(roars_grid_with_offset):.2f} dBZ")
        
        # Create diagnostic plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Raw LP data in polar coordinates (corrected orientation)
        # Data shape: (121 beams, 6656 range bins) so lp_ref[beam, range]
        im1 = ax1.pcolormesh(lp_ranges, azimuths, lp_ref, vmin=-20, vmax=60, cmap='NWSRef', shading='nearest')
        ax1.set_title('Raw ROARS LP Data (Polar)\nCorrect Orientation')
        ax1.set_xlabel('Range (km)')
        ax1.set_ylabel('Azimuth (degrees)')
        plt.colorbar(im1, ax=ax1, label='dBZ')
        
        # Plot 2: LP data without offset (using larger extent)
        if np.sum(valid_no_offset) > 0:
            im2 = ax2.imshow(roars_grid_no_offset, extent=[-extent_large, extent_large, -extent_large, extent_large], 
                           vmin=-20, vmax=60, cmap='NWSRef', origin='lower')
            ax2.set_title(f'LP Data Grid (No Offset)\nValid: {np.sum(valid_no_offset)} points')
            plt.colorbar(im2, ax=ax2, shrink=0.7, label='dBZ')
        else:
            ax2.text(0.5, 0.5, 'No valid data\nafter gridding', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('LP Data Grid (No Offset) - EMPTY')
        ax2.set_xlabel('Distance East (km)')
        ax2.set_ylabel('Distance North (km)')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: LP data with offset (using larger extent)
        if np.sum(valid_with_offset) > 0:
            im3 = ax3.imshow(roars_grid_with_offset, extent=[-extent_large, extent_large, -extent_large, extent_large], 
                           vmin=-20, vmax=60, cmap='NWSRef', origin='lower')
            ax3.set_title(f'LP Data Grid (With Offset: {offset_x:.1f}, {offset_y:.1f} km)\nValid: {np.sum(valid_with_offset)} points')
            plt.colorbar(im3, ax=ax3, shrink=0.7, label='dBZ')
        else:
            ax3.text(0.5, 0.5, 'No valid data\nafter offset+gridding', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title(f'LP Data Grid (With Offset) - EMPTY')
        ax3.set_xlabel('Distance East (km)')
        ax3.set_ylabel('Distance North (km)')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: NEXRAD for comparison (using larger extent)
        nexrad_ref = ds_nexrad['REF'].values
        if nexrad_ref.ndim == 3:
            nexrad_ref = nexrad_ref[0]
        nexrad_resized = cv2.resize(nexrad_ref, (grid_size, grid_size))
        
        im4 = ax4.imshow(nexrad_resized, extent=[-extent_large, extent_large, -extent_large, extent_large], 
                        vmin=-20, vmax=60, cmap='NWSRef', origin='lower')
        plt.colorbar(im4, ax=ax4, shrink=0.7, label='dBZ')
        ax4.set_title('NEXRAD (Reference)')
        ax4.set_xlabel('Distance East (km)')
        ax4.set_ylabel('Distance North (km)')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roars_lp_debug.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Debug plot saved: roars_lp_debug.png")
        
        # Recommendations
        print(f"\n=== RECOMMENDATIONS ===")
        if np.sum(valid_with_offset) == 0:
            print("❌ MAJOR ISSUE: No ROARS LP data visible after offset+gridding")
            if abs(offset_x) > extent/2 or abs(offset_y) > extent/2:
                print(f"   → The coordinate offset ({offset_x:.1f}, {offset_y:.1f} km) is too large for the grid extent (±{extent} km)")
                print(f"   → Solution: Increase grid extent to ±{max(abs(offset_x), abs(offset_y)) + 50:.0f} km")
            else:
                print("   → Issue may be with interpolation method or data quality")
        
        if np.nanmax(lp_ref) < 0:
            print("❌ All ROARS LP reflectivity values are negative - may be noise floor")
            
        if np.sum(lp_ref > 10) < 100:
            print("⚠️  Very few ROARS LP values > 10 dBZ - weak echoes or no precipitation")
        
        ds_roars.close()
        ds_nexrad.close()
        
    except Exception as e:
        print(f"Error in debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_lp_data()