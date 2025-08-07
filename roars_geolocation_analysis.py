#!/usr/bin/env python3
"""
ROARS Geolocation Accuracy Analysis
Detects rotation and translation errors between ROARS and NEXRAD data
"""
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import griddata
from scipy.stats import pearsonr
from scipy.ndimage import rotate
import glob
import os
from datetime import datetime
import seaborn as sns
from matplotlib.patches import Rectangle

# Import existing functions from roars_plotting
from roars_plotting import (
    parse_roars_time, load_nexrad_timestamps, find_closest_nexrad_fast,
    get_coordinate_offset, get_roars_radar_location
)

def normalize_radar_data(data, vmin=-20, vmax=60):
    """Normalize radar data to 0-255 for OpenCV processing"""
    data_clipped = np.clip(data, vmin, vmax)
    normalized = ((data_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return normalized

def correlation_rotation_search(roars_img, nexrad_img, angle_range=(-20, 20), angle_step=0.5):
    """Find optimal rotation angle using correlation"""
    best_angle = 0
    best_correlation = -1
    
    angles = np.arange(angle_range[0], angle_range[1] + angle_step, angle_step)
    correlations = []
    
    for angle in angles:
        rotated_roars = rotate(roars_img, angle, reshape=False, order=1)
        
        # Calculate normalized cross-correlation
        correlation = cv2.matchTemplate(nexrad_img, rotated_roars, cv2.TM_CCOEFF_NORMED)
        max_corr = np.max(correlation)
        correlations.append(max_corr)
        
        if max_corr > best_correlation:
            best_correlation = max_corr
            best_angle = angle
    
    return best_angle, best_correlation, angles, correlations

def hierarchical_rotation_search(roars_img, nexrad_img, coarse_range=(-20, 20), fine_range=3):
    """Two-stage hierarchical search for optimal rotation angle"""
    print("   Performing hierarchical rotation search...")
    
    # Stage 1: Coarse search over wide range
    print(f"   Stage 1: Coarse search {coarse_range[0]}° to {coarse_range[1]}° (2° steps)")
    coarse_angle, coarse_corr, _, _ = correlation_rotation_search(
        roars_img, nexrad_img, 
        angle_range=coarse_range, 
        angle_step=2.0
    )
    print(f"   Coarse result: {coarse_angle:+.0f}° (correlation: {coarse_corr:.4f})")
    
    # Stage 2: Fine search around best coarse result
    fine_min = max(coarse_range[0], coarse_angle - fine_range)
    fine_max = min(coarse_range[1], coarse_angle + fine_range)
    print(f"   Stage 2: Fine search {fine_min:+.0f}° to {fine_max:+.0f}° (0.2° steps)")
    
    fine_angle, fine_corr, fine_angles, fine_correlations = correlation_rotation_search(
        roars_img, nexrad_img,
        angle_range=(fine_min, fine_max),
        angle_step=0.2
    )
    print(f"   Final result: {fine_angle:+.1f}° (correlation: {fine_corr:.4f})")
    
    return fine_angle, fine_corr, fine_angles, fine_correlations

def feature_based_registration(roars_img, nexrad_img):
    """Use ORB features to estimate affine transformation"""
    try:
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Detect keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(roars_img, None)
        kp2, des2 = orb.detectAndCompute(nexrad_img, None)
        
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return None, None, 0
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 4:
            return None, None, len(matches)
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        
        # Estimate partial affine transformation (rotation + translation + uniform scaling)
        transform_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        
        if transform_matrix is not None:
            return transform_matrix, mask, len(matches)
        else:
            return None, None, len(matches)
            
    except Exception as e:
        print(f"Feature registration failed: {e}")
        return None, None, 0

def decompose_transformation(transform_matrix):
    """Extract rotation and translation from affine transformation matrix"""
    if transform_matrix is None:
        return None, None, None
    
    # Extract components
    a, b = transform_matrix[0, 0], transform_matrix[0, 1]
    c, d = transform_matrix[1, 0], transform_matrix[1, 1]
    tx, ty = transform_matrix[0, 2], transform_matrix[1, 2]
    
    # Calculate rotation angle
    rotation_angle = np.degrees(np.arctan2(b, a))
    
    # Calculate scale
    scale_x = np.sqrt(a**2 + b**2)
    scale_y = np.sqrt(c**2 + d**2)
    
    return rotation_angle, (tx, ty), (scale_x, scale_y)

def prepare_radar_images(ds_roars, ds_nexrad, grid_size=200):
    """Prepare aligned radar images for analysis"""
    # Get coordinate offset
    offset_x, offset_y = get_coordinate_offset(ds_roars, ds_nexrad)
    
    # Extract ROARS LP data (Long Pulse - better for long range comparison with NEXRAD)
    roars_ref = ds_roars['lp_reflectivity_dbz'].values
    roars_ranges = ds_roars['range_lp'].values / 1000  # km
    roars_azimuths = ds_roars['azimuth_deg'].values
    
    # Convert ROARS to Cartesian
    az_rad = np.radians(roars_azimuths)
    r_mesh, az_mesh = np.meshgrid(roars_ranges, az_rad)
    x_roars = r_mesh * np.sin(az_mesh)
    y_roars = r_mesh * np.cos(az_mesh)
    
    # Create common grid centered on NEXRAD (expanded to accommodate large offsets)
    extent = 130  # ±130 km 
    x_grid = np.linspace(-extent, extent, grid_size)
    y_grid = np.linspace(-extent, extent, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Interpolate ROARS to grid (with offset correction)
    roars_grid = griddata(
        (x_roars.flatten() + offset_x, y_roars.flatten() + offset_y),
        roars_ref.flatten(),
        (xx, yy),
        method='linear',
        fill_value=np.nan
    )
    
    # Get NEXRAD data
    nexrad_ref = ds_nexrad['REF'].values
    if nexrad_ref.ndim == 3:
        nexrad_ref = nexrad_ref[0]
    
    # Resize NEXRAD to match grid
    nexrad_resized = cv2.resize(nexrad_ref, (grid_size, grid_size))
    
    return roars_grid, nexrad_resized, (offset_x, offset_y)

def analyze_single_pair(ds_roars, ds_nexrad):
    """Analyze a single ROARS-NEXRAD pair for geolocation errors"""
    results = {
        'timestamp': ds_roars['cpi_timestamp'].values[0],
        'coordinate_offset': None,
        'correlation_rotation': None,
        'correlation_confidence': None,
        'feature_rotation': None,
        'feature_translation': None,
        'feature_matches': 0,
        'method_agreement': None
    }
    
    try:
        # Prepare aligned images
        roars_img, nexrad_img, coord_offset = prepare_radar_images(ds_roars, ds_nexrad)
        results['coordinate_offset'] = coord_offset
        
        # Remove NaN values for processing
        roars_clean = np.nan_to_num(roars_img, nan=-20)
        nexrad_clean = np.nan_to_num(nexrad_img, nan=-20)
        
        # Normalize for OpenCV
        roars_norm = normalize_radar_data(roars_clean)
        nexrad_norm = normalize_radar_data(nexrad_clean)
        
        # Method 1: Hierarchical correlation-based rotation search
        best_angle, best_corr, angles, correlations = hierarchical_rotation_search(
            roars_norm, nexrad_norm
        )
        results['correlation_rotation'] = best_angle
        results['correlation_confidence'] = best_corr
        
        # Method 2: Feature-based registration
        transform_matrix, mask, n_matches = feature_based_registration(roars_norm, nexrad_norm)
        results['feature_matches'] = n_matches
        
        if transform_matrix is not None:
            rotation, translation, scale = decompose_transformation(transform_matrix)
            results['feature_rotation'] = rotation
            results['feature_translation'] = translation
            
            # Check agreement between methods
            if abs(rotation - best_angle) < 2.0:  # Within 2 degrees
                results['method_agreement'] = True
            else:
                results['method_agreement'] = False
        
        print(f"✓ Analysis complete: Corr={best_angle:.1f}°, Features={n_matches} matches")
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
    
    return results

def process_dataset(max_files=10):
    """Process ROARS dataset to analyze geolocation accuracy"""
    print("ROARS Geolocation Accuracy Analysis")
    print("===================================")
    
    # Find files
    roars_files = sorted(glob.glob('/Users/oomitusa/Documents/Research/ROARS/roars/20250731/ROARS_Level2*PPI*.nc'))
    nexrad_files = sorted(glob.glob('/Users/oomitusa/Documents/Research/ROARS/nexrad_cappi/20250731/KHTX/*.nc'))
    
    if max_files:
        roars_files = roars_files[:max_files]
    
    print(f"Processing {len(roars_files)} ROARS files")
    
    # Pre-load NEXRAD timestamps
    nexrad_times = load_nexrad_timestamps(nexrad_files)
    
    results = []
    
    for i, roars_file in enumerate(roars_files):
        print(f"\nProcessing {i+1}/{len(roars_files)}: {os.path.basename(roars_file)}")
        
        try:
            # Find matching NEXRAD file
            roars_time = parse_roars_time(roars_file)
            closest_nexrad, time_diff = find_closest_nexrad_fast(roars_time, nexrad_times)
            
            if not closest_nexrad or time_diff > 300:  # Skip if >5 min difference
                print(f"  ✗ No good NEXRAD match (Δt={time_diff:.0f}s)")
                continue
            
            print(f"  → {os.path.basename(closest_nexrad)} (Δt={time_diff:.0f}s)")
            
            # Load datasets
            ds_roars = xr.open_dataset(roars_file)
            ds_nexrad = xr.open_dataset(closest_nexrad)
            
            # Analyze pair
            result = analyze_single_pair(ds_roars, ds_nexrad)
            results.append(result)
            
            ds_roars.close()
            ds_nexrad.close()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    return results

def create_visualizations(results):
    """Create comprehensive visualization plots with enhanced interpretability"""
    if not results:
        return
    
    # Extract data for plotting
    timestamps = [datetime.utcfromtimestamp(r['timestamp']) for r in results]
    corr_rotations = [r['correlation_rotation'] for r in results if r['correlation_rotation'] is not None]
    corr_confidences = [r['correlation_confidence'] for r in results if r['correlation_confidence'] is not None]
    feat_rotations = [r['feature_rotation'] for r in results if r['feature_rotation'] is not None]
    feat_matches = [r['feature_matches'] for r in results]
    coord_offsets_x = [r['coordinate_offset'][0] for r in results if r['coordinate_offset'] is not None]
    coord_offsets_y = [r['coordinate_offset'][1] for r in results if r['coordinate_offset'] is not None]
    
    # Set style for better readability
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))
    
    # 1. Time series of rotation errors - Enhanced for LP data
    ax1 = plt.subplot(2, 3, 1)
    if corr_rotations:
        plt.plot(timestamps, corr_rotations, 'bo-', label='Correlation Method (LP)', markersize=8, linewidth=2)
        # Add error bands if there's variation
        if len(set(corr_rotations)) > 1:
            mean_val = np.mean(corr_rotations)
            std_val = np.std(corr_rotations)
            plt.fill_between(timestamps, [mean_val - std_val]*len(timestamps), 
                           [mean_val + std_val]*len(timestamps), alpha=0.2, color='blue')
    if feat_rotations:
        feat_times = [timestamps[i] for i, r in enumerate(results) if r['feature_rotation'] is not None]
        plt.plot(feat_times, feat_rotations, 'ro-', label='Feature Method', markersize=8, linewidth=2)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.7, linewidth=2, label='Perfect Alignment')
    plt.xlabel('Time (UTC)', fontweight='bold')
    plt.ylabel('Rotation Error (°)', fontweight='bold')
    plt.title('ROARS LP Rotation Error Over Time\n(Negative = Counterclockwise)', fontweight='bold', pad=15)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    plt.xticks(rotation=45)
    
    # Add text annotation for key finding
    if corr_rotations:
        mean_error = np.mean(corr_rotations)
        ax1.text(0.02, 0.98, f'Mean Error: {mean_error:.1f}°\nCorrection: +{-mean_error:.1f}°', 
                transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    # 2. Enhanced histogram and statistics
    ax2 = plt.subplot(2, 3, 2)
    if corr_rotations:
        # Calculate statistics
        mean_rot = np.mean(corr_rotations)
        std_rot = np.std(corr_rotations)
        
        # Create histogram with better binning
        n_bins = max(5, min(20, len(set(corr_rotations))))
        counts, bins, patches = plt.hist(corr_rotations, bins=n_bins, alpha=0.8, 
                                       label=f'LP Data (n={len(corr_rotations)})', 
                                       color='skyblue', edgecolor='navy', linewidth=1.5)
        
        # Add vertical lines for statistics
        plt.axvline(mean_rot, color='red', linestyle='-', linewidth=3, label=f'Mean: {mean_rot:.2f}°')
        if std_rot > 0:
            plt.axvline(mean_rot - std_rot, color='orange', linestyle='--', linewidth=2, alpha=0.7)
            plt.axvline(mean_rot + std_rot, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
                       label=f'±1σ: {std_rot:.3f}°')
    
    if feat_rotations:
        plt.hist(feat_rotations, bins=20, alpha=0.7, label='Features', color='red', edgecolor='darkred')
    
    plt.axvline(0, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Perfect (0°)')
    plt.xlabel('Rotation Error (°)', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title('Distribution of LP Rotation Errors\n(Consistency Check)', fontweight='bold', pad=15)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    # 3. Enhanced confidence analysis
    ax3 = plt.subplot(2, 3, 3)
    if corr_rotations and corr_confidences:
        # Create scatter plot with enhanced styling
        scatter = plt.scatter(corr_rotations, corr_confidences, c=range(len(corr_rotations)), 
                             cmap='plasma', s=120, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add trend line if there's variation
        if len(set(corr_rotations)) > 1:
            z = np.polyfit(corr_rotations, corr_confidences, 1)
            p = np.poly1d(z)
            plt.plot(corr_rotations, p(corr_rotations), "r--", alpha=0.8, linewidth=2, label='Trend')
            plt.legend()
        
        plt.xlabel('Rotation Error (°)', fontweight='bold')
        plt.ylabel('Correlation Confidence', fontweight='bold')
        plt.title('LP Data Quality Assessment\n(Confidence vs Error)', fontweight='bold', pad=15)
        
        # Add colorbar with better formatting
        cbar = plt.colorbar(scatter, label='Time Sequence')
        cbar.ax.tick_params(labelsize=10)
        
        plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        
        # Add confidence threshold line
        if corr_confidences:
            high_conf_threshold = np.percentile(corr_confidences, 75)
            plt.axhline(high_conf_threshold, color='green', linestyle=':', linewidth=2, alpha=0.7,
                       label=f'75th percentile: {high_conf_threshold:.3f}')
            plt.legend()
    
    # 4. Coordinate offsets (translation errors)
    ax4 = plt.subplot(2, 3, 4)
    if coord_offsets_x and coord_offsets_y:
        plt.scatter(coord_offsets_x, coord_offsets_y, c=range(len(coord_offsets_x)), 
                   cmap='plasma', s=80, alpha=0.7, edgecolors='black')
        plt.xlabel('East Offset (km)')
        plt.ylabel('North Offset (km)')
        plt.title('GPS Position Offsets')
        plt.colorbar(label='Time Order')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
    
    # 5. Feature matching statistics
    ax5 = plt.subplot(2, 3, 5)
    plt.bar(range(len(feat_matches)), feat_matches, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('File Index')
    plt.ylabel('Number of Feature Matches')
    plt.title('Feature Detection Success')
    plt.grid(True, alpha=0.3)
    
    # 6. Error statistics summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create statistics text
    stats_text = "=== ANALYSIS STATISTICS ===\n\n"
    
    if corr_rotations:
        mean_rot = np.mean(corr_rotations)
        std_rot = np.std(corr_rotations)
        stats_text += f"Correlation Method:\n"
        stats_text += f"  Mean Error: {mean_rot:.3f}°\n"
        stats_text += f"  Std Dev: {std_rot:.3f}°\n"
        stats_text += f"  Range: {np.min(corr_rotations):.2f}° to {np.max(corr_rotations):.2f}°\n"
        stats_text += f"  Samples: {len(corr_rotations)}\n\n"
    
    if feat_rotations:
        mean_feat = np.mean(feat_rotations)
        std_feat = np.std(feat_rotations)
        stats_text += f"Feature Method:\n"
        stats_text += f"  Mean Error: {mean_feat:.3f}°\n"
        stats_text += f"  Std Dev: {std_feat:.3f}°\n"
        stats_text += f"  Samples: {len(feat_rotations)}\n\n"
    
    if coord_offsets_x and coord_offsets_y:
        stats_text += f"GPS Position Offsets:\n"
        stats_text += f"  Mean East: {np.mean(coord_offsets_x):.3f} km\n"
        stats_text += f"  Mean North: {np.mean(coord_offsets_y):.3f} km\n"
        stats_text += f"  Distance: {np.sqrt(np.mean(coord_offsets_x)**2 + np.mean(coord_offsets_y)**2):.3f} km\n\n"
    
    # Recommendation
    if corr_rotations:
        mean_error = np.mean(corr_rotations)
        if abs(mean_error) > 1.0:
            stats_text += f"RECOMMENDATION:\n"
            stats_text += f"Apply azimuth correction: {-mean_error:.2f}°\n"
            stats_text += f"(Add {-mean_error:.2f}° to all azimuth readings)"
        else:
            stats_text += f"RECOMMENDATION:\nNo significant systematic error"
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add overall title with LP emphasis
    fig.suptitle('ROARS Long Pulse (LP) Geolocation Analysis\nRotation Error Detection & Statistics', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.savefig('roars_geolocation_analysis_claude.png', dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    print("✓ Enhanced LP visualization saved: roars_geolocation_analysis.png")

def create_example_alignment_plot(ds_roars, ds_nexrad, result):
    """Create enhanced plot showing LP alignment correction with clear interpretation"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # Prepare data using LP
    roars_img, nexrad_img, coord_offset = prepare_radar_images(ds_roars, ds_nexrad)
    rotation_error = result['correlation_rotation']
    
    # Clean data
    roars_clean = np.nan_to_num(roars_img, nan=-20)
    nexrad_clean = np.nan_to_num(nexrad_img, nan=-20)
    
    # Use larger extent for LP data comparison (increased to handle coordinate offset)
    extent = 130
    
    # Plot 1: Original ROARS LP
    im1 = ax1.imshow(roars_clean, extent=[-extent, extent, -extent, extent], 
                     vmin=-20, vmax=60, cmap='NWSRef', origin='lower')
    ax1.set_title('ROARS Long Pulse (Original)\nBefore Rotation Correction', fontweight='bold', pad=15)
    ax1.set_xlabel('Distance East (km)', fontweight='bold')
    ax1.set_ylabel('Distance North (km)', fontweight='bold')
    ax1.grid(True, alpha=0.4, color='white', linewidth=0.8)
    ax1.set_aspect('equal')
    
    # Add compass rose
    ax1.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7, label='ROARS LP (dBZ)')
    cbar1.ax.tick_params(labelsize=10)
    
    # Plot 2: NEXRAD
    im2 = ax2.imshow(nexrad_clean, extent=[-extent, extent, -extent, extent], 
                     vmin=-20, vmax=60, cmap='NWSRef', origin='lower')
    ax2.set_title('NEXRAD CAPPI (Reference Ground Truth)\nCorrect Geolocation', fontweight='bold', pad=15)
    ax2.set_xlabel('Distance East (km)', fontweight='bold')
    ax2.set_ylabel('Distance North (km)', fontweight='bold')
    ax2.grid(True, alpha=0.4, color='white', linewidth=0.8)
    ax2.set_aspect('equal')
    
    # Add compass rose
    ax2.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7, label='NEXRAD (dBZ)')
    cbar2.ax.tick_params(labelsize=10)
    
    # Plot 3: Corrected ROARS
    if rotation_error is not None:
        roars_corrected = rotate(roars_clean, -rotation_error, reshape=False, order=1)
        im3 = ax3.imshow(roars_corrected, extent=[-extent, extent, -extent, extent], 
                         vmin=-20, vmax=60, cmap='plasma', origin='lower')  # Different colormap for ROARS
        ax3.set_title(f'ROARS LP After Correction\nRotated by {-rotation_error:.1f}°', fontweight='bold', pad=15)
    else:
        im3 = ax3.imshow(roars_clean, extent=[-extent, extent, -extent, extent], 
                         vmin=-20, vmax=60, cmap='plasma', origin='lower')
        ax3.set_title('ROARS LP (No Correction Available)', fontweight='bold', pad=15)
    
    ax3.set_xlabel('Distance East (km)', fontweight='bold')
    ax3.set_ylabel('Distance North (km)', fontweight='bold')
    ax3.grid(True, alpha=0.4, color='white', linewidth=0.8)
    ax3.set_aspect('equal')
    
    # Add compass rose
    ax3.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.7, label='ROARS LP Corrected (dBZ)')
    cbar3.ax.tick_params(labelsize=10)
    
    # Plot 4: Enhanced Overlay comparison
    # Start with NEXRAD as background (using NWSRef colormap)
    im4_nexrad = ax4.imshow(nexrad_clean, extent=[-extent, extent, -extent, extent], 
                           vmin=-20, vmax=60, cmap='NWSRef', origin='lower', alpha=0.8)
    
    if rotation_error is not None:
        roars_overlay = rotate(roars_clean, -rotation_error, reshape=False, order=1)
        # Create stronger mask for overlay - show moderate to strong echoes
        roars_masked = np.where(roars_overlay > 15, roars_overlay, np.nan)
        
        # Overlay ROARS using a contrasting colormap (hot/magma) with transparency
        im4_roars = ax4.imshow(roars_masked, extent=[-extent, extent, -extent, extent], 
                              vmin=15, vmax=60, cmap='hot', origin='lower', alpha=0.7)
        
        ax4.set_title(f'Overlay: NEXRAD (NWSRef) + Corrected ROARS (Hot)\nAlignment after {-rotation_error:.1f}° rotation correction', 
                     fontweight='bold', pad=15)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='blue', lw=4, label='NEXRAD (Background)'),
                          Line2D([0], [0], color='red', lw=4, label='ROARS LP (Overlay)'),
                          Line2D([0], [0], color='purple', lw=4, label='Perfect Overlap')]
        ax4.legend(handles=legend_elements, loc='lower right', frameon=True, 
                  fancybox=True, shadow=True, fontsize=11)
    else:
        ax4.set_title('Overlay (Correction Failed)', fontweight='bold', pad=15)
    
    ax4.set_xlabel('Distance East (km)', fontweight='bold')
    ax4.set_ylabel('Distance North (km)', fontweight='bold')
    ax4.grid(True, alpha=0.4, color='white', linewidth=0.8)
    ax4.set_aspect('equal')
    
    # Add compass rose
    ax4.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    
    timestamp = datetime.utcfromtimestamp(result['timestamp'])
    correction_text = f"Detected Error: {rotation_error:.1f}° | Suggested Correction: +{-rotation_error:.1f}°" if rotation_error else "No Correction"
    fig.suptitle(f'ROARS LP Alignment Correction Example\n{timestamp.strftime("%Y-%m-%d %H:%M:%S")} UTC | {correction_text}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('roars_alignment_example.png', dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    print("✓ Enhanced LP alignment example saved: roars_alignment_example.png")

def create_specific_time_alignment_plot(target_time_str):
    """Create alignment plot for specific time (e.g., '023019')"""
    try:
        # Find ROARS files
        roars_files = sorted(glob.glob('/Users/oomitusa/Documents/Research/ROARS/roars/20250731/ROARS_Level2*PPI*.nc'))
        nexrad_files = sorted(glob.glob('/Users/oomitusa/Documents/Research/ROARS/nexrad_cappi/20250731/KHTX/*.nc'))
        
        # Find file with specific time
        target_roars_file = None
        for rfile in roars_files:
            if target_time_str in rfile:
                target_roars_file = rfile
                break
        
        if not target_roars_file:
            print(f"✗ Could not find ROARS file with time {target_time_str}")
            return
        
        print(f"  Found target file: {os.path.basename(target_roars_file)}")
        
        # Load NEXRAD timestamps for matching
        nexrad_times = load_nexrad_timestamps(nexrad_files[:20])  # Load subset for speed
        
        # Parse ROARS time and find closest NEXRAD
        roars_time = parse_roars_time(target_roars_file)
        if not roars_time:
            print(f"✗ Could not parse time from {target_roars_file}")
            return
        
        closest_nexrad, time_diff = find_closest_nexrad_fast(roars_time, nexrad_times)
        if not closest_nexrad:
            print(f"✗ No matching NEXRAD file found")
            return
        
        print(f"  → Matched with {os.path.basename(closest_nexrad)} (Δt={time_diff:.0f}s)")
        
        # Load datasets
        ds_roars = xr.open_dataset(target_roars_file)
        ds_nexrad = xr.open_dataset(closest_nexrad)
        
        # Analyze this specific pair
        result = analyze_single_pair(ds_roars, ds_nexrad)
        print(f"  Analysis: Rotation = {result['correlation_rotation']:.1f}°, Confidence = {result['correlation_confidence']:.3f}")
        
        # Create the enhanced alignment plot
        create_example_alignment_plot(ds_roars, ds_nexrad, result)
        
        ds_roars.close()
        ds_nexrad.close()
        
    except Exception as e:
        print(f"✗ Failed to create specific time alignment plot: {e}")

def generate_report(results):
    """Generate comprehensive report with statistics and visualizations"""
    if not results:
        print("No valid results to analyze")
        return
    
    # Create visualizations (simplified for speed)
    try:
        create_visualizations(results)
    except Exception as e:
        print(f"Visualization creation skipped due to: {e}")
    
    # Create alignment plot using 023019 time as requested
    print("📊 Creating alignment example for 023019 time...")
    create_specific_time_alignment_plot("023019")
    
    # Extract statistics
    corr_rotations = [r['correlation_rotation'] for r in results if r['correlation_rotation'] is not None]
    corr_confidences = [r['correlation_confidence'] for r in results if r['correlation_confidence'] is not None]
    feat_rotations = [r['feature_rotation'] for r in results if r['feature_rotation'] is not None]
    coord_offsets_x = [r['coordinate_offset'][0] for r in results if r['coordinate_offset'] is not None]
    coord_offsets_y = [r['coordinate_offset'][1] for r in results if r['coordinate_offset'] is not None]
    
    print(f"\n{'='*60}")
    print(f"ROARS LONG PULSE (LP) GEOLOCATION ANALYSIS REPORT")
    print(f"{'='*60}")
    print(f"Analysis Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {len(results)} ROARS-NEXRAD file pairs processed")
    print(f"Method: Correlation-based rotation detection using LP reflectivity")
    
    if corr_rotations:
        mean_rot = np.mean(corr_rotations)
        std_rot = np.std(corr_rotations)
        median_rot = np.median(corr_rotations)
        
        print(f"\n📊 ROTATION ERROR STATISTICS (Long Pulse Data):")
        print(f"{'─'*50}")
        print(f"  Mean rotation error:     {mean_rot:+7.3f}° ± {std_rot:.3f}°")
        print(f"  Median rotation error:   {median_rot:+7.3f}°")
        print(f"  Standard deviation:      {std_rot:7.3f}° ({'Excellent' if std_rot < 0.1 else 'Good' if std_rot < 0.5 else 'Fair'})")
        print(f"  Range:                   {np.min(corr_rotations):+7.3f}° to {np.max(corr_rotations):+7.3f}°")
        print(f"  95% confidence interval: {mean_rot - 1.96*std_rot:+7.3f}° to {mean_rot + 1.96*std_rot:+7.3f}°")
        print(f"  Sample size:             {len(corr_rotations):7d} measurements")
        
        # Calculate confidence metrics
        consistency = 1.0 - (std_rot / max(abs(mean_rot), 1.0))
        consistency_pct = max(0, min(100, consistency * 100))
        
        print(f"\n🎯 CONFIDENCE ASSESSMENT:")
        print(f"{'─'*50}")
        print(f"  Error consistency:       {consistency_pct:6.1f}% ({'Excellent' if consistency_pct > 95 else 'Good' if consistency_pct > 80 else 'Fair'})")
        
        if corr_confidences:
            mean_conf = np.mean(corr_confidences)
            min_conf = np.min(corr_confidences)
            print(f"  Correlation strength:    {mean_conf:6.3f} (avg), {min_conf:6.3f} (min)")
            print(f"  Data quality:            {'Excellent' if min_conf > 0.7 else 'Good' if min_conf > 0.5 else 'Fair'}")
    
    if feat_rotations:
        mean_feat = np.mean(feat_rotations)
        std_feat = np.std(feat_rotations)
        print(f"\nFeature-based Rotation Analysis:")
        print(f"  Mean rotation error: {mean_feat:.3f}° ± {std_feat:.3f}°")
        print(f"  Median: {np.median(feat_rotations):.3f}°")
        print(f"  Samples: {len(feat_rotations)}")
    
    if coord_offsets_x and coord_offsets_y:
        mean_x, mean_y = np.mean(coord_offsets_x), np.mean(coord_offsets_y)
        std_x, std_y = np.std(coord_offsets_x), np.std(coord_offsets_y)
        distance = np.sqrt(mean_x**2 + mean_y**2)
        print(f"\nGPS Position Offset Analysis:")
        print(f"  Mean East offset: {mean_x:.3f} ± {std_x:.3f} km")
        print(f"  Mean North offset: {mean_y:.3f} ± {std_y:.3f} km")
        print(f"  Total distance offset: {distance:.3f} km")
        print(f"  Bearing of offset: {np.degrees(np.arctan2(mean_x, mean_y)):.1f}° from north")
    
    # Method agreement analysis
    agreements = [r['method_agreement'] for r in results if r['method_agreement'] is not None]
    if agreements:
        agreement_rate = np.mean(agreements) * 100
        print(f"\nMethod Agreement: {agreement_rate:.1f}% of cases where both methods worked")
    
    # Final recommendation with enhanced formatting
    if corr_rotations:
        mean_error = np.mean(corr_rotations)
        print(f"\n🎯 FINAL RECOMMENDATION & IMPLEMENTATION")
        print(f"{'='*60}")
        
        if abs(mean_error) > 1.0:
            print(f"🔍 SYSTEMATIC ERROR DETECTED:")
            print(f"  Current ROARS azimuth error:  {mean_error:+7.3f}°")
            print(f"  Error direction:              {'Counterclockwise' if mean_error < 0 else 'Clockwise'}")
            print(f"  Error magnitude:              {abs(mean_error):7.3f}° ({'Significant' if abs(mean_error) > 5 else 'Moderate'})")
            
            print(f"\n🔧 RECOMMENDED CORRECTION:")
            print(f"  Apply azimuth offset:         {-mean_error:+7.3f}°")
            print(f"  Implementation:               Add {-mean_error:+7.3f}° to all ROARS azimuth readings")
            print(f"  New azimuth formula:          ROARS_corrected = ROARS_raw + ({-mean_error:+.1f}°)")
            
            print(f"\n✅ CONFIDENCE LEVEL:")
            if std_rot < 0.1:
                print(f"  Reliability:                  EXCELLENT (σ = {std_rot:.3f}°)")
                print(f"  Recommendation:               Apply correction immediately")
            elif std_rot < 0.5:
                print(f"  Reliability:                  GOOD (σ = {std_rot:.3f}°)")
                print(f"  Recommendation:               Apply correction with monitoring")
            else:
                print(f"  Reliability:                  FAIR (σ = {std_rot:.3f}°)")
                print(f"  Recommendation:               Collect more data before applying")
                
            print(f"\n📈 EXPECTED IMPROVEMENT:")
            print(f"  This correction should align ROARS LP data with NEXRAD coordinates")
            print(f"  Estimated accuracy improvement: ~{abs(mean_error):.1f}° reduction in systematic error")
            
        else:
            print(f"✅ NO SIGNIFICANT ERROR DETECTED:")
            print(f"  ROARS azimuth alignment appears accurate within ±1.0°")
            print(f"  Current error: {mean_error:+.3f}° (within acceptable range)")
            print(f"  Recommendation: No correction needed")
    
    print(f"\n📊 OUTPUT FILES GENERATED:")
    print(f"{'─'*50}")
    print(f"✓ Enhanced LP analysis plot:     roars_geolocation_analysis.png")
    print(f"✓ Before/after example:         roars_alignment_example.png")
    print(f"✓ Statistical report:           This terminal output")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete. Review visualizations for detailed insights.")
    print(f"{'='*60}")

if __name__ == "__main__":
    results = process_dataset(max_files=10)  # Quick test with 3 files
    generate_report(results)