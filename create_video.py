#!/usr/bin/env python3
"""
Create a video illustrating changes in soil moisture over time.

This script reads daily soil moisture GeoTIFF files from a directory, sorts them
chronologically, renders them as styled maps with a consistent scale and colorbar,
and compiles them into an MPEG-4 (MP4) video with a 1-second delay between frames.
"""

import argparse
import warnings
# Suppress deprecation warnings from rasterio/numpy 2.5 shape assignment
warnings.filterwarnings("ignore", category=DeprecationWarning)
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib
# Use non-interactive Agg backend for script execution
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
import imageio
from PIL import Image


def parse_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract the 8-digit date (YYYYMMDD) from a filename.
    
    Example: 'SPL4SMGP.008_sm_surface_20260301.tif' -> datetime(2026, 3, 1)
    """
    match = re.search(r'(\d{8})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d')
        except ValueError:
            pass
    return None


def get_sorted_tiff_files(directory: Path) -> List[Tuple[Path, datetime]]:
    """Find all TIFF files in the directory and sort them by date."""
    tiff_extensions = {'.tif', '.tiff'}
    files_with_dates = []
    
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in tiff_extensions:
            date = parse_date_from_filename(p.name)
            if date:
                files_with_dates.append((p, date))
            else:
                print(f"Warning: Skipping {p.name} (could not parse date from filename)")
                
    # Sort chronologically by date
    files_with_dates.sort(key=lambda x: x[1])
    return files_with_dates



def plot_geojson(ax, geojson_path: Path, color: str = '#ffffff', linewidth: float = 1.0, alpha: float = 0.5):
    """
    Parse and plot features from a GeoJSON file onto a matplotlib axis.
    Does not require geopandas or shapely.
    """
    if not geojson_path.exists():
        return
        
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for feature in data.get('features', []):
            geom = feature.get('geometry', {})
            geom_type = geom.get('type')
            coords = geom.get('coordinates', [])
            
            if geom_type == 'Polygon':
                for ring in coords:
                    x, y = zip(*ring)
                    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)
            elif geom_type == 'MultiPolygon':
                for polygon in coords:
                    for ring in polygon:
                        x, y = zip(*ring)
                        ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)
    except Exception as e:
        print(f"Warning: Could not plot GeoJSON boundary: {e}")


def generate_frame(
    file_path: Path,
    date: datetime,
    vmin: float,
    vmax: float,
    colormap: str,
    geojson_path: Optional[Path] = None
) -> np.ndarray:
    """
    Render a single map frame using Matplotlib and return it as an RGB numpy array.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        
        # Mask nodata and values outside physical limits [0, 1]
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        data = np.where((data < 0.0) | (data > 1.0), np.nan, data)
        
        # Get bounding box of the raster in CRS coords
        left, bottom, right, top = src.bounds
        extent = [left, right, bottom, top]

    # Create dark-themed visualization
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150, facecolor='#121212')
    ax.set_facecolor('#1c1c1c')
    
    # Hide axis ticks but leave room for labels/layout
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Render the soil moisture raster
    # Use standard interpolation for smooth visual transition
    im = ax.imshow(
        data,
        extent=extent,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        origin='upper',
        interpolation='nearest'
    )
    
    # Plot boundary of Namibia if available
    if geojson_path:
        plot_geojson(ax, geojson_path, color='#ffffff', linewidth=1.2, alpha=0.6)
        
    # Standard geographic grid lines (faint)
    ax.grid(True, color='#ffffff', alpha=0.08, linestyle='--')
    
    # Set coordinates extent slightly padded
    pad_x = (right - left) * 0.05
    pad_y = (top - bottom) * 0.05
    ax.set_xlim(left - pad_x, right + pad_x)
    ax.set_ylim(bottom - pad_y, top + pad_y)
    
    # Add title and text overlays
    ax.text(
        0.02, 0.95,
        "NASA SMAP Surface Soil Moisture",
        transform=ax.transAxes,
        color='#e0e0e0',
        fontsize=14,
        fontweight='bold',
        va='top'
    )
    
    ax.text(
        0.02, 0.90,
        "Namibia Region",
        transform=ax.transAxes,
        color='#888888',
        fontsize=10,
        va='top'
    )
    
    formatted_date = date.strftime('%B %d, %Y')
    ax.text(
        0.02, 0.05,
        formatted_date,
        transform=ax.transAxes,
        color='#ffffff',
        fontsize=18,
        fontweight='bold',
        va='bottom'
    )
    
    # Add Colorbar (horizontal or vertical, let's do a thin right-aligned vertical bar)
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.04, aspect=25)
    cbar.set_label(
        'Volumetric Soil Moisture ($m^3/m^3$)',
        color='#e0e0e0',
        fontsize=9,
        labelpad=10
    )
    cbar.ax.yaxis.set_tick_params(color='#e0e0e0', labelcolor='#e0e0e0', labelsize=8)
    cbar.outline.set_edgecolor('#333333')
    
    plt.tight_layout()
    
    # Convert matplotlib figure to RGB numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    rgb_frame = np.array(img.convert('RGB'))
    
    plt.close(fig)
    return rgb_frame


def main():
    parser = argparse.ArgumentParser(
        description="Compile soil moisture daily GeoTIFFs into a video history."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing the daily .tif files"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video filename (default: input_dir/soil_moisture_history.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second. 1 means 1 second per day/frame (default: 1)"
    )
    parser.add_argument(
        "--colormap",
        default="YlGnBu",
        help="Matplotlib colormap to use (default: YlGnBu)"
    )
    parser.add_argument(
        "--geojson",
        default="namibia.geojson",
        help="Path to region boundary GeoJSON (default: namibia.geojson)"
    )
    parser.add_argument(
        "--max-moisture",
        type=float,
        default=0.4,
        help="Maximum soil moisture value for the colormap scale (default: 0.4)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
        
    if args.output is None:
        output_path = input_path / "soil_moisture_history.mp4"
    else:
        output_path = Path(args.output)
        
    geojson_path = Path(args.geojson)
    if not geojson_path.exists():
        print(f"Warning: GeoJSON boundary '{args.geojson}' not found. Map will be drawn without boundary outline.")
        geojson_path = None
        
    # Get sorted files
    files_with_dates = get_sorted_tiff_files(input_path)
    if not files_with_dates:
        print(f"Error: No daily GeoTIFF files with parseable dates found in '{args.input_dir}'.")
        sys.exit(1)
        
    print(f"Found {len(files_with_dates)} daily observations.")
    
    # Set limits: min is always 0.0, max is user-defined
    vmin = 0.0
    vmax = args.max_moisture
    
    # Generate frames
    frames = []
    print("Generating frames...")
    for idx, (file_path, date) in enumerate(files_with_dates, 1):
        # Skip the output video file itself if it happens to end in .tif (unlikely)
        if file_path.suffix.lower() not in {'.tif', '.tiff'}:
            continue
        print(f"[{idx}/{len(files_with_dates)}] Processing {file_path.name} ({date.strftime('%Y-%m-%d')})...")
        try:
            frame = generate_frame(
                file_path=file_path,
                date=date,
                vmin=vmin,
                vmax=vmax,
                colormap=args.colormap,
                geojson_path=geojson_path
            )
            frames.append(frame)
        except Exception as e:
            print(f"Error generating frame for {file_path.name}: {e}")
            
    if not frames:
        print("Error: No frames could be generated.")
        sys.exit(1)
        
    # Save as video
    print(f"Compiling video to {output_path} (fps={args.fps})...")
    try:
        # Save output using imageio
        imageio.mimsave(str(output_path), frames, fps=args.fps)
        print(f"Success! Video saved to {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
