"""
Download completed SPL4SMGP AppEEARS results.

The script downloads the AppEEARS statistics CSV, uses it to identify
the latest surface-soil-moisture observation for each UTC calendar day,
and downloads only those GeoTIFF files.

Usage:
    python download_appeears.py OUTPUT_DIRECTORY [SUBMISSION_FILE]

Example:
    python download_appeears.py \
        ./soil-moisture

Required .env variables:
    EARTHDATA_USERNAME
    EARTHDATA_PASSWORD
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv

import warnings
# Suppress deprecation warnings from rasterio/numpy 2.5 shape assignment
warnings.filterwarnings("ignore", category=DeprecationWarning)
import io
import numpy as np
import matplotlib
# Use non-interactive Agg backend for script execution
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from PIL import Image


API_URL = "https://appeears.earthdatacloud.nasa.gov/api"
REQUEST_TIMEOUT = 60
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

DEFAULT_PRODUCT = "SPL4SMGP.008"


@dataclass(frozen=True)
class StatisticsRecord:
    """One raster observation listed in the AppEEARS statistics CSV."""

    source_stem: str
    dataset: str
    observed_at: datetime

    @property
    def day(self):
        return self.observed_at.date()

    @property
    def layer_name(self) -> str:
        """
        Convert a statistics dataset name such as

            Geophysical_Data_sm_surface

        to:

            sm_surface
        """
        prefix = "Geophysical_Data_"

        if self.dataset.startswith(prefix):
            return self.dataset[len(prefix):]

        return self.dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the latest daily SPL4SMGP observation from a "
            "completed AppEEARS task."
        )
    )

    parser.add_argument(
        "output_directory",
        type=Path,
        help="Directory in which downloaded files will be stored.",
    )

    parser.add_argument(
        "submission_file",
        nargs="?",
        type=Path,
        default=Path("appeears_submission.json"),
        help=(
            "JSON metadata file created by submit_appeears.py."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )

    parser.add_argument(
        "--method",
        choices=["bundle", "csv"],
        default="bundle",
        help=(
            "Method to determine the daily files: 'bundle' (parse "
            "filenames directly) or 'csv' (use statistics CSV with "
            "filename translation). Default is 'bundle'."
        ),
    )

    parser.add_argument(
        "--no-jpegs",
        action="store_true",
        help="Disable generating JPEG visualizations alongside the GeoTIFFs.",
    )

    parser.add_argument(
        "--colormap",
        default="YlGnBu",
        help="Matplotlib colormap to use for JPEGs (default: YlGnBu)",
    )

    parser.add_argument(
        "--geojson",
        default="namibia.geojson",
        help="Path to region boundary GeoJSON for JPEGs (default: namibia.geojson)",
    )

    parser.add_argument(
        "--max-moisture",
        type=float,
        default=0.4,
        help="Maximum soil moisture value for the colormap scale (default: 0.4)",
    )

    return parser.parse_args()


def translate_csv_filename_to_bundle_stem(
    csv_stem: str,
    product: str,
) -> str:
    """
    Translate a statistics CSV file stem (DOY format with underscore)
    to a bundle GeoTIFF file stem (ISO datetime format with dot).
    """
    match = re.search(r"_doy(\d{4})(\d{3})(\d{6})_", csv_stem)
    if not match:
        return csv_stem

    year = int(match.group(1))
    doy = int(match.group(2))
    time_str = match.group(3)

    try:
        date_obj = datetime.strptime(f"{year}-{doy:03d}", "%Y-%j")
        iso_date_str = date_obj.strftime("%Y%m%d")
        iso_timestamp = f"{iso_date_str}T{time_str}"
    except ValueError:
        return csv_stem

    start_idx = match.start()
    end_idx = match.end()
    translated = csv_stem[:start_idx] + f"_{iso_timestamp}_" + csv_stem[end_idx:]

    product_underscore = product.replace(".", "_")
    if translated.startswith(product_underscore):
        translated = product + translated[len(product_underscore):]

    return translated


def extract_records_from_bundle(
    bundle_files: list[dict[str, Any]],
    product: str,
) -> list[StatisticsRecord]:
    """Parse TIFF filenames in the bundle directly to extract observations."""
    records: list[StatisticsRecord] = []

    pattern = re.compile(
        rf"^{re.escape(product)}_(.*)_(\d{{8}}T\d{{6}})_([A-Za-z0-9_.-]+)\.tiff?$"
    )

    for bundle_file in bundle_files:
        filename = str(bundle_file.get("file_name", ""))
        path = Path(filename)

        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue

        match = pattern.match(path.name)
        if not match:
            continue

        layer_dataset = match.group(1)
        timestamp_str = match.group(2)

        try:
            observed_at = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
        except ValueError:
            continue

        records.append(
            StatisticsRecord(
                source_stem=path.stem,
                dataset=layer_dataset,
                observed_at=observed_at,
            )
        )

    return records


def authenticate(
    session: requests.Session,

    username: str,
    password: str,
) -> str:
    """Authenticate with AppEEARS and return a bearer token."""
    response = session.post(
        f"{API_URL}/login",
        auth=(username, password),
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    result = response.json()
    token = result.get("token")

    if not token:
        raise RuntimeError(
            "AppEEARS authentication succeeded but returned no token."
        )

    return token


def load_submission(path: Path) -> dict[str, Any]:
    """Load the submission metadata produced by the request script."""
    if not path.exists():
        raise FileNotFoundError(
            f"Submission file not found: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        submission = json.load(file)

    if not submission.get("task_id"):
        raise ValueError(
            f"No task_id was found in {path.resolve()}."
        )

    return submission


def get_task(
    session: requests.Session,
    token: str,
    task_id: str,
) -> dict[str, Any]:
    """Retrieve current AppEEARS task information."""
    response = session.get(
        f"{API_URL}/task/{task_id}",
        headers={
            "Authorization": f"Bearer {token}",
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()
    return response.json()


def get_bundle(
    session: requests.Session,
    token: str,
    task_id: str,
) -> dict[str, Any]:
    """Retrieve the list of files generated for a completed task."""
    response = session.get(
        f"{API_URL}/bundle/{task_id}",
        headers={
            "Authorization": f"Bearer {token}",
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()
    return response.json()


def find_statistics_file(
    bundle_files: list[dict[str, Any]],
) -> dict[str, Any]:
    """Find the AppEEARS statistics CSV in the task bundle."""
    candidates = []

    for bundle_file in bundle_files:
        filename = str(bundle_file.get("file_name", ""))
        filename_lower = filename.lower()

        if (
            filename_lower.endswith(".csv")
            and "statistic" in filename_lower
        ):
            candidates.append(bundle_file)

    if not candidates:
        available_csvs = [
            str(item.get("file_name"))
            for item in bundle_files
            if str(item.get("file_name", "")).lower().endswith(".csv")
        ]

        csv_description = (
            "\n".join(f"  - {name}" for name in available_csvs)
            if available_csvs
            else "  No CSV files were found."
        )

        raise RuntimeError(
            "Could not find the AppEEARS statistics CSV.\n"
            f"Available CSV files:\n{csv_description}"
        )

    if len(candidates) > 1:
        names = "\n".join(
            f"  - {item['file_name']}"
            for item in candidates
        )

        raise RuntimeError(
            "More than one possible statistics CSV was found:\n"
            f"{names}"
        )

    return candidates[0]


def download_bundle_file(
    session: requests.Session,
    token: str,
    task_id: str,
    bundle_file: dict[str, Any],
    destination: Path,
    overwrite: bool,
) -> None:
    """Download one bundle file to the requested destination."""
    if destination.exists() and not overwrite:
        print(f"Already exists, skipping: {destination.name}")
        return

    file_id = bundle_file.get("file_id")

    if not file_id:
        raise RuntimeError(
            f"Bundle file has no file_id: {bundle_file}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = destination.with_name(
        f"{destination.name}.part"
    )

    print(f"Downloading: {destination.name}")

    with session.get(
        f"{API_URL}/bundle/{task_id}/{file_id}",
        headers={
            "Authorization": f"Bearer {token}",
        },
        allow_redirects=True,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()

        with temporary_path.open("wb") as file:
            for chunk in response.iter_content(
                chunk_size=DOWNLOAD_CHUNK_SIZE
            ):
                if chunk:
                    file.write(chunk)

    expected_size = bundle_file.get("file_size")

    if (
        expected_size is not None
        and temporary_path.stat().st_size != int(expected_size)
    ):
        actual_size = temporary_path.stat().st_size
        temporary_path.unlink(missing_ok=True)

        raise RuntimeError(
            f"Downloaded size mismatch for {destination.name}: "
            f"expected {expected_size} bytes, received "
            f"{actual_size} bytes."
        )

    temporary_path.replace(destination)


def parse_utc_datetime(value: str) -> datetime:
    """
    Parse an AppEEARS statistics timestamp.

    Example:
        2026-03-15 22:30:00 UTC
    """
    value = value.strip()

    supported_formats = (
        "%Y-%m-%d %H:%M:%S UTC",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
    )

    for date_format in supported_formats:
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            continue

    raise ValueError(
        f"Unsupported AppEEARS Date value: {value!r}"
    )


def read_statistics(
    statistics_path: Path,
    product: str,
) -> list[StatisticsRecord]:
    """Read raster filenames and observation times from the CSV."""
    records: list[StatisticsRecord] = []

    with statistics_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "File Name",
            "Dataset",
            "Date",
        }

        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                "Statistics CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            source_stem = (row.get("File Name") or "").strip()
            dataset = (row.get("Dataset") or "").strip()
            date_value = (row.get("Date") or "").strip()

            if not source_stem or not dataset or not date_value:
                print(
                    f"Warning: skipping incomplete statistics row "
                    f"{row_number}.",
                    file=sys.stderr,
                )
                continue

            translated_stem = translate_csv_filename_to_bundle_stem(
                Path(source_stem).stem,
                product,
            )

            records.append(
                StatisticsRecord(
                    source_stem=translated_stem,
                    dataset=dataset,
                    observed_at=parse_utc_datetime(date_value),
                )
            )

    if not records:
        raise RuntimeError(
            "The statistics CSV contains no usable observations."
        )

    return records



def select_latest_per_day(
    records: list[StatisticsRecord],
) -> list[StatisticsRecord]:
    """Select the chronologically latest observation for each UTC day."""
    selected: dict[Any, StatisticsRecord] = {}

    for record in records:
        existing = selected.get(record.day)

        if (
            existing is None
            or record.observed_at > existing.observed_at
        ):
            selected[record.day] = record

    return sorted(
        selected.values(),
        key=lambda record: record.observed_at,
    )


def find_matching_tiff(
    bundle_files: list[dict[str, Any]],
    source_stem: str,
) -> dict[str, Any]:
    """Match a statistics CSV record to its bundle GeoTIFF."""
    exact_matches = []

    for bundle_file in bundle_files:
        filename = str(bundle_file.get("file_name", ""))
        path = Path(filename)

        if (
            path.suffix.lower() in {".tif", ".tiff"}
            and path.stem == source_stem
        ):
            exact_matches.append(bundle_file)

    if len(exact_matches) == 1:
        return exact_matches[0]

    if len(exact_matches) > 1:
        raise RuntimeError(
            f"More than one TIFF matched {source_stem!r}."
        )

    # Some AppEEARS output names may have minor suffix differences.
    partial_matches = []

    for bundle_file in bundle_files:
        filename = str(bundle_file.get("file_name", ""))
        path = Path(filename)

        if (
            path.suffix.lower() in {".tif", ".tiff"}
            and (
                path.stem.startswith(source_stem)
                or source_stem.startswith(path.stem)
            )
        ):
            partial_matches.append(bundle_file)

    if len(partial_matches) == 1:
        return partial_matches[0]

    raise RuntimeError(
        "Could not match the statistics record to a TIFF:\n"
        f"  {source_stem}"
    )


def clean_filename_part(value: str) -> str:
    """Remove characters that are unsafe or awkward in filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned.strip("._")


def make_output_tiff_name(
    product: str,
    record: StatisticsRecord,
) -> str:
    """
    Create a filename such as:

        SPL4SMGP.008_sm_surface_20260315.tif
    """
    product_part = clean_filename_part(product)
    layer_part = clean_filename_part(record.layer_name)
    date_part = record.observed_at.strftime("%Y%m%d")

    return (
        f"{product_part}_{layer_part}_{date_part}.tif"
    )


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


def save_jpeg_frame(
    file_path: Path,
    output_path: Path,
    date: datetime,
    vmin: float,
    vmax: float,
    colormap: str,
    geojson_path: Optional[Path] = None
) -> None:
    """
    Render a single map frame using Matplotlib and save it as a JPEG image.
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
    
    # Plot boundary if available
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
    
    # Save directly to output path as JPEG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='jpeg', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


def main() -> int:
    args = parse_arguments()
    load_dotenv()

    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")

    if not username or not password:
        print(
            "EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be defined "
            "in the .env file.",
            file=sys.stderr,
        )
        return 1

    try:
        submission = load_submission(args.submission_file)

        task_id = str(submission["task_id"])
        product = str(
            submission.get("product", DEFAULT_PRODUCT)
        )

        output_directory = args.output_directory.resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        with requests.Session() as session:
            print("Authenticating with AppEEARS...")

            token = authenticate(
                session,
                username,
                password,
            )

            print(f"Checking task: {task_id}")

            task = get_task(
                session,
                token,
                task_id,
            )

            status = task.get("status")

            if status != "done":
                if status == "error":
                    raise RuntimeError(
                        "The AppEEARS task ended with an error."
                    )

                print(
                    f"Task is not ready. Current status: {status}"
                )
                return 2

            print("Retrieving task bundle...")

            bundle = get_bundle(
                session,
                token,
                task_id,
            )

            bundle_files = bundle.get("files", [])

            if not bundle_files:
                raise RuntimeError(
                    "The completed task bundle contains no files."
                )

            # Try to download statistics file for convenience
            try:
                statistics_file = find_statistics_file(bundle_files)
                statistics_destination = (
                    output_directory
                    / f"{product}_Statistics.csv"
                )
                download_bundle_file(
                    session=session,
                    token=token,
                    task_id=task_id,
                    bundle_file=statistics_file,
                    destination=statistics_destination,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                print(
                    f"Warning: Could not download statistics CSV: {exc}",
                    file=sys.stderr,
                )
                statistics_destination = None

            if args.method == "bundle":
                print("Identifying daily files directly from bundle file names...")
                records = extract_records_from_bundle(bundle_files, product)
            else:
                if not statistics_destination or not statistics_destination.exists():
                    raise RuntimeError(
                        "Statistics CSV file is required for 'csv' method but "
                        "could not be downloaded."
                    )
                print("Identifying daily files using statistics CSV...")
                records = read_statistics(statistics_destination, product)

            selected_records = select_latest_per_day(records)

            print(
                f"\nFound {len(records)} observations covering "
                f"{len(selected_records)} day(s)."
            )


            print("Latest observation selected for each day:")

            for record in selected_records:
                print(
                    "  "
                    f"{record.observed_at:%Y-%m-%d %H:%M:%S} UTC"
                )

            print()

            # Determine if JPEGs should be generated and load GeoJSON boundary if needed
            if not args.no_jpegs:
                geojson_path = Path(args.geojson)
                if not geojson_path.exists():
                    print(f"Warning: GeoJSON boundary '{args.geojson}' not found. JPEGs will be drawn without boundary outline.")
                    geojson_path = None
            else:
                geojson_path = None

            for record in selected_records:
                tiff_file = find_matching_tiff(
                    bundle_files,
                    record.source_stem,
                )

                destination = (
                    output_directory
                    / make_output_tiff_name(product, record)
                )

                download_bundle_file(
                    session=session,
                    token=token,
                    task_id=task_id,
                    bundle_file=tiff_file,
                    destination=destination,
                    overwrite=args.overwrite,
                )

                if not args.no_jpegs:
                    jpeg_destination = output_directory / "images" / destination.with_suffix(".jpg").name
                    if not jpeg_destination.exists() or args.overwrite:
                        print(f"Generating JPEG: {jpeg_destination.name}")
                        try:
                            save_jpeg_frame(
                                file_path=destination,
                                output_path=jpeg_destination,
                                date=record.observed_at,
                                vmin=0.0,
                                vmax=args.max_moisture,
                                colormap=args.colormap,
                                geojson_path=geojson_path
                            )
                        except Exception as exc:
                            print(f"Warning: Could not generate JPEG for {destination.name}: {exc}", file=sys.stderr)
                    else:
                        print(f"JPEG already exists, skipping: {jpeg_destination.name}")

        print("\nDownload complete.")
        print(f"Output directory: {output_directory}")
        print(
            f"Daily TIFFs retained: {len(selected_records)}"
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        requests.RequestException,
        json.JSONDecodeError,
    ) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())