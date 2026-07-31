"""
Submit an AppEEARS area request for SPL4SMGP surface soil moisture.

Usage:
    python submit_appeears.py 2026-01-10

    python submit_appeears.py 2026-01-10 2026-01-20

The first form requests a single day. The second requests an inclusive
date range.

AppEEARS will return all three-hourly observations in the requested period.
A separate processing script will later retain only the final observation
from each day.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


API_URL = "https://appeears.earthdatacloud.nasa.gov/api"

PRODUCT = "SPL4SMGP.008"
LAYER = "Geophysical_Data_sm_surface"

DEFAULT_POLYGON_FILE = Path("namibia.geojson")
DEFAULT_OUTPUT_FILE = Path("appeears_submission.json")

REQUEST_TIMEOUT = 60


def parse_date(value: str) -> date:
    """Parse a command-line date in YYYY-MM-DD format."""
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}. Use YYYY-MM-DD."
        ) from exc

    if parsed > datetime.now(timezone.utc).date():
        raise argparse.ArgumentTypeError(
            f"Date {value!r} is in the future."
        )

    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit an AppEEARS request for SPL4SMGP surface "
            "soil moisture for a single date or date range."
        )
    )

    parser.add_argument(
        "start_date",
        type=parse_date,
        metavar="START_DATE",
        help="Start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "end_date",
        type=parse_date,
        nargs="?",
        metavar="END_DATE",
        help=(
            "Optional end date in YYYY-MM-DD format. "
            "Omit it to request a single day."
        ),
    )

    parser.add_argument(
        "--polygon",
        type=Path,
        default=DEFAULT_POLYGON_FILE,
        help=(
            "GeoJSON polygon file. "
            f"Default: {DEFAULT_POLYGON_FILE}"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=(
            "File in which to save the submission metadata. "
            f"Default: {DEFAULT_OUTPUT_FILE}"
        ),
    )

    args = parser.parse_args()

    # A missing end date means a one-day request.
    if args.end_date is None:
        args.end_date = args.start_date

    if args.end_date < args.start_date:
        parser.error("END_DATE cannot be earlier than START_DATE.")

    return args


def load_polygon(path: Path) -> dict[str, Any]:
    """Load GeoJSON and normalise it to a FeatureCollection."""
    if not path.exists():
        raise FileNotFoundError(
            f"Polygon file not found: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        geojson = json.load(file)

    geojson_type = geojson.get("type")

    if geojson_type == "FeatureCollection":
        feature_collection = geojson

    elif geojson_type == "Feature":
        feature_collection = {
            "type": "FeatureCollection",
            "features": [geojson],
        }

    elif geojson_type in {"Polygon", "MultiPolygon"}:
        feature_collection = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": geojson,
                }
            ],
        }

    else:
        raise ValueError(
            "GeoJSON must be a FeatureCollection, Feature, "
            "Polygon or MultiPolygon."
        )

    if not feature_collection.get("features"):
        raise ValueError("The GeoJSON contains no features.")

    feature_collection.setdefault("fileName", path.stem)

    return feature_collection


def authenticate(
    session: requests.Session,
    username: str,
    password: str,
) -> str:
    """Authenticate with AppEEARS and return the bearer token."""
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


def submit_task(
    session: requests.Session,
    token: str,
    polygon: dict[str, Any],
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Submit the AppEEARS area request."""
    if start_date == end_date:
        task_name = f"Namibia SPL4SMGP {start_date.isoformat()}"
    else:
        task_name = (
            f"Namibia SPL4SMGP "
            f"{start_date.isoformat()} to {end_date.isoformat()}"
        )

    task = {
        "task_type": "area",
        "task_name": task_name,
        "params": {
            "dates": [
                {
                    "startDate": start_date.strftime("%m-%d-%Y"),
                    "endDate": end_date.strftime("%m-%d-%Y"),
                }
            ],
            "layers": [
                {
                    "product": PRODUCT,
                    "layer": LAYER,
                }
            ],
            "geo": polygon,
            "output": {
                "format": {
                    "type": "geotiff",
                    "filename_date": "calendar",
                },
                "projection": "geographic",
            },
        },
    }

    response = session.post(
        f"{API_URL}/task",
        headers={
            "Authorization": f"Bearer {token}",
        },
        json=task,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 202:
        try:
            error_details = response.json()
        except ValueError:
            error_details = response.text

        raise RuntimeError(
            "AppEEARS rejected the task.\n"
            f"HTTP {response.status_code}: {error_details}"
        )

    return response.json()


def main() -> int:
    args = parse_arguments()

    load_dotenv()

    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")

    if not username or not password:
        print(
            "EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be defined in .env",
            file=sys.stderr,
        )
        return 1

    try:
        polygon = load_polygon(args.polygon)

        if args.start_date == args.end_date:
            print(f"Requested date: {args.start_date.isoformat()}")
        else:
            print(
                "Requested date range: "
                f"{args.start_date.isoformat()} to "
                f"{args.end_date.isoformat()}"
            )

        with requests.Session() as session:
            print("Authenticating with AppEEARS...")

            token = authenticate(
                session,
                username,
                password,
            )

            print("Submitting task...")

            result = submit_task(
                session=session,
                token=token,
                polygon=polygon,
                start_date=args.start_date,
                end_date=args.end_date,
            )

        task_id = result["task_id"]

        submission = {
            "task_id": task_id,
            "status": result.get("status"),
            "product": PRODUCT,
            "layer": LAYER,
            "start_date": args.start_date.isoformat(),
            "end_date": args.end_date.isoformat(),
            "selection_rule": "latest_observation_per_day",
            "projection": "geographic",
            "status_url": f"{API_URL}/status/{task_id}",
            "task_url": f"{API_URL}/task/{task_id}",
            "bundle_url": f"{API_URL}/bundle/{task_id}",
            "submitted_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open("w", encoding="utf-8") as file:
            json.dump(submission, file, indent=2)

        print("\nRequest submitted successfully.")
        print(f"Task ID: {task_id}")
        print(f"Status:  {result.get('status')}")
        print(f"Saved:   {args.output.resolve()}")

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