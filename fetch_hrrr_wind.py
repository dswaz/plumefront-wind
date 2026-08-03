"""
fetch_hrrr_wind.py

Fetches the latest HRRR 10m wind U/V components from NOAA NOMADS via
byte-range requests (a few MB, not the ~700MB full GRIB2 file), reprojects
from the native Lambert Conformal Conic grid to a regular lat/lon grid, and
encodes the result into a single RGBA PNG texture + a metadata JSON file.

Output (overwritten each run):
  wind.png       - RGBA PNG: R=U (east-west wind), G=V (north-south wind)
  wind_meta.json - bounds, run time, U/V min/max for decoding on the client

Usage:
    python fetch_hrrr_wind.py
    python fetch_hrrr_wind.py --run 12          # force a specific run hour (UTC)
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.interpolate import griddata as scipy_griddata

NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"
USER_AGENT = "plumefront-wind-fetcher (github.com/dswaz/plumefront-wind)"

OUT_NX, OUT_NY = 1000, 600
OUT_BBOX = {"west": -134.0, "east": -61.0, "south": 21.0, "north": 53.0}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=".", help="Output directory")
    p.add_argument("--run", default=None, type=int, help="Override run hour (0-23 UTC)")
    p.add_argument("--forecast", default="00", help="Forecast hour: 00=analysis")
    p.add_argument("--date", default=None, help="Override date YYYYMMDD")
    return p.parse_args()


def latest_run_target(now_utc):
    # HRRR needs ~45min to process; back off one hour from "now" to find a
    # run that's actually finished and posted. Must roll the *date* back too
    # when this crosses midnight UTC, not just wrap the hour with `% 24` —
    # otherwise just after 00:00 UTC this asks for "today at 23z", which
    # doesn't exist yet (it's actually yesterday's 23z run).
    return now_utc - timedelta(hours=1)


def build_urls(date_str, run_hour, fhour):
    fname = f"hrrr.t{run_hour:02d}z.wrfsfcf{fhour}.grib2"
    base = f"{NOMADS_BASE}/hrrr.{date_str}/conus/{fname}"
    return base, base + ".idx"


def fetch_index(idx_url):
    r = requests.get(idx_url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    records = []
    for line in r.text.strip().splitlines():
        parts = line.split(":")
        if len(parts) < 5:
            continue
        records.append({"num": int(parts[0]), "offset": int(parts[1]), "var": parts[3], "level": parts[4]})
    return records


def find_wind_bands(records):
    ugrd = vgrd = None
    for rec in records:
        if rec["var"] == "UGRD" and "10 m above ground" in rec["level"]:
            ugrd = rec
        if rec["var"] == "VGRD" and "10 m above ground" in rec["level"]:
            vgrd = rec
        if ugrd and vgrd:
            break
    if not ugrd or not vgrd:
        raise RuntimeError("Could not find UGRD/VGRD 10m bands in .idx file")
    return ugrd, vgrd


def fetch_grib_band(grib_url, records, target_rec):
    idx = target_rec["num"] - 1
    start = target_rec["offset"]
    end = records[idx + 1]["offset"] - 1 if idx + 1 < len(records) else None
    range_header = f"bytes={start}-{end}" if end else f"bytes={start}-"
    r = requests.get(grib_url, headers={"User-Agent": USER_AGENT, "Range": range_header}, timeout=120)
    r.raise_for_status()
    return r.content


def parse_grib2_data(grib_bytes):
    """Return (data_2d, lats_2d, lons_2d) from raw GRIB2 bytes via cfgrib."""
    import cfgrib

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib_bytes)
        tmp_path = f.name
    try:
        ds = cfgrib.open_dataset(tmp_path)
        key = list(ds.data_vars)[0]
        data = ds[key].values.astype(np.float32)
        lats = ds.latitude.values.astype(np.float32)
        lons = ds.longitude.values.astype(np.float32)
        if lats.ndim == 1:
            lons, lats = np.meshgrid(lons, lats)
        lons = np.where(lons > 180, lons - 360, lons)
        return data, lats, lons
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def reproject_to_latlon(data, src_lats, src_lons):
    src_points = np.column_stack([src_lons.ravel(), src_lats.ravel()])
    tgt_lons = np.linspace(OUT_BBOX["west"], OUT_BBOX["east"], OUT_NX)
    tgt_lats = np.linspace(OUT_BBOX["south"], OUT_BBOX["north"], OUT_NY)
    tgt_lon_grid, tgt_lat_grid = np.meshgrid(tgt_lons, tgt_lats)
    result = scipy_griddata(src_points, data.ravel(), (tgt_lon_grid, tgt_lat_grid), method="linear")
    return np.nan_to_num(result, nan=0.0).astype(np.float32)


def encode_wind_png(u, v):
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())

    def norm(arr, lo, hi):
        if hi == lo:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)

    r = norm(u, u_min, u_max)
    g = norm(v, v_min, v_max)
    b = np.zeros_like(r)
    a = np.full_like(r, 255)
    rgba = np.stack([r, g, b, a], axis=-1)
    img = Image.fromarray(rgba, "RGBA")
    meta = {"u_min": u_min, "u_max": u_max, "v_min": v_min, "v_max": v_max, "width": img.width, "height": img.height}
    return img, meta


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    target = latest_run_target(now)
    date_str = args.date or target.strftime("%Y%m%d")
    run_hour = args.run if args.run is not None else target.hour

    grib_url, idx_url = build_urls(date_str, run_hour, args.forecast)
    print(f"HRRR run: {date_str} t{run_hour:02d}z f{args.forecast}", file=sys.stderr)
    records = fetch_index(idx_url)
    ugrd_rec, vgrd_rec = find_wind_bands(records)

    u_bytes = fetch_grib_band(grib_url, records, ugrd_rec)
    v_bytes = fetch_grib_band(grib_url, records, vgrd_rec)
    print(f"Fetched U ({len(u_bytes)//1024} KB) and V ({len(v_bytes)//1024} KB)", file=sys.stderr)

    u, u_lats, u_lons = parse_grib2_data(u_bytes)
    v, _, _ = parse_grib2_data(v_bytes)

    u_ll = reproject_to_latlon(u, u_lats, u_lons)
    v_ll = reproject_to_latlon(v, u_lats, u_lons)

    img, meta = encode_wind_png(u_ll, v_ll)
    meta.update({
        "run_date": date_str,
        "run_hour": run_hour,
        "forecast": args.forecast,
        "generated": now.isoformat(),
        "bbox": OUT_BBOX,
        "source": "NOAA HRRR 3km CONUS (reprojected to lat/lon)",
    })

    img.save(out_dir / "wind.png", "PNG")
    (out_dir / "wind_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved wind.png ({(out_dir / 'wind.png').stat().st_size // 1024} KB) + wind_meta.json", file=sys.stderr)


if __name__ == "__main__":
    main()
