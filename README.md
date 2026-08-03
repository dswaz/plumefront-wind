# plumefront-wind

Hourly NOAA HRRR wind data for [plumefront.com](https://plumefront.com)'s wind
visualization layer — decoupled from the main app's repos/deploy pipeline on
purpose, so it can update independently on its own schedule.

A scheduled GitHub Action (`.github/workflows/fetch-wind.yml`) runs
`fetch_hrrr_wind.py` every hour, which pulls the latest 10m wind U/V bands
from NOAA NOMADS (byte-range fetch, not the full ~700MB GRIB file),
reprojects to a regular lat/lon grid, and commits two files:

- `wind.png` — RGBA texture: R=U wind component, G=V wind component, normalized 0-255
- `wind_meta.json` — bounding box, U/V min/max (for decoding), run time

The app fetches these directly from `raw.githubusercontent.com` (which sends
permissive CORS headers), so no hosting/Pages setup is needed here.

Public repo → unlimited free GitHub Actions minutes.
