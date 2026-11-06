"""Download WorldClim 2.1 (5 arc-min), crop to the extent, delete the originals.

The drive is nearly full, so only one global file exists at a time.
"""
import zipfile

import rasterio
import requests
from rasterio.windows import from_bounds

from config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, PROCESSED, RAW, USER_AGENT

BASE = "https://geodata.ucdavis.edu"
GCMS = ["ACCESS-CM2", "MPI-ESM1-2-HR", "MIROC6", "IPSL-CM6A-LR", "EC-Earth3-Veg"]
SSPS = ["ssp245", "ssp585"]
PERIOD = "2041-2060"


def fetch(url, dest):
    with requests.get(url, stream=True, timeout=300,
                      headers={"User-Agent": USER_AGENT}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def crop(sources, out):
    """Write the cropped bands of one or more rasters to a single GeoTIFF."""
    bands, profile = [], None
    for src_path in sources:
        with rasterio.open(src_path) as src:
            win = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, src.transform)
            win = win.round_offsets().round_lengths()
            data = src.read(window=win).astype("float32")
            if src.nodata is not None:
                data[data == src.nodata] = -9999
            data[data < -1e30] = -9999
            bands.extend(data)
            if profile is None:
                profile = src.profile
                profile.update(height=data.shape[1], width=data.shape[2],
                               transform=src.window_transform(win), dtype="float32",
                               nodata=-9999, compress="deflate", predictor=2,
                               tiled=False)
    profile["count"] = len(bands)
    with rasterio.open(out, "w", **profile) as dst:
        for i, band in enumerate(bands, 1):
            dst.write(band, i)


def current():
    out = PROCESSED / "climate_current.tif"
    if out.exists():
        return
    zip_path = RAW / "bio.zip"
    fetch(f"{BASE}/climate/worldclim/2_1/base/wc2.1_5m_bio.zip", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW / "bio")
    zip_path.unlink()
    files = [RAW / "bio" / f"wc2.1_5m_bio_{i}.tif" for i in range(1, 20)]
    crop(files, out)
    for f in (RAW / "bio").iterdir():
        f.unlink()
    (RAW / "bio").rmdir()
    print("current climate done", flush=True)


def elevation():
    out = PROCESSED / "elevation.tif"
    if out.exists():
        return
    zip_path = RAW / "elev.zip"
    fetch(f"{BASE}/climate/worldclim/2_1/base/wc2.1_5m_elev.zip", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW / "elev")
    zip_path.unlink()
    tif = RAW / "elev" / "wc2.1_5m_elev.tif"
    crop([tif], out)
    tif.unlink()
    (RAW / "elev").rmdir()
    print("elevation done", flush=True)


def future():
    for ssp in SSPS:
        for gcm in GCMS:
            out = PROCESSED / f"climate_{gcm}_{ssp}_{PERIOD}.tif"
            if out.exists():
                continue
            name = f"wc2.1_5m_bioc_{gcm}_{ssp}_{PERIOD}.tif"
            tmp = RAW / name
            try:
                fetch(f"{BASE}/cmip6/5m/{gcm}/{ssp}/{name}", tmp)
                crop([tmp], out)
                print("done", gcm, ssp, flush=True)
            except requests.HTTPError as e:
                print("FAILED", gcm, ssp, e, flush=True)
            finally:
                tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    current()
    elevation()
    future()
