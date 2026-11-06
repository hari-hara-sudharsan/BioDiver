"""Shared settings for the BioDiver pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED, CACHE, OUTPUTS = (ROOT / "data/raw", ROOT / "data/processed",
                                  ROOT / "data/cache", ROOT / "outputs")
for _p in (RAW, PROCESSED, CACHE, OUTPUTS):
    _p.mkdir(parents=True, exist_ok=True)

# Modelling extent: South Asia plus the wider Himalaya and Tibetan Plateau.
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 60, 100, 5, 38

MIN_OCCURRENCES = 30          # raw-record screen before cleaning
USER_AGENT = "BioDiver-EthnoHack/0.1 (research prototype)"
GBIF = "https://api.gbif.org/v1"
