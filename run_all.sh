#!/usr/bin/env bash
# Rebuild BioDiver from public sources. Expect several hours: GBIF is rate limited.
# Run from the repository root after: pip install -r requirements.txt
set -euo pipefail
PY=${PYTHON:-python}

$PY src/01_species.py          # Wikidata medicinal taxa matched to GBIF
$PY src/01b_priority.py        # add curated priority species
$PY src/01c_redlist.py         # Red List status from the IUCN archive
$PY src/02_climate.py          # WorldClim baseline and CMIP6 futures
$PY src/03_occurrences.py      # GBIF records; rerun to retry any failures
$PY src/03b_global_counts.py   # global record counts
$PY src/04_sdm.py              # models, validation and projections
$PY src/05_chemistry.py        # compound links from Wikidata
$PY src/06_stress_test.py      # formula, chemistry and relatives analyses
$PY src/07_hindcast.py         # temporal hindcast
$PY src/08_triage.py           # triage with release gate
$PY src/09_export.py           # public dashboard data
