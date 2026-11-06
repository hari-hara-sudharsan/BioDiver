# BioDiver

A climate stress test for South Asian medicinal plants, and for the traditional formulas that depend on them.
Built by Team NothingButCode (Hari Hara Sudharsan J, Vendhan E T) for EthnoHack 2026, Track 3: Biodiversity and Sustainability.

**Live dashboard:** https://hari-hara-sudharsan.github.io/BioDiver/

> Research prototype. These are model projections, not forecasts, medical advice or Red List assessments.
> Nothing here says a plant is safe, or that one plant can replace another.

## What it does

- Matches medicinal plants from Wikidata to GBIF, and adds global threat status from the IUCN archive.
- Models each species' climate niche under spatial block cross-validation, and projects it to 2041-2060 across 5 climate models and 2 emissions scenarios.
- Grades every model (standard, low, no skill, data gap), so weak results are shown as weak.
- Tests whole formulas: the sourcing footprint is the area where all ingredients stay suitable together.
- Checks predictions with a temporal hindcast against a distance-only baseline.
- Suggests which unassessed plants experts might review first. Predictions are released only if the model passes a gate fixed in advance.

## The dashboard

`docs/` is a static site with no build step. To run it locally:

```bash
cd docs
python -m http.server 8765
```

Then open http://127.0.0.1:8765/. Opening `index.html` directly will not work, because the page fetches `data/app.json`.

If `summary.preview` is true in `docs/data/app.json`, the page shows a banner: those figures come from a partial run.

## Rebuild from source

```bash
pip install -r requirements.txt
bash run_all.sh
python -m pytest tests
```

Expect several hours, because GBIF is rate limited and every species is modelled separately. Raw data are not stored in this repository.

## Safeguards

- Public layers stop at 0.5-degree blocks, and no occurrence coordinates are published.
- Threatened species (CR, EN, VU, EW) have no public map, no location list and no share in any map layer.
- Formula membership comes only from cited literature. Ingredients without a source are marked as such.

See [DATA_SOURCES.md](DATA_SOURCES.md) for sources, licences, traditional knowledge and sensitive-location policy.
Derived outputs depend on CC BY-NC records and WorldClim, so they are for non-commercial research use only.

## Known limits

Climate suitability is not presence, abundance or supply. Soil, land use, harvest pressure, dispersal and biotic interactions are not modelled.
Occurrence data reflect where people look. Hindi and Tamil text in the grower's guide is a draft awaiting native-speaker review.
