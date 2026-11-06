# Data sources and licences

Raw data are not stored in this repository. `run_all.sh` downloads them from the original providers.

| Source | Used for | Licence or terms | In this repository? |
|---|---|---|---|
| [GBIF](https://www.gbif.org) occurrences | Species records, 1970 onward | Per record: CC0, CC BY or CC BY-NC | No raw records |
| [IUCN Red List via GBIF](https://www.gbif.org/dataset/19491596-35ae-4a91-9a98-85cf505f1bd3) | Global threat status and assessment year | Listed on GBIF as CC BY 4.0; check IUCN terms before reuse | Status codes only |
| [WorldClim 2.1](https://www.worldclim.org) | Baseline climate, CMIP6 futures, elevation | Academic and non-commercial use; check current terms; no redistribution | No |
| [Wikidata](https://www.wikidata.org) | Medicinal taxa and compound-to-plant links curated by LOTUS | CC0 | Summary outputs only |
| [OpenStreetMap](https://www.openstreetmap.org/copyright) | Dashboard base map | ODbL, with attribution | Tiles load at runtime |

## Non-commercial notice

Derived outputs depend on CC BY-NC occurrence records and WorldClim. They are for non-commercial research use only.
Commercial use would need the analysis rebuilt from suitably licensed data, or permission from the providers.

An open-licence-only filter was tested first, but it removed most records of threatened Himalayan species
(for example, 1 of 102 records of *Aconitum heterophyllum* remained). Each record's licence is stored so the analysis can be repeated.

## Traditional knowledge

Formula membership comes only from published literature, cited in `data/curated/formulas.csv`.
No community-held or unpublished knowledge is used. Published use does not remove community rights or interests,
and field use should be guided by local knowledge holders, following the
[CARE Principles for Indigenous Data Governance](https://doi.org/10.5334/dsj-2020-043).

## Sensitive locations

Exact locations of threatened plants can help illegal collection. Public outputs therefore:

- stop at 0.5-degree blocks
- contain no occurrence coordinates
- give threatened species (CR, EN, VU, EW) no individual map or location list

## Key references

- Fick SE, Hijmans RJ (2017). WorldClim 2. *International Journal of Climatology* 37:4302-4315.
- Rutz A et al. (2022). The LOTUS initiative for open knowledge management in natural products research. *eLife* 11:e70780.
- Roberts DR et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40:913-929.
- Phillips SJ et al. (2009). Sample selection bias and presence-only distribution models. *Ecological Applications* 19:181-197.
- Carroll SR et al. (2020). The CARE Principles for Indigenous Data Governance. *Data Science Journal* 19:43.
