# Example profile set (Fast / Strong / Ultra)

The three original ClusterLM tiers as importable execution profiles (`*.profile.json`) plus the illustrative routing alias
`auto.alias.json`. They are what settings migration produces from the tier catalog and what a fresh install is seeded with
(`tests/migration` and `tests/config` prove the three sources agree).

* These are data, not measurements: models are unpinned (`expected_root_hash: null`), every performance goal is
  `pending_qualification`, and nothing here claims hardware qualification.
* Worker slots are bound by the old role names `node:laptop-class` and `node:designated-3060`; after import, bind them to
  your machines (or replace the selectors with resource requirements).
* Importing never enables API exposure or LAN visibility.
* Copies of `docs/interfaces/examples/profile-{fast,strong,ultra}.example.json` and `routing-alias.example.json`; a test
  fails if they drift.
