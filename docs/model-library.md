# Model library

`orchestrator/library` (module `clusterlm_library`) records the models the Host knows about.

- Ids are `mdl_<24 hex>`, derived from a structure digest: file names and sizes plus the GGUF tensor directory. Only the GGUF
  header is read; tensor data is never touched.
- `scan_directory` identifies models; `ModelLibrary::rescan` merges results. Identity resolution never guesses: ambiguity is an
  error, and an unpinned match needs user confirmation.
- `pinned_root` is a user confirmation. `root_hash` is set only by verification (`set_verified_root`), never by scanning.
- Every tensor type in a model must be executable by the chosen backend (`check_model` in `runtime/domain`), otherwise the model
  is not distributable and `max_workers` is 0.
- Library documents that fail validation are quarantined by settings, not deleted.
