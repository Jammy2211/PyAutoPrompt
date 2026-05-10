# Extend scaling_relation examples with CSV loading via autoconf.csvable

## Background

The two scaling_relation examples shipped by `feature/scaling-relation-update` (issue
[autolens_workspace#141](https://github.com/PyAutoLabs/autolens_workspace/issues/141)) currently use
**hardcoded Python lists** for the extra-galaxy / scaling-galaxy luminosities:

```python
extra_galaxies_luminosity_list = [0.9, 0.9]
scaling_galaxies_luminosity_list = [0.45, 0.45]
```

The centres are loaded from JSON files written by the simulators
(`extra_galaxies_centres.json`, `scaling_galaxies_centres.json`), but the luminosities are not.
That is fine for a tutorial of fixed length, but production users with 20+ galaxies want a single
file describing the full population.

`autoconf` already provides a generic CSV reader/writer at `autoconf/csvable.py`:

- `output_to_csv(rows, file_path, headers=None)` — write list-of-dicts or list-of-sequences
- `list_from_csv(file_path)` — read as ordered list-of-dicts

There is no autolens/autogalaxy-specific schema layer yet — `autolens.point.dataset` does this for
its own format. We probably want a similar (very thin) schema layer for galaxy populations.

## Goal

Both `imaging/features/scaling_relation/modeling.py` and `group/features/scaling_relation/modeling.py`
should be able to drop a CSV like this in the dataset folder:

```csv
y,x,luminosity,redshift
3.5,2.5,0.9,0.5
-4.4,-5.0,0.9,0.5
```

…and load centres + luminosities (+ optional redshifts) in one call. The hardcoded lists in the
modeling scripts stay as a fallback path for the tutorial flow but the CSV path is shown alongside.

## Proposed work

1. **Decide where the schema layer lives.** Two options:
   - (a) Extend `autoconf/csvable.py` with a thin typed reader (e.g. `galaxy_table_from_csv`)
     returning `(centres: Grid2DIrregular, luminosities: list[float], redshifts: list[float] | None)`.
     Pro: every workspace gets it. Con: `autoconf` does not depend on `autoarray`, so the
     `Grid2DIrregular` return type would push it down a dependency edge.
   - (b) Keep the schema layer in `autolens` (or `autoarray`). Add e.g. `al.util.galaxy_table_from_csv`
     that wraps `autoconf.csvable.list_from_csv` and produces the typed outputs. Probably the right call.

2. **Extend both simulators** (`scripts/imaging/features/scaling_relation/...` borrows
   `dataset/group/simple`, so update `scripts/group/simulator.py`; the new
   `scripts/group/features/scaling_relation/simulator.py` writes its own `scaling_relation` dataset)
   to also write `extra_galaxies.csv` and/or `scaling_galaxies.csv` next to the centre JSONs.
   Keep the JSON files for backward compatibility — the CSV is additive.

3. **Update both modeling scripts** to load from the CSV instead of (or alongside) the JSON +
   hardcoded list. Show the CSV path as the primary modern flow; document the JSON-fallback inline
   for users on older datasets.

4. **Tests.** `test_autoconf/test_csvable.py` already covers the generic round-trip. Add tests for
   the new schema layer (in whichever repo owns it), exercising:
   - missing optional column (`redshift` absent → `None`)
   - extra columns (silently ignored)
   - row order preserved
   - empty file → empty population

## Out of scope

- Refactoring the `autolens.point.dataset` CSV layer to use this new helper. That should be a
  separate task — `point.dataset` has its own column conventions and we should not couple them.
- Migrating other workspace features to CSV. This task is scoped to the two scaling_relation
  examples plus their simulators.

## Reference reads

- `autoconf/csvable.py` — generic CSV I/O
- `autolens.point.dataset` — example of an existing CSV schema layer
- `autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py` — current consumer
  (post issue #141)
- `autolens_workspace/scripts/group/features/scaling_relation/{simulator,modeling}.py` — current
  consumer (post issue #141)
