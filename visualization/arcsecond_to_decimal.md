# Claude Development Prompt: Arcsecond Tick Label Decimal Placement

You are working in the PyAutoLens / PyAutoArray plotting codebase.

## Goal

Add a boolean option to the plotting API that changes arcsecond tick labels from the current format:

```text
0.45"
-2.2"
3.1"
```

to:

```text
0."45
-2."2
3."1
```

This should be implemented inside the plotting source code, not by user scripts editing Matplotlib tick labels after a figure has been drawn.

## Requirements

- Add a clearly named boolean option, for example `arcsec_after_decimal`, defaulting to `False` so existing plots are unchanged.
- The option should be available from high-level plotting functions such as `autolens.plot.plot_array(...)`.
- When enabled, only labels that are already formatted as arcseconds should change.
- Preserve all existing tick locations, precision choices, rotation, font size, and other style settings.
- Do not change colorbars or non-arcsecond axis labels.
- Add focused tests or examples covering:
  - default behavior remains `0.45"`;
  - enabled behavior becomes `0."45`;
  - negative values become `-2."2`;
  - integer-like labels without a decimal either remain unchanged or use a documented behavior.

## Suggested Implementation Direction

Find where axis tick labels are formatted in the AutoArray / PyAutoLens plotting stack. Start from:

- `autolens.plot.plot_array(...)`
- `autoarray.plot.plot_array(...)`
- the axis / tick helper functions used by `autoarray.plot.utils`
- any config-driven tick-label formatter that appends the arcsecond symbol

Implement this as close as possible to the tick-label formatter that appends `"`, rather than modifying labels after plotting. Ideally the formatter should receive the boolean and place the arcsecond marker either at the end or immediately after the decimal point.

## Acceptance Criteria

This example should work without post-processing:

```python
import autolens.plot as aplt

aplt.plot_array(
    array=array,
    title="VIS Data",
    arcsec_after_decimal=True,
)
```

and should produce axis labels like:

```text
-2."2   0."45   3."1
```

The same call without `arcsec_after_decimal=True` should preserve the current default labels.

## Notes

This request comes from paper-figure generation, where editing Matplotlib labels after `aplt.plot_array` works but creates fragile layout and whitespace side effects. The source-level option should keep normal PyAutoLens layout behavior intact.
