# Diagnostics

The `xcs_code_stockpile.sanity` module provides lightweight checks for beam and detector behavior. These functions do not apply shot filtering; subset the metadata before calling them when appropriate.

```python
from xcs_code_stockpile.sanity import (
    intensity_vs_i0,
    plot_i0_vs_time,
    plot_intensity_vs_i0,
    plot_intensity_vs_time,
)

filtered = metadata.loc[shots].copy()
filtered.attrs.update(metadata.attrs)

fit_data = intensity_vs_i0(filtered)
figure = plot_intensity_vs_i0(filtered)
figure.savefig("intensity-vs-i0.png")
```

`intensity_vs_i0` uses only pump-off rows and stores fit results in DataFrame attributes:

- `beta`: fitted slope;
- `intercept`: fitted intercept;
- `R^2`: coefficient of determination.

Plotting functions return a Matplotlib `Figure`. The caller is responsible for displaying, saving, and closing it.
