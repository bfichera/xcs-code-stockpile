# Diagnostics

The public diagnostic routines provide lightweight checks for beam and detector behavior. Each accepts an optional boolean `filter` aligned with the metadata; omitting it uses every row. `intensity_vs_i0` and its plot additionally select pump-off rows.

```python
import xcs_code_stockpile as xcs

fit_data = xcs.intensity_vs_i0(metadata, filter=shots)
figure = xcs.plot_intensity_vs_i0(metadata, filter=shots)
figure.savefig("intensity-vs-i0.png")
```

`intensity_vs_i0` uses only pump-off rows and stores fit results in DataFrame attributes:

- `beta`: fitted slope;
- `intercept`: fitted intercept;
- `R^2`: coefficient of determination.

Plotting functions return a Matplotlib `Figure`. The caller is responsible for displaying, saving, and closing it.
