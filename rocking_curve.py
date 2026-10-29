import pickle
from pathlib import Path

import xcs_code_stockpile as xcs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
proposal_number = {{proposal_number}}  # noqa
run = {{run}}  # noqa

# %%
h5 = xcs.load_smalldata(run, proposal_number)
h5dict = xcs.H5Dictionary(
    stack='/jungfrau1M/ROI_area',
    i0=lambda h5: h5['diodeGon/channels'][..., 0],
    laser_on='evr/code_41',
    angle='scan/var0'
)

# %%
roi = (slice(100, 130), slice(None))
metadata = xcs.create_metadata(h5, run, h5dict, roi=None)
valid = xcs.i0_min_max_filter(metadata, 0, 100)

# %%
xcs.plot_detector_image(h5, h5dict, samplerate=100).show()

# %%
xcs.plot_i0_vs_time(metadata, valid).show()

# %%
xcs.plot_intensity_vs_i0(metadata, valid).show()

# %%
xcs.plot_intensity_vs_time(metadata, valid).show()

# %%
on_metadata = metadata[metadata['laser_on']]
valid_on = valid[metadata['laser_on']]
off_metadata = metadata[~metadata['laser_on']]
valid_off = valid[~metadata['laser_on']]
rockings = []
for meta, color, label, valid_ in zip([on_metadata, off_metadata], ['red', 'blue'], ['on', 'off'], [valid_on, valid_off]):
    rocking = xcs.calc_rocking_stats(h5, h5dict, 'angle', meta, valid_, roi=None)
    plt.errorbar(rocking.index.to_numpy(), rocking['intensity_mean'].to_numpy(), rocking['intensity_stderr'].to_numpy(), color=color, label=label)
    plt.xlabel('angle')
    plt.ylabel('intensity / i0')
    rockings.append(rocking)
plt.title(f'Run {run:04d}')
plt.legend()
plt.show()

# %%
# Save rocking curve to a file
save_dir = Path('/sdf/home/b/bfichera/xcs101927227/results/bfichera/xcs101927227-analysis/rocking_curve_data')
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)
with open(save_dir / f'run_{run:04d}.pkl', 'wb') as fh:
    to_save = {'angle_on': rockings[0].index.to_numpy(),
               'intensity_on': rockings[0]['intensity_mean'].to_numpy(),
               'angle_off': rockings[1].index.to_numpy(),
               'intensity_off': rockings[1]['intensity_mean'].to_numpy()}
    pickle.dump(to_save, fh)

# %%
for valid_, meta, tag in zip([valid_on, valid_off], [on_metadata, off_metadata], ['on', 'off']):
    xcs.save_rocking_movie(
        h5,
        h5dict,
        'angle',
        meta,
        valid_,
        f'/sdf/home/b/bfichera/xcs101927227/results/bfichera/xcs101927227-analysis/movies/run_{run:04d}',
        f'rocking_{tag}.gif',
        roi=None
    )
