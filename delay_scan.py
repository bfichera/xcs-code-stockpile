import xcs_code_stockpile as xcs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# %%
def slow_mean(stack, valid):
    n = 0
    out = np.zeros(stack.shape[1:])
    for i in tqdm(list(range(stack.shape[0]))):
        if valid[i]:
            out += stack[i, ...]
            n += 1
    return out / n

# %%
proposal_number = {{proposal_number}}  # noqa
run = {{run}}  # noqa

# %%
h5 = xcs.load_smalldata(run, proposal_number)
h5dict = xcs.H5Dictionary(
    stack='/jungfrau1M/ROI_area',
    i0=lambda h5: h5['diodeGon/channels'][..., 0],
    laser_on='evr/code_41',
    delay='scan/awg_delay_ns_delay',
    measured_voltage='ai/ch04'
)

# %%
metadata = xcs.create_metadata(h5, run, h5dict, roi=None)
valid = xcs.sensible_filter(metadata, 50, 99.5, 0.5, 99.5)

# %%
on_meta = metadata[metadata['laser_on']]
off_meta = metadata[~metadata['laser_on']]
fig, axs = plt.subplots(1, 2)
for ax in axs:
    ax.plot(on_meta['measured_voltage'], label='on', color='red')
    ax.plot(off_meta['measured_voltage'], label='off', color='blue')
    ax.set_xlabel('frame')
    ax.set_ylabel('measured_voltage')
axs[1].set_ylim(bottom=0.0, top=0.3)
plt.legend()
plt.show()

# %%
rois = [
    (slice(40, 160, None), slice(25, 150, None)),
    (slice(103, 123, None), slice(52, 67, None)),
    (slice(103, 123, None), slice(73, 97, None))
]


# %%
xcs.plot_detector_image(h5, h5dict, samplerate=1)
roi_colors = ['black', 'red', 'green', 'blue', 'purple', 'brown']
for i, roi in enumerate(rois):
    plt.plot([roi[1].start, roi[1].stop], [roi[0].start, roi[0].start], color=roi_colors[i])
    plt.plot([roi[1].start, roi[1].stop], [roi[0].stop, roi[0].stop], color=roi_colors[i])
    plt.plot([roi[1].start, roi[1].start], [roi[0].start, roi[0].stop], color=roi_colors[i])
    plt.plot([roi[1].stop, roi[1].stop], [roi[0].start, roi[0].stop], color=roi_colors[i])

# %%
xcs.plot_i0_vs_time(metadata, valid).show()

# %%
xcs.plot_intensity_vs_i0(metadata, valid).show()

# %%
xcs.plot_intensity_vs_time(metadata, valid).show()

# %%
delays = h5[h5dict.delay][...]
diff = np.absolute(np.diff(delays))
u_diffs, counts = np.unique(diff, return_counts=True)
typical_dt = u_diffs[u_diffs > 0][np.argmax(counts[u_diffs > 0])]
print(f'{typical_dt=:0.2f}')

# %%

fig, axd = plt.subplot_mosaic([['intensity_mean', 'com_i', 'com_j'], ['fwhm_i', 'fwhm_j', '_skip']])
for i, (tag, ax) in enumerate(axd.items()):
    if tag == '_skip':
        continue
    for j, (roi, color) in enumerate(zip(rois, roi_colors)):
        trxrd = xcs.calc_trxrd_stats(h5, h5dict, metadata, valid, typical_dt, roi=roi)
        label = f'on/off (roi {j})' if i == 0 else None
        ax.plot(trxrd.index, trxrd[f'on_{tag}']/trxrd[f'off_{tag}'], color=color, linestyle='-', label=label)
    ax.set_title(tag)
fig.legend(loc='lower right')
fig.suptitle(f'Run {run}')
fig.tight_layout()
plt.show()

# %%


def normalize(a):
    return a / np.amax(a)


fig, axd = plt.subplot_mosaic([['intensity_mean', 'com_i', 'com_j'], ['fwhm_i', 'fwhm_j', '_skip']])
for i, (tag, ax) in enumerate(axd.items()):
    if tag == '_skip':
        continue
    for j, (roi, color) in enumerate(zip(rois, roi_colors)):
        trxrd = xcs.calc_trxrd_stats(h5, h5dict, metadata, valid, typical_dt, roi=roi)
        label = f'off (roi {j})' if j == i else None
        ax.plot(trxrd.index, normalize(trxrd[f'off_{tag}']), color=color, linestyle=':', label=label)
    ax.set_title(tag)
fig.legend(loc='lower right')
fig.suptitle(f'Run {run}')
fig.tight_layout()
plt.show()

# %%
voltage_high = metadata['measured_voltage'] > 1.0
laser_on = metadata['laser_on']
high_and_on = (voltage_high) & (laser_on)
off = (~voltage_high) & (~laser_on)
stack = h5[h5dict.stack]

# %%
mean_on = slow_mean(stack, high_and_on)
mean_off = slow_mean(stack, off)
diff = mean_on / mean_off - 1.0

# %%
fig, axd = plt.subplot_mosaic([['diff', 'on', 'off']], figsize=(12, 12))
axd['diff'].imshow(diff, cmap='RdBu_r', vmin=np.nanpercentile(diff, 1), vmax=np.nanpercentile(diff, 99))
axd['on'].imshow(mean_on, vmin=np.nanpercentile(mean_on, 1), vmax=np.nanpercentile(mean_on, 99))
axd['off'].imshow(mean_off, vmin=0.0, vmax=3.0)

axd['diff'].set_title('on/off-1: jungfrau (zoomed)')
axd['on'].set_title('on shots: jungfrau (zoomed)')
axd['off'].set_title('off shots: jungfrau (zoomed)')


for i, roi in enumerate(rois):
    axd['diff'].plot([roi[1].start, roi[1].stop], [roi[0].start, roi[0].start], color=roi_colors[i])
    axd['diff'].plot([roi[1].start, roi[1].stop], [roi[0].stop, roi[0].stop], color=roi_colors[i])
    axd['diff'].plot([roi[1].start, roi[1].start], [roi[0].start, roi[0].stop], color=roi_colors[i])
    axd['diff'].plot([roi[1].stop, roi[1].stop], [roi[0].start, roi[0].stop], color=roi_colors[i])

fig.tight_layout()
fig.show()

# %%
xcs.save_movie(
    h5,
    h5dict,
    metadata,
    valid,
    typical_dt,
    f'/sdf/home/b/bfichera/xcs101927227/results/bfichera/xcs101927227-analysis/movies/run_{run:04d}',
    '',
    roi=None
)
