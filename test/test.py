import h5py
import numpy as np

import xcs_code_stockpile as xcs

n_shots = 300
n_pi = 100
n_pj = 100
com_i = 50
com_j = 50
sigma = 30 / 2.335
i, j = np.meshgrid(np.arange(n_pi), np.arange(n_pj))

stack = np.exp(
    ((i - com_i)**2 + (j - com_j)**2) / 2 / sigma**2) + np.random.uniform(
        0, 1, size=n_pi * n_pj).reshape(i.shape)

delay = np.sin(np.arange(n_shots) / 20)
labtime = np.arange(n_shots) * 4
laser_on = np.arange(n_shots) // 2
xray_on = np.ones(shape=n_shots)
i0 = np.random.uniform(-0.9, 1.1, size=n_shots)

with h5py.File('test.h5', 'w') as h5:
    h5.create_dataset('/zyla_ladm/ROI_area',
                      shape=stack.shape,
                      dtype=stack.dtype,
                      data=stack)
    h5.create_dataset('/event_time',
                      shape=labtime.shape,
                      dtype=labtime.dtype,
                      data=labtime)
    h5.create_dataset('laser_on',
                      shape=laser_on.shape,
                      dtype=laser_on.dtype,
                      data=laser_on)
    h5.create_dataset('xray_on',
                      shape=xray_on.shape,
                      dtype=xray_on.dtype,
                      data=xray_on)
    h5.create_dataset('i0', shape=i0.shape, dtype=i0.dtype, data=i0)
    h5.create_dataset('delay',
                      shape=delay.shape,
                      dtype=delay.dtype,
                      data=delay)

roi = slice(None), slice(None)
with h5py.File('test.h5', 'r') as h5:
    h5dict = xcs.H5Dictionary()
    metadata = xcs.create_metadata(h5, 144, h5dict)
    filter = xcs.sensible_filter(metadata, 0, 100, 0, 100)
    xcs._strong_reversibility_diffcorr(h5, h5dict, metadata, filter, 0.1, 64)
