"""Shared synthetic pump-probe run used by the test modules."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter, shift

matplotlib.use('Agg')

import xcs_code_stockpile as xcs


def _forbid_breakpoint(*args, **kwargs):
    raise AssertionError('breakpoint() was called from library code')


class SyntheticRunTestCase(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch.object(sys, 'breakpointhook', _forbid_breakpoint)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / 'smalldata.h5'
        self.h5dict = xcs.H5Dictionary()

        self.n_shots = 500
        self.image_size = 100
        self.correlation_roi = (slice(48, 53), slice(48, 53))
        self.i0 = np.linspace(0.8, 1.2, self.n_shots)
        self.event_time = np.arange(self.n_shots, dtype=float)
        self.laser_on = np.tile([0, 1, 0, 1, 1], 100).astype(bool)

        triangle = np.concatenate((np.arange(-20., 41.),
                                   np.arange(40., -21., -1.)))
        self.delay = np.resize(triangle, self.n_shots)
        self.delay_values = np.arange(-20., 41.)

        pixels = np.arange(self.image_size)
        rows, cols = np.meshgrid(pixels, pixels, indexing='ij')
        sigma = 13.

        def gaussian(center_j):
            return np.exp(-((rows - 49.5)**2 + (cols - center_j)**2) /
                          (2 * sigma**2))

        self.equilibrium_envelope = gaussian(45.)
        self.delay_envelopes = {}
        self.pump_speckle_displacement = {}
        for delay in self.delay_values:
            if delay < 0:
                amplitude = 1.
                center_j = 45.
                displacement = 0.
            else:
                amplitude = (0.72 + 0.12 * np.exp(-delay / 60) *
                             np.cos(2 * np.pi * delay / 60))
                center_j = 55.
                displacement = 2 * np.sin(2 * np.pi * delay / 30)
            self.delay_envelopes[delay] = amplitude * gaussian(center_j)
            self.pump_speckle_displacement[delay] = displacement

        rng = np.random.default_rng(1)
        rho = np.exp(-5 / 40)
        field = np.zeros((self.image_size, self.image_size), dtype=complex)
        self.labtime_speckles = []
        for _ in range(100):
            innovation = (rng.normal(size=field.shape) +
                          1j * rng.normal(size=field.shape))
            innovation = gaussian_filter(innovation.real, 0.3) + 1j * (
                gaussian_filter(innovation.imag, 0.3))
            innovation /= np.sqrt(np.mean(np.abs(innovation)**2))
            field = rho * field + np.sqrt(1 - rho**2) * innovation
            speckle = np.abs(field)**2
            self.labtime_speckles.append(speckle / speckle.mean())
        self.labtime_speckles = np.array(self.labtime_speckles)

        normalized_stack = np.empty(
            (self.n_shots, self.image_size, self.image_size))
        for shot in range(self.n_shots):
            delay = self.delay[shot]
            envelope = (self.delay_envelopes[delay]
                        if self.laser_on[shot]
                        else self.equilibrium_envelope)
            displacement = (self.pump_speckle_displacement[delay]
                            if self.laser_on[shot] else 0.)
            speckle = shift(self.labtime_speckles[shot // 5],
                            shift=(0, displacement),
                            order=1,
                            mode='wrap')
            image = envelope * speckle
            normalized_stack[shot] = image * envelope.sum() / image.sum()
        self.normalized_stack = normalized_stack
        self.stack = normalized_stack * self.i0[:, None, None]
        self.off_image = normalized_stack[0]
        self.delay_images = {}
        for delay in self.delay_values:
            speckle = shift(self.labtime_speckles[0],
                            shift=(0,
                                   self.pump_speckle_displacement[delay]),
                            order=1,
                            mode='wrap')
            image = self.delay_envelopes[delay] * speckle
            self.delay_images[delay] = (
                image * self.delay_envelopes[delay].sum() / image.sum())

        with h5py.File(self.path, 'w') as h5:
            h5.create_dataset(self.h5dict.stack, data=self.stack)
            h5.create_dataset(self.h5dict.event_time, data=self.event_time)
            h5.create_dataset(self.h5dict.laser_on, data=self.laser_on)
            h5.create_dataset(self.h5dict.xray_on,
                              data=np.ones(self.n_shots, dtype=int))
            h5.create_dataset(self.h5dict.i0, data=self.i0)
            h5.create_dataset(self.h5dict.delay, data=self.delay)

        self.h5 = h5py.File(self.path, 'r')
        self.metadata = xcs.create_metadata(self.h5, 144, self.h5dict)
        self.shot_filter = xcs.sensible_filter(self.metadata, 0, 100, 0,
                                                100)

        self.static_filter = self.shot_filter & ~self.metadata['laser_on']

        self.rocking_path = Path(self.tempdir.name) / 'rocking.h5'
        self.rocking_angle = np.repeat(np.linspace(-1., 1., 100), 5)
        rocking_envelope = (0.05 + np.exp(
            -(self.rocking_angle / 0.22)**2 / 2))
        self.rocking_stack = (self.off_image[None, ...] *
                              rocking_envelope[:, None, None])
        with h5py.File(self.rocking_path, 'w') as h5:
            h5.create_dataset(self.h5dict.stack, data=self.rocking_stack)
            h5.create_dataset(self.h5dict.event_time,
                              data=np.arange(self.n_shots, dtype=float))
            h5.create_dataset(self.h5dict.laser_on,
                              data=np.zeros(self.n_shots, dtype=int))
            h5.create_dataset(self.h5dict.xray_on,
                              data=np.ones(self.n_shots, dtype=int))
            h5.create_dataset(self.h5dict.i0,
                              data=np.ones(self.n_shots))
            h5.create_dataset(self.h5dict.delay,
                              data=np.zeros(self.n_shots))

        self.rocking_h5 = h5py.File(self.rocking_path, 'r')
        self.rocking_metadata = xcs.create_metadata(
            self.rocking_h5, 145, self.h5dict)
        self.rocking_metadata['angle'] = self.rocking_angle
        self.rocking_filter = np.ones(self.n_shots, dtype=bool)

    def tearDown(self):
        self.h5.close()
        self.rocking_h5.close()
        self.tempdir.cleanup()

    @staticmethod
    def correlation(im1, im2):
        return np.mean(im1 * im2) / np.mean(im1) / np.mean(im2) - 1
