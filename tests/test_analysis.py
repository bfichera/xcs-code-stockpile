import tempfile
import unittest
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter, shift

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import xcs_code_stockpile as xcs
from xcs_code_stockpile.sanity import (intensity_vs_i0,
                                       plot_i0_vs_time,
                                       plot_intensity_vs_i0,
                                       plot_intensity_vs_time)


class AnalysisTestCase(unittest.TestCase):

    def setUp(self):
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

    def test_synthetic_run(self):
        np.testing.assert_array_equal(self.delay[:61], np.arange(-20., 41.))
        np.testing.assert_array_equal(self.delay[61:122],
                                      np.arange(40., -21., -1.))

        off_cols = self.delay_envelopes[-1.].sum(axis=0)
        pumped_cols = self.delay_envelopes[0.].sum(axis=0)
        pixels = np.arange(self.image_size)
        off_com_j = np.sum(pixels * off_cols) / off_cols.sum()
        pumped_com_j = np.sum(pixels * pumped_cols) / pumped_cols.sum()
        self.assertGreater(pumped_com_j, off_com_j)
        self.assertLess(self.delay_images[0.].sum(), self.off_image.sum())

        pumped_intensity = np.array([
            self.delay_envelopes[delay].sum()
            for delay in self.delay_values[self.delay_values >= 0]
        ])
        first_difference = np.diff(pumped_intensity)
        self.assertTrue(np.any(first_difference > 0))
        self.assertTrue(np.any(first_difference < 0))

        pumped_com = []
        for delay in self.delay_values[self.delay_values >= 0]:
            cols = self.delay_envelopes[delay].sum(axis=0)
            pumped_com.append(np.sum(pixels * cols) / cols.sum())
        np.testing.assert_allclose(pumped_com, pumped_com[0])

        displacement = np.array([
            self.pump_speckle_displacement[delay]
            for delay in self.delay_values[self.delay_values >= 0]
        ])
        self.assertAlmostEqual(displacement.max(), -displacement.min())
        np.testing.assert_allclose(displacement[:11], displacement[30:41],
                                   atol=1e-14)

    def test_create_metadata(self):
        expected_intensity = self.stack.sum(axis=(-1, -2))
        np.testing.assert_allclose(self.metadata['intensity_in_roi'],
                                   expected_intensity)
        np.testing.assert_array_equal(self.metadata['shot_idx'],
                                      np.arange(self.n_shots))
        self.assertEqual(self.metadata.attrs['run'], 144)
        self.assertEqual(self.metadata['laser_on'].dtype, bool)
        self.assertEqual(self.metadata['xray_on'].dtype, bool)

        roi_metadata = xcs.create_metadata(
            self.h5, 144, self.h5dict, roi=[20, 80, 10, 90])
        expected = self.stack[:, 10:90, 20:80].sum(axis=(-1, -2))
        np.testing.assert_allclose(roi_metadata['intensity_in_roi'], expected)

    def test_intensity_vs_i0(self):
        result = intensity_vs_i0(self.metadata)
        self.assertEqual(len(result), np.count_nonzero(~self.laser_on))
        np.testing.assert_allclose(result['i0'], self.i0[~self.laser_on])
        np.testing.assert_allclose(result['intensity'],
                                   self.off_image.sum() *
                                   self.i0[~self.laser_on])
        self.assertAlmostEqual(result.attrs['beta'], self.off_image.sum())
        self.assertAlmostEqual(result.attrs['intercept'], 0.)
        self.assertAlmostEqual(result.attrs['R^2'], 1.)

    def test_sensible_filter(self):
        self.assertTrue(self.shot_filter.all())

        metadata = self.metadata.copy()
        metadata.loc[0, 'xray_on'] = False
        metadata.loc[1, 'i0'] = 0
        metadata.loc[2, 'i0'] = np.nan
        shot_filter = xcs.sensible_filter(metadata, 0, 100, 0, 100)
        self.assertFalse(shot_filter.iloc[:3].any())
        self.assertTrue(shot_filter.iloc[3:].all())

    def test_static_correlation(self):
        g2_minus_one, twotime = xcs.calc_static_correlation(
            self.h5,
            self.h5dict,
            self.metadata,
            self.static_filter,
            roi=self.correlation_roi,
            ret_twotime=True,
        )
        static_shots = self.metadata.loc[
            self.static_filter, 'shot_idx'].to_numpy()
        roi_stack = self.stack[static_shots][..., *self.correlation_roi]
        flat = roi_stack.reshape(static_shots.size, -1)
        means = flat.mean(axis=1)
        expected_twotime = flat @ flat.T / flat.shape[1]
        expected_twotime /= means[:, None] * means
        expected_twotime -= 1
        np.testing.assert_allclose(twotime, expected_twotime)
        self.assertTrue(np.isnan(g2_minus_one[0]))
        short_lag_contrast = np.mean(g2_minus_one[1:11])
        long_lag_contrast = np.mean(g2_minus_one[100:150])
        self.assertGreater(short_lag_contrast, 0.1)
        self.assertLess(abs(long_lag_contrast), 0.1)
        expected = [
            np.diagonal(expected_twotime, offset=i).mean()
            for i in range(1, static_shots.size)
        ]
        np.testing.assert_allclose(g2_minus_one[1:], expected)

        std_correlation, std_twotime = xcs.calc_static_correlation(
            self.h5,
            self.h5dict,
            self.metadata,
            self.static_filter,
            roi=self.correlation_roi,
            denominator_type='std',
            ret_twotime=True,
        )
        centered = flat - means[:, None]
        std = flat.std(axis=1)
        expected_std_twotime = centered @ centered.T / flat.shape[1]
        expected_std_twotime /= std[:, None] * std
        np.testing.assert_allclose(std_twotime, expected_std_twotime)
        expected_std = [
            np.diagonal(expected_std_twotime, offset=i).mean()
            for i in range(1, static_shots.size)
        ]
        np.testing.assert_allclose(std_correlation[1:], expected_std,
                                   atol=1e-11)

    def test_differential_correlations(self):
        expected_by_delay = {delay: [] for delay in self.delay_values}
        expected_by_labtime = {delay: {} for delay in self.delay_values}
        for shot in np.flatnonzero(self.laser_on):
            labtime_bin = shot // 5
            labtime_bin_start = labtime_bin * 5
            on_image = self.normalized_stack[shot][*self.correlation_roi]
            off_image = self.normalized_stack[
                labtime_bin_start][*self.correlation_roi]
            correlation = self.correlation(on_image, off_image)
            delay = self.delay[shot]
            expected_by_delay[delay].append(correlation)
            expected_by_labtime[delay].setdefault(
                labtime_bin, []).append(correlation)
        expected_none = np.array([
            np.mean(expected_by_delay[delay]) for delay in self.delay_values
        ])
        expected_reversible = np.array([
            np.mean([np.mean(values) for values in
                     expected_by_labtime[delay].values()])
            for delay in self.delay_values
        ])
        for reversibility_type in ['strong', 'weak', 'none']:
            kwargs = {}
            if reversibility_type != 'none':
                kwargs['labtime_bin_interval'] = 5
            result = xcs.differential_correlation(
                reversibility_type,
                self.h5,
                self.h5dict,
                self.metadata,
                self.shot_filter,
                delay_bin_interval=1,
                roi=self.correlation_roi,
                **kwargs,
            )
            np.testing.assert_allclose(result.index.to_numpy(dtype=float),
                                       self.delay_values)
            expected = (expected_none if reversibility_type == 'none' else
                        expected_reversible)
            np.testing.assert_allclose(result.to_numpy(), expected)

    def test_trxrd_stats(self):
        for laser_on in [False, True]:
            shot_filter = self.shot_filter & (self.metadata['laser_on'] ==
                                              laser_on)
            result = xcs.calc_trxrd_stats(
                self.h5,
                self.h5dict,
                self.metadata,
                shot_filter,
                bin_interval=1,
            )
            metadata = self.metadata[shot_filter]
            delay_group = np.floor(metadata['delay'])
            for delay_bin, group in metadata.groupby(delay_group):
                actual = result.loc[delay_bin]
                shots = group['shot_idx'].to_numpy()
                sums = self.stack[shots].sum(axis=(-1, -2))
                mean_i0 = group['i0'].mean()
                self.assertAlmostEqual(actual['intensity_mean'],
                                       np.mean(sums) / mean_i0)
                self.assertAlmostEqual(
                    actual['intensity_stderr'],
                    np.std(sums) / np.sqrt(len(group)) / mean_i0)
                self.assertEqual(actual['n_shots'], len(group))

                mean_im = self.stack[shots].mean(axis=0)
                rows = mean_im.sum(axis=1)
                cols = mean_im.sum(axis=0)
                pixels = np.arange(self.image_size)
                com_i = np.sum(pixels * rows) / rows.sum()
                com_j = np.sum(pixels * cols) / cols.sum()
                fwhm_i = np.sqrt(
                    np.sum(rows * (pixels - com_i)**2) /
                    rows.sum()) * 2.335
                fwhm_j = np.sqrt(
                    np.sum(cols * (pixels - com_j)**2) /
                    cols.sum()) * 2.335
                self.assertAlmostEqual(actual['com_i'], com_i)
                self.assertAlmostEqual(actual['com_j'], com_j)
                self.assertAlmostEqual(actual['fwhm_i'], fwhm_i)
                self.assertAlmostEqual(actual['fwhm_j'], fwhm_j)

    def test_rocking_stats(self):
        result = xcs.calc_rocking_stats(
            self.rocking_h5,
            self.h5dict,
            'angle',
            self.rocking_metadata,
            self.rocking_filter,
        )
        for angle, group in self.rocking_metadata.groupby('angle'):
            shots = group['shot_idx'].to_numpy()
            sums = self.rocking_stack[shots].sum(axis=(-1, -2))
            actual = result.loc[angle]
            self.assertAlmostEqual(actual['intensity_mean'], np.mean(sums))
            self.assertAlmostEqual(actual['intensity_stderr'],
                                   np.std(sums) / np.sqrt(len(group)))
            self.assertEqual(actual['n_shots'], len(group))

        peak_angle = result['intensity_mean'].idxmax()
        angle_step = np.diff(np.unique(self.rocking_angle))[0]
        self.assertLessEqual(abs(peak_angle), angle_step)
        self.assertGreater(result['intensity_mean'].max(),
                           10 * result['intensity_mean'].iloc[0])

    def test_visualizations(self):
        plot_dir = Path(__file__).parent / 'plots'
        plot_dir.mkdir(exist_ok=True)

        g2_minus_one, twotime = xcs.calc_static_correlation(
            self.h5,
            self.h5dict,
            self.metadata,
            self.static_filter,
            roi=self.correlation_roi,
            ret_twotime=True,
        )
        differential = {}
        for reversibility_type in ['strong', 'weak', 'none']:
            kwargs = {}
            if reversibility_type != 'none':
                kwargs['labtime_bin_interval'] = 5
            differential[reversibility_type] = xcs.differential_correlation(
                reversibility_type,
                self.h5,
                self.h5dict,
                self.metadata,
                self.shot_filter,
                delay_bin_interval=1,
                roi=self.correlation_roi,
                **kwargs,
            )
        trxrd = {}
        for laser_state in [False, True]:
            shot_filter = self.shot_filter & (
                self.metadata['laser_on'] == laser_state)
            trxrd[laser_state] = xcs.calc_trxrd_stats(
                self.h5,
                self.h5dict,
                self.metadata,
                shot_filter,
                bin_interval=1,
            )
        rocking = xcs.calc_rocking_stats(
            self.rocking_h5,
            self.h5dict,
            'angle',
            self.rocking_metadata,
            self.rocking_filter,
        )

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        first_off_shot = self.metadata.loc[
            self.static_filter, 'shot_idx'].iloc[0]
        axes[0, 0].imshow(self.stack[first_off_shot])
        axes[0, 0].set_title('Example speckled frame')
        image = axes[0, 1].imshow(twotime, origin='lower', aspect='auto')
        axes[0, 1].set_title('Two-time correlation')
        fig.colorbar(image, ax=axes[0, 1])
        lag = np.arange(1, g2_minus_one.size)
        positive = g2_minus_one[1:] > 0
        axes[0, 2].loglog(lag[positive], g2_minus_one[1:][positive])
        axes[0, 2].set_title('Static correlation')
        axes[0, 2].set_xlabel('Retained frame lag')
        axes[0, 2].set_ylabel('g2 - 1')
        for name, result in differential.items():
            axes[1, 0].plot(result.index, result, marker='o', label=name)
        axes[1, 0].set_title('Differential correlation')
        axes[1, 0].set_xlabel('Delay-bin left edge')
        axes[1, 0].legend()
        for laser_state, color, label in [(False, 'blue', 'Laser off'),
                                          (True, 'red', 'Laser on')]:
            result = trxrd[laser_state]
            axes[1, 1].errorbar(result.index,
                                result['intensity_mean'],
                                yerr=result['intensity_stderr'],
                                marker='o',
                                color=color,
                                label=label)
        axes[1, 1].set_title('tr-XRD intensity / i0')
        axes[1, 1].set_xlabel('Delay-bin left edge')
        axes[1, 1].legend()
        axes[1, 2].errorbar(rocking.index,
                            rocking['intensity_mean'],
                            yerr=rocking['intensity_stderr'],
                            marker='o')
        axes[1, 2].set_title('Rocking curve')
        axes[1, 2].set_xlabel('Angle index')
        fig.tight_layout()
        figures = {'analysis_overview.png': fig}

        differential_fig, differential_axes = plt.subplots(
            1, 3, figsize=(15, 4), sharex=True, sharey=True)
        colors = {'strong': 'tab:blue', 'weak': 'tab:orange',
                  'none': 'tab:green'}
        for axis, name in zip(differential_axes, ['strong', 'weak', 'none']):
            result = differential[name]
            axis.plot(result.index,
                      result,
                      marker='o',
                      color=colors[name])
            axis.set_title(f'{name.capitalize()} reversibility')
            axis.set_xlabel('Delay-bin left edge')
        differential_axes[0].set_ylabel('Differential correlation')
        differential_fig.tight_layout()
        figures['differential_correlations.png'] = differential_fig

        model_fig, model_axes = plt.subplots(1, 3, figsize=(14, 4))
        model_axes[0].plot(self.event_time, self.delay, marker='o')
        model_axes[0].set_title('Triangle-wave delay scan')
        model_axes[0].set_xlabel('Lab time')
        model_axes[0].set_ylabel('Delay')
        on_shots = self.metadata['laser_on']
        normalized_intensity = (
            self.metadata.loc[on_shots, 'intensity_in_roi'] /
            self.metadata.loc[on_shots, 'i0'])
        model_axes[1].scatter(self.metadata.loc[on_shots, 'delay'],
                              normalized_intensity)
        model_axes[1].set_title('Pump response')
        model_axes[1].set_xlabel('Delay')
        model_axes[1].set_ylabel('Integrated intensity / i0')
        center_row = self.image_size // 2
        model_axes[2].plot(self.off_image[center_row], label='Pump off')
        model_axes[2].plot(self.delay_images[0.][center_row], label='Delay 0')
        model_axes[2].set_title('Gaussian peak shift and drop')
        model_axes[2].set_xlabel('Detector column')
        model_axes[2].legend()
        model_fig.tight_layout()
        figures['synthetic_run.png'] = model_fig

        figures.update({
            'intensity_vs_i0.png': plot_intensity_vs_i0(self.metadata),
            'i0_vs_time.png': plot_i0_vs_time(self.metadata),
            'intensity_vs_time.png': plot_intensity_vs_time(self.metadata),
        })
        for filename, figure in figures.items():
            path = plot_dir / filename
            figure.savefig(path, dpi=120)
            plt.close(figure)
            self.assertGreater(path.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
