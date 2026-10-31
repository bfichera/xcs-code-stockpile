import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

import xcs_code_stockpile as xcs
from xcs_code_stockpile import _utils, rocking, trxrd
from xcs_code_stockpile.datahandler import SmartH5
from xcs_code_stockpile.roi import _ROI
from xcs_code_stockpile.sanity import fit_linear_i_vs_i0

from synthetic import SyntheticRunTestCase


class ROITestCase(unittest.TestCase):

    def test_from_array_is_x_then_y(self):
        roi = _ROI.from_array([10, 20, 30, 40])
        self.assertEqual(tuple(roi), (slice(30, 40), slice(10, 20)))
        self.assertEqual(roi._as_list, [10, 20, 30, 40])

    def test_from_array_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            _ROI.from_array([1, 2, 3])

    def test_from_slices_keeps_steps(self):
        roi = _ROI.from_slices(slice(1, 9, 2), slice(3, 7, 3))
        self.assertEqual(tuple(roi), (slice(1, 9, 2), slice(3, 7, 3)))

    def test_guess(self):
        roi = _ROI.from_array([0, 1, 2, 3])
        self.assertIs(_ROI._guess(roi), roi)
        self.assertEqual(tuple(_ROI._guess([0, 1, 2, 3])),
                         (slice(2, 3), slice(0, 1)))
        self.assertEqual(tuple(_ROI._guess((slice(5, 6), slice(7, 8)))),
                         (slice(5, 6), slice(7, 8)))
        with self.assertRaises(TypeError):
            _ROI._guess([1, 2, 3])

    def test_parsed_roi_positional_keyword_and_default(self):

        @_utils.parsed_roi
        def f(a, roi=None):
            return a[..., *roi]

        a = np.arange(100).reshape(10, 10)
        np.testing.assert_array_equal(f(a), a)
        np.testing.assert_array_equal(f(a, [1, 3, 4, 7]), a[4:7, 1:3])
        np.testing.assert_array_equal(f(a, roi=[1, 3, 4, 7]), a[4:7, 1:3])
        np.testing.assert_array_equal(
            f(a, roi=(slice(2, 5), slice(None, None, 2))), a[2:5, ::2])


class DataHandlerTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / 'data.h5'
        rng = np.random.default_rng(0)
        self.stack = rng.random((6, 4, 5))
        self.raw_delay = np.arange(6.)
        h5dict = xcs.H5Dictionary()
        with h5py.File(self.path, 'w') as h5:
            h5.create_dataset(h5dict.stack, data=self.stack)
            h5.create_dataset(h5dict.event_time, data=np.arange(6.))
            h5.create_dataset(h5dict.laser_on, data=[0, 1, 0, 1, 0, 1])
            h5.create_dataset(h5dict.xray_on, data=np.ones(6, dtype=int))
            h5.create_dataset(h5dict.i0, data=np.ones(6))
            h5.create_dataset('/raw/delay', data=self.raw_delay)

    def test_h5dictionary_allows_extra_fields_and_callables(self):
        h5dict = xcs.H5Dictionary(extra='/some/path',
                                  delay=lambda h5: h5['/raw/delay'][...])
        dumped = h5dict.model_dump()
        self.assertEqual(dumped['extra'], '/some/path')
        self.assertTrue(callable(dumped['delay']))

    def test_smart_h5_accepts_callable_keys(self):
        with SmartH5(self.path, 'r') as h5:
            np.testing.assert_array_equal(
                h5[lambda f: f['/raw/delay'][...] * 2], self.raw_delay * 2)
            np.testing.assert_array_equal(h5['/raw/delay'][...],
                                          self.raw_delay)

    def test_create_metadata_with_callable_mapping(self):
        h5dict = xcs.H5Dictionary(
            delay=lambda h5: h5['/raw/delay'][...] * 10)
        with SmartH5(self.path, 'r') as h5:
            metadata = xcs.create_metadata(h5, 1, h5dict)
        np.testing.assert_array_equal(metadata['delay'],
                                      self.raw_delay * 10)

    def test_create_metadata_extra_field_becomes_column(self):
        h5dict = xcs.H5Dictionary(delay='/raw/delay', raw='/raw/delay')
        with h5py.File(self.path, 'r') as h5:
            metadata = xcs.create_metadata(h5, 1, h5dict)
        np.testing.assert_array_equal(metadata['raw'], self.raw_delay)

    def test_create_metadata_missing_key_is_skipped(self):
        h5dict = xcs.H5Dictionary(delay='/raw/delay', missing='/not/here')
        with h5py.File(self.path, 'r') as h5:
            metadata = xcs.create_metadata(h5, 1, h5dict)
        self.assertNotIn('missing', metadata.columns)


class CreateMetadataTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / 'data.h5'
        self.h5dict = xcs.H5Dictionary()
        self.stack = np.ones((10, 3, 3))
        self.stack[2, 1, 1] = np.nan
        self.laser_on = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
        with h5py.File(self.path, 'w') as h5:
            h5.create_dataset(self.h5dict.stack, data=self.stack)
            h5.create_dataset(self.h5dict.event_time, data=np.arange(10.))
            h5.create_dataset(self.h5dict.laser_on, data=self.laser_on)
            h5.create_dataset(self.h5dict.xray_on,
                              data=np.ones(10, dtype=int))
            h5.create_dataset(self.h5dict.i0, data=np.ones(10))
            h5.create_dataset(self.h5dict.delay, data=np.arange(10.))
        self.h5 = h5py.File(self.path, 'r')
        self.addCleanup(self.h5.close)

    def test_nan_policy(self):
        with self.assertRaises(ValueError):
            xcs.create_metadata(self.h5, 1, self.h5dict)
        metadata = xcs.create_metadata(self.h5, 1, self.h5dict,
                                       nan_policy='ignore')
        self.assertEqual(metadata['intensity_in_roi'].iloc[2], 8.)
        self.assertEqual(metadata['intensity_in_roi'].iloc[0], 9.)
        with self.assertRaises(ValueError):
            xcs.create_metadata(self.h5, 1, self.h5dict, nan_policy='bad')

    def test_calc_intensity_false(self):
        metadata = xcs.create_metadata(self.h5, 1, self.h5dict,
                                       calc_intensity=False)
        self.assertTrue(metadata['intensity_in_roi'].isna().all())
        self.assertEqual(len(metadata), 10)
        np.testing.assert_array_equal(metadata['shot_idx'], np.arange(10))

    def test_default_laser_off_type(self):
        metadata = xcs.create_metadata(self.h5, 1, self.h5dict,
                                       nan_policy='ignore')
        expected = ['on' if lon else 'off' for lon in self.laser_on]
        self.assertEqual(metadata['laser_off_type'].tolist(), expected)

    def test_laser_off_type_func_receives_extra_arguments(self):
        calls = []

        def off_type_func(h5, *args, **kwargs):
            calls.append((h5, args, kwargs))
            return xcs.basic_off_type_encoder(h5['/lightStatus/laser'][...])

        metadata = xcs.create_metadata(self.h5, 1, self.h5dict, None,
                                       'ignore', True, off_type_func, 'x',
                                       key='y')
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], self.h5)
        self.assertEqual(calls[0][1], ('x',))
        self.assertEqual(calls[0][2], {'key': 'y'})
        self.assertEqual(metadata['laser_off_type'].tolist(),
                         xcs.basic_off_type_encoder(self.laser_on))


class OffTypeEncoderTestCase(unittest.TestCase):

    def test_counts_shots_since_last_on_shot(self):
        laser_on = np.array([1, 0, 0, 1, 0, 1, 1, 0])
        self.assertEqual(xcs.basic_off_type_encoder(laser_on), [
            '000_on', '001_off', '002_off', '000_on', '001_off', '000_on',
            '000_on', '001_off'
        ])

    def test_leading_off_shots_borrow_from_first_full_period(self):
        laser_on = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])
        self.assertEqual(xcs.basic_off_type_encoder(laser_on), [
            '001_off', '002_off', '000_on', '001_off', '002_off', '000_on',
            '001_off', '002_off', '000_on'
        ])

    def test_leading_off_shots_with_multiple_on_shots_per_period(self):
        laser_on = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
        self.assertEqual(xcs.basic_off_type_encoder(laser_on), [
            '001_off', '000_on', '001_off', '000_on', '000_on', '001_off',
            '000_on', '001_off', '000_on', '000_on'
        ])

    def test_long_leading_off_run_is_filled_across_periods(self):
        laser_on = np.tile([1, 0, 0, 0], 50)[3:]
        encoded = xcs.basic_off_type_encoder(laser_on)
        self.assertEqual(encoded[:6], [
            '003_off', '000_on', '001_off', '002_off', '003_off', '000_on'
        ])
        laser_on = np.tile([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 50)[1:]
        encoded = xcs.basic_off_type_encoder(laser_on)
        self.assertEqual(encoded[:10], [f'{i:03d}_off' for i in range(1, 10)] +
                         ['000_on'])

    def test_no_on_shots(self):
        self.assertEqual(xcs.basic_off_type_encoder(np.array([0, 0, 0])),
                         ['unknown_off'] * 3)

    def test_empty_input(self):
        self.assertEqual(xcs.basic_off_type_encoder(np.array([], dtype=int)),
                         [])

    def test_not_enough_data_for_period_gives_unknown(self):
        laser_on = np.array([0, 0, 1, 0, 0])
        self.assertEqual(xcs.basic_off_type_encoder(laser_on), [
            'unknown_off', 'unknown_off', '000_on', '001_off', '002_off'
        ])

    def test_off_shot_never_labelled_on(self):
        rng = np.random.default_rng(0)
        for laser_on in [
                np.array([0, 0, 1, 0, 1]),
                np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1]),
                rng.random(200) < 0.5,
                np.tile([0, 0, 1, 1], 30),
        ]:
            encoded = xcs.basic_off_type_encoder(laser_on)
            for lon, ot in zip(laser_on, encoded):
                self.assertEqual(ot.endswith('_on'), bool(lon))

    def test_leading_shot_that_should_be_on_is_unknown(self):
        laser_on = np.tile([0, 0, 1], 20)
        laser_on[2] = 0
        encoded = xcs.basic_off_type_encoder(laser_on)
        self.assertEqual(encoded[:6], [
            '001_off', '002_off', 'unknown_off', '001_off', '002_off',
            '000_on'
        ])

    def test_unknown_label(self):
        encoded = xcs.basic_off_type_encoder(np.array([0, 0]),
                                             unknown_label='?')
        self.assertEqual(encoded, ['?', '?'])

    def test_period_robust_to_dropped_on_shots(self):
        from xcs_code_stockpile._utils import _estimate_period
        rng = np.random.default_rng(0)
        for pattern in [[0, 1, 0, 1, 1], [1, 0, 0, 0], [1, 0],
                        [1] + [0] * 11, [1] + [0] * 99]:
            base = np.tile(pattern, 2000 // len(pattern)).astype(bool)
            for p_drop in [0., 0.02, 0.1]:
                laser_on = base & (rng.random(base.size) >= p_drop)
                self.assertEqual(_estimate_period(laser_on), len(pattern),
                                 (pattern, p_drop))
        self.assertIsNone(_estimate_period(rng.random(1000) < 0.5))

    def test_max_period(self):
        from xcs_code_stockpile._utils import _estimate_period
        laser_on = np.tile([1] + [0] * 199, 5).astype(bool)
        self.assertEqual(_estimate_period(laser_on), 200)
        self.assertIsNone(_estimate_period(laser_on, max_period=100))
        encoded = xcs.basic_off_type_encoder(laser_on[1:], max_period=100)
        self.assertEqual(encoded[0], 'unknown_off')
        encoded = xcs.basic_off_type_encoder(laser_on[1:])
        self.assertEqual(encoded[0], '001_off')

    def test_accepts_boolean_input(self):
        laser_on = np.tile([0, 1, 0, 1, 1], 4).astype(bool)
        encoded = xcs.basic_off_type_encoder(laser_on)
        self.assertEqual(len(encoded), laser_on.size)
        for lon, ot in zip(laser_on, encoded):
            self.assertEqual(ot.endswith('_on'), bool(lon))
        self.assertEqual(set(encoded), {'000_on', '001_off'})


class FilterTestCase(unittest.TestCase):

    def make_metadata(self, n=100):
        return pd.DataFrame({
            'xray_on': np.ones(n, dtype=bool),
            'i0': np.linspace(1., 2., n),
            'intensity_in_roi': 3 * np.linspace(1., 2., n),
            'laser_off_type': ['on' if i % 2 else 'off' for i in range(n)],
        })

    def test_no_filter(self):
        metadata = self.make_metadata().iloc[5:]
        result = xcs.no_filter(metadata)
        self.assertTrue(result.all())
        self.assertTrue(result.index.equals(metadata.index))

    def test_common_enough_off_type_filter_drops_rare_types(self):
        metadata = self.make_metadata()
        metadata.loc[7, 'laser_off_type'] = 'rare'
        result = xcs.common_enough_off_type_filter(metadata)
        self.assertFalse(result.loc[7])
        self.assertEqual(result.sum(), 99)
        self.assertTrue(result.index.equals(metadata.index))

    def test_common_enough_off_type_filter_keeps_balanced_types(self):
        result = xcs.common_enough_off_type_filter(self.make_metadata())
        self.assertTrue(result.all())

    def test_i0_min_max_filter(self):
        metadata = self.make_metadata()
        metadata.loc[0, 'xray_on'] = False
        metadata.loc[1, 'i0'] = 0.
        metadata.loc[2, 'i0'] = np.nan
        metadata.loc[3, 'i0'] = -1.
        result = xcs.i0_min_max_filter(metadata, 10, 90)
        self.assertFalse(result.iloc[:4].any())
        valid_i0 = metadata['i0'].iloc[4:]
        low, high = np.percentile(valid_i0, [10, 90])
        expected = (metadata['i0'] >= low) & (metadata['i0'] <= high)
        expected.iloc[:4] = False
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_i0_min_max_filter_drops_rare_off_types(self):
        metadata = self.make_metadata()
        metadata.loc[50, 'laser_off_type'] = 'rare'
        result = xcs.i0_min_max_filter(metadata, 0, 100)
        self.assertFalse(result.loc[50])
        self.assertEqual(result.sum(), 99)

    def test_sensible_filter_drops_rare_off_types(self):
        metadata = self.make_metadata()
        metadata.loc[50, 'laser_off_type'] = 'rare'
        result = xcs.sensible_filter(metadata, 0, 100, 0, 100)
        self.assertFalse(result.loc[50])
        self.assertEqual(result.sum(), 99)

    def test_filters_preserve_non_default_index(self):
        metadata = self.make_metadata()
        metadata.index = metadata.index + 1000
        for result in [
                xcs.i0_min_max_filter(metadata, 0, 100),
                xcs.sensible_filter(metadata, 0, 100, 0, 100),
                xcs.common_enough_off_type_filter(metadata),
        ]:
            self.assertTrue(result.index.equals(metadata.index))
            self.assertTrue(result.all())


class UtilsTestCase(unittest.TestCase):

    def test_ts_to_s(self):
        ts = np.array([(5 << 32) | 500_000_000, (7 << 32)])
        np.testing.assert_allclose(_utils.ts_to_s(ts), [5.5, 7.])

    def test_save_array_to_movie_regulates_and_clips(self):
        ims = np.arange(24, dtype=float).reshape(2, 3, 4)
        regulated = _utils.save_array_to_movie(ims, vmin=4., vmax=19.)
        expected = (np.clip(ims, 4., 19.) - 4.) / 15.
        np.testing.assert_allclose(regulated, expected)

        regulated = _utils.save_array_to_movie(ims)
        np.testing.assert_allclose(regulated, ims / 23.)

    def test_save_array_to_movie_writes_gif(self):
        import imageio.v3 as iio
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / 'movie.gif'
            ims = np.random.default_rng(0).random((3, 8, 9))
            _utils.save_array_to_movie(ims, path)
            frames = iio.imread(path)
            self.assertEqual(frames.shape[1:3], (8, 9))
            self.assertGreaterEqual(frames.shape[0], 3)

    def test_save_array_to_movie_without_path_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tempdir:
            cwd = os.getcwd()
            os.chdir(tempdir)
            try:
                _utils.save_array_to_movie(np.ones((2, 3, 3)) *
                                           np.arange(2)[:, None, None])
                self.assertEqual(os.listdir(tempdir), [])
            finally:
                os.chdir(cwd)

    def test_slow_helpers_match_numpy(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / 'data.h5'
            h5dict = xcs.H5Dictionary()
            stack = np.random.default_rng(0).random((8, 5, 6))
            with h5py.File(path, 'w') as h5:
                h5.create_dataset(h5dict.stack, data=stack)
            shots = np.array([1, 4, 6])
            roi = [1, 4, 2, 5]
            with h5py.File(path, 'r') as h5:
                np.testing.assert_allclose(
                    _utils.slow_mean(h5, h5dict, shots, roi=roi),
                    stack[shots, 2:5, 1:4].mean(axis=0))
                np.testing.assert_allclose(
                    _utils.slow_sum(h5, h5dict, shots, roi=roi),
                    stack[shots, 2:5, 1:4].sum(axis=0))
                np.testing.assert_allclose(
                    _utils.slow_intensities(h5, h5dict, shots, roi=roi),
                    stack[shots, 2:5, 1:4].sum(axis=(-1, -2)))
                np.testing.assert_allclose(
                    _utils.slow_mean(h5, h5dict, shots),
                    stack[shots].mean(axis=0))


class SanityTestCase(SyntheticRunTestCase):

    def test_fit_linear_i_vs_i0(self):
        beta, intercept, r2 = fit_linear_i_vs_i0(self.metadata,
                                                 self.static_filter)
        self.assertAlmostEqual(beta, self.off_image.sum())
        self.assertAlmostEqual(intercept, 0., places=6)
        self.assertAlmostEqual(r2, 1.)

        _, _, r2_all = fit_linear_i_vs_i0(self.metadata)
        self.assertLess(r2_all, 1.)

    def test_plot_intensity_vs_i0_marks_rejected_shots(self):
        selected = np.arange(self.n_shots) < 50
        fig = xcs.plot_intensity_vs_i0(self.metadata, filter=selected)
        scatter = fig.axes[0].collections[0]
        self.assertEqual(scatter.get_offsets().shape[0], self.n_shots)
        edgecolors = scatter.get_edgecolors()
        red = np.all(np.isclose(edgecolors[:, :3], [1, 0, 0]), axis=1)
        np.testing.assert_array_equal(red, ~selected)
        plt.close(fig)

    def test_plot_detector_image(self):
        fig = xcs.plot_detector_image(self.h5, self.h5dict, 7)
        image = fig.axes[0].images[0]
        expected = self.stack[::7].mean(axis=0)
        np.testing.assert_allclose(image.get_array(), expected)
        np.testing.assert_allclose(image.get_clim(),
                                   np.percentile(expected, [1, 99]))
        plt.close(fig)

        fig = xcs.plot_detector_image(self.h5, self.h5dict, 50, vmin=0,
                                      vmax=1, cmap='magma')
        image = fig.axes[0].images[0]
        self.assertEqual(image.get_clim(), (0, 1))
        self.assertEqual(image.get_cmap().name, 'magma')
        plt.close(fig)

    def test_plot_roi_image(self):
        roi = [10, 30, 20, 50]
        fig = xcs.plot_roi_image(self.h5, self.h5dict, self.metadata,
                                 self.static_filter, roi=roi)
        shots = self.metadata.loc[self.static_filter, 'shot_idx']
        expected = self.stack[shots][:, 20:50, 10:30].mean(axis=0)
        np.testing.assert_allclose(fig.axes[0].images[0].get_array(),
                                   expected)
        plt.close(fig)

        fig = xcs.plot_roi_image(self.h5, self.h5dict, self.metadata)
        np.testing.assert_allclose(fig.axes[0].images[0].get_array(),
                                   self.stack.mean(axis=0))
        plt.close(fig)


class OffTypeMixin:

    def encoded_metadata(self):
        metadata = self.metadata.copy()
        metadata['laser_off_type'] = xcs.basic_off_type_encoder(
            self.laser_on)
        return metadata


class TrxrdFeatureTestCase(OffTypeMixin, SyntheticRunTestCase):

    def test_columns_are_prefixed_by_off_type(self):
        result = trxrd.calc_trxrd_stats(self.h5, self.h5dict, self.metadata,
                                        self.shot_filter, 1)
        stats = ['intensity_mean', 'intensity_stderr', 'n_shots', 'com_i',
                 'com_j', 'fwhm_i', 'fwhm_j']
        expected = {f'{t}_{s}' for t in ['on', 'off'] for s in stats}
        self.assertEqual(set(result.columns), expected)

    def test_values_per_encoded_off_type(self):
        metadata = self.encoded_metadata()
        result = xcs.calc_trxrd_stats(self.h5, self.h5dict, metadata, None,
                                      10)
        delay_bin = np.floor((metadata['delay'] + 20) / 10) * 10 - 20
        for (bin_, off_type), group in metadata.groupby(
            [delay_bin, 'laser_off_type']):
            shots = group['shot_idx'].to_numpy()
            sums = self.stack[shots].sum(axis=(-1, -2))
            mean_i0 = group['i0'].mean()
            row = result.loc[bin_]
            self.assertAlmostEqual(row[f'{off_type}_intensity_mean'],
                                   sums.mean() / mean_i0)
            self.assertAlmostEqual(
                row[f'{off_type}_intensity_stderr'],
                sums.std() / np.sqrt(len(shots)) / mean_i0)
            self.assertEqual(row[f'{off_type}_n_shots'], len(shots))

    def test_compute_image_stats_false(self):
        result = xcs.calc_trxrd_stats(self.h5, self.h5dict, self.metadata,
                                      self.shot_filter, 10,
                                      compute_image_stats=False)
        for column in result.columns:
            if column.split('_', 1)[1] in ['com_i', 'com_j', 'fwhm_i',
                                           'fwhm_j']:
                self.assertTrue(result[column].isna().all(), column)
            else:
                self.assertFalse(result[column].isna().any(), column)

    def test_roi_and_none_filter(self):
        roi = [40, 60, 30, 70]
        result = xcs.calc_trxrd_stats(self.h5, self.h5dict, self.metadata,
                                      None, 100, roi=roi)
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        for off_type, laser_on in [('on', True), ('off', False)]:
            group = self.metadata[self.metadata['laser_on'] == laser_on]
            shots = group['shot_idx'].to_numpy()
            substack = self.stack[shots][:, 30:70, 40:60]
            self.assertAlmostEqual(
                row[f'{off_type}_intensity_mean'],
                substack.sum(axis=(-1, -2)).mean() / group['i0'].mean())
            self.assertEqual(row[f'{off_type}_n_shots'], len(shots))
            mean_im = substack.mean(axis=0)
            rows = mean_im.sum(axis=1)
            self.assertAlmostEqual(
                row[f'{off_type}_com_i'],
                np.sum(np.arange(rows.size) * rows) / rows.sum())


class TrxrdImsTestCase(OffTypeMixin, SyntheticRunTestCase):

    def expected_ims(self, metadata, bin_starts, bin_interval, on_label='on',
                     off_label='off', roi_slices=(slice(None), slice(None))):
        on_ims, off_ims = [], []
        for start in bin_starts:
            in_bin = ((metadata['delay'] >= start) &
                      (metadata['delay'] < start + bin_interval))
            frames = []
            for label in (on_label, off_label):
                group = metadata[in_bin & (metadata['laser_off_type'] ==
                                           label)]
                shots = group['shot_idx'].to_numpy()
                frames.append(self.stack[shots][(slice(None), *roi_slices)]
                              .mean(axis=0) / group['i0'].mean())
            on_ims.append(frames[0])
            off_ims.append(frames[1])
        return np.array(on_ims), np.array(off_ims)

    def test_values_match_numpy(self):
        delays, on_ims, off_ims = xcs.trxrd_ims(self.h5, self.h5dict,
                                                self.metadata, None, 10)
        starts = np.arange(-20., 41., 10.)
        np.testing.assert_allclose(delays, starts)
        expected_on, expected_off = self.expected_ims(self.metadata, starts,
                                                      10)
        self.assertEqual(on_ims.shape, (len(starts), 100, 100))
        self.assertEqual(off_ims.shape, (len(starts), 100, 100))
        np.testing.assert_allclose(on_ims, expected_on)
        np.testing.assert_allclose(off_ims, expected_off)

    def test_filter_is_applied(self):
        delays, on_ims, off_ims = xcs.trxrd_ims(
            self.h5, self.h5dict, self.metadata, self.shot_filter, 10)
        metadata = self.metadata[self.shot_filter]
        expected_on, expected_off = self.expected_ims(metadata, delays, 10)
        np.testing.assert_allclose(on_ims, expected_on)
        np.testing.assert_allclose(off_ims, expected_off)

    def test_roi_list_and_slices_agree(self):
        roi = [40, 60, 30, 70]
        delays, on_ims, off_ims = xcs.trxrd_ims(self.h5, self.h5dict,
                                                self.metadata, None, 20,
                                                roi=roi)
        self.assertEqual(on_ims.shape[1:], (40, 20))
        expected_on, expected_off = self.expected_ims(
            self.metadata, delays, 20,
            roi_slices=(slice(30, 70), slice(40, 60)))
        np.testing.assert_allclose(on_ims, expected_on)
        np.testing.assert_allclose(off_ims, expected_off)
        _, on_slices, off_slices = xcs.trxrd_ims(
            self.h5, self.h5dict, self.metadata, None, 20,
            roi=(slice(30, 70), slice(40, 60)))
        np.testing.assert_allclose(on_slices, on_ims)
        np.testing.assert_allclose(off_slices, off_ims)

    def test_empty_delay_bins_are_skipped(self):
        delay = self.metadata['delay']
        keep = (delay < -10) | (delay >= 20)
        delays, on_ims, off_ims = xcs.trxrd_ims(self.h5, self.h5dict,
                                                self.metadata, keep, 10)
        np.testing.assert_allclose(delays, [-20., 20., 30., 40.])
        self.assertEqual(on_ims.shape[0], 4)
        self.assertFalse(np.isnan(on_ims).any())
        self.assertFalse(np.isnan(off_ims).any())

    def test_custom_labels_in_either_order(self):
        reference = xcs.trxrd_ims(self.h5, self.h5dict, self.metadata, None,
                                  10)
        metadata = self.metadata.copy()
        metadata['laser_off_type'] = np.where(metadata['laser_on'],
                                              'pump_on', 'pump_off')
        for first_shot in (0, 1):
            keep = metadata['shot_idx'] >= first_shot
            ref = xcs.trxrd_ims(self.h5, self.h5dict, self.metadata, keep,
                                10)
            result = xcs.trxrd_ims(self.h5, self.h5dict, metadata, keep, 10)
            for a, b in zip(result, ref):
                np.testing.assert_allclose(a, b)
        self.assertEqual(len(reference), 3)

    def test_requires_exactly_two_off_types(self):
        metadata = self.metadata.copy()
        metadata['laser_off_type'] = np.where(
            metadata['laser_on'], 'on',
            np.where(metadata['shot_idx'] % 2 == 0, 'off', 'other_off'))
        with self.assertRaises(NotImplementedError):
            xcs.trxrd_ims(self.h5, self.h5dict, metadata, None, 10)
        only_on = self.metadata['laser_on']
        with self.assertRaises(NotImplementedError):
            xcs.trxrd_ims(self.h5, self.h5dict, self.metadata, only_on, 10)


class SaveMovieTestCase(OffTypeMixin, SyntheticRunTestCase):

    def capture(self, module, func, *args, **kwargs):
        calls = []

        def fake(ims, path=None, **kws):
            calls.append({'ims': np.array(ims), 'path': Path(path), **kws})

        with mock.patch.object(module, 'save_array_to_movie', fake):
            func(*args, **kwargs)
        return calls

    def expected_frames(self, metadata, on_type, off_type, bin_interval):
        delay = metadata['delay']
        bins = np.floor((delay - delay.min()) / bin_interval)
        on, off = [], []
        for _, group in metadata.groupby(bins):
            frames = []
            for type_ in [on_type, off_type]:
                sel = group[group['laser_off_type'] == type_]
                frames.append(self.stack[sel['shot_idx']].mean(axis=0) /
                              sel['i0'].mean())
            on.append(frames[0])
            off.append(frames[1])
        on, off = np.array(on), np.array(off)
        return on, off, on / off - 1

    def test_save_movie_writes_gifs(self):
        import imageio.v3 as iio
        save_dir = Path(self.tempdir.name) / 'nested' / 'movies'
        xcs.save_movie(self.h5, self.h5dict, self.metadata, self.shot_filter,
                       10, save_dir, fname_prefix='run_')
        names = sorted(p.name for p in save_dir.iterdir())
        self.assertEqual(
            names, ['run_diff_on_off.gif', 'run_off_off.gif',
                    'run_on_on.gif'])
        frames = iio.imread(save_dir / 'run_on_on.gif')
        self.assertEqual(frames.shape[1:], (self.image_size,
                                            self.image_size, 3))

    def test_save_movie_frames(self):
        save_dir = Path(self.tempdir.name) / 'movies'
        calls = self.capture(trxrd, xcs.save_movie, self.h5, self.h5dict,
                             self.metadata, self.shot_filter, 10, save_dir,
                             roi=[40, 60, 30, 70])
        self.assertEqual([c['path'].name for c in calls],
                         ['on_on.gif', 'off_off.gif', 'diff_on_off.gif'])
        expected = self.expected_frames(self.metadata, 'on', 'off', 10)
        for call, frames in zip(calls, expected):
            np.testing.assert_allclose(call['ims'],
                                       frames[:, 30:70, 40:60])
            lo = np.nanpercentile(call['ims'], 1)
            hi = np.nanpercentile(call['ims'], 99)
            if call['path'].name.startswith('diff_'):
                lo, hi = -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
            self.assertAlmostEqual(call['vmin'], lo)
            self.assertAlmostEqual(call['vmax'], hi)

    def test_save_movie_diff_uses_diverging_colormap(self):
        calls = self.capture(trxrd, xcs.save_movie, self.h5, self.h5dict,
                             self.metadata, self.shot_filter, 10,
                             self.tempdir.name)
        cmaps = {c['path'].name: c['cmap'] for c in calls}
        self.assertEqual(cmaps['on_on.gif'], 'viridis')
        self.assertEqual(cmaps['off_off.gif'], 'viridis')
        self.assertEqual(cmaps['diff_on_off.gif'], 'RdBu_r')

    def test_save_movie_multiple_off_types(self):
        laser_on = np.tile([1, 0, 0, 0, 0], 100).astype(bool)
        metadata = self.metadata.copy()
        metadata['laser_on'] = laser_on
        metadata['laser_off_type'] = xcs.basic_off_type_encoder(laser_on)
        off_types = ['001_off', '003_off']
        calls = self.capture(trxrd, xcs.save_movie, self.h5, self.h5dict,
                             metadata, self.shot_filter, 10,
                             self.tempdir.name, on_types=['000_on'],
                             off_types=off_types)
        names = [c['path'].name for c in calls]
        self.assertEqual(names, [
            'on_000_on.gif', 'off_001_off.gif', 'diff_000_on_001_off.gif',
            'on_000_on.gif', 'off_003_off.gif', 'diff_000_on_003_off.gif'
        ])
        for i, off_type in enumerate(off_types):
            expected = self.expected_frames(metadata, '000_on', off_type, 10)
            for call, frames in zip(calls[3 * i:3 * i + 3], expected):
                np.testing.assert_allclose(call['ims'], frames)

    def test_save_rocking_movie(self):
        import imageio.v3 as iio
        metadata = self.rocking_metadata.copy()
        metadata['i0'] = np.linspace(1., 3., self.n_shots)
        save_dir = Path(self.tempdir.name) / 'rocking'
        roi = [10, 30, 20, 50]
        calls = self.capture(rocking, xcs.save_rocking_movie,
                             self.rocking_h5, self.h5dict, 'angle', metadata,
                             None, save_dir, fname='rocking.gif', roi=roi)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['path'], save_dir / 'rocking.gif')
        expected = np.array([
            self.rocking_stack[g['shot_idx']][:, 20:50, 10:30].mean(axis=0) /
            g['i0'].mean() for _, g in metadata.groupby('angle')
        ])
        np.testing.assert_allclose(calls[0]['ims'], expected)

        xcs.save_rocking_movie(self.rocking_h5, self.h5dict, 'angle',
                               metadata, None, save_dir,
                               fname='rocking.gif', roi=roi)
        frames = iio.imread(save_dir / 'rocking.gif')
        self.assertEqual(frames.shape[1:], (30, 20, 3))

    def test_save_rocking_movie_default_fname(self):
        save_dir = Path(self.tempdir.name) / 'rocking_default'
        xcs.save_rocking_movie(self.rocking_h5, self.h5dict, 'angle',
                               self.rocking_metadata, None, save_dir)
        self.assertEqual(len(list(save_dir.iterdir())), 1)


class RockingFeatureTestCase(SyntheticRunTestCase):

    def test_rocking_stats_normalized_by_i0_with_roi(self):
        metadata = self.rocking_metadata.copy()
        metadata['i0'] = np.linspace(1., 3., self.n_shots)
        roi = [10, 30, 20, 50]
        selected = np.arange(self.n_shots) % 3 != 0
        result = xcs.calc_rocking_stats(self.rocking_h5, self.h5dict,
                                        'angle', metadata, selected,
                                        roi=roi)
        for angle, group in metadata[selected].groupby('angle'):
            shots = group['shot_idx'].to_numpy()
            sums = self.rocking_stack[shots][:, 20:50, 10:30].sum(
                axis=(-1, -2))
            mean_i0 = group['i0'].mean()
            row = result.loc[angle]
            self.assertAlmostEqual(row['intensity_mean'],
                                   sums.mean() / mean_i0)
            self.assertAlmostEqual(row['intensity_stderr'],
                                   sums.std() / np.sqrt(len(shots)) / mean_i0)
            self.assertEqual(row['n_shots'], len(shots))


class StaticCorrelationFeatureTestCase(SyntheticRunTestCase):

    def test_nan_denominators_ignore_nan_pixels(self):
        roi = (slice(40, 60), slice(40, 60))
        stack = self.stack.copy()
        stack[:, 45, 45] = np.nan
        path = Path(self.tempdir.name) / 'nan.h5'
        with h5py.File(path, 'w') as h5:
            h5.create_dataset(self.h5dict.stack, data=stack)
        clean = self.stack.copy()
        clean = np.delete(clean[:, 40:60, 40:60].reshape(self.n_shots, -1),
                          5 * 20 + 5, axis=1)
        shots = self.metadata.loc[self.static_filter, 'shot_idx'].to_numpy()
        flat = clean[shots]
        means = flat.mean(axis=1)
        expected_mean = flat @ flat.T / flat.shape[1] / (
            means[:, None] * means) - 1
        centered = flat - means[:, None]
        std = flat.std(axis=1)
        expected_std = centered @ centered.T / flat.shape[1] / (std[:, None] *
                                                                std)
        np.fill_diagonal(expected_mean, np.nan)
        np.fill_diagonal(expected_std, np.nan)
        with h5py.File(path, 'r') as h5:
            for denominator_type, expected in [('nanmean', expected_mean),
                                               ('nanstd', expected_std)]:
                g2, twotime = xcs.calc_static_correlation(
                    h5, self.h5dict, self.metadata, self.static_filter,
                    roi=roi, denominator_type=denominator_type,
                    ret_twotime=True)
                np.testing.assert_allclose(twotime, expected,
                                           err_msg=denominator_type)
                self.assertFalse(np.isnan(g2[1:]).any())

    def test_twotime_diagonal_is_nan(self):
        _, twotime = xcs.calc_static_correlation(self.h5, self.h5dict,
                                                 self.metadata,
                                                 self.static_filter,
                                                 roi=self.correlation_roi,
                                                 ret_twotime=True)
        self.assertTrue(np.isnan(np.diagonal(twotime)).all())

    def test_roi_list_matches_slices(self):
        from_slices = xcs.calc_static_correlation(self.h5, self.h5dict,
                                                  self.metadata,
                                                  self.static_filter,
                                                  roi=self.correlation_roi)
        from_list = xcs.calc_static_correlation(self.h5, self.h5dict,
                                                self.metadata,
                                                self.static_filter,
                                                roi=[48, 53, 48, 53])
        np.testing.assert_allclose(from_slices, from_list)

    def test_invalid_denominator(self):
        with self.assertRaises(NotImplementedError):
            xcs.calc_static_correlation(self.h5, self.h5dict, self.metadata,
                                        self.static_filter,
                                        roi=self.correlation_roi,
                                        denominator_type='median')


class DifferentialFeatureTestCase(OffTypeMixin, SyntheticRunTestCase):

    def diffcorr(self, reversibility_type, metadata=None, **kwargs):
        if reversibility_type != 'none':
            kwargs.setdefault('labtime_bin_interval', 5)
        kwargs.setdefault('roi', self.correlation_roi)
        return xcs.differential_correlation(
            reversibility_type, self.h5, self.h5dict,
            self.metadata if metadata is None else metadata,
            self.shot_filter, 1, **kwargs)

    def test_nan_denominators_match_plain_without_nans(self):
        for reversibility_type in ['strong', 'weak', 'none']:
            for plain in ['mean', 'std']:
                expected = self.diffcorr(reversibility_type,
                                         denominator_type=plain)
                actual = self.diffcorr(reversibility_type,
                                       denominator_type='nan' + plain)
                pd.testing.assert_series_equal(actual, expected)

    def test_std_denominator(self):
        expected = {}
        for shot in np.flatnonzero(self.laser_on):
            preceding_off = shot - 1 if not self.laser_on[shot - 1] else (
                shot - 2)
            on_image = self.stack[shot][*self.correlation_roi]
            off_image = self.stack[preceding_off][*self.correlation_roi]
            corr = np.corrcoef(on_image.ravel(), off_image.ravel())[0, 1]
            expected.setdefault(self.delay[shot], []).append(corr)
        result = self.diffcorr('none', denominator_type='std')
        np.testing.assert_allclose(
            result.to_numpy(),
            [np.mean(expected[d]) for d in self.delay_values])

    def test_roi_list_matches_slices(self):
        for reversibility_type in ['strong', 'weak', 'none']:
            pd.testing.assert_series_equal(
                self.diffcorr(reversibility_type),
                self.diffcorr(reversibility_type, roi=[48, 53, 48, 53]))

    def test_invalid_types(self):
        with self.assertRaises(NotImplementedError):
            self.diffcorr('medium')
        with self.assertRaises(NotImplementedError):
            self.diffcorr('none', denominator_type='median')

    def test_more_than_two_off_types_rejected(self):
        laser_on = np.tile([1, 0, 0], 167)[:self.n_shots].astype(bool)
        metadata = self.metadata.copy()
        metadata['laser_on'] = laser_on
        metadata['laser_off_type'] = xcs.basic_off_type_encoder(laser_on)
        for reversibility_type in ['strong', 'weak', 'none']:
            with self.assertRaises(NotImplementedError):
                self.diffcorr(reversibility_type, metadata=metadata)

    def test_two_encoded_off_types_accepted(self):
        expected = self.diffcorr('none')
        actual = self.diffcorr('none', metadata=self.encoded_metadata())
        pd.testing.assert_series_equal(actual, expected)


class DifferentialTwotimeTestCase(SyntheticRunTestCase):

    def run_in_tempdir(self, func, *args, **kwargs):
        cwd = os.getcwd()
        workdir = Path(self.tempdir.name) / 'cwd'
        workdir.mkdir()
        os.chdir(workdir)
        try:
            result = func(*args, **kwargs)
        finally:
            os.chdir(cwd)
        return result, sorted(os.listdir(workdir))

    def test_single_labtime_bin_matches_direct_calculation(self):
        twotime, stray_files = self.run_in_tempdir(
            xcs.differential_twotime, 'strong', self.h5, self.h5dict,
            self.metadata, self.shot_filter, 10, 1000,
            roi=self.correlation_roi)
        self.assertEqual(stray_files, [])

        bins = np.arange(-20., 41., 10.)
        self.assertEqual(twotime.shape, (bins.size, bins.size))
        np.testing.assert_allclose(twotime, twotime.T)

        mean_on = []
        for left in bins:
            shots = np.flatnonzero(self.laser_on & (self.delay >= left) &
                                   (self.delay < left + 10))
            mean_on.append(
                self.stack[shots][:, *self.correlation_roi].mean(axis=0))
        for i in range(bins.size):
            for j in range(bins.size):
                if i == j:
                    continue
                self.assertAlmostEqual(
                    twotime[i, j], self.correlation(mean_on[i], mean_on[j]))

    def test_std_denominator(self):
        twotime, _ = self.run_in_tempdir(
            xcs.differential_twotime, 'strong', self.h5, self.h5dict,
            self.metadata, self.shot_filter, 20, 1000,
            roi=self.correlation_roi, denominator_type='std')
        bins = np.arange(-20., 41., 20.)
        mean_on = []
        for left in bins:
            shots = np.flatnonzero(self.laser_on & (self.delay >= left) &
                                   (self.delay < left + 20))
            mean_on.append(
                self.stack[shots][:, *self.correlation_roi].mean(axis=0))
        self.assertAlmostEqual(
            twotime[1, 0],
            np.corrcoef(mean_on[1].ravel(), mean_on[0].ravel())[0, 1])

    def test_many_labtime_bins_nan_only_where_delays_never_cooccur(self):
        twotime, _ = self.run_in_tempdir(
            xcs.differential_twotime, 'strong', self.h5, self.h5dict,
            self.metadata, self.shot_filter, 10, 5,
            roi=self.correlation_roi)
        n_bins = twotime.shape[0]
        cooccur = np.zeros((n_bins, n_bins), dtype=bool)
        for start in range(0, self.n_shots, 5):
            shots = np.arange(start, start + 5)
            if self.laser_on[shots].all() or not self.laser_on[shots].any():
                continue
            on_shots = shots[self.laser_on[shots]]
            bins = np.unique(((self.delay[on_shots] + 20) // 10).astype(int))
            cooccur[np.ix_(bins, bins)] = True
        np.fill_diagonal(cooccur, False)
        self.assertTrue(cooccur.any())
        self.assertFalse(np.isnan(twotime[cooccur]).any())
        off_diagonal = ~np.eye(n_bins, dtype=bool)
        self.assertTrue(np.isnan(twotime[off_diagonal & ~cooccur]).all())
        np.testing.assert_allclose(twotime, twotime.T)

    def test_unimplemented_reversibility_types(self):
        for reversibility_type in ['weak', 'none', 'medium']:
            with self.assertRaises(NotImplementedError):
                xcs.differential_twotime(reversibility_type, self.h5,
                                         self.h5dict, self.metadata,
                                         self.shot_filter, 10, 5)


if __name__ == '__main__':
    unittest.main()
