# xcs-code-stockpile

Read the documentation [here](https://bfichera.github.io/xcs-code-stockpile/)!

## Installation

```
pip install git+https://github.com/bfichera/xcs-code-stockpile.git
```

## Requirements

-   `h5py`
-   `matplotlib`
-   `numpy`
-   `pandas`
-   `scikit-learn`
-   `scipy`

## Usage

### H5Dicts

Part of the problem with data analysis at beamtimes is that the h5 keys that you
need to use seem to change all the time.

To remediate this, we use a data type called an H5Dictionary, instantiated as:

```
import xcs_code_stockpiles as xcs
h5dict = xcs.H5Dictionary(
    stack='/jungfrau1M/ROI_area',
    laser_on='evr/code_41',
    delay='scan/awg_delay_ns_delay',
    measured_voltage='ai/ch04'
    i0=lambda h5: h5['diodeGon/channels'][..., 0],
)
```

Note that the keys can either be strings or functions of the form
`func(h5, *args, **kwargs)`. In this case, we needed to use
`h5['diodeGon/channels'][..., 0]` as the `i0` array, which the lambda function
in that signature returns.

Then, entries in the `h5` file can be accessed with, e.g., `h5[h5dict.delay]`.

### ROIs

Multiple ways of reporting ROIs are allowed. For example, I like

```
roi = (slice(i_start, i_end), slice(j_start, j_end))
data = stack[..., *roi]
```

But others like

```
roi = [x0, x1, y0, y1]
data = stack[..., roi[2]:roi[3], roi[0]:roi[1]]
```

Both of these options are supported just choose whatever you like best

### Metadata

First things first, we want to compose a dataframe that compiles metadata about
every shot in the h5, like what was the value of `i0`, was the laser or the xray
on or off, etc. etc. This is done with the function `create_metadata`:

```
metadata = xcs.create_metadata(h5, run, h5dict, roi=None)
```

You can filter the shots in various ways defined in the documentation. But for
instance, if you want to remove any shots where `i0` was less than 20th
percentile or greater than 99th percentile:

```
valid = xcs.i0_min_max_filter(metadata, 20, 99)
valid_metadata = metadata[valid]
```

### Sanity checks

A lot of the time you want to know like, what does my mean detector image look
like? What about the intensity in an roi vs `i0`? Is my intensity varying with
lab time?

There are a bunch of one-liners to do that, like:

```
xcs.plot_detector_image(h5, h5dict, samplerate=10)
xcs.plot_i0_vs_time(metadata, valid)
xcs.plot_intensity_vs_i0(metadata, valid)
xcs.plot_intensity_vs_time(metadata, valid)
```

### Rocking curves

```
xcs.calc_rocking_stats(h5, h5dict, 'angle', meta, valid_, roi=None)
```

where 'angle' is a key in the `h5dict` (see above), outputs a dataframe that
looks like

```
          intensity_mean  intensity_stderr  n_shots
angle
28.22024   245508.733535       6666.424074     86.0
28.22412   221174.520826       5413.673072     94.0
28.22812   209136.930675       5738.881239    101.0
28.23212   205586.143905       4703.099987     98.0
28.23612   227013.007733       6326.169881     69.0
...                  ...               ...      ...
28.60412   235348.223440       6710.338664     60.0
28.60812   200997.281006       5222.136110     87.0
28.61212   224465.517235       5669.345498     82.0
28.61612   230137.176309       6449.770778     66.0
28.62012   221107.678595       6172.491033     97.0

[101 rows x 3 columns]
```

where the `instensity_mean` column is normalized by `i0`.

### Normal time-resolved XRD

The following line:

```
roi = (slice(50,100), slice(150,220))
trxrd = xcs.calc_trxrd_stats(h5, h5dict, metadata, valid, delay_bin, roi=roi)
```

bins all of the delays over a binsize `delay_bin`, and then trxrd is a dataframe
that looks like

<details>
    <summary>Click to view lots of text</summary>

```
               on_intensity_mean  on_intensity_stderr  on_n_shots   on_com_i  \
delay_bin                                                                      
890596.638710      145063.421875          5714.866163        72.0  71.198111   
890613.445433      139287.000000          6080.901729        64.0  71.568496   
890647.058878      146586.531250          5635.672403        74.0  71.263878   
890663.865601      142778.984375          6309.072774        60.0  71.162771   
890680.672323      144911.531250          6049.032423        56.0  71.363782   
...                          ...                  ...         ...        ...   
892697.479046      138200.750000          6618.951488        57.0  71.166981   
892714.285769      143014.484375          5864.359223        70.0  70.982520   
892747.899214      137755.140625          5672.647469        49.0  70.430010   
892764.705937      141625.421875          5581.748941        65.0  70.649551   
892781.512660      137080.796875          6049.570394        68.0  70.968746

                on_com_j  on_fwhm_i  on_fwhm_j  off_intensity_mean  \

delay_bin  
890596.638710 51.767022 50.280028 69.659566 143473.640625  
890613.445433 52.034099 49.934316 70.114963 147393.453125  
890647.058878 51.770082 50.117312 69.260940 138924.390625  
890663.865601 52.879383 52.261808 71.381020 139712.156250  
890680.672323 51.772942 50.078841 69.663883 145630.921875  
... ... ... ... ...  
892697.479046 52.737284 51.865622 71.183383 143169.328125  
892714.285769 53.437609 53.192670 72.166016 144281.296875  
892747.899214 55.600430 57.649685 76.204099 144496.796875  
892764.705937 54.172832 54.853314 73.600426 143429.234375  
892781.512660 52.741135 52.069320 71.333536 149189.046875

               off_intensity_stderr  off_n_shots  off_com_i  off_com_j  \

delay_bin  
890596.638710 6334.963309 52.0 71.205150 51.846850  
890613.445433 6827.275557 57.0 71.346031 51.552704  
890647.058878 5925.720370 61.0 71.205994 52.816548  
890663.865601 6919.280928 61.0 71.087663 53.720580  
890680.672323 6369.125210 52.0 71.249143 51.773438  
... ... ... ... ...  
892697.479046 7219.295342 50.0 71.534665 51.322592  
892714.285769 6987.816800 53.0 71.133122 52.758881  
892747.899214 7523.321369 34.0 70.722398 54.490950  
892764.705937 7052.736121 51.0 70.961802 52.922404  
892781.512660 6001.693333 47.0 70.850774 52.198029

               off_fwhm_i  off_fwhm_j

delay_bin  
890596.638710 50.251148 69.652775  
890613.445433 49.592014 68.970615  
890647.058878 52.141398 71.468876  
890663.865601 53.676705 72.883888  
890680.672323 50.381550 69.275338  
... ... ...  
892697.479046 48.686635 68.426662  
892714.285769 52.087793 71.301205  
892747.899214 55.366924 74.361926  
892764.705937 52.526665 71.570295  
892781.512660 51.789409 70.331759

[100 rows x 14 columns]
```

</details>

Here the index of the dataframe is the left edge of the delay bin.

### Equilibrium correlation

Check out `calc_static_correlation`

### Nonequilibrium correlations

In nonequilibrium, we need to be careful to only correlate images that are not
too separated in time.

Furthermore, the quantity of interest is slightly different. In equilibrium, we
are interested in asking: what is the typical correlation between images
separated by a certain time, $\tau$? But in nonequilibrium, we want to ask: how
correlated is the typical image at delay time $t$ to the typical pump-off image?
To make it even more complicated, we need to be careful to consider the
reversibility of the pump-induced dynamics. For example, some systems obey a
constraint which we call "strong reversibility": at every delay time $t$, the
spatial distribution of scatterers is exactly the same for every shot with that
delay time. Other systems obey "weak reversibility", where the strong
reversibility constraint is true only for the shots before zero delay. The "No
reversibility" case occurs when the system is simply never the same after the
pump has hit the sample.

The reversibility setting dictates how much averaging we're allowed to do of the
images before we compute a correlation. For example, in the case of strong
reversibility:

1. Categorize each image by
    1. what labtime-bin it belongs to, and
    2. what delay bin in belongs to.
2. For each labtime bin
    1. compute the typical laser off image
    2. for each delay bin in that labtime bin, compute the typical laser on
       image
    3. correlate each of the laser on images with the laser off image
3. Average the correlation vs delay plots over labtime bins

In the case of weak reversibility, you can average the laser off images, in the
labtime bins, but you have to compute the correlation between individual pump-on
shots and that average laser off image, and then average THOSE quantities

In the case of no reversibility, each shot needs to be correlated with the laser
off shot that came before it.

In short: yikes

Anyways this is all implemented in `differential_correlation`, look at the documentation for the typical signature!

There is also `differential_twotime` defined as well.
```
