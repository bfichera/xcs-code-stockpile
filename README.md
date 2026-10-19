# xcs-code-stockpile

A collection of calculations of correlations, among others! For the XCS hutch at
LCLS.

## Various definitions defined

### ROIs

Multiple ways of reporting ROIs are allowed. For example, I am partial to using

```
roi = (slice(i_start, i_end), slice(j_start, j_end))
data = stack[..., *roi]
```

But others like

```
roi = [x0, x1, y0, y1]
data = stack[..., roi[2]:roi[3], roi[0]:roi[1]]
```

Why you would prefer the second choice is beyond me. But the good news is that
here we detect what kind of ROI you are trying to use and try to use it, or
`raise` when it can't, so everyone can do what they prefer!

> [!WARNING] In all cases though, the ROI coordinates are defined relative to
> the `smalldata` frame, which!!!! might not be the same as the detector
> coordinate system!

### Equilibrium correlations

There are WAY too many ways of defining correlations in the XPCS literature.
It's important that we clearly differentiate them.

For example, let `stack` be a collection of frames, of shape
`(n_frames, n_pi, n_pj)`

We can normalize the frames in various ways, like with a savitsky-golov filter,
or a fitted gaussian, etc. Also, the frames should all be pre-normalized by
`i0`.

Then we have a choice of what kind of twotime to compute. When should we use
which one? Who knows!

```
def _twotime_option1(stack):
    wf = stack.reshape(stack.shape[0], -1).T  # n_pixels x n_frames
    mn = np.mean(wf, axis=0)
    numerator = (wf.T @ wf) / wf.shape[0]
    denominator = mn[:, None] * mn
    return numerator / denominator

def _twotime_option2(stack):
    wf = stack.reshape(stack.shape[0], -1).T  # n_pixels x n_frames
    mn = np.mean(wf, axis=0)
    std = np.std(wf, axis=0)
    numerator = (wf.T @ wf) / wf.shape[0] - mn[:, None] * mn
    denominator = std[:, None] * std
    return numerator / denominator
```

Only once we have the twotime do we then look at it, see if it's more or less
constant along the diagonals (it should be, if the system is truly in
equilibrium), and then average the off-diagonals.

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

### Normal time-resolved XRD stuff

In addition to the XPCS information, we also get all the tr-XRD information for
free, like: how does the peak COM change vs delay time? What about its height
and FWHM? All of this should be reported for each run number.

Also, we should be able to quickly plot a rocking curve

### Filtering and sanity checks

Sometimes, there are shots with not enough x-ray photons. Other times there are
shots with too many photons! Obviously, we have to make sure we filter out the
bad shots. But it's also often useful at this stage to ask questions like: is
my average intensity evolving over time? How does the scatter plot of measured
intensity vs. `i0` look like? A pandas dataframe is usually the best way to
implement this so that's how it's done here.
