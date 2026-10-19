# Regions of interest

Functions that accept `roi` support two equivalent forms.

## Slice form

Use a `(row_slice, column_slice)` tuple:

```python
roi = (slice(row_start, row_stop), slice(column_start, column_stop))
selected = stack[..., *roi]
```

Slices use normal Python semantics. Omit `roi` or pass `None` to select the complete stored frame.

## Coordinate-list form

Use four coordinates ordered as `[x0, x1, y0, y1]`, where `x` is the detector column and `y` is the detector row:

```python
roi = [x0, x1, y0, y1]
selected = stack[..., y0:y1, x0:x1]
```

The stop coordinates are excluded, as with normal Python slices.

!!! warning "Coordinate system"
    Coordinates are relative to the frame in the smalldata detector dataset. If smalldata already stores a cropped frame, these do not refer directly to the full detector.

## Consistency

It is usually best to use the same ROI when creating metadata and running an analysis. Otherwise, `intensity_in_roi` and the frame data used by the analysis describe different detector regions.
