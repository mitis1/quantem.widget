# ShowStrain

Map strain across a 4D-STEM scan with the cepstral (autocorrelation) method of
`quantem.diffraction.StrainMapAutocorrelation`. Every pattern is intensity-scaled
and Fourier transformed. In the transform, the crystal periodicity shows up as a
lattice of peaks whose positions track the real-space lattice. You pick two lattice
vectors on the mean transform, tune the preprocessing and the multi-peak refinement
with sliders, and fit every scan position for the strain maps.

```python
from quantem.widget import ShowStrain

widget = ShowStrain(
    dataset,                      # Dataset4dstem, StrainMapAutocorrelation, or a 4D array
    mode="gamma", gamma=0.5,      # preprocess(...)
    refine_all_peaks=True,        # choose_lattice_vector(...)
    threshold_percentile=0.98,
    min_peak_spacing=10,
    refine_radius_px=1.5,
    fit_method="cft",             # fit_lattice_vectors(method=..., refine_radius_px=...)
    fit_refine_radius_px=3,
    device="cuda:0",              # fit_lattice_vectors(device=...)
)
```

Click two lattice peaks on the transform, or pass `g1=(10, 10), g2=(-10, 10)`, then
press **Fit** (or call `widget.fit()`):

```python
strain = widget.fit()             # quantem StrainMap
strain.plot_strain()              # the usual quantem figures and saving
widget.g1_refined, widget.g2_refined  # refined basis on the mean transform
widget.peaks, widget.peak_weights     # peaks in the multi-peak least squares
widget.workflow                       # the StrainMapAutocorrelation, for further steps
```

Drag a box on any strain map to choose the reference region. This is
`update_reference(strain_mask=...)` with `ref_mask[r0:r1, c0:c1] = True`, and it
recomputes the strain from the existing fit without refitting. The bottom 100 rows
of a 512-row scan, for example, are `reference_region=[412, 512, 0, 512]`. Without a
region, the reference is quantem's default: the median weighted by the fit amplitude
(`mask_weight`).

## What runs where

All science runs in quantem. Each committed control is the same-named keyword of
one `StrainMapAutocorrelation` call:

| Step | quantem call | Traits |
|---|---|---|
| Detector mask (optional) | `diffraction_mask` | `diffraction_mask`, `mask_threshold_percentile`, `mask_edge_blend` |
| Mean transform | `preprocess` | `mode`, `gamma`, `log_offset`, `log_offset_relative`, `hann_window`, `skip`, `q_to_r_rotation_ccw_deg`, `q_transpose` |
| Lattice basis | `choose_lattice_vector` | `g1`, `g2`, `refine_gaussian`, `refine_radius_px`, `refine_all_peaks`, `threshold_percentile`, `min_peak_spacing` |
| Per-position fit | `fit_lattice_vectors`, `calculate_strain_map` | `fit_method`, `fit_refine_radius_px` (defaults to `refine_radius_px`), `refine_dft`, `upsample`, `batch_size`, `device` |
| Reference lattice | `StrainMap.update_reference` | `reference_region`, `calculation_metric` |
| Display frame | `StrainMap.rotate_strain` (`plot_strain(rotation_angle=)`) | `rotation_angle` |

The strain panels follow `plot_strain`. The ranges are symmetric, so zero strain
is white in `RdBu_r`. Each panel has a colorbar, and a scale bar in scan units when
the panel is wide enough. `mask_range` dims low-weight positions exactly as
`plot_strain(mask_range=)` does: black at or below `low`, full color at or above
`high`. A non-zero `rotation_angle` shows `e_uu`, `e_vv` and `e_uv` instead of
`e_rr`, `e_cc` and `e_rc`.

Vectors are `(row, col)` offsets from the transform centre `(dp_row // 2, dp_col // 2)`
in transform pixels of the unrotated detector frame. This is the convention of
`choose_lattice_vector(define_in_rotated=False)`. Zoom and pan never change them.
`threshold_percentile` is a fraction (0.98), while `mask_threshold_percentile` is a
percent (50), matching the quantem arguments. The data are never copied, cropped,
binned or cast. `skip` subsamples only the mean transform, exactly as `preprocess(skip=)`
does, and the fit always uses every position. `device="cuda"` or `"cuda:N"` raises when
that GPU is unavailable and never falls back to the CPU. Warnings from quantem, such as
missing rotation metadata or a clamped `edge_blend`, are shown under the status line.

ShowStrain needs a quantem build that includes `StrainMapAutocorrelation`. Without it,
constructing the widget raises `ImportError`, while importing `quantem.widget` still works.

## Memory

quantem's `fit_lattice_vectors(save_to_gpu=True)` copies the whole scan to the GPU, about
19 GB for a 512 x 512 x 192 x 192 float32 scan. ShowStrain releases that copy and returns
the allocator's cached blocks as soon as each fit ends, including when the fit fails or is
interrupted. You don't need to restart the kernel or call `free()` yourself.

- `keep_on_gpu=True` keeps the copy between fits for faster refits. It is then released
  after `gpu_idle_timeout_s` (default 600 s) with no fit or parameter change.
- `widget.free()` releases the copy now; the widget stays usable.
- `widget.close()` also drops the workflow and closes the widget.
- Rerunning a `ShowStrain(dataset)` cell closes the widget it replaces (same input object)
  and prints that it did. `close_previous=False` keeps both.

torch's CUDA context (about 0.5 GB per GPU it touched) stays until the kernel exits.

## Previews and commits

Dragging a slider or a vector changes only the browser view. The slider value, the
dashed candidate peaks, the masked detector pixels and the lattice the dragged basis
predicts are redrawn once per animation frame. Releasing commits the trait, and
quantem reruns only the steps that trait feeds. The candidate peaks and masked
pixels in the preview use the same rule as quantem on the float32 image the browser
holds. The committed peaks, weights and vectors always come from Python. Changing
any parameter after a fit dims the strain maps and marks them stale; `strain_map`
returns `None` until you fit again.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.showstrain.ShowStrain
   :members: fit, workflow, strain_map, peaks, peak_weights, state_dict, load_state_dict, save, export_html
   :show-inheritance:
```

## Interactive controls

| Control | Trait | Expected effect |
|---|---|---|
| Click the transform (a vector missing) | `g1`, then `g2` | Places the seed; quantem refines it and the refined arrow replaces it |
| Drag a vector tip | `g1` / `g2` | Arrow and predicted lattice follow the pointer; release refines from the new seed |
| Clear Vectors | `g1`, `g2` | Removes both vectors and the peak fit |
| Mode / Gamma / Offset / Relative | `mode`, `gamma`, `log_offset`, `log_offset_relative` | Recomputes the mean transform on release |
| Hann | `hann_window` | Hann window before the FFT (needed by `cft` and DFT refinement) |
| Skip | `skip` | Mean transform over every n-th scan row and column |
| Mask / Mask below / Feather | `diffraction_mask`, `mask_threshold_percentile`, `mask_edge_blend` | Cyan preview of masked pixels while dragging; red feathered mask after release |
| Gaussian / Radius | `refine_gaussian`, `refine_radius_px` | Peak refinement model and window (dashed circle at each seed) |
| Multi-peak | `refine_all_peaks` | Basis from every detected peak by weighted least squares; yellow circles sized by weight |
| Threshold / Spacing | `threshold_percentile`, `min_peak_spacing` | Dashed cyan candidates while dragging; refined peaks after release |
| Contrast | browser only | Transform display maximum (default: brightest lattice peak) |
| Method / Fit radius / DFT | `fit_method`, `fit_refine_radius_px`, `refine_dft` | Settings of the next fit; mark an existing fit stale |
| Fit N Positions | `fit_request` | Runs the per-position fit on `device` and shows `e_rr`, `e_cc`, `e_rc`, `phi`, weight |
| Progress bar (during a fit) | `fit_progress` | Positions done / total, elapsed and estimated time left, from quantem's batch loop |
| Peak color / Line / Size | `peak_color`, `peak_line_width`, `peak_size` | Style of the fitted-peak circles (and the dashed preview) |
| Strain ± / Rotation ± | `strain_range_percent`, `rotation_range_degrees` | Live symmetric color range of the strain (%) and rotation (deg) maps and their colorbars |
| Mask | `mask_range` | Live dimming of low-weight positions, as `plot_strain(mask_range=)` |
| Frame | `rotation_angle` | Strain tensor in a rotated frame (`e_uu`, `e_vv`, `e_uv`); no refit |
| Drag a box on a strain map | `reference_region` | Dashed box on all five maps while dragging; release recomputes the strain against that region |
| Metric / Reset Reference | `calculation_metric`, `reference_region` | Median or mean reference; back to the weighted default |
| Hover transform / strain map | browser only | Offset, length (px and real units), intensity; scan position and value |
| Wheel / drag / double-click | view only | Zoom about the cursor / pan / reset to `cropping_factor` |
| Export HTML | `export_request` | Standalone page with the float32 views; the 4D scan is not embedded |
