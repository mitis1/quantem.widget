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
    refine_all_peaks=True,        # choose_basis_vectors(...)
    threshold_percentile=0.98,
    min_peak_spacing=10,
    refine_radius_px=1.5,
    fit_method="cft",             # fit_lattice(method=..., refine_radius_px=...)
    fit_refine_radius_px=3,
    device="cuda:0",              # fit_lattice(device=...)
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
| Lattice basis | `choose_basis_vectors` | `g1`, `g2`, `refine_gaussian`, `refine_radius_px`, `refine_all_peaks`, `threshold_percentile`, `min_peak_spacing` |
| Per-position fit | `fit_lattice`, `calculate_strain_map` | `fit_method`, `fit_refine_radius_px` (defaults to `refine_radius_px`), `refine_dft`, `upsample`, `batch_size`, `device` |
| Reference lattice | `StrainMap.update_reference` | `reference_region`, `calculation_metric` |
| Display frame | `StrainMap.rotate_strain` (`plot_strain(rotation_angle=)`) | `rotation_angle` |

## Layout

The widget has two large panels:

- **Left:** the *Cepstral transform*, where you pick the lattice, and everything in
  scan space.
  - *Real space* is a virtual ADF or BF image of the scan, computed when you press
    *Compute ADF Image* because it reads every pattern once. Its detector comes from
    quantem's `estimate_central_beam` on the mean pattern: the bright-field disk is
    that radius plus 2 px, and ADF is everything outside it, as in the strain
    notebooks. The sum runs in blocks of scan rows, so a memory-mapped scan is never
    copied whole.
  - `e_uu`, `e_vv`, `e_uv`, `planar` (`e_uu + e_vv`, as
    `plot_strain(plot_dilation=True)`) and `rotation`. These tabs open after the
    first fit, and the panel then switches to `e_uu`.
- **Right:** diffraction space, to compare against: the *Mean diffraction* pattern (log
  scale, with the detector mask) and its *Fourier transform*. The Fourier transform is
  the plain |FFT| of the mean pattern, with no intensity scaling or mask. It uses the
  cepstral transform's zoom, pan and Contrast and shows the same g1/g2. The
  *Diffraction panel* switch (`show_diffraction`) hides this panel, leaving only the
  left one.

Under the panels, all five components and the fit weight are shown side by side.
Click one to show it large on the left.

The strain maps follow `plot_strain`:

- The ranges are symmetric, so zero is white. The colormaps are `plot_strain`'s
  defaults: `RdBu_r` for strain and `PiYG` for rotation.
- Each map has a colorbar, and a scale bar in scan units when it is wide enough.
- `e_uu`, `e_vv` and `e_uv` are the tensor in the frame rotated by `rotation_angle`.
  At 0 they equal `StrainMap.e_rr`, `e_cc` and `e_rc`.
- **Mask** turns on the `plot_strain(mask_range=)` dimming: black at or below `low`,
  full color at or above `high`. The dimming follows the Range slider while you drag.
  It is off by default, with a window of (0, 0.01), so switching it on changes almost
  nothing until the slider moves. Passing `mask_range=` turns it on.

After a fit you can keep changing any preprocessing, lattice or fit setting. The maps
dim and are marked stale; *Refit* fits again with the new settings, and the reference
region carries over.

Vectors are `(row, col)` offsets from the transform centre `(dp_row // 2, dp_col // 2)`
in transform pixels of the unrotated detector frame. This is the convention of
`choose_basis_vectors(define_in_rotated_frame=False)`. Zoom and pan never change them.
`threshold_percentile` is a fraction (0.98), while `mask_threshold_percentile` is a
percent (50), matching the quantem arguments. The data are never copied, cropped,
binned or cast. `skip` subsamples only the mean transform, exactly as `preprocess(skip=)`
does, and the fit always uses every position. `device="cuda"` or `"cuda:N"` raises when
that GPU is unavailable and never falls back to the CPU. Warnings from quantem, such as
missing rotation metadata or a clamped `edge_blend`, are shown under the status line.

ShowStrain needs a quantem build that includes `StrainMapAutocorrelation`. Without it,
constructing the widget raises `ImportError`, while importing `quantem.widget` still works.

## Memory

quantem's `fit_lattice(save_to_gpu=True)` copies the whole scan to the GPU, about
19 GB for a 512 x 512 x 192 x 192 float32 scan. ShowStrain releases that copy and returns
the allocator's cached blocks as soon as each fit ends, including when the fit fails or is
interrupted. You don't need to restart the kernel or call `free()` yourself.

- `keep_on_gpu=True` keeps the copy between fits for faster refits. It is then released
  after `gpu_idle_timeout_s` (default 600 s) with no fit or parameter change.
- `widget.free()` releases the copy now; the widget stays usable.
- `widget.close()` also drops the workflow and closes the widget.
- Rerunning a `ShowStrain(dataset)` cell closes the widget it replaces (same input object)
  and prints that it did. `close_previous=False` keeps both.

Interrupting a step (the kernel's stop button) never leaves quantem's arrays alive.
IPython keeps the last traceback for `%debug`, and quantem's frames in it would hold
full-size arrays, such as a float64 copy of the scan in `preprocess`, until the kernel
restarted. ShowStrain re-raises the interrupt without those frames.

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
| Detector mask / Below / Feather | `diffraction_mask`, `mask_threshold_percentile`, `mask_edge_blend` | Cyan preview of masked pixels on the diffraction panel while dragging; red feathered mask after release |
| Gaussian / Radius | `refine_gaussian`, `refine_radius_px` | Peak refinement model and window (dashed circle at each seed) |
| Multi-peak | `refine_all_peaks` | Basis from every detected peak by weighted least squares; yellow circles sized by weight |
| Threshold / Spacing | `threshold_percentile`, `min_peak_spacing` | Dashed cyan candidates while dragging; refined peaks after release |
| Contrast | browser only | Transform display maximum (default: brightest lattice peak) |
| Method / Fit radius / DFT | `fit_method`, `fit_refine_radius_px`, `refine_dft` | Settings of the next fit; mark an existing fit stale |
| Fit / Refit N Positions | `fit_request` | Runs the per-position fit on `device` and opens the strain tabs |
| Left tabs: Cepstral transform / Real space / e_uu / e_vv / e_uv / planar / rotation | browser only | Switches the left panel; the strain tabs open after a fit |
| Compute ADF Image / Detector | `real_space_mode` | Computes the virtual ADF or BF image on the Real space tab |
| Right tabs: Mean diffraction / Fourier transform; Diffraction panel | `show_diffraction` | Switches or hides the diffraction-space panel |
| Progress bar (during a fit) | `fit_progress` | Positions done / total, elapsed and estimated time left, from quantem's batch loop |
| Peak color / Line / Size | `peak_color`, `peak_line_width`, `peak_size` | Style of the fitted-peak circles (and the dashed preview) |
| Strain ± / Rotation ± | `strain_range_percent`, `rotation_range_degrees` | Live symmetric color range of the strain (%) and rotation (deg) maps and their colorbars |
| Mask / Range | `apply_mask`, `mask_range` | Off by default; on, low-weight positions dim live with the Range slider, as `plot_strain(mask_range=)` |
| Rotation angle | `rotation_angle` | Strain tensor in a rotated frame (`e_uu`, `e_vv`, `e_uv`); no refit |
| Drag a box on a strain or real-space map | `reference_region` | Dashed box on every scan map while dragging; release recomputes the strain against that region |
| Metric / Reset Reference | `calculation_metric`, `reference_region` | Median or mean reference; back to the weighted default |
| Hover transform / strain map | browser only | Offset, length (px and real units), intensity; scan position and value |
| Wheel / drag / double-click | view only | Zoom about the cursor / pan / reset to `cropping_factor` |
| Export HTML | `export_request` | Standalone page with the float32 views; the 4D scan is not embedded |
