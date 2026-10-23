"""ShowStrain: interactive cepstral (autocorrelation) strain mapping of a 4D-STEM scan.

The widget drives quantem's ``StrainMapAutocorrelation`` and keeps every scientific step
in that class: the detector mask, the intensity transform of each pattern, the refinement
of the picked lattice vectors against the mean transform (optionally against every
detected peak), the per-position fit and the strain tensor. The browser shows the mean
diffraction pattern, the mean transform with the lattice overlay, and the strain maps,
and turns clicks, drags and slider releases into the same keyword arguments the
notebook API takes.
"""

import base64
import contextlib
import inspect
import io
import json
import math
import pathlib
import time
import warnings
import weakref

import anywidget
import numpy as np
import traitlets
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from quantem.widget.adapters import core as core_adapter
from quantem.widget.colormap import colormap_lut
from quantem.widget.device import resolve_device
from quantem.widget.export import HtmlExportMixin, ensure_mobile_viewport, export_slug
from quantem.widget.fallback import StaticFallbackMixin
from quantem.widget.state import SavedStateMixin
from quantem.widget.utils.array import _b64_safe, to_numpy
from quantem.widget.utils.state_io import resolve_widget_version, unwrap_state_payload

_REQUIREMENT = (
    "ShowStrain needs quantem with quantem.diffraction.StrainMapAutocorrelation "
    "(choose_lattice_vector(g1=..., g2=..., min_peak_spacing=...)). Install a quantem build that includes it."
)
MODES = ("linear", "log", "log-min", "gamma")
FIT_METHODS = ("gaussian", "cft")
# The shown strain components, in the order of strain_bytes: plot_strain's e_uu, e_vv, e_uv in the
# frame rotated by rotation_angle (e_rr, e_cc, e_rc at 0), the planar strain e_uu + e_vv
# (plot_strain(plot_dilation=True)), and the rotation phi (radians).
STRAIN_COMPONENTS = ("e_uu", "e_vv", "e_uv", "planar", "phi")
REAL_SPACE_MODES = ("adf", "bf")
# The virtual bright-field disk is the central beam grown by this margin, as in the strain notebooks.
_BF_MARGIN_PX = 2.0
# |sin| of the angle between the picked vectors below which they are refused as collinear (~3 degrees).
_MIN_BASIS_SINE = 0.05

# A committed change to one of these reruns the whole chain (mask, transform, lattice).
_PREPROCESS_TRAITS = (
    "mode", "gamma", "log_offset", "log_offset_relative", "hann_window", "skip",
    "diffraction_mask", "mask_threshold_percentile", "mask_edge_blend",
    "q_to_r_rotation_ccw_deg", "q_transpose",
)
# A committed change to one of these refines the lattice against the existing transform.
_LATTICE_TRAITS = (
    "g1", "g2", "refine_gaussian", "refine_radius_px", "refine_all_peaks",
    "threshold_percentile", "min_peak_spacing",
)
# These only change how the next fit runs.
_FIT_TRAITS = ("fit_method", "fit_refine_radius_px", "refine_dft", "upsample", "batch_size")
# The reference lattice of StrainMap.update_reference and the plot_strain frame rotation;
# they recompute the strain from the existing fit, never the fit.
_REFERENCE_TRAITS = ("reference_region", "calculation_metric", "rotation_angle")
# Display-only settings; saved with the state but never recomputed.
_DISPLAY_TRAITS = (
    "cropping_factor", "strain_range_percent", "rotation_range_degrees", "apply_mask", "mask_range", "title",
    "peak_color", "peak_line_width", "peak_size", "show_diffraction",
)
_STATE_KEYS = _PREPROCESS_TRAITS + _LATTICE_TRAITS + _FIT_TRAITS + _REFERENCE_TRAITS + _DISPLAY_TRAITS
CALCULATION_METRICS = ("median", "mean")
# What the saved-notebook preview shows; a change to any of these redraws it at the next save.
_PREVIEW_TRAITS = ("transform_bytes", "g1_refined", "g2_refined", "peaks_bytes", "strain_bytes", "cropping_factor")


def _workflow_class():
    """quantem's StrainMapAutocorrelation, imported on first use.

    quantem core is optional for quantem.widget (it does not install on an Intel Mac or
    Windows on ARM), so this module imports without it.
    """
    try:
        from quantem.diffraction import StrainMapAutocorrelation
    except ImportError as exc:
        raise ImportError(_REQUIREMENT) from exc
    params = inspect.signature(StrainMapAutocorrelation.choose_lattice_vector).parameters
    if not {"g1", "g2", "min_peak_spacing", "threshold_percentile"} <= set(params):
        raise ImportError(_REQUIREMENT)
    return StrainMapAutocorrelation


def _as_workflow(data):
    """A StrainMapAutocorrelation for ``data``: passed through, or built from a dataset or 4D array."""
    cls = _workflow_class()
    if isinstance(data, cls):
        return data
    if core_adapter.is_dataset(data, ndim=4):
        from quantem.core.datastructures import Dataset4dstem

        if isinstance(data, Dataset4dstem):
            return cls.from_dataset(data)
        data = core_adapter.as_array(data)
    array = to_numpy(data)
    if array.ndim != 4:
        raise ValueError(
            f"ShowStrain expects a 4D-STEM dataset (scan_row, scan_col, dp_row, dp_col); got shape {array.shape!r}."
        )
    return cls.from_array(array)


def _real_units(label: str) -> str:
    """quantem's mathtext Angstrom label as plain text for the browser."""
    return "Å" if "AA" in label or label in {"A", "Å"} else label


def _vector_or_none(value):
    """A length-2 trait list as a float tuple, or None when it is empty."""
    return (float(value[0]), float(value[1])) if len(value) == 2 else None


def _base64_buffer(name, data):
    """Embedded-state buffer holding ``data`` for the Bytes trait ``name``."""
    return {"encoding": "base64", "path": [name], "data": base64.b64encode(data).decode("ascii")}


class ShowStrain(SavedStateMixin, HtmlExportMixin, StaticFallbackMixin, anywidget.AnyWidget):
    """Pick lattice vectors on the mean cepstral transform and map strain across a 4D-STEM scan.

    Every scientific step runs in :class:`quantem.diffraction.StrainMapAutocorrelation`,
    with the widget's traits as its keyword arguments:

    1. ``diffraction_mask(threshold_percentile=mask_threshold_percentile,
       edge_blend=mask_edge_blend)`` when ``diffraction_mask`` is on;
    2. ``preprocess(mode, gamma, log_offset, log_offset_relative, hann_window, skip,
       q_to_r_rotation_ccw_deg, q_transpose)``, the mean transform shown in the centre panel;
    3. ``choose_lattice_vector(g1, g2, refine_gaussian, refine_radius_px, refine_all_peaks,
       threshold_percentile, min_peak_spacing)`` once both vectors are picked;
    4. :meth:`fit`: ``fit_lattice_vectors(...)`` on ``device`` and ``calculate_strain_map()``;
    5. ``StrainMap.update_reference(strain_mask=reference_region, calculation_metric)``
       when a reference region is dragged on the strain maps (no refit).

    A slider release, a dropped vector or a switch reruns the steps it feeds; dragging only
    previews. Vectors are ``(row, col)`` offsets from the transform centre
    ``(dp_row // 2, dp_col // 2)`` in transform pixels of the unrotated detector frame,
    the convention of ``choose_lattice_vector(define_in_rotated=False)``.

    Parameters
    ----------
    data : Dataset4dstem, StrainMapAutocorrelation or array_like
        A 4D-STEM scan ``(scan_row, scan_col, dp_row, dp_col)``: a quantem ``Dataset4dstem``
        (its sampling, units and rotation metadata are used), an existing workflow
        (the widget's parameters replace its mask, transform and lattice), or a NumPy,
        PyTorch or CuPy array. The data are not copied, cropped, binned or cast.
    g1, g2 : tuple of float, optional
        Initial lattice vectors ``(d_row, d_col)``. Without them, click two peaks.
    device : str, default "auto"
        Torch device for :meth:`fit` (``"cuda:N"``, ``"mps"``, ``"cpu"``). An explicit
        GPU that is not available raises; nothing falls back to the CPU.
    title : str, default ""
        Title above the panels.
    save_state : bool, default False
        Embed the images in a saved notebook so it reopens interactive without a kernel.
        Off, the notebook keeps a static picture.
    close_previous : bool, default True
        Close the previous ShowStrain built on the same dataset object (a rerun cell), so the
        replaced widget does not keep its workflow and GPU memory alive.
    keep_on_gpu : bool, default False
        Keep quantem's GPU copy of the scan after a fit for faster refits. Off, it is released
        as soon as each fit ends. Kept, it is released after ``gpu_idle_timeout_s`` (default
        600 s) with no fit or parameter change, or by :meth:`free` / :meth:`close`.
    **kwargs
        Any parameter trait (``mode="gamma"``, ``gamma=0.5``, ``refine_all_peaks=True``,
        ``threshold_percentile=0.98``, ``min_peak_spacing=10``, ...).

    Examples
    --------
    >>> widget = ShowStrain(dataset, mode="gamma", gamma=0.5, g1=(10, 10), g2=(-10, 10),
    ...                     refine_all_peaks=True, device="cuda:0")
    >>> strain = widget.fit()           # or press Fit in the widget
    >>> strain.plot_strain()
    """

    _esm = pathlib.Path(__file__).parent / "static" / "showstrain.js"
    _static_fallback_image = traitlets.Unicode("").tag(sync=True)
    _static_fallback_mime = traitlets.Unicode("image/png").tag(sync=True)
    _UNSAVED_HEAVY_KEYS = (
        "transform_bytes", "dp_bytes", "dp_fft_bytes", "mask_bytes", "strain_bytes", "weight_bytes", "real_space_bytes",
        "export_payload",
    )

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    # The right panel (mean diffraction pattern / its Fourier transform) beside the cepstral transform.
    show_diffraction = traitlets.Bool(True).tag(sync=True)
    offline = traitlets.Bool(False).tag(sync=True)

    # --- data shapes and calibration (Python -> browser) ---
    scan_rows = traitlets.Int(0).tag(sync=True)
    scan_cols = traitlets.Int(0).tag(sync=True)
    dp_rows = traitlets.Int(0).tag(sync=True)
    dp_cols = traitlets.Int(0).tag(sync=True)
    data_dtype = traitlets.Unicode("").tag(sync=True)
    sampling_real = traitlets.List(traitlets.Float(), default_value=[1.0, 1.0]).tag(sync=True)
    real_units = traitlets.Unicode("1/pixels").tag(sync=True)
    scan_sampling = traitlets.Float(1.0).tag(sync=True)
    scan_units = traitlets.Unicode("pixels").tag(sync=True)
    device = traitlets.Unicode("cpu").tag(sync=True)

    # --- preprocessing (browser -> Python; quantem's keyword names) ---
    mode = traitlets.Unicode("linear").tag(sync=True)
    gamma = traitlets.Float(0.5).tag(sync=True)
    log_offset = traitlets.Float(0.1).tag(sync=True)
    log_offset_relative = traitlets.Bool(False).tag(sync=True)
    hann_window = traitlets.Bool(False).tag(sync=True)
    skip = traitlets.Int(1).tag(sync=True)
    diffraction_mask = traitlets.Bool(False).tag(sync=True)
    mask_threshold_percentile = traitlets.Float(50.0).tag(sync=True)
    mask_edge_blend = traitlets.Float(64.0).tag(sync=True)
    q_to_r_rotation_ccw_deg = traitlets.Float(None, allow_none=True).tag(sync=True)
    q_transpose = traitlets.Bool(None, allow_none=True).tag(sync=True)

    # --- lattice picking and multi-peak refinement ---
    g1 = traitlets.List(traitlets.Float(), default_value=[], maxlen=2).tag(sync=True)
    g2 = traitlets.List(traitlets.Float(), default_value=[], maxlen=2).tag(sync=True)
    refine_gaussian = traitlets.Bool(True).tag(sync=True)
    refine_radius_px = traitlets.Float(2.0).tag(sync=True)
    refine_all_peaks = traitlets.Bool(False).tag(sync=True)
    threshold_percentile = traitlets.Float(0.9975).tag(sync=True)
    min_peak_spacing = traitlets.Float(0.0).tag(sync=True)

    # --- per-position fit ---
    fit_method = traitlets.Unicode("gaussian").tag(sync=True)
    fit_refine_radius_px = traitlets.Float(None, allow_none=True).tag(sync=True)
    refine_dft = traitlets.Bool(False).tag(sync=True)
    upsample = traitlets.Int(16).tag(sync=True)
    batch_size = traitlets.Int(None, allow_none=True).tag(sync=True)
    fit_request = traitlets.Int(0).tag(sync=True)

    # --- reference lattice ---
    reference_region = traitlets.List(traitlets.Int(), default_value=[], maxlen=4).tag(sync=True)
    calculation_metric = traitlets.Unicode("median").tag(sync=True)
    rotation_angle = traitlets.Float(0.0).tag(sync=True)
    reference_vectors = traitlets.List(traitlets.List(traitlets.Float()), default_value=[]).tag(sync=True)
    reference_message = traitlets.Unicode("").tag(sync=True)

    # --- display only ---
    cropping_factor = traitlets.Float(0.25).tag(sync=True)
    strain_range_percent = traitlets.List(traitlets.Float(), default_value=[-3.0, 3.0], minlen=2, maxlen=2).tag(sync=True)
    rotation_range_degrees = traitlets.List(traitlets.Float(), default_value=[-2.0, 2.0], minlen=2, maxlen=2).tag(sync=True)
    # plot_strain(mask_range=...) dimming of low-weight positions, applied only while apply_mask is on.
    # Off by default; passing mask_range turns it on. The default window starts at (0, 0.01), so
    # switching it on changes almost nothing until the slider moves.
    apply_mask = traitlets.Bool(False).tag(sync=True)
    mask_range = traitlets.List(traitlets.Float(), default_value=[0.0, 0.01], minlen=2, maxlen=2).tag(sync=True)
    # Virtual real-space image of the scan: "" until first asked for, then "adf" or "bf".
    real_space_mode = traitlets.Unicode("").tag(sync=True)
    real_space_bytes = traitlets.Bytes(b"").tag(sync=True)
    real_space_info = traitlets.Unicode("").tag(sync=True)
    peak_color = traitlets.Unicode("#ffd54f").tag(sync=True)
    peak_line_width = traitlets.Float(1.5).tag(sync=True)
    peak_size = traitlets.Float(1.0).tag(sync=True)

    # --- results (Python -> browser) ---
    dp_bytes = traitlets.Bytes(b"").tag(sync=True)
    # |FFT| of the mean diffraction pattern (fft-shifted, no intensity scaling or mask): the plain
    # Fourier transform to compare the cepstral transform against.
    dp_fft_bytes = traitlets.Bytes(b"").tag(sync=True)
    dp_fft_vmax = traitlets.Float(1.0).tag(sync=True)
    mask_bytes = traitlets.Bytes(b"").tag(sync=True)
    transform_bytes = traitlets.Bytes(b"").tag(sync=True)
    transform_vmax = traitlets.Float(1.0).tag(sync=True)
    g1_refined = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    g2_refined = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    n_peaks = traitlets.Int(0).tag(sync=True)
    peaks_bytes = traitlets.Bytes(b"").tag(sync=True)
    strain_bytes = traitlets.Bytes(b"").tag(sync=True)
    weight_bytes = traitlets.Bytes(b"").tag(sync=True)
    fit_stale = traitlets.Bool(False).tag(sync=True)
    computing = traitlets.Bool(False).tag(sync=True)
    status_message = traitlets.Unicode("").tag(sync=True)
    warning_messages = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    timings_ms = traitlets.Dict(default_value={}).tag(sync=True)
    # {"done": positions fitted, "total": positions, "elapsed_s": seconds} while a fit runs; {} otherwise
    fit_progress = traitlets.Dict(default_value={}).tag(sync=True)

    # --- memory ---
    # Keep quantem's GPU copy of the scan between fits (faster refits); released after gpu_idle_timeout_s.
    keep_on_gpu = traitlets.Bool(False)
    gpu_idle_timeout_s = traitlets.Float(600.0)

    # Open widgets by the id of their dataset, so rerunning a cell closes the one it replaces.
    _open_by_dataset: dict = {}

    def __init__(
        self, data, *, device: str = "auto", save_state: bool = False, close_previous: bool = True, **kwargs
    ) -> None:
        unknown = sorted(k for k in kwargs if k not in set(type(self).class_trait_names()))
        if unknown:
            raise TypeError(f"ShowStrain() got unexpected keyword argument {unknown[0]!r}.")
        self._save_state = bool(save_state)
        self._configure_static_fallback(notebook_preview_format="png")
        self._workflow = _as_workflow(data)
        self._source = data  # what the cell passed in: a rerun passes the same object
        self._strain = None
        self._peaks = None
        self._last_preprocess_key = None
        self._last_lattice_key = None
        self._idle_handle = None
        self._initializing = True
        super().__init__()
        dataset = self._workflow.dataset
        if close_previous:
            self._close_previous(data)
        scan_rows, scan_cols, dp_rows, dp_cols = (int(n) for n in dataset.shape)
        with self.hold_sync():
            self.widget_version = resolve_widget_version()
            self.device = resolve_device(device)
            self.scan_rows, self.scan_cols, self.dp_rows, self.dp_cols = scan_rows, scan_cols, dp_rows, dp_cols
            self.data_dtype = str(dataset.array.dtype)
            self.scan_sampling = float(dataset.sampling[0])
            self.scan_units = str(dataset.units[0])
            self.title = str(kwargs.pop("title", "") or getattr(dataset, "name", "") or "")
            if "mask_range" in kwargs and "apply_mask" not in kwargs:
                kwargs["apply_mask"] = True
            for name, value in kwargs.items():
                as_list = ("g1", "g2", "strain_range_percent", "rotation_range_degrees", "mask_range", "reference_region")
                setattr(self, name, list(value) if name in as_list else value)
            # The pattern diffraction_mask thresholds, computed as it does (float32 for a float32 scan).
            dp_mean = getattr(dataset, "dp_mean", None)
            dp_mean = np.mean(dataset.array, axis=(0, 1)) if dp_mean is None else core_adapter.as_array(dp_mean)
            self.dp_bytes = _b64_safe(np.asarray(dp_mean, dtype=np.float32).tobytes())
            dp_fft = np.abs(np.fft.fftshift(np.fft.fft2(np.asarray(dp_mean, dtype=np.float64))))
            self.dp_fft_bytes = _b64_safe(dp_fft.astype(np.float32).tobytes())
            self.dp_fft_vmax = _brightest_lattice_peak(dp_fft)
        self._initializing = False
        self.observe(self._on_export_request_change, names=["export_request"])
        self.observe(self._invalidate_preview, names=list(_PREVIEW_TRAITS))
        self.observe(self._on_parameter_change, names=list(_PREPROCESS_TRAITS + _LATTICE_TRAITS + _FIT_TRAITS))
        self.observe(self._on_fit_request, names=["fit_request"])
        self.observe(self._on_reference_change, names=list(_REFERENCE_TRAITS))
        self.observe(self._on_real_space_mode, names=["real_space_mode"])
        self._run(preprocess=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @traitlets.validate("mode")
    def _validate_mode(self, proposal):
        """One of the four intensity transforms ``preprocess`` takes."""
        value = str(proposal["value"]).strip().lower()
        if value not in MODES:
            raise traitlets.TraitError(f"mode must be one of {MODES}, got {proposal['value']!r}.")
        return value

    @traitlets.validate("fit_method")
    def _validate_fit_method(self, proposal):
        """``fit_lattice_vectors(method=...)``: "gaussian" or "cft"."""
        value = str(proposal["value"]).strip().lower()
        if value not in FIT_METHODS:
            raise traitlets.TraitError(f"fit_method must be one of {FIT_METHODS}, got {proposal['value']!r}.")
        return value

    @traitlets.validate("calculation_metric")
    def _validate_calculation_metric(self, proposal):
        """``update_reference(calculation_metric=...)``: "median" or "mean"."""
        value = str(proposal["value"]).strip().lower()
        if value not in CALCULATION_METRICS:
            raise traitlets.TraitError(f"calculation_metric must be one of {CALCULATION_METRICS}, got {proposal['value']!r}.")
        return value

    @traitlets.validate("reference_region")
    def _validate_reference_region(self, proposal):
        """Empty (the weighted default) or ``[row0, row1, col0, col1]``, half-open, inside the scan."""
        value = [int(v) for v in proposal["value"]]
        if not value:
            return value
        if len(value) != 4:
            raise traitlets.TraitError(f"reference_region must be [] or [row0, row1, col0, col1], got {value!r}.")
        r0, r1, c0, c1 = value
        if not (0 <= r0 < r1 <= self.scan_rows and 0 <= c0 < c1 <= self.scan_cols):
            raise traitlets.TraitError(
                f"reference_region={value} must satisfy 0 <= row0 < row1 <= {self.scan_rows} "
                f"and 0 <= col0 < col1 <= {self.scan_cols}."
            )
        return value

    @traitlets.validate("g1", "g2")
    def _validate_vector(self, proposal):
        """Empty (not picked) or a finite ``(d_row, d_col)`` inside the transform."""
        value = [float(v) for v in proposal["value"]]
        name = proposal["trait"].name
        if len(value) not in (0, 2) or not all(math.isfinite(v) for v in value):
            raise traitlets.TraitError(f"{name} must be [] or a finite (d_row, d_col), got {value!r}.")
        if value and self.dp_rows and (abs(value[0]) > self.dp_rows // 2 or abs(value[1]) > self.dp_cols // 2):
            raise traitlets.TraitError(f"{name}={tuple(value)} lies outside the {self.dp_rows}x{self.dp_cols} transform.")
        return value

    @traitlets.validate("skip", "upsample")
    def _validate_positive_int(self, proposal):
        """Scan stride and DFT upsampling are whole numbers of at least 1."""
        if int(proposal["value"]) < 1:
            raise traitlets.TraitError(f"{proposal['trait'].name} must be >= 1, got {proposal['value']!r}.")
        return int(proposal["value"])

    @traitlets.validate("threshold_percentile")
    def _validate_threshold(self, proposal):
        """A fraction in (0, 1), as ``choose_lattice_vector`` takes it (not a percent)."""
        value = float(proposal["value"])
        if not 0.0 < value < 1.0:
            raise traitlets.TraitError(f"threshold_percentile is a fraction in (0, 1), got {value!r}.")
        return value

    @traitlets.validate("mask_threshold_percentile")
    def _validate_mask_percentile(self, proposal):
        """A percent in [0, 100], as ``diffraction_mask`` takes it."""
        value = float(proposal["value"])
        if not 0.0 <= value <= 100.0:
            raise traitlets.TraitError(f"mask_threshold_percentile is a percent in [0, 100], got {value!r}.")
        return value

    @traitlets.validate("refine_radius_px", "fit_refine_radius_px", "mask_edge_blend", "gamma", "log_offset")
    def _validate_positive_float(self, proposal):
        """Window radii, feather width, gamma exponent and log offset are positive (a fit radius of None follows refine_radius_px)."""
        if proposal["value"] is None and proposal["trait"].name == "fit_refine_radius_px":
            return None
        value = float(proposal["value"])
        if not (math.isfinite(value) and value > 0):
            raise traitlets.TraitError(f"{proposal['trait'].name} must be a positive number, got {value!r}.")
        return value

    @traitlets.validate("min_peak_spacing")
    def _validate_spacing(self, proposal):
        """Pixels; 0 turns the spacing filter off."""
        value = float(proposal["value"])
        if not (math.isfinite(value) and value >= 0):
            raise traitlets.TraitError(f"min_peak_spacing must be >= 0 pixels, got {value!r}.")
        return value

    @traitlets.validate("mask_range")
    def _validate_mask_range(self, proposal):
        """``plot_strain(mask_range=(low, high))``: weights <= low are black, >= high full color."""
        low, high = (float(v) for v in proposal["value"])
        if not (0.0 <= low < high <= 1.0):
            raise traitlets.TraitError(f"mask_range must satisfy 0 <= low < high <= 1, got {[low, high]!r}.")
        return [low, high]

    @traitlets.validate("peak_line_width", "peak_size")
    def _validate_peak_style(self, proposal):
        """Overlay circle line width (px) and size scale: positive numbers."""
        value = float(proposal["value"])
        if not (math.isfinite(value) and value > 0):
            raise traitlets.TraitError(f"{proposal['trait'].name} must be a positive number, got {value!r}.")
        return value

    @traitlets.validate("cropping_factor")
    def _validate_cropping_factor(self, proposal):
        """Fraction of the transform shown when the view resets; the data are never cropped."""
        value = float(proposal["value"])
        if not 0.0 < value <= 1.0:
            raise traitlets.TraitError(f"cropping_factor must be in (0, 1], got {value!r}.")
        return value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def workflow(self):
        """The ``StrainMapAutocorrelation`` the widget drives, in its current state."""
        return self._workflow

    @property
    def strain_map(self):
        """The ``StrainMap`` from the last :meth:`fit`, or None before a fit or after a parameter change."""
        return None if self.fit_stale else self._strain

    @property
    def peaks(self) -> np.ndarray | None:
        """Refined peak offsets ``(N, 2)`` in transform pixels when ``refine_all_peaks`` is on, else None."""
        return None if self._peaks is None else self._peaks.copy()

    @property
    def peak_weights(self) -> np.ndarray | None:
        """Fitted amplitude of each refined peak ``(N,)``, the weight in the lattice least squares."""
        weights = getattr(self._workflow, "mean_img_weights", None)
        return None if self._peaks is None or weights is None else np.asarray(weights, dtype=float).ravel().copy()

    def fit(self):
        """Fit the lattice vectors at every scan position and compute the strain map.

        Runs ``fit_lattice_vectors`` on ``device`` with the current ``refine_gaussian``,
        ``refine_radius_px``, ``refine_all_peaks``, ``fit_method``, ``refine_dft``,
        ``upsample`` and ``batch_size``, then ``calculate_strain_map()``. The strain maps
        appear under the transform.

        Returns
        -------
        StrainMap
            The strain map, also kept as :attr:`strain_map`.
        """
        if not (self.g1_refined and self.g2_refined):
            raise ValueError("Pick g1 and g2 on the transform (or pass g1=, g2=) before fitting.")
        self._compute("Fitting lattice vectors...", self._fit_and_strain)
        if self.status_message.startswith("Error"):
            raise RuntimeError(self.status_message)
        return self._strain

    def reference_mask(self) -> np.ndarray | None:
        """The boolean ``(scan_row, scan_col)`` ``strain_mask`` of ``reference_region``, or None when unset."""
        if not self.reference_region:
            return None
        r0, r1, c0, c1 = self.reference_region
        mask = np.zeros((self.scan_rows, self.scan_cols), dtype=bool)
        mask[r0:r1, c0:c1] = True
        return mask

    def free(self) -> float:
        """Release quantem's GPU copy of the scan now; the widget stays usable (the next fit reloads it).

        Returns
        -------
        float
            GB released.
        """
        released = self._release_gpu()
        if released:
            print(f"ShowStrain: freed {released:.2f} GB on {self.device}")
        return released

    def close(self) -> None:
        """Release the GPU copy and the workflow, then close the widget (its view stops working)."""
        self._cancel_idle_release()
        if getattr(self, "_workflow", None) is not None:
            self._release_gpu()
        self._workflow = None
        self._source = None
        self._strain = None
        self._peaks = None
        for key, entry in list(type(self)._open_by_dataset.items()):
            if entry() is self:
                del type(self)._open_by_dataset[key]
        super().close()

    def state_dict(self) -> dict:
        """JSON-safe parameters and display settings (no pixels): what ``save`` writes."""
        state = {}
        for key in _STATE_KEYS:
            value = getattr(self, key)
            state[key] = list(value) if isinstance(value, (list, tuple)) else value
        return state

    def load_state_dict(self, state: dict | str | pathlib.Path) -> None:
        """Restore parameters from a ``save`` file, its JSON payload, or a ``state_dict``, then recompute once."""
        if isinstance(state, (str, pathlib.Path)):
            payload = json.loads(pathlib.Path(state).read_text())
            state = unwrap_state_payload(payload, require_envelope=True, expected_widget="ShowStrain")
        else:
            state = unwrap_state_payload(state, expected_widget="ShowStrain")
        self._initializing = True
        try:
            with self.hold_trait_notifications():
                for key, value in state.items():
                    if key in _STATE_KEYS:
                        setattr(self, key, value)
        finally:
            self._initializing = False
        self._run(preprocess=True)

    # ------------------------------------------------------------------
    # Recompute chain
    # ------------------------------------------------------------------
    def _preprocess_key(self):
        return tuple(getattr(self, name) for name in _PREPROCESS_TRAITS)

    def _lattice_key(self):
        return tuple(tuple(v) if isinstance(v, list) else v for v in (getattr(self, n) for n in _LATTICE_TRAITS))

    def _on_parameter_change(self, change):
        """A committed parameter: rerun only the steps it feeds (once per batch of browser edits)."""
        if self._initializing:
            return
        if self._idle_handle is not None:
            self._arm_idle_release()  # still in use: restart the idle clock
        name = change["name"]
        if name in _PREPROCESS_TRAITS and self._preprocess_key() != self._last_preprocess_key:
            self._run(preprocess=True)
        elif name in _LATTICE_TRAITS and self._lattice_key() != self._last_lattice_key:
            self._run(preprocess=False)
        elif name in _FIT_TRAITS:
            self._mark_fit_stale()

    def _on_reference_change(self, change):
        """A dragged reference region, metric or frame angle: recompute the strain from the existing fit."""
        if self._initializing or self._strain is None or self.fit_stale:
            return
        self._compute("Updating reference...", self._apply_reference)

    def _apply_reference(self) -> str:
        """``StrainMap.update_reference`` with the region (or the default weights) and the metric."""
        strain = self._strain
        mask = self.reference_mask()
        strain.update_reference(strain_mask=mask, calculation_metric=self.calculation_metric)
        # plot_strain(rotation_angle=...): the tensor in a frame rotated by the angle; phi is invariant
        e_uu, e_vv, e_uv = strain.rotate_strain(self.rotation_angle)
        planar = e_uu + e_vv
        maps = np.stack([np.asarray(m, dtype=np.float32) for m in (e_uu, e_vv, e_uv, planar, strain.phi.array)])
        if mask is None:
            message = f"Reference: {self.calculation_metric} weighted by the fit amplitude (mask_weight)"
        else:
            r0, r1, c0, c1 = self.reference_region
            message = (f"Reference: {self.calculation_metric} over rows {r0}-{r1 - 1}, cols {c0}-{c1 - 1} "
                       f"({int(mask.sum())} positions)")
        with self.hold_sync():
            self.strain_bytes = _b64_safe(maps.tobytes())
            self.reference_vectors = [[float(v) for v in strain.g1_ref], [float(v) for v in strain.g2_ref]]
            self.reference_message = message
        return message

    def _on_fit_request(self, change):
        """The browser's Fit button."""
        if self._initializing or not (self.g1_refined and self.g2_refined):
            return
        self._compute("Fitting lattice vectors...", self._fit_and_strain)

    def _run(self, *, preprocess: bool) -> None:
        """Rerun preprocessing (when asked) and the lattice refinement, with a busy flag and status."""
        def steps():
            messages = []
            if preprocess:
                messages.append(self._preprocess())
            messages.append(self._refine_lattice())
            return " · ".join(m for m in messages if m)

        self._compute("Computing transform..." if preprocess else "Refining lattice...", steps)

    def _compute(self, busy_message: str, step) -> None:
        """Run ``step`` with ``computing`` set; a quantem warning is shown, an exception becomes the status."""
        self.computing = True
        self.status_message = busy_message
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                message = step()
            self.warning_messages = list(dict.fromkeys(str(w.message) for w in caught))
            self.status_message = message or ""
        except Exception as exc:  # the widget shows it; the kernel keeps running
            exc.__traceback__ = None  # quantem's frames (full-size arrays) must not outlive the error
            self.status_message = f"Error: {exc}"
        except KeyboardInterrupt:
            # IPython keeps the last traceback for %debug; quantem's frames in it hold full-size
            # arrays (a float64 copy of the scan in preprocess) that nothing could free until the
            # kernel restarts. Re-raise without them.
            self.status_message = "Interrupted"
            raise KeyboardInterrupt("ShowStrain step interrupted") from None
        finally:
            self.computing = False

    @traitlets.validate("real_space_mode")
    def _validate_real_space_mode(self, proposal):
        """"" (not computed yet), "adf" or "bf"."""
        value = str(proposal["value"]).strip().lower()
        if value and value not in REAL_SPACE_MODES:
            raise traitlets.TraitError(f"real_space_mode must be one of {REAL_SPACE_MODES}, got {proposal['value']!r}.")
        return value

    def _on_real_space_mode(self, change):
        """The Real space tab asked for (or switched) the virtual image."""
        if self._initializing or not self.real_space_mode or self._workflow is None:
            return
        self._compute("Computing real-space image...", self._real_space_image)

    def _real_space_image(self) -> str:
        """Virtual ADF or BF image of the scan, on the central-beam geometry quantem estimates.

        The detector is quantem's: ``estimate_central_beam`` on the mean pattern, a bright-field
        disk of that radius plus ``_BF_MARGIN_PX``, and the annulus outside it for ADF, as the
        strain notebooks build ``get_virtual_image``. The masked sum runs over blocks of scan rows
        rather than ``array * mask`` at once, so a memory-mapped scan is never copied whole.
        """
        from quantem.diffraction.disk_detection import estimate_central_beam

        dataset = self._workflow.dataset
        start = time.perf_counter()
        dp_mean = np.frombuffer(self.dp_bytes, np.float32, self.dp_rows * self.dp_cols).reshape(self.dp_rows, self.dp_cols)
        (center_row, center_col), radius = estimate_central_beam(dp_mean)
        rr, cc = np.mgrid[: self.dp_rows, : self.dp_cols]
        inside = (rr - center_row) ** 2 + (cc - center_col) ** 2 <= (radius + _BF_MARGIN_PX) ** 2
        detector = (inside if self.real_space_mode == "bf" else ~inside).astype(np.float32)
        array = dataset.array
        image = np.empty((self.scan_rows, self.scan_cols), dtype=np.float64)
        step = max(1, int(256 * 1024**2 // max(1, array[0].nbytes)))  # ~256 MB of patterns per block
        for row in range(0, self.scan_rows, step):
            block = np.asarray(array[row : row + step], dtype=np.float32)
            image[row : row + step] = np.tensordot(block, detector, axes=([2, 3], [0, 1]))
        elapsed = (time.perf_counter() - start) * 1e3
        name = "BF" if self.real_space_mode == "bf" else "ADF"
        info = (f"Virtual {name}: center ({center_row:.1f}, {center_col:.1f}) px, "
                f"{'r <= ' if name == 'BF' else 'r > '}{radius + _BF_MARGIN_PX:.1f} px")
        with self.hold_sync():
            self.real_space_bytes = _b64_safe(image.astype(np.float32).tobytes())
            self.real_space_info = info
        return f"{info} in {elapsed / 1e3:.1f} s"

    def _preprocess(self) -> str:
        """The detector mask and the mean transform, as ``diffraction_mask`` + ``preprocess``."""
        work = self._workflow
        start = time.perf_counter()
        if self.diffraction_mask:
            work.diffraction_mask(
                threshold_percentile=self.mask_threshold_percentile, edge_blend=self.mask_edge_blend, plot_mask=False
            )
            mask = np.asarray(work.mask_diffraction, dtype=np.float32)
        else:
            # StrainMapAutocorrelation's own no-mask state
            work.mask_diffraction = np.ones(work.dataset.array.shape[2:])
            work.mask_diffraction_inv = np.zeros(work.dataset.array.shape[2:])
            mask = None
        work.preprocess(
            mode=self.mode,
            gamma=self.gamma,
            log_offset=self.log_offset,
            log_offset_relative=self.log_offset_relative,
            hann_window=self.hann_window,
            skip=None if self.skip == 1 else self.skip,
            q_to_r_rotation_ccw_deg=self.q_to_r_rotation_ccw_deg,
            q_transpose=self.q_transpose,
            plot_transform=False,
        )
        elapsed = (time.perf_counter() - start) * 1e3
        transform = np.asarray(work.transform.array, dtype=np.float64)
        with self.hold_sync():
            self.mask_bytes = b"" if mask is None else _b64_safe(mask.tobytes())
            self.transform_bytes = _b64_safe(transform.astype(np.float32).tobytes())
            self.transform_vmax = _brightest_lattice_peak(transform)
            self.sampling_real = [float(s) for s in work.metadata["sampling_real"]]
            self.real_units = _real_units(str(work.metadata.get("real_units", "")))
            self.timings_ms = {**self.timings_ms, "preprocess": round(elapsed, 1)}
        self._last_preprocess_key = self._preprocess_key()
        self._last_lattice_key = None  # a new transform needs a new refinement
        self._mark_fit_stale()
        positions = math.ceil(self.scan_rows / self.skip) * math.ceil(self.scan_cols / self.skip)
        return f"Transform of {positions} patterns in {elapsed:.0f} ms"

    def _refine_lattice(self) -> str:
        """``choose_lattice_vector`` with the picked vectors, or clear the result when one is missing."""
        work = self._workflow
        g1, g2 = _vector_or_none(self.g1), _vector_or_none(self.g2)
        self._last_lattice_key = self._lattice_key()
        self._mark_fit_stale()
        if g1 is None or g2 is None:
            self._peaks = None
            with self.hold_sync():
                self.g1_refined, self.g2_refined = [], []
                self.peaks_bytes, self.n_peaks = b"", 0
            return "Click a peak to place g1" if g1 is None else "Click a second peak to place g2"
        # within ~3 degrees of collinear the two seeds refine onto one row of peaks
        sine = abs(g1[0] * g2[1] - g1[1] * g2[0]) / max(math.hypot(*g1) * math.hypot(*g2), 1e-12)
        if sine < _MIN_BASIS_SINE:
            raise ValueError(f"g1={g1} and g2={g2} are (nearly) parallel; pick two non-collinear peaks.")
        start = time.perf_counter()
        work.mean_img_peaks = work.mean_img_weights = work.mean_img_ab = None
        work.choose_lattice_vector(
            g1=g1,
            g2=g2,
            refine_gaussian=self.refine_gaussian,
            refine_radius_px=self.refine_radius_px,
            refine_all_peaks=self.refine_all_peaks,
            threshold_percentile=self.threshold_percentile,
            min_peak_spacing=self.min_peak_spacing,
            plot=False,
        )
        elapsed = (time.perf_counter() - start) * 1e3
        packed = b""
        self._peaks = None
        if self.refine_all_peaks and work.mean_img_peaks is not None:
            self._peaks = np.asarray(work.mean_img_peaks, dtype=float).reshape(-1, 2)
            weights = np.asarray(work.mean_img_weights, dtype=float).reshape(-1, 1)
            ab = np.asarray(work.mean_img_ab, dtype=float).reshape(-1, 2)
            # (row, col, weight, a, b) per peak
            packed = _b64_safe(np.hstack([self._peaks, weights, ab]).astype(np.float32).tobytes())
        with self.hold_sync():
            self.g1_refined = [float(v) for v in np.asarray(work.g1).ravel()[:2]]
            self.g2_refined = [float(v) for v in np.asarray(work.g2).ravel()[:2]]
            self.peaks_bytes = packed
            self.n_peaks = 0 if self._peaks is None else int(self._peaks.shape[0])
            self.timings_ms = {**self.timings_ms, "refine": round(elapsed, 1)}
        peaks = f" against {self.n_peaks} peaks" if self.refine_all_peaks else ""
        return f"Lattice refined{peaks} in {elapsed:.0f} ms"

    def _fit_and_strain(self) -> str:
        """``fit_lattice_vectors`` on ``device``, then ``calculate_strain_map``."""
        work = self._workflow
        start = time.perf_counter()
        self._cancel_idle_release()
        # The GPU copy of the scan quantem makes for the fit is released when it ends (also on an
        # error or an interrupt), unless keep_on_gpu holds it for the next fit until it idles out.
        with _progress_into(self), self._gpu_copy_scope():
            work.fit_lattice_vectors(
                refine_gaussian=self.refine_gaussian,
                refine_dft=self.refine_dft,
                refine_all_peaks=self.refine_all_peaks,
                refine_radius_px=self.refine_radius_px if self.fit_refine_radius_px is None else self.fit_refine_radius_px,
                upsample=self.upsample,
                device=self.device,
                batch_size=self.batch_size,
                method=self.fit_method,
                progressbar=True,
            )
        strain = work.calculate_strain_map(calculation_metric=self.calculation_metric)
        elapsed = (time.perf_counter() - start) * 1e3
        weight = np.asarray(work.mask_weight, dtype=np.float32)
        self._strain = strain
        with self.hold_sync():
            self.weight_bytes = _b64_safe(weight.tobytes())
            self.fit_stale = False
            self.timings_ms = {**self.timings_ms, "fit": round(elapsed, 1)}
            # the dragged reference region outlives a refit; without one this is calculate_strain_map's own reference
            self._apply_reference()
        return f"Fit {self.scan_rows * self.scan_cols} positions on {self.device} in {elapsed / 1e3:.2f} s"

    @contextlib.contextmanager
    def _gpu_copy_scope(self):
        """Release (or, with ``keep_on_gpu``, schedule the release of) the GPU scan copy after the block."""
        try:
            yield
        finally:
            if self.keep_on_gpu:
                self._arm_idle_release()
            else:
                self._release_gpu()

    def _release_gpu(self) -> float:
        """Drop ``StrainMapAutocorrelation._gpu_cache`` and return the allocator's cached blocks; GB released."""
        work = self._workflow
        cache = getattr(work, "_gpu_cache", None) if work is not None else None
        if cache is None:
            return 0.0
        released = cache.nelement() * cache.element_size() / 1e9
        device = cache.device
        work._gpu_cache = None
        del cache
        import torch

        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        return released

    def _arm_idle_release(self) -> None:
        """Release the kept GPU copy after ``gpu_idle_timeout_s`` with no fit or parameter change."""
        self._cancel_idle_release()
        try:
            from tornado.ioloop import IOLoop

            loop = IOLoop.current()
        except Exception:  # no event loop (a plain script): the copy lives until free() or close()
            return
        ref = weakref.ref(self)

        def release():
            widget = ref()
            if widget is not None and widget._workflow is not None:
                widget._idle_handle = None
                released = widget._release_gpu()
                if released:
                    widget.status_message = (
                        f"Released the {released:.1f} GB GPU copy of the scan after "
                        f"{widget.gpu_idle_timeout_s / 60:.0f} min idle; the next fit reloads it"
                    )

        self._idle_handle = (loop, loop.call_later(self.gpu_idle_timeout_s, release))

    def _cancel_idle_release(self) -> None:
        handle = getattr(self, "_idle_handle", None)
        if handle is not None:
            loop, timeout = handle
            loop.remove_timeout(timeout)
            self._idle_handle = None

    def _close_previous(self, source) -> None:
        """Close the ShowStrain a rerun cell replaces (same input object), so it stops holding memory."""
        registry = type(self)._open_by_dataset
        key = id(source)
        previous = registry.get(key)
        previous = previous() if previous is not None else None
        same_input = previous is not None and getattr(previous, "_source", None) is source
        if same_input and previous is not self and getattr(previous, "comm", None) is not None:
            previous.close()
            print("ShowStrain: closed the previous widget on this dataset (close_previous=False keeps it open)")
        registry[key] = weakref.ref(self)

    def _mark_fit_stale(self) -> None:
        """Parameters changed after a fit: the shown strain no longer matches them."""
        if self.strain_bytes:
            self.fit_stale = True

    # ------------------------------------------------------------------
    # Saved-notebook preview
    # ------------------------------------------------------------------
    def _invalidate_preview(self, change=None) -> None:
        """Drop the cached preview so the next save redraws it."""
        self._static_fallback_image = ""

    def _static_png_b64(self, max_px: int = 512) -> str | None:
        """The transform with the refined vectors and, after a fit, the strain maps, as a base64 PNG."""
        work = self._workflow
        if work.transform is None:
            return None
        has_strain = self._strain is not None and not self.fit_stale
        figure = Figure(figsize=((2 if has_strain else 1) * max_px / 100, max_px / 100), dpi=100, layout="constrained")
        FigureCanvasAgg(figure)
        if has_strain:
            grid = figure.add_gridspec(1, 2)
            axes = figure.add_subplot(grid[0])
            strain_grid = grid[1].subgridspec(1, 4)
        else:
            axes = figure.add_subplot()
        image = np.asarray(work.transform.array)
        rows, cols = image.shape
        center = (rows // 2, cols // 2)
        axes.imshow(image, cmap=ListedColormap(colormap_lut("gray")), vmin=0.0, vmax=self.transform_vmax)
        half_r, half_c = rows * self.cropping_factor / 2, cols * self.cropping_factor / 2
        axes.set_xlim(center[1] - half_c, center[1] + half_c)
        axes.set_ylim(center[0] + half_r, center[0] - half_r)
        if self._peaks is not None:
            axes.scatter(self._peaks[:, 1] + center[1], self._peaks[:, 0] + center[0], s=30 * self.peak_size**2,
                         facecolors="none", edgecolors=self.peak_color, linewidths=self.peak_line_width)
        for vector, color, label in ((self.g1_refined, "#ff4d4f", "g1"), (self.g2_refined, "#40a9ff", "g2")):
            if vector:
                axes.annotate("", xy=(center[1] + vector[1], center[0] + vector[0]), xytext=(center[1], center[0]),
                              arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5})
                axes.text(center[1] + vector[1], center[0] + vector[0], f" {label}", color=color, fontsize=9)
        axes.set_title(f"{self.title or 'Mean transform'} (central {self.cropping_factor:.0%})", fontsize=9)
        axes.set_xticks([])
        axes.set_yticks([])
        if has_strain:
            # the shown maps: reference and frame rotation applied
            count = len(STRAIN_COMPONENTS)
            maps = np.frombuffer(self.strain_bytes, np.float32, count * self.scan_rows * self.scan_cols)
            maps = maps.reshape(count, self.scan_rows, self.scan_cols)
            for index, name in enumerate(("e_uu", "e_vv", "e_uv", "phi")):
                ax = figure.add_subplot(strain_grid[index])
                values = maps[STRAIN_COMPONENTS.index(name)]
                if name == "phi":
                    values, (vmin, vmax), unit = np.degrees(values), self.rotation_range_degrees, "deg"
                else:
                    values, (vmin, vmax), unit = values * 100.0, self.strain_range_percent, "%"
                # plot_strain's defaults: cmap_strain="RdBu_r", cmap_rotation="PiYG"
                cmap = "PiYG" if name == "phi" else ListedColormap(colormap_lut("RdBu_r"))
                ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_title(f"{name} ({unit})", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    # ------------------------------------------------------------------
    # Standalone HTML
    # ------------------------------------------------------------------
    def export_html(self, path: str | pathlib.Path | None = None, *, title: str | None = None) -> pathlib.Path:
        """Write a standalone HTML page of the current view, with no kernel behind it.

        The page holds the mean pattern, the float32 transform, the refined vectors and
        peaks and, after a fit, the float32 strain maps, so zoom, pan, hover readouts and
        the strain color ranges work. Controls that need the 4D data (preprocessing,
        picking, fitting) are locked; the raw scan is never embedded.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination file. Defaults to ``<title>_<scan_rows>x<scan_cols>.html`` in the
            working directory.
        title : str, optional
            Browser page title. Defaults to the widget ``title`` or "ShowStrain".

        Returns
        -------
        pathlib.Path
            The written HTML file.
        """
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path()
        self._write_html_export(export_path, title=title)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, {self._export_mode_label()})"
        return export_path

    def _export_mode_label(self, **options) -> str:
        """The page holds float32 pixels, never quantized."""
        return "float32 transform and strain"

    def _default_html_export_path(self, **options) -> pathlib.Path:
        """``<title>_<scan_rows>x<scan_cols>.html`` in the working directory."""
        return pathlib.Path.cwd() / f"{export_slug(self.title, 'showstrain')}_{self.scan_rows}x{self.scan_cols}.html"

    def _write_html_export(self, path: str | pathlib.Path, *, title: str | None = None, **options) -> pathlib.Path:
        """Embed this widget's current state (not a clone: the 4D data never re-enter a widget)."""
        from ipywidgets.embed import dependency_state, embed_minimal_html

        export_path = pathlib.Path(path)
        save_state, self._save_state = self._save_state, True
        try:
            state = dependency_state([self], drop_defaults=False)
        finally:
            self._save_state = save_state
        entry = state.get(self.model_id, {})
        model_state = entry.get("state")
        if isinstance(model_state, dict):
            model_state.update(
                export_enabled=False, export_request="", export_status="", export_payload_id="",
                export_filename="", _static_fallback_image="", offline=True, computing=False,
            )
            buffers = [b for b in entry.get("buffers", []) if (b.get("path") or [None])[0] != "export_payload"]
            buffers.append(_base64_buffer("export_payload", b""))
            entry["buffers"] = buffers
        export_path.parent.mkdir(parents=True, exist_ok=True)
        embed_minimal_html(str(export_path), views=[self], title=title or self.title or "ShowStrain",
                           drop_defaults=False, state=state)
        ensure_mobile_viewport(export_path)
        return export_path


def _brightest_lattice_peak(transform: np.ndarray) -> float:
    """Display maximum of ``plot_transform``: the transform at the max of transform x |q|.

    Weighting by the distance from the centre keeps the central (DC) peak from setting
    the contrast, so the lattice peaks are visible at the default range.
    """
    rows, cols = transform.shape
    kr = (np.arange(rows, dtype=float) - rows // 2)[:, None]
    kc = (np.arange(cols, dtype=float) - cols // 2)[None, :]
    weighted = transform * np.sqrt(kr * kr + kc * kc)
    index = np.unravel_index(np.nanargmax(weighted), weighted.shape)
    value = float(transform[index])
    return value if math.isfinite(value) and value > 0 else 1.0


@contextlib.contextmanager
def _progress_into(widget, interval_s: float = 0.25):
    """Route quantem's ``tqdm`` progress into ``widget.fit_progress`` for the duration of a fit.

    ``fit_lattice_vectors(progressbar=True)`` imports ``tqdm.auto.tqdm`` when it starts, so a
    stand-in bound there for the call reports each finished batch to the browser (at most every
    ``interval_s``, and once at the end) and prints nothing. The original is always restored.
    """
    import io

    try:
        import tqdm.auto
        import tqdm.std
    except ImportError:  # tqdm is optional: the fit runs, without a progress bar
        yield
        return

    original = tqdm.auto.tqdm
    start = time.perf_counter()

    class _WidgetTqdm(tqdm.std.tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.update(file=io.StringIO(), mininterval=0, miniters=1)
            super().__init__(*args, **kwargs)
            self._last_sent = -1.0
            self._send(force=True)

        def update(self, n=1):
            shown = super().update(n)
            self._send(force=self.total is not None and self.n >= self.total)
            return shown

        def _send(self, force=False):
            now = time.perf_counter()
            if force or now - self._last_sent >= interval_s:
                self._last_sent = now
                widget.fit_progress = {"done": int(self.n), "total": int(self.total or 0),
                                       "elapsed_s": round(now - start, 1)}

    tqdm.auto.tqdm = _WidgetTqdm
    try:
        yield
    finally:
        tqdm.auto.tqdm = original
        widget.fit_progress = {}
