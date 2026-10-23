"""ShowStrain: the widget's results are exactly StrainMapAutocorrelation's for the same arguments."""

import json
import warnings

import numpy as np
import pytest
import torch
import traitlets

from quantem.widget.showstrain import STRAIN_COMPONENTS, ShowStrain, _workflow_class

try:
    StrainMapAutocorrelation = _workflow_class()
except ImportError:
    pytest.skip("needs quantem's StrainMapAutocorrelation", allow_module_level=True)


def lattice_4dstem(scan_shape=(6, 5), det=64, spacing=8.0, max_strain=0.01, dtype=np.float32, seed=0):
    """Gaussian Bragg disks on a square reciprocal lattice; real-space e_cc ramps 0 -> max_strain along scan cols."""
    rng = np.random.default_rng(seed)
    rr, cc = np.mgrid[:det, :det].astype(np.float64)
    center = det / 2
    data = np.empty((*scan_shape, det, det), dtype=np.float64)
    for j in range(scan_shape[1]):
        e_cc = max_strain * j / max(1, scan_shape[1] - 1)
        dp = np.zeros((det, det))
        for h in range(-3, 4):
            for k in range(-3, 4):
                g = np.array([h * spacing, k * spacing / (1 + e_cc)])
                dp += np.exp(-np.hypot(*g) / 20.0) * np.exp(-((rr - center - g[0]) ** 2 + (cc - center - g[1]) ** 2) / 2.0)
        for i in range(scan_shape[0]):
            data[i, j] = rng.poisson(dp * 400) / 400.0
    return data.astype(dtype)


DATA = lattice_4dstem()
SEEDS = {"g1": (8, 0), "g2": (0, 8)}
PEAKS = {"refine_all_peaks": True, "threshold_percentile": 0.99, "min_peak_spacing": 4.0, "refine_radius_px": 1.5}


def direct(data=DATA, *, mode="linear", gamma=0.5, choose=None, mask=None):
    """The notebook path: the same calls the widget makes, by hand."""
    work = StrainMapAutocorrelation.from_array(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mask:
            work.diffraction_mask(plot_mask=False, **mask)
        work.preprocess(mode=mode, gamma=gamma, plot_transform=False)
        if choose:
            work.choose_lattice_vector(plot=False, **choose)
    return work


def test_constructor_reproduces_preprocess_and_multi_peak_refinement():
    widget = ShowStrain(DATA, device="cpu", mode="gamma", gamma=0.5, **SEEDS, **PEAKS)
    work = direct(mode="gamma", choose={**SEEDS, **PEAKS})
    assert (widget.scan_rows, widget.scan_cols, widget.dp_rows, widget.dp_cols) == DATA.shape
    np.testing.assert_array_equal(np.frombuffer(widget.transform_bytes, np.float32, 64 * 64).reshape(64, 64),
                                  work.transform.array.astype(np.float32))
    np.testing.assert_allclose(widget.g1_refined, work.g1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(widget.g2_refined, work.g2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(widget.peaks, work.mean_img_peaks)
    assert widget.n_peaks == len(work.mean_img_peaks) > 4
    assert widget.status_message.startswith("Transform of 30 patterns")
    # the refined basis lands on the synthetic lattice (transform spacing det / spacing = 8 px)
    assert np.hypot(*widget.g1_refined) == pytest.approx(8.0, abs=0.2)
    assert abs(np.dot(widget.g1_refined, widget.g2_refined)) < 0.5


def test_peak_threshold_and_spacing_commits_rerun_the_refinement():
    widget = ShowStrain(DATA, device="cpu", **SEEDS, **PEAKS)
    loose = widget.n_peaks
    widget.threshold_percentile = 0.998
    assert 0 < widget.n_peaks < loose
    np.testing.assert_allclose(widget.peaks, direct(choose={**SEEDS, **PEAKS, "threshold_percentile": 0.998}).mean_img_peaks)
    widget.threshold_percentile = 0.99
    widget.min_peak_spacing = 10.0
    spaced = widget.peaks
    assert 2 <= len(spaced) < loose
    distances = np.hypot(*(spaced[:, None, :] - spaced[None, :, :]).transpose(2, 0, 1))
    assert distances[~np.eye(len(spaced), dtype=bool)].min() >= 10.0 - 1.0  # integer candidates, then sub-pixel refined
    # one packed record per peak: row, col, weight, a, b
    packed = np.frombuffer(widget.peaks_bytes, np.float32, widget.n_peaks * 5).reshape(-1, 5)
    np.testing.assert_allclose(packed[:, :2], spaced, atol=1e-4)
    np.testing.assert_allclose(packed[:, 2], widget.peak_weights, rtol=1e-6)


def test_clicks_seed_the_vectors_and_a_parallel_pick_is_reported():
    widget = ShowStrain(DATA, device="cpu")
    assert widget.g1_refined == [] and widget.status_message.endswith("Click a peak to place g1")
    widget.g1 = [8.2, 0.3]  # first click
    assert widget.g1_refined == [] and widget.status_message.endswith("Click a second peak to place g2")
    widget.g2 = [-0.4, 7.9]  # second click
    assert np.hypot(*widget.g1_refined) == pytest.approx(8.0, abs=0.2)
    widget.g2 = [16.0, 0.6]  # dragged onto the 2*g1 site
    assert widget.status_message.startswith("Error: ") and "parallel" in widget.status_message
    widget.g1, widget.g2 = [], []
    assert widget.g1_refined == widget.g2_refined == []


def test_preprocessing_and_mask_commits_match_the_workflow():
    widget = ShowStrain(DATA, device="cpu", **SEEDS)
    widget.mode = "log"
    np.testing.assert_array_equal(np.frombuffer(widget.transform_bytes, np.float32, 64 * 64),
                                  direct(mode="log").transform.array.astype(np.float32).ravel())
    widget.diffraction_mask = True
    widget.mask_edge_blend = 4.0
    work = direct(mode="log", mask={"threshold_percentile": 50.0, "edge_blend": 4.0})
    np.testing.assert_array_equal(np.frombuffer(widget.mask_bytes, np.float32, 64 * 64),
                                  work.mask_diffraction.astype(np.float32).ravel())
    np.testing.assert_array_equal(widget.workflow.transform.array, work.transform.array)
    widget.diffraction_mask = False
    assert widget.mask_bytes == b"" and np.all(widget.workflow.mask_diffraction == 1)


def test_quantem_warnings_are_shown_not_silenced():
    widget = ShowStrain(DATA, device="cpu", diffraction_mask=True, mask_edge_blend=500.0)
    assert any("edge_blend" in message for message in widget.warning_messages)
    assert any("q_to_r_rotation_ccw_deg" in message for message in widget.warning_messages)


def test_fit_returns_the_workflow_strain_map_and_goes_stale_on_change():
    widget = ShowStrain(DATA, device="cpu", **SEEDS, **PEAKS)
    strain = widget.fit()
    work = direct(choose={**SEEDS, **PEAKS})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        work.fit_lattice_vectors(refine_radius_px=1.5, refine_all_peaks=True, device="cpu", progressbar=False)
    expected = work.calculate_strain_map()
    maps = np.frombuffer(widget.strain_bytes, np.float32, 5 * 30).reshape(5, 6, 5)
    # shown: e_uu, e_vv, e_uv (= e_rr, e_cc, e_rc at frame angle 0), planar e_uu + e_vv, rotation phi
    shown = {"e_uu": expected.e_rr.array, "e_vv": expected.e_cc.array, "e_uv": expected.e_rc.array,
             "planar": expected.e_rr.array + expected.e_cc.array, "phi": expected.phi.array}
    assert tuple(shown) == STRAIN_COMPONENTS
    for name in ("e_rr", "e_cc", "e_rc", "phi"):
        np.testing.assert_allclose(getattr(strain, name).array, getattr(expected, name).array, atol=1e-12)
    for index, values in enumerate(shown.values()):
        np.testing.assert_allclose(maps[index], values.astype(np.float32), atol=1e-7)
    # e_cc follows the synthetic ramp: it increases monotonically along the scan columns
    assert np.all(np.diff(strain.e_cc.array.mean(axis=0)) > 0)
    assert widget.strain_map is strain and not widget.fit_stale
    widget.min_peak_spacing = 5.0
    assert widget.fit_stale and widget.strain_map is None
    widget.fit_request += 1  # the Fit button
    assert not widget.fit_stale and widget.status_message.startswith("Fit 30 positions on cpu")


def test_fit_without_vectors_is_refused():
    with pytest.raises(ValueError, match="Pick g1 and g2"):
        ShowStrain(DATA, device="cpu").fit()


def test_inputs_keep_their_dtype_and_reject_bad_shapes():
    for dtype in (np.float32, np.uint16):
        data = (DATA * 1000).astype(dtype)
        widget = ShowStrain(data, device="cpu")
        assert widget.data_dtype == np.dtype(dtype).name
        assert widget.workflow.dataset.array.dtype == dtype
    widget = ShowStrain(torch.from_numpy(DATA), device="cpu")
    assert widget.data_dtype == "float32"
    with pytest.raises(ValueError, match="4D-STEM"):
        ShowStrain(DATA[0], device="cpu")


def test_a_workflow_or_dataset_is_used_in_place():
    work = StrainMapAutocorrelation.from_array(DATA)
    widget = ShowStrain(work, device="cpu", **SEEDS)
    assert widget.workflow is work and work.g1 is not None
    widget = ShowStrain(work.dataset, device="cpu")
    assert widget.workflow.dataset is work.dataset


@pytest.mark.parametrize(("name", "value"), [
    ("mode", "sqrt-ish"), ("threshold_percentile", 99.0), ("mask_threshold_percentile", 101.0),
    ("fit_method", "adam"), ("skip", 0), ("refine_radius_px", 0.0), ("g1", [80.0, 0.0]), ("cropping_factor", 0.0),
])
def test_invalid_parameters_are_refused_by_name(name, value):
    widget = ShowStrain(DATA, device="cpu")
    with pytest.raises(traitlets.TraitError, match=name):
        setattr(widget, name, value)


def test_unknown_keyword_and_unavailable_gpu_raise():
    with pytest.raises(TypeError, match="threshold"):
        ShowStrain(DATA, device="cpu", threshold=0.9)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            ShowStrain(DATA, device="cuda")


def test_state_round_trip_restores_parameters_and_results(tmp_path):
    widget = ShowStrain(DATA, device="cpu", mode="gamma", gamma=0.7, **SEEDS, **PEAKS, strain_range_percent=(-1, 1))
    path = tmp_path / "strain.json"
    widget.save(path)
    assert json.loads(path.read_text())["state"]["threshold_percentile"] == 0.99
    other = ShowStrain(DATA, device="cpu")
    other.load_state_dict(path)
    assert other.state_dict() == widget.state_dict()
    np.testing.assert_allclose(other.g1_refined, widget.g1_refined)
    assert other.n_peaks == widget.n_peaks


def test_export_html_holds_the_view_but_not_the_scan(tmp_path):
    widget = ShowStrain(DATA, device="cpu", title="Synthetic strain", **SEEDS)
    widget.fit()
    path = widget.export_html(tmp_path / "strain.html")
    html = path.read_text()
    assert '"offline": true' in html and '"export_enabled": false' in html
    # only the float32 views travel: transform, mean pattern and its |FFT|, mask, strain maps, weight, real space
    state = json.loads(html.split('<script type="application/vnd.jupyter.widget-state+json">')[1].split("</script>")[0])
    entry = state["state"][widget.model_id]
    names = {buffer["path"][0] for buffer in entry["buffers"] if buffer["data"]}
    assert names <= {"transform_bytes", "dp_bytes", "dp_fft_bytes", "mask_bytes", "peaks_bytes", "strain_bytes", "weight_bytes",
                     "real_space_bytes"}
    embedded = sum(len(buffer["data"]) * 3 // 4 for buffer in entry["buffers"])
    assert embedded < 4 * 64 * 64 * 4 + 7 * 30 * 4 + 64  # vs DATA.nbytes = 30 * 64 * 64 * 4
    assert widget.offline is False and widget.export_enabled is True


def test_reference_region_matches_update_reference_without_refitting():
    widget = ShowStrain(DATA, device="cpu", **SEEDS, **PEAKS)
    strain = widget.fit()
    fitted = widget.workflow.g1_array.copy()
    widget.reference_region = [0, 6, 0, 2]  # dragged over the two left (least strained) columns
    expected = direct(choose={**SEEDS, **PEAKS})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected.fit_lattice_vectors(refine_radius_px=1.5, refine_all_peaks=True, device="cpu", progressbar=False)
    reference = expected.calculate_strain_map()
    mask = np.zeros((6, 5), dtype=bool)
    mask[:, 0:2] = True
    reference.update_reference(strain_mask=mask)
    np.testing.assert_array_equal(widget.workflow.g1_array, fitted)  # no refit
    maps = np.frombuffer(widget.strain_bytes, np.float32, 5 * 30).reshape(5, 6, 5)
    np.testing.assert_allclose(maps[1], reference.e_cc.array.astype(np.float32), atol=1e-7)
    np.testing.assert_allclose(widget.reference_vectors, [reference.g1_ref, reference.g2_ref])
    assert widget.reference_message.startswith("Reference: median over rows 0-5, cols 0-1 (12 positions)")
    # the left columns are now the zero of the ramp; the right columns are stretched
    assert abs(maps[1][:, :2].mean()) < abs(maps[1][:, -1].mean()) and maps[1][:, -1].mean() > 0
    widget.calculation_metric = "mean"
    reference.update_reference(strain_mask=mask, calculation_metric="mean")
    np.testing.assert_allclose(np.frombuffer(widget.strain_bytes, np.float32, 30 * 5).reshape(5, 6, 5)[1],
                               reference.e_cc.array.astype(np.float32), atol=1e-7)
    widget.reference_region = []
    assert widget.reference_message.startswith("Reference: mean weighted by the fit amplitude")
    # the region survives a refit
    widget.reference_region = [0, 6, 0, 2]
    widget.fit()
    assert widget.reference_message.startswith("Reference: mean over rows 0-5")
    with pytest.raises(traitlets.TraitError, match="reference_region"):
        widget.reference_region = [0, 7, 0, 2]
    assert widget.strain_map is not strain  # a refit builds a new StrainMap


def test_frame_rotation_is_plot_strains_rotate_strain():
    widget = ShowStrain(DATA, device="cpu", **SEEDS, **PEAKS)
    strain = widget.fit()
    widget.rotation_angle = 12.0
    e_uu, e_vv, e_uv = strain.rotate_strain(12.0)
    maps = np.frombuffer(widget.strain_bytes, np.float32, 5 * 30).reshape(5, 6, 5)
    for index, values in enumerate((e_uu, e_vv, e_uv, e_uu + e_vv, strain.phi.array)):
        np.testing.assert_allclose(maps[index], values.astype(np.float32), atol=1e-7)
    assert not widget.fit_stale


def test_fit_radius_is_separate_from_the_lattice_radius():
    widget = ShowStrain(DATA, device="cpu", fit_method="cft", hann_window=True, fit_refine_radius_px=3.0, **SEEDS, **PEAKS)
    widget.fit()
    assert widget.workflow.metadata["fit_refine_radius_px"] == 3.0
    assert widget.workflow.metadata["choose_refine_radius_px"] == 1.5
    with pytest.raises(traitlets.TraitError, match="mask_range"):
        widget.mask_range = [0.5, 0.2]


def test_fit_reports_progress_to_the_browser_and_restores_tqdm():
    import tqdm.auto

    original = tqdm.auto.tqdm
    widget = ShowStrain(DATA, device="cpu", batch_size=7, **SEEDS)
    seen = []
    widget.observe(lambda change: seen.append(dict(change["new"])), names=["fit_progress"])
    widget.fit()
    reports = [r for r in seen if r]
    assert reports[0]["done"] == 0 and reports[-1] == {**reports[-1], "done": 30, "total": 30}
    assert [r["done"] for r in reports] == sorted(r["done"] for r in reports)
    assert seen[-1] == {} and widget.fit_progress == {}  # cleared when the fit ends
    assert tqdm.auto.tqdm is original


def test_peak_overlay_style_is_validated_and_saved():
    widget = ShowStrain(DATA, device="cpu", peak_color="#00e5ff", peak_line_width=3, peak_size=2)
    assert widget.state_dict()["peak_color"] == "#00e5ff" and widget.state_dict()["peak_line_width"] == 3.0
    with pytest.raises(traitlets.TraitError, match="peak_line_width"):
        widget.peak_line_width = 0


def _stage_gpu_copy(widget, nbytes=4_000_000):
    """Make the next fit leave a scan copy in _gpu_cache, as fit_lattice_vectors(save_to_gpu=True) does on a GPU."""
    work = widget.workflow
    fit = work.fit_lattice_vectors

    def fit_with_copy(**kwargs):
        work._gpu_cache = torch.zeros(nbytes // 4, dtype=torch.float32)
        return fit(**kwargs)

    work.fit_lattice_vectors = fit_with_copy


def test_the_gpu_copy_is_released_when_a_fit_ends():
    widget = ShowStrain(DATA, device="cpu", **SEEDS)
    _stage_gpu_copy(widget)
    widget.fit()
    assert widget.workflow._gpu_cache is None


def test_keep_on_gpu_holds_the_copy_until_free_or_the_idle_timeout():
    import asyncio

    from tornado.ioloop import IOLoop

    widget = ShowStrain(DATA, device="cpu", keep_on_gpu=True, gpu_idle_timeout_s=0.05, **SEEDS)
    _stage_gpu_copy(widget)
    widget.fit()
    assert widget.workflow._gpu_cache is not None
    assert widget.free() == pytest.approx(0.004)
    assert widget.workflow._gpu_cache is None and widget.free() == 0.0
    widget.fit()  # kept again; nothing happens for 50 ms, then the idle release runs on the kernel loop
    assert widget.workflow._gpu_cache is not None
    IOLoop.current().run_sync(lambda: asyncio.sleep(0.2))
    assert widget.workflow._gpu_cache is None
    assert widget.status_message.startswith("Released the 0.0 GB GPU copy")


def test_rerunning_the_cell_closes_the_widget_it_replaces(capsys):
    data = DATA.copy()
    first = ShowStrain(data, device="cpu", **SEEDS)
    second = ShowStrain(data, device="cpu", **SEEDS)  # the same cell, run again
    assert first.comm is None and first.workflow is None
    assert "closed the previous widget" in capsys.readouterr().out
    other = ShowStrain(DATA.copy(), device="cpu")  # a different scan stays open
    kept = ShowStrain(data, device="cpu", close_previous=False)
    assert second.comm is not None and other.comm is not None and kept.comm is not None
    for widget in (second, other, kept):
        widget.close()
        assert widget.comm is None and widget.workflow is None


def test_real_space_tab_matches_quantems_virtual_images():
    from quantem.core.datastructures import Dataset4dstem
    from quantem.diffraction.disk_detection import estimate_central_beam

    dataset = Dataset4dstem.from_array(DATA)
    widget = ShowStrain(dataset, device="cpu")
    assert widget.real_space_bytes == b""  # nothing computed until the tab asks
    (row, col), radius = estimate_central_beam(np.mean(DATA, axis=(0, 1)))
    for mode, detector in (("adf", ("annular", ((row, col), (radius + 2.0, 1e9)))), ("bf", ("circle", ((row, col), radius + 2.0)))):
        widget.real_space_mode = mode  # the Real space tab, then its Detector menu
        expected = dataset.get_virtual_image(mode=detector[0], geometry=detector[1], attach=False).array
        shown = np.frombuffer(widget.real_space_bytes, np.float32, 30).reshape(6, 5)
        np.testing.assert_allclose(shown, expected.astype(np.float32), rtol=1e-5)
        assert widget.real_space_info.startswith(f"Virtual {mode.upper()}: center")


def test_strain_mask_toggle_starts_off_unless_a_range_is_given():
    widget = ShowStrain(DATA, device="cpu")
    assert widget.apply_mask is False and widget.mask_range == [0.0, 0.01]
    widget = ShowStrain(DATA, device="cpu", mask_range=(0.2, 0.25))
    assert widget.apply_mask is True and widget.mask_range == [0.2, 0.25]
    widget = ShowStrain(DATA, device="cpu", mask_range=(0.2, 0.25), apply_mask=False)
    assert widget.apply_mask is False


def test_an_interrupted_step_keeps_no_quantem_frames_alive():
    widget = ShowStrain(DATA, device="cpu", **SEEDS)

    def interrupted_preprocess(**kwargs):
        scratch = np.ones(1_000_000)  # stands in for preprocess's full-size intermediates  # noqa: F841
        raise KeyboardInterrupt

    widget.workflow.preprocess = interrupted_preprocess
    with pytest.raises(KeyboardInterrupt) as raised:
        widget.mode = "log"
    error = raised.value
    assert error.__cause__ is None and error.__suppress_context__
    frames = []
    tb = error.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    # the traceback IPython keeps stops at the widget: quantem's frame and its arrays are gone
    assert "interrupted_preprocess" not in frames
    assert widget.status_message == "Interrupted" and widget.computing is False


def test_fourier_tab_is_the_plain_fft_of_the_mean_pattern():
    widget = ShowStrain(DATA, device="cpu")
    expected = np.abs(np.fft.fftshift(np.fft.fft2(np.mean(DATA, axis=(0, 1)).astype(np.float64))))
    shown = np.frombuffer(widget.dp_fft_bytes, np.float32, 64 * 64).reshape(64, 64)
    np.testing.assert_allclose(shown, expected.astype(np.float32), rtol=1e-6)
    assert 0 < widget.dp_fft_vmax < expected.max()  # the DC peak does not set the contrast
