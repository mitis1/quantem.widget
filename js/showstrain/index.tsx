/**
 * ShowStrain: pick lattice vectors on the mean cepstral transform of a 4D-STEM
 * scan, tune the preprocessing and multi-peak refinement, and view the strain maps.
 *
 * Python (quantem's StrainMapAutocorrelation) owns every scientific result. This
 * view draws the float32 images it receives and turns interactions into trait
 * commits: a slider release, a dropped vector, a switch. While a pointer or a
 * slider moves, only browser-local previews change (the dragged vector and the
 * lattice it predicts, the peak candidates, the masked detector pixels, the
 * reference box), drawn from refs once per animation frame; nothing reaches the
 * kernel until release.
 *
 * Layout: two large panels. Left: the mean diffraction pattern or a virtual
 * real-space image of the scan. Right: the mean transform (lattice picking) or one
 * strain component. Under them, every strain component side by side.
 *
 * Vectors are (row, col) offsets from the transform centre (rows // 2, cols // 2)
 * in transform pixels, whatever the zoom or pan.
 */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import LinearProgress from "@mui/material/LinearProgress";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Slider, { type SliderProps } from "../shared/Slider";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import Typography from "@mui/material/Typography";
import { useTheme, WIDGET_TYPOGRAPHY as typography, WIDGET_UI_FONT, type ThemeColors } from "../theme";
import { SPACING, compactButton, controlRow, sliderStyles, switchStyles } from "../controlStyles";
import { downloadBlob, extractBytes, extractFloat32, preserveRestoredWidgetModelsOnSave } from "../format";
import { useHideStaticFallback } from "../staticFallback";
import { COLORMAPS } from "../display/colormaps";
import { createColormapEngine, type DisplayColormapEngine } from "../display/cpuColormap";
import { RenderPathBadge } from "../shared/RenderPathBadge";
import { drawScaleBarInRegion } from "../figure";
import { findPeakCandidates, latticeSites, maskedPixels, percentileLinear, type PeakCandidate } from "./peaks";
import { PIYG_LUT } from "./piyg";

const PANEL_PX = 448;
const SMALL_MAX_H = 200;
const SMALL_MAX_W = 170;
const COLORBAR_W = 44;
const HIT_PX = 10;
const CLICK_MOVE_PX = 4;
const DOUBLE_CLICK_GRACE_MS = 300;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 40;
const VECTOR_COLORS = ["#ff4d4f", "#40a9ff"];
const VECTOR_NAMES = ["g1", "g2"] as const;
const PREVIEW_COLOR = "#36cfc9";
const STRAIN_CMAP = "RdBu_r";
// plot_strain's defaults: cmap_strain="RdBu_r", cmap_rotation="PiYG" (PiYG ships in this folder)
const ROTATION_CMAP = "PiYG";
const ROTATION_INDEX = 4;
const LUTS: Record<string, Uint8Array> = { ...COLORMAPS, [ROTATION_CMAP]: PIYG_LUT };
// strain_bytes order (STRAIN_COMPONENTS in showstrain.py: e_uu, e_vv, e_uv, planar, phi), then the fit weight
const STRAIN_PANELS = [
  { key: "e_uu", label: "e_uu", unit: "%", scale: 100 },
  { key: "e_vv", label: "e_vv", unit: "%", scale: 100 },
  { key: "e_uv", label: "e_uv", unit: "%", scale: 100 },
  { key: "planar", label: "planar", unit: "%", scale: 100, title: "Planar strain e_uu + e_vv" },
  { key: "rotation", label: "rotation", unit: "deg", scale: 180 / Math.PI },
  { key: "weight", label: "weight", unit: "", scale: 1 },
] as const;
const N_STRAIN = 5;  // components in strain_bytes
const WEIGHT_INDEX = 5;
const SLOT_TRANSFORM = 0;
const SLOT_DP = 1;
const SLOT_REAL = 2;
const SLOT_STRAIN = 3;  // 3..8
const SLOT_FFT = 9;

type Vec = [number, number];
type View = { zoom: number; panX: number; panY: number };
type Drag =
  | { mode: "pending"; startX: number; startY: number; startPanX: number; startPanY: number }
  | { mode: "pan"; startX: number; startY: number; startPanX: number; startPanY: number }
  | { mode: "vector"; index: 0 | 1; at: Vec };
// Left: the cepstral transform and everything in scan space. Right: the diffraction-space
// comparisons, hideable.
type LeftTab = "transform" | "real" | 0 | 1 | 2 | 3 | 4;
type RightTab = "diffraction" | "fourier";

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

function asVec(value: number[] | undefined | null): Vec | null {
  return value && value.length === 2 ? [value[0], value[1]] : null;
}

function round2(value: number): number {
  return Math.round(value * 100) / 100;
}

// MUI commits its last value at the end of every press, even one that moved nothing; pass a
// commit on only after a real change, so a click on the rail does not rerun the transform twice.
function CommitSlider({ onChange, onChangeCommitted, ...props }: SliderProps) {
  const changedRef = React.useRef(false);
  return (
    <Slider
      {...props}
      onChange={(e, v, thumb) => { changedRef.current = true; onChange?.(e, v, thumb); }}
      onChangeCommitted={(e, v) => {
        if (!changedRef.current) return;
        changedRef.current = false;
        onChangeCommitted?.(e, v);
      }}
    />
  );
}

/** Label, slider and value as one unit that never splits when the row wraps (the toolbar rule). */
function SliderPair(props: {
  label: string; ariaLabel: string; value: number | number[]; min: number; max: number; step: number;
  format: (v: number | number[]) => string; disabled?: boolean; width?: number; colors: ThemeColors;
  onChange: (v: number | number[]) => void; onCommit: (v: number | number[]) => void; title?: string;
}) {
  const { colors } = props;
  return (
    <Box sx={{ display: "inline-flex", alignItems: "center", gap: "6px", whiteSpace: "nowrap" }} title={props.title}>
      <Typography sx={{ ...typography.label, color: colors.text }}>{props.label}</Typography>
      <CommitSlider
        size="small"
        aria-label={props.ariaLabel}
        value={props.value}
        min={props.min}
        max={props.max}
        step={props.step}
        disabled={props.disabled}
        onChange={(_, v) => props.onChange(v)}
        onChangeCommitted={(_, v) => props.onCommit(v)}
        sx={{ ...sliderStyles.small, width: props.width ?? 100, color: colors.accent }}
      />
      <Typography sx={{ ...typography.value, minWidth: 52, color: colors.textMuted, fontVariantNumeric: "tabular-nums" }}>
        {props.format(props.value)}
      </Typography>
    </Box>
  );
}

function SwitchPair(props: { label: string; checked: boolean; disabled?: boolean; colors: ThemeColors; onChange: (v: boolean) => void; title?: string }) {
  return (
    <Box sx={{ display: "inline-flex", alignItems: "center", gap: "2px", whiteSpace: "nowrap" }} title={props.title}>
      <Typography sx={{ ...typography.label, color: props.colors.text }}>{props.label}</Typography>
      <Switch
        size="small"
        checked={props.checked}
        disabled={props.disabled}
        sx={switchStyles.small}
        slotProps={{ input: { "aria-label": props.label } }}
        onChange={(e) => props.onChange(e.target.checked)}
      />
    </Box>
  );
}

/** A row of text tabs above a panel; the active one is underlined in the accent color. */
function TabBar<T extends string | number>(props: {
  tabs: { value: T; label: string; disabled?: boolean; title?: string }[]; value: T; onChange: (v: T) => void;
  colors: ThemeColors; ariaLabel: string;
}) {
  const { colors } = props;
  return (
    <Box role="tablist" aria-label={props.ariaLabel} sx={{ display: "flex", flexWrap: "wrap", gap: "2px", mb: "2px" }}>
      {props.tabs.map((tab) => {
        const active = tab.value === props.value;
        return (
          <Box
            key={String(tab.value)}
            component="button"
            role="tab"
            aria-selected={active}
            disabled={tab.disabled}
            title={tab.title}
            onClick={() => !tab.disabled && props.onChange(tab.value)}
            sx={{
              ...typography.label, px: "8px", py: "3px", cursor: tab.disabled ? "default" : "pointer",
              border: "none", background: "none", color: tab.disabled ? colors.textMuted : active ? colors.accent : colors.text,
              opacity: tab.disabled ? 0.5 : 1, borderBottom: `2px solid ${active ? colors.accent : "transparent"}`,
              fontWeight: active ? 600 : 400,
            }}
          >
            {tab.label}
          </Box>
        );
      })}
    </Box>
  );
}

/** Colormap one float32 image on the shared engine (WebGPU, or the matching JS path). */
async function colormapToBitmap(
  engine: DisplayColormapEngine, slot: number, data: Float32Array, width: number, height: number,
  vmin: number, vmax: number, cmap: string,
): Promise<ImageBitmap | null> {
  engine.uploadData(slot, data, width, height);
  const out = await engine.renderSlotsToImageBitmapAsync([slot], [{ vmin, vmax }], false, cmap, LUTS[cmap] ?? COLORMAPS.gray);
  return out?.[0] ?? null;
}

/** Log display of the mean pattern, as diffraction_mask's diagnostic plot: log(max(I, smallest positive)). */
function logDisplay(values: Float32Array): { data: Float32Array; vmin: number; vmax: number } {
  let floor = Infinity;
  for (const v of values) if (v > 0 && v < floor) floor = v;
  if (!Number.isFinite(floor)) floor = 1e-12;
  const data = new Float32Array(values.length);
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = Math.log(Math.max(values[i], floor));
    data[i] = v;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return { data, vmin: lo, vmax: hi > lo ? hi : lo + 1 };
}

/** A vertical colorbar beside a panel: its colormap, range and the zero tick. */
function drawColorbar(canvas: HTMLCanvasElement | null, height: number, lut: Uint8Array, vmin: number, vmax: number, colors: ThemeColors) {
  const ctx = canvas?.getContext("2d");
  if (!ctx) return;
  const barW = 8;
  ctx.clearRect(0, 0, COLORBAR_W, height);
  for (let y = 0; y < height; y++) {
    const entry = Math.round((1 - y / Math.max(height - 1, 1)) * 255) * 3;
    ctx.fillStyle = `rgb(${lut[entry]},${lut[entry + 1]},${lut[entry + 2]})`;
    ctx.fillRect(0, y, barW, 1);
  }
  ctx.strokeStyle = colors.border;
  ctx.strokeRect(0.5, 0.5, barW - 1, height - 1);
  ctx.fillStyle = colors.text;
  ctx.font = `9px ${WIDGET_UI_FONT}`;
  ctx.textAlign = "left";
  const fmt = (v: number) => (Math.abs(v) >= 10 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2));
  ctx.textBaseline = "top";
  ctx.fillText(fmt(vmax), barW + 3, 0);
  ctx.textBaseline = "bottom";
  ctx.fillText(fmt(vmin), barW + 3, height);
  if (vmin < 0 && vmax > 0 && height >= 40) {
    const y = (vmax / (vmax - vmin)) * height;
    ctx.textBaseline = "middle";
    ctx.fillText("0", barW + 3, y);
    ctx.fillRect(barW - 3, y, 3, 1);
  }
}

function ShowStrainWidget() {
  const model = useModel();
  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);
  const [offline] = useModelState<boolean>("offline");
  const { colors } = useTheme(!!offline);

  const [title] = useModelState<string>("title");
  const [showControls] = useModelState<boolean>("show_controls");
  const [scanRows] = useModelState<number>("scan_rows");
  const [scanCols] = useModelState<number>("scan_cols");
  const [dpRows] = useModelState<number>("dp_rows");
  const [dpCols] = useModelState<number>("dp_cols");
  const [dataDtype] = useModelState<string>("data_dtype");
  const [samplingReal] = useModelState<number[]>("sampling_real");
  const [realUnits] = useModelState<string>("real_units");
  const [scanSampling] = useModelState<number>("scan_sampling");
  const [scanUnits] = useModelState<string>("scan_units");
  const [device] = useModelState<string>("device");

  const [mode, setMode] = useModelState<string>("mode");
  const [gamma, setGamma] = useModelState<number>("gamma");
  const [logOffset, setLogOffset] = useModelState<number>("log_offset");
  const [logOffsetRelative, setLogOffsetRelative] = useModelState<boolean>("log_offset_relative");
  const [hannWindow, setHannWindow] = useModelState<boolean>("hann_window");
  const [skip, setSkip] = useModelState<number>("skip");
  const [diffractionMask, setDiffractionMask] = useModelState<boolean>("diffraction_mask");
  const [maskPercentile, setMaskPercentile] = useModelState<number>("mask_threshold_percentile");
  const [maskEdgeBlend, setMaskEdgeBlend] = useModelState<number>("mask_edge_blend");

  const [g1] = useModelState<number[]>("g1");
  const [g2] = useModelState<number[]>("g2");
  const [refineGaussian, setRefineGaussian] = useModelState<boolean>("refine_gaussian");
  const [refineRadius, setRefineRadius] = useModelState<number>("refine_radius_px");
  const [refineAllPeaks, setRefineAllPeaks] = useModelState<boolean>("refine_all_peaks");
  const [thresholdPercentile, setThresholdPercentile] = useModelState<number>("threshold_percentile");
  const [minPeakSpacing, setMinPeakSpacing] = useModelState<number>("min_peak_spacing");

  const [fitMethod, setFitMethod] = useModelState<string>("fit_method");
  const [refineDft, setRefineDft] = useModelState<boolean>("refine_dft");
  const [fitRequest, setFitRequest] = useModelState<number>("fit_request");
  const [fitRadius, setFitRadius] = useModelState<number | null>("fit_refine_radius_px");
  const [fitProgress] = useModelState<{ done?: number; total?: number; elapsed_s?: number }>("fit_progress");

  const [croppingFactor] = useModelState<number>("cropping_factor");
  const [strainRange, setStrainRange] = useModelState<number[]>("strain_range_percent");
  const [rotationRange, setRotationRange] = useModelState<number[]>("rotation_range_degrees");
  const [applyMask, setApplyMask] = useModelState<boolean>("apply_mask");
  const [maskRange, setMaskRange] = useModelState<number[]>("mask_range");
  const [rotationAngle, setRotationAngle] = useModelState<number>("rotation_angle");
  const [referenceRegion, setReferenceRegion] = useModelState<number[]>("reference_region");
  const [calculationMetric, setCalculationMetric] = useModelState<string>("calculation_metric");
  const [referenceVectors] = useModelState<number[][]>("reference_vectors");
  const [referenceMessage] = useModelState<string>("reference_message");
  const [peakColor, setPeakColor] = useModelState<string>("peak_color");
  const [peakLineWidth, setPeakLineWidth] = useModelState<number>("peak_line_width");
  const [peakSize, setPeakSize] = useModelState<number>("peak_size");
  const [realSpaceMode, setRealSpaceMode] = useModelState<string>("real_space_mode");
  const [realSpaceInfo] = useModelState<string>("real_space_info");
  const [showDiffraction, setShowDiffraction] = useModelState<boolean>("show_diffraction");

  const [dpBytes] = useModelState<DataView>("dp_bytes");
  const [dpFftBytes] = useModelState<DataView>("dp_fft_bytes");
  const [dpFftVmax] = useModelState<number>("dp_fft_vmax");
  const [maskBytes] = useModelState<DataView>("mask_bytes");
  const [transformBytes] = useModelState<DataView>("transform_bytes");
  const [transformVmax] = useModelState<number>("transform_vmax");
  const [g1Refined] = useModelState<number[]>("g1_refined");
  const [g2Refined] = useModelState<number[]>("g2_refined");
  const [nPeaks] = useModelState<number>("n_peaks");
  const [peaksBytes] = useModelState<DataView>("peaks_bytes");
  const [strainBytes] = useModelState<DataView>("strain_bytes");
  const [weightBytes] = useModelState<DataView>("weight_bytes");
  const [realSpaceBytes] = useModelState<DataView>("real_space_bytes");
  const [fitStale] = useModelState<boolean>("fit_stale");
  const [computing] = useModelState<boolean>("computing");
  const [statusMessage] = useModelState<string>("status_message");
  const [warningMessages] = useModelState<string[]>("warning_messages");

  const [exportEnabled] = useModelState<boolean>("export_enabled");
  const [exportStatus] = useModelState<string>("export_status");
  const [, setExportRequest] = useModelState<string>("export_request");
  const [exportPayload] = useModelState<DataView>("export_payload");
  const [exportPayloadId] = useModelState<string>("export_payload_id");
  const [exportFilename] = useModelState<string>("export_filename");

  const locked = !!offline;  // no kernel: nothing that needs the 4D data can change

  // ---------------------------------------------------------------- decoded data
  const transform = React.useMemo(() => extractFloat32(transformBytes, dpRows * dpCols), [transformBytes, dpRows, dpCols]);
  const dp = React.useMemo(() => extractFloat32(dpBytes, dpRows * dpCols), [dpBytes, dpRows, dpCols]);
  const dpFft = React.useMemo(() => extractFloat32(dpFftBytes, dpRows * dpCols), [dpFftBytes, dpRows, dpCols]);
  const mask = React.useMemo(() => extractFloat32(maskBytes, dpRows * dpCols), [maskBytes, dpRows, dpCols]);
  const peaks = React.useMemo(() => {
    const raw = extractFloat32(peaksBytes, nPeaks * 5);
    if (!raw) return [] as { row: number; col: number; weight: number }[];
    const out = [];
    for (let i = 0; i < nPeaks; i++) out.push({ row: raw[i * 5], col: raw[i * 5 + 1], weight: raw[i * 5 + 2] });
    return out;
  }, [peaksBytes, nPeaks]);
  const strain = React.useMemo(() => extractFloat32(strainBytes, N_STRAIN * scanRows * scanCols), [strainBytes, scanRows, scanCols]);
  const weight = React.useMemo(() => extractFloat32(weightBytes, scanRows * scanCols), [weightBytes, scanRows, scanCols]);
  const realSpace = React.useMemo(() => extractFloat32(realSpaceBytes, scanRows * scanCols), [realSpaceBytes, scanRows, scanCols]);
  const hasStrain = !!strain;

  // ---------------------------------------------------------------- engine
  const [engine, setEngine] = React.useState<DisplayColormapEngine | null>(null);
  React.useEffect(() => {
    let disposed = false;
    let created: DisplayColormapEngine | null = null;
    void createColormapEngine().then((next) => {
      if (disposed) { next.destroy(); return; }
      created = next;
      setEngine(next);
    });
    return () => { disposed = true; created?.destroy(); };
  }, []);

  // ---------------------------------------------------------------- tabs
  const [leftTab, setLeftTab] = React.useState<LeftTab>("transform");
  const [rightTab, setRightTab] = React.useState<RightTab>("diffraction");
  // the first fit opens e_uu; losing the strain (a refit pending) goes back to the transform
  const hadStrainRef = React.useRef(hasStrain);
  React.useEffect(() => {
    if (hasStrain && !hadStrainRef.current) setLeftTab(0);
    if (!hasStrain && typeof leftTab === "number") setLeftTab("transform");
    hadStrainRef.current = hasStrain;
  }, [hasStrain, leftTab]);

  // ---------------------------------------------------------------- local (preview) control values
  const useLocal = <T,>(value: T): [T, (v: T) => void] => {
    const [local, setLocal] = React.useState<T>(value);
    React.useEffect(() => setLocal(value), [value]);
    return [local, setLocal];
  };
  const [localGamma, setLocalGamma] = useLocal(gamma);
  const [localMaskPercentile, setLocalMaskPercentile] = useLocal(maskPercentile);
  const [localEdgeBlend, setLocalEdgeBlend] = useLocal(maskEdgeBlend);
  const [localRadius, setLocalRadius] = useLocal(refineRadius);
  const [localThreshold, setLocalThreshold] = useLocal(thresholdPercentile);
  const [localSpacing, setLocalSpacing] = useLocal(minPeakSpacing);
  const [localStrainRange, setLocalStrainRange] = useLocal<number[]>(strainRange);
  const [localRotationRange, setLocalRotationRange] = useLocal<number[]>(rotationRange);
  const [localMaskRange, setLocalMaskRange] = useLocal<number[]>(maskRange);
  const [localRotationAngle, setLocalRotationAngle] = useLocal(rotationAngle);
  const [localFitRadius, setLocalFitRadius] = useLocal(fitRadius ?? refineRadius);
  const [localPeakLineWidth, setLocalPeakLineWidth] = useLocal(peakLineWidth);
  const [localPeakSize, setLocalPeakSize] = useLocal(peakSize);
  const [localLogOffsetExp, setLocalLogOffsetExp] = useLocal(Math.log10(logOffset));
  const [contrast, setContrast] = React.useState(1);

  // Peak candidates the threshold / spacing sliders would give, while they differ from the committed values.
  const peakPreviewActive = refineAllPeaks && (localThreshold !== thresholdPercentile || localSpacing !== minPeakSpacing);
  const peakPreview = React.useMemo<PeakCandidate[] | null>(() => {
    if (!peakPreviewActive || !transform) return null;
    return findPeakCandidates(transform, dpRows, dpCols, localThreshold, localSpacing);
  }, [peakPreviewActive, transform, dpRows, dpCols, localThreshold, localSpacing]);
  const maskPreviewActive = diffractionMask && localMaskPercentile !== maskPercentile;
  // the mask preview belongs to the diffraction panel; show it while its slider moves
  React.useEffect(() => { if (maskPreviewActive) { setShowDiffraction(true); setRightTab("diffraction"); } }, [maskPreviewActive]);
  React.useEffect(() => { if (peakPreviewActive) setLeftTab("transform"); }, [peakPreviewActive]);

  // ---------------------------------------------------------------- transform panel: refs + one rAF draw
  const transformCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const transformBitmapRef = React.useRef<ImageBitmap | null>(null);
  const viewRef = React.useRef<View>({ zoom: 1, panX: 0, panY: 0 });
  const dragRef = React.useRef<Drag | null>(null);
  const hoverRef = React.useRef<{ x: number; y: number } | null>(null);
  const readoutRef = React.useRef<HTMLSpanElement>(null);
  const frameRef = React.useRef<number | null>(null);
  const drawRef = React.useRef<() => void>(() => {});
  const requestDraw = React.useCallback(() => {
    if (frameRef.current !== null) return;
    frameRef.current = window.requestAnimationFrame(() => {
      frameRef.current = null;
      drawRef.current();
    });
  }, []);
  React.useEffect(() => () => { if (frameRef.current !== null) window.cancelAnimationFrame(frameRef.current); }, []);

  const baseScale = dpRows > 0 && dpCols > 0 ? PANEL_PX / Math.max(dpRows, dpCols) : 1;
  const resetView = React.useCallback(() => {
    viewRef.current = { zoom: clamp(1 / (croppingFactor || 1), MIN_ZOOM, MAX_ZOOM), panX: 0, panY: 0 };
    requestDraw();
  }, [croppingFactor, requestDraw]);
  React.useEffect(() => { resetView(); }, [resetView, dpRows, dpCols]);

  // offset (row, col) from the centre pixel <-> canvas px; pixel centres sit at index + 0.5
  const centerRow = Math.floor(dpRows / 2);
  const centerCol = Math.floor(dpCols / 2);
  const offsetToScreen = React.useCallback((row: number, col: number): [number, number] => {
    const { zoom, panX, panY } = viewRef.current;
    const s = baseScale * zoom;
    return [
      PANEL_PX / 2 + (centerCol + col + 0.5 - dpCols / 2) * s + panX,
      PANEL_PX / 2 + (centerRow + row + 0.5 - dpRows / 2) * s + panY,
    ];
  }, [baseScale, centerRow, centerCol, dpRows, dpCols]);
  const screenToOffset = React.useCallback((x: number, y: number): Vec => {
    const { zoom, panX, panY } = viewRef.current;
    const s = baseScale * zoom;
    return [
      (y - PANEL_PX / 2 - panY) / s + dpRows / 2 - 0.5 - centerRow,
      (x - PANEL_PX / 2 - panX) / s + dpCols / 2 - 0.5 - centerCol,
    ];
  }, [baseScale, centerRow, centerCol, dpRows, dpCols]);

  // |r| in transform pixels and, when calibrated, in real-space units (sampling_real per pixel)
  const calibrated = !!realUnits && realUnits !== "1/pixels" && samplingReal?.length === 2;
  const realStep = calibrated ? (samplingReal[0] + samplingReal[1]) / 2 : 1;
  const lengthLabel = (px: number) => (calibrated ? `|r| ${px.toFixed(2)} px = ${(px * realStep).toFixed(3)} ${realUnits}` : `|r| ${px.toFixed(2)} px`);

  // Shown vectors: the refined ones once Python returns them, else the picked seeds.
  const shownVectors: (Vec | null)[] = [asVec(g1Refined) ?? asVec(g1), asVec(g2Refined) ?? asVec(g2)];
  const seeds: (Vec | null)[] = [asVec(g1), asVec(g2)];

  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftBitmapRef = React.useRef<ImageBitmap | null>(null);
  const fftHoverRef = React.useRef<{ x: number; y: number } | null>(null);
  const fftDragRef = React.useRef<{ startX: number; startY: number; startPanX: number; startPanY: number } | null>(null);

  // The plain |FFT| of the mean pattern, on the transform's zoom and pan, with the same basis,
  // so the cepstral peaks can be compared spot for spot.
  const drawFourierCanvas = () => {
    const ctx = fftCanvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, PANEL_PX, PANEL_PX);
    const { zoom, panX, panY } = viewRef.current;
    const s = baseScale * zoom;
    const bitmap = fftBitmapRef.current;
    if (bitmap) ctx.drawImage(bitmap, PANEL_PX / 2 - (dpCols / 2) * s + panX, PANEL_PX / 2 - (dpRows / 2) * s + panY, dpCols * s, dpRows * s);
    const [cx, cy] = offsetToScreen(0, 0);
    shownVectors.forEach((vec, i) => {
      if (!vec) return;
      const [x, y] = offsetToScreen(vec[0], vec[1]);
      ctx.strokeStyle = VECTOR_COLORS[i];
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = `bold 11px ${WIDGET_UI_FONT}`;
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.85)";
      ctx.strokeText(VECTOR_NAMES[i], x + 7, y - 5);
      ctx.fillStyle = "#fff";
      ctx.fillText(VECTOR_NAMES[i], x + 7, y - 5);
    });
    drawScaleBarInRegion(ctx, { x: 0, y: 0, width: PANEL_PX, height: PANEL_PX }, zoom, s, realStep, calibrated ? realUnits : "px");
    const readout = readoutRef.current;
    const hover = fftHoverRef.current;
    if (readout && hover && dpFft) {
      const [row, col] = screenToOffset(hover.x, hover.y);
      const pr = Math.round(row) + centerRow;
      const pc = Math.round(col) + centerCol;
      const value = pr >= 0 && pc >= 0 && pr < dpRows && pc < dpCols ? dpFft[pr * dpCols + pc] : NaN;
      readout.textContent = `|FFT| (${row.toFixed(2)}, ${col.toFixed(2)}) px · ${lengthLabel(Math.hypot(row, col))} · ${Number.isFinite(value) ? value.toPrecision(4) : "-"}`;
    }
  };

  const drawTransformCanvas = () => {
    const canvas = transformCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, PANEL_PX, PANEL_PX);
    const bitmap = transformBitmapRef.current;
    const { zoom, panX, panY } = viewRef.current;
    const s = baseScale * zoom;
    if (bitmap) {
      ctx.drawImage(bitmap, PANEL_PX / 2 - (dpCols / 2) * s + panX, PANEL_PX / 2 - (dpRows / 2) * s + panY, dpCols * s, dpRows * s);
    }
    const drag = dragRef.current;
    const vectors = shownVectors.slice();
    if (drag?.mode === "vector") vectors[drag.index] = drag.at;
    const [cx, cy] = offsetToScreen(0, 0);

    // lattice the (dragged) basis predicts
    if (vectors[0] && vectors[1]) {
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.lineWidth = 1;
      for (const [row, col] of latticeSites(vectors[0], vectors[1], dpRows / 2, dpCols / 2)) {
        const [x, y] = offsetToScreen(row, col);
        if (x < -4 || y < -4 || x > PANEL_PX + 4 || y > PANEL_PX + 4) continue;
        ctx.beginPath();
        ctx.moveTo(x - 3, y); ctx.lineTo(x + 3, y);
        ctx.moveTo(x, y - 3); ctx.lineTo(x, y + 3);
        ctx.stroke();
      }
    }
    // committed refined peaks, radius by weight, with a dark halo so the ring reads on bright peaks
    if (refineAllPeaks && peaks.length && !peakPreview) {
      const wmax = Math.max(...peaks.map((p) => Math.abs(p.weight)), 1e-30);
      for (const p of peaks) {
        const [x, y] = offsetToScreen(p.row, p.col);
        const radius = localPeakSize * (3 + 6 * Math.sqrt(Math.abs(p.weight) / wmax));
        ctx.strokeStyle = "rgba(0,0,0,0.75)";
        ctx.lineWidth = localPeakLineWidth + 2;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.strokeStyle = peakColor || "#ffd54f";
        ctx.lineWidth = localPeakLineWidth;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }
    // slider-drag preview of the candidates
    if (peakPreview) {
      ctx.strokeStyle = PREVIEW_COLOR;
      ctx.lineWidth = localPeakLineWidth;
      ctx.setLineDash([3, 3]);
      for (const p of peakPreview) {
        const [x, y] = offsetToScreen(p.row, p.col);
        ctx.beginPath();
        ctx.arc(x, y, 6 * localPeakSize, 0, 2 * Math.PI);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    // refinement window of each seed
    seeds.forEach((seed, i) => {
      if (!seed || (drag?.mode === "vector" && drag.index === i)) return;
      const [x, y] = offsetToScreen(seed[0], seed[1]);
      ctx.strokeStyle = VECTOR_COLORS[i];
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.arc(x, y, Math.max(3, localRadius * s), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    });
    // vectors
    vectors.forEach((vec, i) => {
      if (!vec) return;
      const [x, y] = offsetToScreen(vec[0], vec[1]);
      ctx.strokeStyle = VECTOR_COLORS[i];
      ctx.fillStyle = VECTOR_COLORS[i];
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.stroke();
      const angle = Math.atan2(y - cy, x - cx);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x - 8 * Math.cos(angle - 0.4), y - 8 * Math.sin(angle - 0.4));
      ctx.lineTo(x - 8 * Math.cos(angle + 0.4), y - 8 * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fill();
      ctx.font = `bold 11px ${WIDGET_UI_FONT}`;
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.85)";
      ctx.strokeText(VECTOR_NAMES[i], x + 7, y - 5);
      ctx.fillStyle = "#fff";
      ctx.fillText(VECTOR_NAMES[i], x + 7, y - 5);
    });
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(cx, cy, 2, 0, 2 * Math.PI);
    ctx.fill();
    drawScaleBarInRegion(ctx, { x: 0, y: 0, width: PANEL_PX, height: PANEL_PX }, zoom, s, realStep, calibrated ? realUnits : "px");

    // hover readout, written straight to the DOM so a moving pointer never rerenders React
    const readout = readoutRef.current;
    if (readout) {
      const hover = drag?.mode === "vector" ? offsetToScreen(drag.at[0], drag.at[1]) : hoverRef.current ? [hoverRef.current.x, hoverRef.current.y] : null;
      if (!hover || !transform) {
        readout.textContent = "";
      } else {
        const [row, col] = screenToOffset(hover[0], hover[1]);
        const pr = Math.round(row) + centerRow;
        const pc = Math.round(col) + centerCol;
        const value = pr >= 0 && pc >= 0 && pr < dpRows && pc < dpCols ? transform[pr * dpCols + pc] : NaN;
        readout.textContent = `transform (${row.toFixed(2)}, ${col.toFixed(2)}) px · ${lengthLabel(Math.hypot(row, col))} · I ${Number.isFinite(value) ? value.toPrecision(4) : "-"}`;
      }
    }
  };
  drawRef.current = () => {
    drawTransformCanvas();
    drawFourierCanvas();
  };
  // redraw whenever anything the drawing reads changes (the canvas remounts when its tab returns)
  React.useEffect(() => { requestDraw(); });

  // colormap the transform (display only: vmin 0, vmax = brightest lattice peak x contrast)
  React.useEffect(() => {
    if (!engine || !transform || !dpRows || !dpCols) return;
    let cancelled = false;
    void colormapToBitmap(engine, SLOT_TRANSFORM, transform, dpCols, dpRows, 0, (transformVmax || 1) / contrast, "gray").then((bitmap) => {
      if (cancelled) { bitmap?.close(); return; }
      transformBitmapRef.current?.close();
      transformBitmapRef.current = bitmap;
      requestDraw();
    });
    return () => { cancelled = true; };
  }, [engine, transform, dpRows, dpCols, transformVmax, contrast, requestDraw]);

  // the plain |FFT|: vmin 0, vmax = its brightest lattice peak (as the transform), same Contrast
  React.useEffect(() => {
    if (!engine || !dpFft || !dpRows || !dpCols) return;
    let cancelled = false;
    void colormapToBitmap(engine, SLOT_FFT, dpFft, dpCols, dpRows, 0, (dpFftVmax || 1) / contrast, "gray").then((bitmap) => {
      if (cancelled) { bitmap?.close(); return; }
      fftBitmapRef.current?.close();
      fftBitmapRef.current = bitmap;
      requestDraw();
    });
    return () => { cancelled = true; };
  }, [engine, dpFft, dpRows, dpCols, dpFftVmax, contrast, requestDraw]);

  // ---------------------------------------------------------------- transform pointer handling
  const pendingClickRef = React.useRef<number | null>(null);
  const cancelPendingClick = () => {
    if (pendingClickRef.current !== null) { window.clearTimeout(pendingClickRef.current); pendingClickRef.current = null; }
  };
  React.useEffect(() => () => cancelPendingClick(), []);

  const panelPoint = (e: React.PointerEvent | React.WheelEvent | React.MouseEvent, size: number): [number, number] => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return [(e.clientX - rect.left) * (size / rect.width), (e.clientY - rect.top) * (size / rect.height)];
  };

  const commitVectors = (next: (Vec | null)[]) => {
    model.set("g1", next[0] ? [round2(next[0][0]), round2(next[0][1])] : []);
    model.set("g2", next[1] ? [round2(next[1][0]), round2(next[1][1])] : []);
    model.save_changes();
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    if (e.detail >= 2) { cancelPendingClick(); dragRef.current = null; return; }
    const [x, y] = panelPoint(e, PANEL_PX);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    if (!locked) {
      for (let i = 1; i >= 0; i--) {
        const vec = shownVectors[i];
        if (!vec) continue;
        const [vx, vy] = offsetToScreen(vec[0], vec[1]);
        if (Math.hypot(x - vx, y - vy) <= HIT_PX) {
          dragRef.current = { mode: "vector", index: i as 0 | 1, at: vec };
          requestDraw();
          return;
        }
      }
    }
    const { panX, panY } = viewRef.current;
    dragRef.current = { mode: "pending", startX: x, startY: y, startPanX: panX, startPanY: panY };
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    const [x, y] = panelPoint(e, PANEL_PX);
    hoverRef.current = { x, y };
    const drag = dragRef.current;
    if (drag?.mode === "vector") {
      const [row, col] = screenToOffset(x, y);
      drag.at = [clamp(row, -centerRow, dpRows - 1 - centerRow), clamp(col, -centerCol, dpCols - 1 - centerCol)];
    } else if (drag && Math.hypot(x - drag.startX, y - drag.startY) > CLICK_MOVE_PX) {
      dragRef.current = { ...drag, mode: "pan" };
      viewRef.current = { ...viewRef.current, panX: drag.startPanX + x - drag.startX, panY: drag.startPanY + y - drag.startY };
    }
    requestDraw();
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag) return;
    if (drag.mode === "vector") {
      const next = seeds.slice();
      next[drag.index] = drag.at;
      commitVectors(next);
    } else if (drag.mode === "pending" && !locked && (!seeds[0] || !seeds[1])) {
      const [x, y] = panelPoint(e, PANEL_PX);
      const at = screenToOffset(x, y);
      cancelPendingClick();
      pendingClickRef.current = window.setTimeout(() => {
        pendingClickRef.current = null;
        const next = seeds.slice();
        next[next[0] ? 1 : 0] = at;
        commitVectors(next);
      }, DOUBLE_CLICK_GRACE_MS);
    }
    requestDraw();
  };

  // wheel zoom about the cursor; the native non-passive listener keeps the page from scrolling
  const transformBoxRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const box = transformBoxRef.current;
    if (!box) return;
    const prevent = (e: WheelEvent) => e.preventDefault();
    box.addEventListener("wheel", prevent, { passive: false });
    return () => box.removeEventListener("wheel", prevent);
  }, [leftTab]);
  const fftBoxRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const box = fftBoxRef.current;
    if (!box) return;
    const prevent = (e: WheelEvent) => e.preventDefault();
    box.addEventListener("wheel", prevent, { passive: false });
    return () => box.removeEventListener("wheel", prevent);
  }, [rightTab, showDiffraction]);
  const handleWheel = (e: React.WheelEvent) => {
    const [x, y] = panelPoint(e, PANEL_PX);
    const { zoom, panX, panY } = viewRef.current;
    const next = clamp(zoom * (e.deltaY > 0 ? 0.9 : 1.1), MIN_ZOOM, MAX_ZOOM);
    const px = (x - PANEL_PX / 2 - panX) / zoom;
    const py = (y - PANEL_PX / 2 - panY) / zoom;
    viewRef.current = { zoom: next, panX: x - PANEL_PX / 2 - px * next, panY: y - PANEL_PX / 2 - py * next };
    requestDraw();
  };

  // ---------------------------------------------------------------- mean diffraction pattern + detector mask
  const dpCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const [dpBitmap, setDpBitmap] = React.useState<ImageBitmap | null>(null);
  React.useEffect(() => {
    if (!engine || !dp || !dpRows || !dpCols) return;
    let cancelled = false;
    const { data, vmin, vmax } = logDisplay(dp);
    void colormapToBitmap(engine, SLOT_DP, data, dpCols, dpRows, vmin, vmax, "gray").then((bitmap) => {
      if (!cancelled) setDpBitmap(bitmap);
    });
    return () => { cancelled = true; };
  }, [engine, dp, dpRows, dpCols]);
  // masked pixels: the slider's preview while it moves, else the committed feathered mask
  const maskOverlay = React.useMemo(() => {
    if (!dpRows || !dpCols || !diffractionMask) return null;
    const rgba = new Uint8ClampedArray(dpRows * dpCols * 4);
    if (maskPreviewActive && dp) {
      const masked = maskedPixels(dp, dpRows, dpCols, localMaskPercentile);
      for (let i = 0; i < masked.length; i++) if (masked[i]) { rgba[i * 4] = 54; rgba[i * 4 + 1] = 207; rgba[i * 4 + 2] = 201; rgba[i * 4 + 3] = 150; }
    } else if (mask) {
      for (let i = 0; i < mask.length; i++) { rgba[i * 4] = 255; rgba[i * 4 + 1] = 77; rgba[i * 4 + 2] = 79; rgba[i * 4 + 3] = Math.round(170 * (1 - mask[i])); }
    } else {
      return null;
    }
    const canvas = document.createElement("canvas");
    canvas.width = dpCols;
    canvas.height = dpRows;
    canvas.getContext("2d")?.putImageData(new ImageData(rgba, dpCols, dpRows), 0, 0);
    return canvas;
  }, [dp, mask, dpRows, dpCols, diffractionMask, maskPreviewActive, localMaskPercentile]);
  const dpScale = dpRows && dpCols ? PANEL_PX / Math.max(dpRows, dpCols) : 1;
  React.useLayoutEffect(() => {
    const ctx = dpCanvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, PANEL_PX, PANEL_PX);
    const w = dpCols * dpScale;
    const h = dpRows * dpScale;
    if (dpBitmap) ctx.drawImage(dpBitmap, (PANEL_PX - w) / 2, (PANEL_PX - h) / 2, w, h);
    if (maskOverlay) ctx.drawImage(maskOverlay, (PANEL_PX - w) / 2, (PANEL_PX - h) / 2, w, h);
    drawScaleBarInRegion(ctx, { x: 0, y: 0, width: PANEL_PX, height: PANEL_PX }, 1, dpScale, 1, "px");
  }, [dpBitmap, maskOverlay, dpRows, dpCols, dpScale, colors.bg, rightTab, showDiffraction]);
  const dpReadoutMove = (e: React.PointerEvent) => {
    const [x, y] = panelPoint(e, PANEL_PX);
    const w = dpCols * dpScale;
    const h = dpRows * dpScale;
    const col = Math.floor((x - (PANEL_PX - w) / 2) / dpScale);
    const row = Math.floor((y - (PANEL_PX - h) / 2) / dpScale);
    const out = readoutRef.current;
    if (!out || !dp || row < 0 || col < 0 || row >= dpRows || col >= dpCols) return;
    const m = mask ? `, mask ${mask[row * dpCols + col].toFixed(2)}` : "";
    out.textContent = `detector (${row}, ${col}) px · mean I ${dp[row * dpCols + col].toPrecision(4)}${m}`;
  };

  // ---------------------------------------------------------------- scan-space maps: real space, strain
  const bigScale = scanRows && scanCols ? Math.min(PANEL_PX / scanRows, (PANEL_PX - COLORBAR_W) / scanCols) : 1;
  const bigW = Math.max(1, Math.round(scanCols * bigScale));
  const bigH = Math.max(1, Math.round(scanRows * bigScale));
  const smallScale = scanRows && scanCols ? Math.min(SMALL_MAX_H / scanRows, SMALL_MAX_W / scanCols) : 1;
  const smallW = Math.max(1, Math.round(scanCols * smallScale));
  const smallH = Math.max(1, Math.round(scanRows * smallScale));

  const strainDisplay = React.useMemo(() => {
    if (!strain) return null;
    const n = scanRows * scanCols;
    const panels: Float32Array[] = [];
    for (let k = 0; k < N_STRAIN; k++) {
      const scale = STRAIN_PANELS[k].scale;  // percent, or degrees for the rotation
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = strain[k * n + i] * scale;
      panels.push(out);
    }
    panels.push(weight ?? new Float32Array(n).fill(1));
    return panels;
  }, [strain, weight, scanRows, scanCols]);
  const strainRanges = [localStrainRange, localStrainRange, localStrainRange, localStrainRange, localRotationRange, [0, 1]];
  const strainCmap = (k: number) => (k === WEIGHT_INDEX ? "gray" : k === ROTATION_INDEX ? ROTATION_CMAP : STRAIN_CMAP);
  const [strainBitmaps, setStrainBitmaps] = React.useState<(ImageBitmap | null)[]>([]);
  React.useEffect(() => {
    if (!engine || !strainDisplay) { setStrainBitmaps([]); return; }
    let cancelled = false;
    void Promise.all(strainDisplay.map((panel, k) => colormapToBitmap(
      engine, SLOT_STRAIN + k, panel, scanCols, scanRows, strainRanges[k][0], strainRanges[k][1], strainCmap(k),
    ))).then((bitmaps) => { if (!cancelled) setStrainBitmaps(bitmaps); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine, strainDisplay, scanRows, scanCols, ...localStrainRange, ...localRotationRange]);

  // the virtual image: robust 0.5-99.5 % window, gray
  const [realBitmap, setRealBitmap] = React.useState<ImageBitmap | null>(null);
  const realRange = React.useMemo(() => {
    if (!realSpace) return null;
    return [percentileLinear(realSpace, 0.5), percentileLinear(realSpace, 99.5)];
  }, [realSpace]);
  React.useEffect(() => {
    if (!engine || !realSpace || !realRange) { setRealBitmap(null); return; }
    let cancelled = false;
    const [lo, hi] = realRange;
    void colormapToBitmap(engine, SLOT_REAL, realSpace, scanCols, scanRows, lo, hi > lo ? hi : lo + 1, "gray")
      .then((bitmap) => { if (!cancelled) setRealBitmap(bitmap); });
    return () => { cancelled = true; };
  }, [engine, realSpace, realRange, scanRows, scanCols]);

  // plot_strain(mask_range) dimming, only while Mask is on: weight <= low is black, >= high full color
  const maskOverlayCanvas = React.useMemo(() => {
    if (!applyMask || !weight || !scanRows || !scanCols) return null;
    const [low, high] = localMaskRange;
    const rgba = new Uint8ClampedArray(scanRows * scanCols * 4);
    for (let i = 0; i < weight.length; i++) {
      const visible = clamp((weight[i] - low) / Math.max(high - low, 1e-12), 0, 1);
      rgba[i * 4 + 3] = Math.round(255 * (1 - visible));
    }
    const canvas = document.createElement("canvas");
    canvas.width = scanCols;
    canvas.height = scanRows;
    canvas.getContext("2d")?.putImageData(new ImageData(rgba, scanCols, scanRows), 0, 0);
    return canvas;
  }, [applyMask, weight, scanRows, scanCols, localMaskRange]);

  // Reference region: the committed box (solid) or the box being dragged (dashed), on every scan map.
  // [row0, row1, col0, col1], half-open.
  const regionDragRef = React.useRef<{ start: Vec; end: Vec } | null>(null);
  const scanHoverRef = React.useRef<{ source: number | "real"; row: number; col: number } | null>(null);
  const scanCanvasRefs = React.useRef<Record<string, HTMLCanvasElement | null>>({});
  const scanFrameRef = React.useRef<number | null>(null);
  const scanDrawRef = React.useRef<() => void>(() => {});
  const requestScanDraw = React.useCallback(() => {
    if (scanFrameRef.current !== null) return;
    scanFrameRef.current = window.requestAnimationFrame(() => {
      scanFrameRef.current = null;
      scanDrawRef.current();
    });
  }, []);
  React.useEffect(() => () => { if (scanFrameRef.current !== null) window.cancelAnimationFrame(scanFrameRef.current); }, []);
  const regionFromDrag = (drag: { start: Vec; end: Vec }): number[] => [
    Math.min(drag.start[0], drag.end[0]), Math.max(drag.start[0], drag.end[0]) + 1,
    Math.min(drag.start[1], drag.end[1]), Math.max(drag.start[1], drag.end[1]) + 1,
  ];
  const scanUnit = scanUnits && scanUnits !== "pixels" ? scanUnits : "px";

  scanDrawRef.current = () => {
    const drag = regionDragRef.current;
    const region = drag ? regionFromDrag(drag) : (referenceRegion?.length === 4 ? referenceRegion : null);
    const drawMap = (key: string, bitmap: ImageBitmap | null, w: number, h: number, masked: boolean) => {
      const ctx = scanCanvasRefs.current[key]?.getContext("2d");
      if (!ctx) return;
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, w, h);
      if (bitmap) ctx.drawImage(bitmap, 0, 0, w, h);
      if (masked && maskOverlayCanvas) ctx.drawImage(maskOverlayCanvas, 0, 0, w, h);
      if (w >= 60) {
        drawScaleBarInRegion(ctx, { x: 0, y: 0, width: w, height: h }, 1, w / scanCols, scanSampling || 1, scanUnit,
          { targetBarPx: Math.min(60, w * 0.35), margin: 8, barThickness: 3, color: "#000", barShadowColor: "#fff", outline: { color: "#fff", width: 3 } });
      }
      if (!region) return;
      const sx = w / scanCols;
      const sy = h / scanRows;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(drag ? [3, 2] : []);
      ctx.strokeStyle = drag ? PREVIEW_COLOR : "#000";
      ctx.strokeRect(region[2] * sx + 0.75, region[0] * sy + 0.75, (region[3] - region[2]) * sx - 1.5, (region[1] - region[0]) * sy - 1.5);
      ctx.setLineDash([]);
    };
    strainBitmaps.forEach((bitmap, k) => drawMap(`small-${k}`, bitmap, smallW, smallH, k !== WEIGHT_INDEX));
    if (typeof leftTab === "number") drawMap("big", strainBitmaps[leftTab] ?? null, bigW, bigH, true);
    if (leftTab === "real") drawMap("real", realBitmap, bigW, bigH, false);

    const out = readoutRef.current;
    const hover = scanHoverRef.current;
    if (!out || !hover) return;
    const position = scanUnit !== "px" ? ` = (${(hover.row * scanSampling).toFixed(2)}, ${(hover.col * scanSampling).toFixed(2)}) ${scanUnit}` : "";
    if (drag) {
      const [r0, r1, c0, c1] = regionFromDrag(drag);
      out.textContent = `reference rows ${r0}-${r1 - 1}, cols ${c0}-${c1 - 1} (${(r1 - r0) * (c1 - c0)} positions) · release to apply`;
    } else if (hover.source === "real" && realSpace) {
      out.textContent = `scan (${hover.row}, ${hover.col})${position} · ${realSpaceMode.toUpperCase()} ${realSpace[hover.row * scanCols + hover.col].toPrecision(4)}`;
    } else if (typeof hover.source === "number" && strainDisplay) {
      const panel = STRAIN_PANELS[hover.source];
      const value = strainDisplay[hover.source][hover.row * scanCols + hover.col];
      const w = weight ? ` · weight ${weight[hover.row * scanCols + hover.col].toFixed(3)}` : "";
      out.textContent = `scan (${hover.row}, ${hover.col})${position} · ${panel.label} ${value.toFixed(hover.source === WEIGHT_INDEX ? 3 : 4)} ${panel.unit}${hover.source === WEIGHT_INDEX ? "" : w}`;
    }
  };
  React.useLayoutEffect(() => { scanDrawRef.current(); });

  const scanPosition = (e: React.PointerEvent): Vec => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return [
      clamp(Math.floor(((e.clientY - rect.top) / rect.height) * scanRows), 0, scanRows - 1),
      clamp(Math.floor(((e.clientX - rect.left) / rect.width) * scanCols), 0, scanCols - 1),
    ];
  };
  // Dragging a box on any scan map sets the reference, once a fit exists.
  const canPickReference = !locked && hasStrain && !fitStale;
  const scanHandlers = (source: number | "real") => ({
    onPointerDown: (e: React.PointerEvent) => {
      if (e.button !== 0 || !canPickReference) return;
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      const at = scanPosition(e);
      regionDragRef.current = { start: at, end: at };
      requestScanDraw();
    },
    onPointerMove: (e: React.PointerEvent) => {
      const [row, col] = scanPosition(e);
      scanHoverRef.current = { source, row, col };
      if (regionDragRef.current) regionDragRef.current.end = [row, col];
      requestScanDraw();
    },
    onPointerUp: () => {
      const drag = regionDragRef.current;
      regionDragRef.current = null;
      // a click without a drag leaves the reference alone
      if (drag && (drag.start[0] !== drag.end[0] || drag.start[1] !== drag.end[1])) setReferenceRegion(regionFromDrag(drag));
      requestScanDraw();
    },
    onPointerLeave: () => { scanHoverRef.current = null; },
  });

  // colorbars beside the big strain panel and every small one
  const colorbarRefs = React.useRef<Record<string, HTMLCanvasElement | null>>({});
  React.useLayoutEffect(() => {
    const draw = (key: string, k: number, height: number) =>
      drawColorbar(colorbarRefs.current[key] ?? null, height, LUTS[strainCmap(k)], strainRanges[k][0], strainRanges[k][1], colors);
    STRAIN_PANELS.forEach((_, k) => draw(`small-${k}`, k, smallH));
    if (typeof leftTab === "number") draw("big", leftTab, bigH);
  });

  // ---------------------------------------------------------------- export
  const pendingExportRef = React.useRef<{ id: string; filename: string } | null>(null);
  const [localExportStatus, setLocalExportStatus] = React.useState("");
  const handleExport = () => {
    const slug = (title || "showstrain").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "") || "showstrain";
    const filename = `${slug}_${scanRows}x${scanCols}.html`;
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    pendingExportRef.current = { id, filename };
    setLocalExportStatus(`Preparing ${filename}...`);
    setExportRequest(JSON.stringify({ mode: "single", download: true, id, filename }));
  };
  React.useEffect(() => {
    const pending = pendingExportRef.current;
    if (!pending || exportPayloadId !== pending.id) return;
    const bytes = extractBytes(exportPayload);
    if (bytes.length === 0) return;
    const payload = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength ? bytes : bytes.slice();
    downloadBlob(new Blob([payload as BlobPart], { type: "text/html;charset=utf-8" }), exportFilename || pending.filename);
    pendingExportRef.current = null;
    setLocalExportStatus(`Saved ${exportFilename || pending.filename} (${(payload.byteLength / 2 ** 20).toFixed(1)} MB, float32)`);
    setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
  }, [exportPayload, exportPayloadId, exportFilename, setExportRequest]);
  React.useEffect(() => {
    if (exportStatus?.startsWith("Export failed")) { pendingExportRef.current = null; setLocalExportStatus(""); }
  }, [exportStatus]);

  // ---------------------------------------------------------------- layout
  const themedSelect = {
    fontSize: 10, height: 24, bgcolor: colors.controlBg, color: colors.text,
    "& .MuiSvgIcon-root": { color: colors.text },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: colors.accent },
  };
  const menuProps = { PaperProps: { sx: { bgcolor: colors.controlBg, color: colors.text, "& .MuiMenuItem-root": { fontSize: 11 } } } };
  // one bordered control row per step, as the other widgets' toolbars
  const row = { ...controlRow, border: `1px solid ${colors.border}`, bgcolor: colors.controlBg, mb: `${SPACING.XS}px` };
  const sectionLabel = (text: string) => (
    <Typography sx={{ ...typography.label, fontWeight: 600, color: colors.text, minWidth: 58 }}>{text}</Typography>
  );
  const button = { ...compactButton, color: colors.accent };
  const isError = /^Error/.test(statusMessage || "");
  const vectorReadout = (i: 0 | 1) => {
    const seed = seeds[i];
    const refined = asVec(i === 0 ? g1Refined : g2Refined);
    if (!seed) return "not placed";
    const fmt = (v: Vec) => `(${v[0].toFixed(2)}, ${v[1].toFixed(2)})`;
    return refined ? `${fmt(refined)} px · ${lengthLabel(Math.hypot(...refined))} · seed ${fmt(seed)}` : `seed ${fmt(seed)}`;
  };
  const angle = (() => {
    const a = asVec(g1Refined);
    const b = asVec(g2Refined);
    if (!a || !b) return null;
    const cos = (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(...a) * Math.hypot(...b));
    return (Math.acos(clamp(cos, -1, 1)) * 180) / Math.PI;
  })();
  const strainLabel = (k: number) => `${STRAIN_PANELS[k].label}${STRAIN_PANELS[k].unit ? ` (${STRAIN_PANELS[k].unit})` : ""}`;
  const panelBox = { border: `1px solid ${colors.border}`, width: PANEL_PX, height: PANEL_PX, position: "relative" as const, boxSizing: "content-box" as const };
  const scanMapBox = { ...panelBox, display: "flex", alignItems: "center", justifyContent: "center", gap: "3px" };

  return (
    <Box className="showstrain-root" sx={{ p: "12px", bgcolor: colors.bg, color: colors.text, fontFamily: WIDGET_UI_FONT, width: "100%", boxSizing: "border-box" }}>
      <Stack direction="row" alignItems="center" sx={{ mb: "6px", flexWrap: "wrap", columnGap: "8px" }}>
        <Typography sx={{ ...typography.title, fontSize: 13, color: colors.text }}>{title || "ShowStrain"}</Typography>
        <RenderPathBadge colors={colors} />
        <Typography sx={{ ...typography.labelSmall, color: colors.textMuted }}>
          {scanRows}×{scanCols} scan · {dpRows}×{dpCols} {dataDtype} · fit on {device}
        </Typography>
        <Box sx={{ flex: 1 }} />
        <Button size="small" sx={button} onClick={resetView}>Reset View</Button>
        {exportEnabled && !offline && (
          <Button size="small" sx={button} onClick={handleExport} title="Standalone HTML: float32 transform and strain maps">Export HTML</Button>
        )}
      </Stack>

      {/* ------------------------------------------------ the two large panels */}
      <Stack direction="row" sx={{ flexWrap: "wrap", gap: "12px", alignItems: "flex-start" }}>
        {/* left: the cepstral transform and the scan-space maps */}
        <Box>
          <TabBar<LeftTab>
            ariaLabel="Left panel" colors={colors} value={leftTab} onChange={setLeftTab}
            tabs={[
              { value: "transform", label: `Cepstral transform (${mode}${mode === "gamma" ? ` ${gamma}` : ""}${hannWindow ? ", Hann" : ""})` },
              { value: "real", label: "Real space", title: "Virtual ADF or BF image of the scan; drag a box on it to set the strain reference" },
              ...([0, 1, 2, 3, 4] as const).map((k) => ({
                value: k as LeftTab, label: STRAIN_PANELS[k].label, disabled: !hasStrain,
                title: hasStrain ? ("title" in STRAIN_PANELS[k] ? (STRAIN_PANELS[k] as { title: string }).title : undefined) : "Fit to see the strain",
              })),
            ]}
          />
          {leftTab === "transform" ? (
            <Box ref={transformBoxRef} sx={panelBox}>
              <canvas
                ref={transformCanvasRef} width={PANEL_PX} height={PANEL_PX} aria-label="Mean transform"
                data-quantem-scientific-output={transform ? "showstrain-transform" : undefined}
                style={{ width: PANEL_PX, height: PANEL_PX, display: "block", cursor: locked ? "grab" : "crosshair", touchAction: "none", imageRendering: "pixelated" }}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={() => { hoverRef.current = null; requestDraw(); }}
                onDoubleClick={(e) => { e.preventDefault(); cancelPendingClick(); resetView(); }}
                onWheel={handleWheel}
              />
              {computing && (
                <Box sx={{ position: "absolute", top: 4, right: 6, fontSize: 10, px: "4px", bgcolor: "rgba(0,0,0,0.6)", color: "#fff" }}>computing…</Box>
              )}
            </Box>
          ) : leftTab === "real" ? (
            <Box sx={scanMapBox}>
              {realBitmap ? (
                <canvas
                  ref={(el) => { scanCanvasRefs.current.real = el; }} width={bigW} height={bigH} aria-label="Real space map"
                  data-quantem-scientific-output="showstrain-real-space"
                  style={{ width: bigW, height: bigH, imageRendering: "pixelated", display: "block", cursor: canPickReference ? "crosshair" : "default", touchAction: "none" }}
                  {...scanHandlers("real")}
                />
              ) : computing && realSpaceMode ? (
                <Typography sx={{ ...typography.label, color: colors.textMuted }}>Computing the real-space image…</Typography>
              ) : locked ? (
                <Typography sx={{ ...typography.label, color: colors.textMuted }}>Not computed before export</Typography>
              ) : (
                <Box sx={{ textAlign: "center", px: "24px" }}>
                  <Button size="small" variant="outlined" disabled={computing} sx={{ ...button, borderColor: colors.border }}
                    onClick={() => setRealSpaceMode(realSpaceMode || "adf")}>
                    Compute {(realSpaceMode || "adf").toUpperCase()} Image
                  </Button>
                  <Typography sx={{ ...typography.labelSmall, color: colors.textMuted, mt: "6px" }}>
                    Reads every pattern of the {scanRows}×{scanCols} scan once
                  </Typography>
                </Box>
              )}
            </Box>
          ) : (
            <Box sx={{ ...scanMapBox, opacity: fitStale ? 0.45 : 1 }}>
              <canvas
                ref={(el) => { scanCanvasRefs.current.big = el; }} width={bigW} height={bigH} aria-label={`${STRAIN_PANELS[leftTab].key} map large`}
                data-quantem-scientific-output={strainBitmaps[leftTab] ? `showstrain-${STRAIN_PANELS[leftTab].key}-large` : undefined}
                style={{ width: bigW, height: bigH, imageRendering: "pixelated", display: "block", cursor: canPickReference ? "crosshair" : "default", touchAction: "none" }}
                {...scanHandlers(leftTab)}
              />
              <canvas ref={(el) => { colorbarRefs.current.big = el; }} width={COLORBAR_W} height={bigH}
                aria-label={`${STRAIN_PANELS[leftTab].label} colorbar large`} style={{ width: COLORBAR_W, height: bigH, display: "block" }} />
            </Box>
          )}
          <Box sx={{ display: "flex", alignItems: "center", gap: "8px", mt: "2px", minHeight: 22, maxWidth: PANEL_PX, flexWrap: "wrap" }}>
            {leftTab === "transform" ? (
              <Typography sx={{ ...typography.labelSmall, color: colors.textMuted }}>
                {locked
                  ? "Standalone page: zoom, pan and hover work; rerun in Jupyter to change the lattice."
                  : !seeds[0] ? "Click a lattice peak to place g1. Scroll to zoom, drag to pan, double-click to reset."
                  : !seeds[1] ? "Click a second, non-collinear peak to place g2."
                  : "Drag a vector tip to re-seed it; the lattice it predicts follows the pointer."}
              </Typography>
            ) : leftTab === "real" ? (
              <>
                <Box sx={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
                  <Typography sx={{ ...typography.label, color: colors.text }}>Detector</Typography>
                  <Select size="small" value={realSpaceMode || "adf"} disabled={locked} sx={themedSelect} MenuProps={menuProps}
                    inputProps={{ "aria-label": "Real space detector" }} onChange={(e) => setRealSpaceMode(String(e.target.value))}>
                    <MenuItem value="adf">ADF</MenuItem>
                    <MenuItem value="bf">BF</MenuItem>
                  </Select>
                </Box>
                <Typography sx={{ ...typography.labelSmall, color: colors.textMuted }}>
                  {realSpaceInfo}{realSpaceInfo && canPickReference ? " · drag a box to set the reference" : ""}
                </Typography>
              </>
            ) : (
              <Typography sx={{ ...typography.labelSmall, color: colors.textMuted }}>
                {`${strainLabel(leftTab)}${fitStale ? " · parameters changed since the fit" : ""}${canPickReference ? " · drag a box to set the reference" : ""}`}
              </Typography>
            )}
          </Box>
        </Box>

        {/* right: diffraction space, to compare against; the switch hides it */}
        <Box>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "8px" }}>
            {showDiffraction ? (
              <TabBar<RightTab>
                ariaLabel="Right panel" colors={colors} value={rightTab} onChange={setRightTab}
                tabs={[
                  { value: "diffraction", label: "Mean diffraction" },
                  { value: "fourier", label: "Fourier transform", title: "|FFT| of the mean diffraction pattern (no intensity scaling or mask), on the cepstral transform's zoom and pan" },
                ]}
              />
            ) : <Box />}
            <SwitchPair label="Diffraction panel" checked={!!showDiffraction} colors={colors} onChange={setShowDiffraction}
              title="Show the mean diffraction pattern and its Fourier transform beside the cepstral transform" />
          </Box>
          {showDiffraction && (rightTab === "diffraction" ? (
            <Box sx={panelBox}>
              <canvas
                ref={dpCanvasRef} width={PANEL_PX} height={PANEL_PX} aria-label="Mean diffraction"
                data-quantem-scientific-output={dpBitmap ? "showstrain-mean-diffraction" : undefined}
                style={{ width: PANEL_PX, height: PANEL_PX, imageRendering: "pixelated", display: "block" }}
                onPointerMove={dpReadoutMove}
              />
            </Box>
          ) : (
            <Box ref={fftBoxRef} sx={panelBox}>
              <canvas
                ref={fftCanvasRef} width={PANEL_PX} height={PANEL_PX} aria-label="Fourier transform"
                data-quantem-scientific-output={dpFft ? "showstrain-fourier-transform" : undefined}
                style={{ width: PANEL_PX, height: PANEL_PX, display: "block", cursor: "grab", touchAction: "none", imageRendering: "pixelated" }}
                onPointerDown={(e) => {
                  if (e.button !== 0) return;
                  (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
                  const [x, y] = panelPoint(e, PANEL_PX);
                  fftDragRef.current = { startX: x, startY: y, startPanX: viewRef.current.panX, startPanY: viewRef.current.panY };
                }}
                onPointerMove={(e) => {
                  const [x, y] = panelPoint(e, PANEL_PX);
                  fftHoverRef.current = { x, y };
                  const drag = fftDragRef.current;
                  if (drag) viewRef.current = { ...viewRef.current, panX: drag.startPanX + x - drag.startX, panY: drag.startPanY + y - drag.startY };
                  requestDraw();
                }}
                onPointerUp={() => { fftDragRef.current = null; }}
                onPointerLeave={() => { fftHoverRef.current = null; }}
                onDoubleClick={(e) => { e.preventDefault(); resetView(); }}
                onWheel={handleWheel}
              />
            </Box>
          ))}
          {showDiffraction && (
            <Typography sx={{ ...typography.labelSmall, color: colors.textMuted, mt: "2px", minHeight: 22, maxWidth: PANEL_PX }}>
              {rightTab === "diffraction"
                ? (diffractionMask ? (maskPreviewActive ? "Cyan: pixels the threshold would mask" : "Red: masked detector (feathered)") : "Log scale · no detector mask")
                : "|FFT| of the mean pattern, no intensity scaling or mask · same zoom, pan and Contrast as the cepstral transform"}
            </Typography>
          )}
        </Box>
      </Stack>

      {/* ------------------------------------------------ readouts and status */}
      <Box sx={{ mt: "2px", mb: "6px" }}>
        <Typography sx={{ ...typography.value, minHeight: 14, color: colors.accent }}>
          <span ref={readoutRef} />
        </Typography>
        <Stack direction="row" sx={{ flexWrap: "wrap", columnGap: "16px", rowGap: "2px", alignItems: "baseline" }}>
          {VECTOR_NAMES.map((name, i) => (
            <Typography key={name} sx={{ ...typography.value, fontSize: 11, color: seeds[i] ? VECTOR_COLORS[i] : colors.textMuted }}>
              {name} {vectorReadout(i as 0 | 1)}
            </Typography>
          ))}
          {angle !== null && <Typography sx={{ ...typography.value, fontSize: 11, color: colors.text }}>angle {angle.toFixed(2)}°</Typography>}
          {refineAllPeaks && (
            <Typography sx={{ ...typography.value, fontSize: 11, color: peakPreview ? PREVIEW_COLOR : peakColor || "#ffd54f" }}>
              {peakPreview ? `${peakPreview.length} candidate peaks (preview)` : `${nPeaks} peaks in the lattice fit`}
            </Typography>
          )}
          <Button size="small" disabled={locked || (!seeds[0] && !seeds[1])} sx={button} onClick={() => commitVectors([null, null])}>Clear Vectors</Button>
        </Stack>
        <Typography aria-label="ShowStrain status" sx={{ ...typography.label, mt: "2px", color: isError ? "#d32f2f" : colors.text, wordBreak: "break-word" }}>
          {statusMessage}
        </Typography>
        {(warningMessages || []).length > 0 && (
          <Typography sx={{ ...typography.labelSmall, color: colors.textMuted }} title={(warningMessages || []).join("\n")}>
            quantem: {(warningMessages || [])[0]}{(warningMessages || []).length > 1 ? ` (+${warningMessages.length - 1} more)` : ""}
          </Typography>
        )}
        {(localExportStatus || exportStatus) && (
          <Typography sx={{ ...typography.labelSmall, color: /^Export failed/.test(exportStatus || "") ? "#d32f2f" : colors.textMuted }}>
            {localExportStatus || exportStatus}
          </Typography>
        )}
      </Box>

      {/* ------------------------------------------------ controls: one bordered row per step */}
      {showControls && (
        <Box>
          <Box sx={row}>
            {sectionLabel("Preprocess")}
            <Box sx={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
              <Typography sx={{ ...typography.label, color: colors.text }}>Mode</Typography>
              <Select size="small" value={mode} disabled={locked} sx={themedSelect} MenuProps={menuProps}
                inputProps={{ "aria-label": "Mode" }} onChange={(e) => setMode(String(e.target.value))}>
                {["linear", "log", "log-min", "gamma"].map((m) => <MenuItem key={m} value={m}>{m}</MenuItem>)}
              </Select>
            </Box>
            {mode === "gamma" && (
              <SliderPair label="Gamma" ariaLabel="Gamma" colors={colors} value={localGamma} min={0.05} max={1.5} step={0.05}
                disabled={locked} format={(v) => (v as number).toFixed(2)}
                onChange={(v) => setLocalGamma(v as number)} onCommit={(v) => setGamma(v as number)} />
            )}
            {mode === "log-min" && (
              <>
                <SliderPair label="Offset" ariaLabel="Log offset" colors={colors} value={localLogOffsetExp} min={-4} max={2} step={0.1}
                  disabled={locked} format={(v) => (10 ** (v as number)).toPrecision(2)}
                  onChange={(v) => setLocalLogOffsetExp(v as number)} onCommit={(v) => setLogOffset(Number((10 ** (v as number)).toPrecision(3)))} />
                <SwitchPair label="Relative" checked={logOffsetRelative} disabled={locked} colors={colors} onChange={setLogOffsetRelative}
                  title="log_offset in units of the mean min-shifted intensity" />
              </>
            )}
            <SwitchPair label="Hann" checked={hannWindow} disabled={locked} colors={colors} onChange={setHannWindow}
              title="Hann window before the FFT (needed for the cft method and DFT refinement)" />
            <Box sx={{ display: "inline-flex", alignItems: "center", gap: "6px" }} title="Average the mean transform over every skip-th scan row and column">
              <Typography sx={{ ...typography.label, color: colors.text }}>Skip</Typography>
              <Select size="small" value={skip} disabled={locked} sx={themedSelect} MenuProps={menuProps}
                inputProps={{ "aria-label": "Skip" }} onChange={(e) => setSkip(Number(e.target.value))}>
                {Array.from(new Set([1, 2, 3, 4, 5, 10, 20, 50, skip])).sort((a, b) => a - b).map((s) => (
                  <MenuItem key={s} value={s}>
                    {s === 1 ? "all" : `1/${s}`} ({(Math.ceil(scanRows / s) * Math.ceil(scanCols / s)).toLocaleString()} patterns)
                  </MenuItem>
                ))}
              </Select>
            </Box>
            <SwitchPair label="Detector mask" checked={diffractionMask} disabled={locked} colors={colors} onChange={setDiffractionMask}
              title="Mask detector pixels below a percentile of the mean pattern (diffraction_mask)" />
            {diffractionMask && (
              <>
                <SliderPair label="Below" ariaLabel="Mask threshold percentile" colors={colors} value={localMaskPercentile} min={0} max={99} step={1}
                  disabled={locked} format={(v) => `${(v as number).toFixed(0)} %`}
                  onChange={(v) => setLocalMaskPercentile(v as number)} onCommit={(v) => setMaskPercentile(v as number)} />
                <SliderPair label="Feather" ariaLabel="Mask edge blend" colors={colors} value={localEdgeBlend} min={1} max={128} step={1}
                  disabled={locked} format={(v) => `${(v as number).toFixed(0)} px`}
                  onChange={(v) => setLocalEdgeBlend(v as number)} onCommit={(v) => setMaskEdgeBlend(v as number)} />
              </>
            )}
          </Box>

          <Box sx={row}>
            {sectionLabel("Lattice")}
            <SwitchPair label="Gaussian" checked={refineGaussian} disabled={locked} colors={colors} onChange={setRefineGaussian}
              title="Refine each peak with an isotropic 2D Gaussian (else parabolic only)" />
            <SliderPair label="Radius" ariaLabel="Refine radius" colors={colors} value={localRadius} min={0.5} max={5} step={0.25}
              disabled={locked} format={(v) => `${(v as number).toFixed(2)} px`}
              onChange={(v) => setLocalRadius(v as number)} onCommit={(v) => setRefineRadius(v as number)} />
            <SwitchPair label="Multi-peak" checked={refineAllPeaks} disabled={locked} colors={colors} onChange={setRefineAllPeaks}
              title="Fit the basis to every detected peak by weighted least squares (refine_all_peaks)" />
            {refineAllPeaks && (
              <>
                <SliderPair label="Threshold" ariaLabel="Peak threshold percentile" colors={colors} value={localThreshold} min={0.9} max={0.9995} step={0.0005}
                  width={130} disabled={locked} format={(v) => (v as number).toFixed(4)}
                  title="Keep local maxima above this quantile of the transform (threshold_percentile)"
                  onChange={(v) => setLocalThreshold(v as number)} onCommit={(v) => setThresholdPercentile(v as number)} />
                <SliderPair label="Spacing" ariaLabel="Minimum peak spacing" colors={colors} value={localSpacing} min={0} max={40} step={0.5}
                  disabled={locked} format={(v) => `${(v as number).toFixed(1)} px`}
                  title="Minimum distance between kept peaks, brightest first (min_peak_spacing; 0 = off)"
                  onChange={(v) => setLocalSpacing(v as number)} onCommit={(v) => setMinPeakSpacing(v as number)} />
              </>
            )}
          </Box>
          <Box sx={row}>
            {sectionLabel("Display")}
            <SliderPair label="Contrast" ariaLabel="Transform contrast" colors={colors} value={contrast} min={0.25} max={8} step={0.25}
              format={(v) => `${(v as number).toFixed(2)}x`} onChange={(v) => setContrast(v as number)} onCommit={() => {}} />
            <Box sx={{ display: "inline-flex", alignItems: "center", gap: "6px" }} title="Color of the fitted peak circles (peak_color)">
              <Typography sx={{ ...typography.label, color: colors.text }}>Peaks</Typography>
              <input type="color" aria-label="Peak color" value={peakColor || "#ffd54f"} onChange={(e) => setPeakColor(e.target.value)}
                style={{ width: 28, height: 20, padding: 0, border: `1px solid ${colors.border}`, background: "none", cursor: "pointer" }} />
            </Box>
            <SliderPair label="Line" ariaLabel="Peak line width" colors={colors} value={localPeakLineWidth} min={0.5} max={6} step={0.5}
              width={70} title="Circle line width in screen px (peak_line_width)" format={(v) => `${(v as number).toFixed(1)} px`}
              onChange={(v) => setLocalPeakLineWidth(v as number)} onCommit={(v) => setPeakLineWidth(v as number)} />
            <SliderPair label="Size" ariaLabel="Peak circle size" colors={colors} value={localPeakSize} min={0.5} max={4} step={0.25}
              width={70} title="Circle radius scale; the radius also grows with the peak weight (peak_size)" format={(v) => `${(v as number).toFixed(2)}x`}
              onChange={(v) => setLocalPeakSize(v as number)} onCommit={(v) => setPeakSize(v as number)} />
          </Box>

          <Box sx={row}>
            {sectionLabel("Fit")}
            <Box sx={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
              <Typography sx={{ ...typography.label, color: colors.text }}>Method</Typography>
              <Select size="small" value={fitMethod} disabled={locked} sx={themedSelect} MenuProps={menuProps}
                inputProps={{ "aria-label": "Fit method" }} onChange={(e) => setFitMethod(String(e.target.value))}>
                <MenuItem value="gaussian">gaussian</MenuItem>
                <MenuItem value="cft">cft</MenuItem>
              </Select>
            </Box>
            <SliderPair label="Fit radius" ariaLabel="Fit refine radius" colors={colors} value={localFitRadius} min={0.5} max={6} step={0.25}
              disabled={locked} title="fit_lattice refine_radius_px (gaussian: fit window; cft: maximum peak move)"
              format={(v) => `${(v as number).toFixed(2)} px`}
              onChange={(v) => setLocalFitRadius(v as number)} onCommit={(v) => setFitRadius(v as number)} />
            <SwitchPair label="DFT" checked={refineDft} disabled={locked || fitMethod === "cft"} colors={colors} onChange={setRefineDft}
              title="DFT-upsampled refinement after the Gaussian fit (needs Hann)" />
            <Button
              size="small" variant="outlined"
              disabled={locked || computing || !asVec(g1Refined) || !asVec(g2Refined)}
              sx={{ ...button, borderColor: colors.border }}
              onClick={() => setFitRequest((fitRequest || 0) + 1)}
            >
              {hasStrain ? "Refit" : "Fit"} {(scanRows * scanCols).toLocaleString()} Positions
            </Button>
            {fitStale && <Typography sx={{ ...typography.label, color: colors.accent }}>Parameters changed since the fit · Refit</Typography>}
            {fitProgress?.total ? (() => {
              const done = fitProgress.done ?? 0;
              const total = fitProgress.total ?? 1;
              const elapsed = fitProgress.elapsed_s ?? 0;
              const left = done > 0 ? (elapsed * (total - done)) / done : NaN;
              const clock = (sec: number) => (sec >= 60 ? `${Math.floor(sec / 60)} min ${Math.round(sec % 60)} s` : `${Math.round(sec)} s`);
              return (
                <Box sx={{ minWidth: 240, flex: "1 1 240px", maxWidth: 520 }} aria-label="Fit progress">
                  <LinearProgress variant="determinate" value={(100 * done) / total} sx={{ height: 6, borderRadius: 0 }} />
                  <Typography sx={{ ...typography.value, color: colors.text, fontVariantNumeric: "tabular-nums", mt: "2px" }}>
                    {done.toLocaleString()} / {total.toLocaleString()} positions · {clock(elapsed)}
                    {Number.isFinite(left) && done < total ? ` · ~${clock(left)} left` : ""} on {device}
                  </Typography>
                </Box>
              );
            })() : null}
          </Box>

          {hasStrain && (
            <Box sx={row}>
              {sectionLabel("Strain")}
              <SliderPair label="Strain ±" ariaLabel="Strain range percent" colors={colors}
                value={Math.max(Math.abs(localStrainRange[0]), Math.abs(localStrainRange[1]))} min={0.05} max={10} step={0.05}
                title="Symmetric color range, so zero strain is white (plot_strain strain_range_percent)"
                format={(v) => `${(v as number).toFixed(2)} %`}
                onChange={(v) => setLocalStrainRange([-(v as number), v as number])} onCommit={(v) => setStrainRange([-(v as number), v as number])} />
              <SliderPair label="Rotation ±" ariaLabel="Rotation range degrees" colors={colors}
                value={Math.max(Math.abs(localRotationRange[0]), Math.abs(localRotationRange[1]))} min={0.05} max={10} step={0.05}
                title="Symmetric color range of the rotation (plot_strain rotation_range_degrees)"
                format={(v) => `${(v as number).toFixed(2)}°`}
                onChange={(v) => setLocalRotationRange([-(v as number), v as number])} onCommit={(v) => setRotationRange([-(v as number), v as number])} />
              <SwitchPair label="Mask" checked={!!applyMask} colors={colors} onChange={setApplyMask}
                title="Dim low-weight positions as plot_strain(mask_range=...) does" />
              <SliderPair label="Range" ariaLabel="Mask range" colors={colors} value={localMaskRange} min={0} max={1} step={0.01}
                disabled={!applyMask} width={110}
                title="plot_strain mask_range: weight <= low is black, >= high full color"
                format={(v) => `${(v as number[])[0].toFixed(2)}…${(v as number[])[1].toFixed(2)}`}
                onChange={(v) => setLocalMaskRange(v as number[])} onCommit={(v) => setMaskRange(v as number[])} />
              <SliderPair label="Rotation angle" ariaLabel="Rotation angle" colors={colors} value={localRotationAngle} min={-90} max={90} step={0.5}
                disabled={locked || fitStale} width={90} title="plot_strain rotation_angle: e_uu, e_vv, e_uv in a frame rotated by this angle"
                format={(v) => `${(v as number).toFixed(1)}°`}
                onChange={(v) => setLocalRotationAngle(v as number)} onCommit={(v) => setRotationAngle(v as number)} />
            </Box>
          )}
          {hasStrain && (
            <Box sx={row}>
              {sectionLabel("Reference")}
              <Box sx={{ display: "inline-flex", alignItems: "center", gap: "6px" }} title="Statistic of the reference lattice over the region (update_reference calculation_metric)">
                <Typography sx={{ ...typography.label, color: colors.text }}>Metric</Typography>
                <Select size="small" value={calculationMetric} disabled={locked || fitStale} sx={themedSelect} MenuProps={menuProps}
                  inputProps={{ "aria-label": "Reference metric" }} onChange={(e) => setCalculationMetric(String(e.target.value))}>
                  <MenuItem value="median">median</MenuItem>
                  <MenuItem value="mean">mean</MenuItem>
                </Select>
              </Box>
              <Button size="small" disabled={locked || fitStale || !(referenceRegion?.length === 4)} sx={button} onClick={() => setReferenceRegion([])}>
                Reset Reference
              </Button>
              <Typography sx={{ ...typography.label, color: colors.text }} aria-label="Reference status">
                {referenceMessage || "Drag a box on a strain or real-space map to choose the reference region."}
              </Typography>
              {referenceVectors?.length === 2 && (
                <Typography sx={{ ...typography.value, color: colors.textMuted }} title="Reference vectors in the scan frame (after the detector-to-scan rotation)">
                  g1_ref ({referenceVectors[0][0].toFixed(3)}, {referenceVectors[0][1].toFixed(3)}) · g2_ref ({referenceVectors[1][0].toFixed(3)}, {referenceVectors[1][1].toFixed(3)}) px
                </Typography>
              )}
            </Box>
          )}
        </Box>
      )}

      {/* ------------------------------------------------ every component at once */}
      {strainDisplay && (
        <Box sx={{ mt: "6px", opacity: fitStale ? 0.45 : 1 }}>
          <Stack direction="row" sx={{ flexWrap: "wrap", gap: "10px" }}>
            {STRAIN_PANELS.map((panel, k) => (
              <Box key={panel.key} title={k < WEIGHT_INDEX ? "Click to show it large on the left" : undefined}>
                <Typography
                  sx={{ ...typography.label, color: leftTab === k ? colors.accent : colors.text, cursor: k < WEIGHT_INDEX ? "pointer" : "default" }}
                  onClick={() => k < WEIGHT_INDEX && setLeftTab(k as LeftTab)}
                >
                  {strainLabel(k)}
                </Typography>
                <Stack direction="row" sx={{ gap: "3px" }}>
                  <canvas
                    ref={(el) => { scanCanvasRefs.current[`small-${k}`] = el; }}
                    width={smallW} height={smallH}
                    data-quantem-scientific-output={strainBitmaps[k] ? `showstrain-${panel.key}` : undefined}
                    aria-label={`${panel.key} map`}
                    style={{ width: smallW, height: smallH, imageRendering: "pixelated", border: `1px solid ${leftTab === k ? colors.accent : colors.border}`, display: "block", cursor: canPickReference ? "crosshair" : "default", touchAction: "none" }}
                    onDoubleClick={() => k < WEIGHT_INDEX && setLeftTab(k as LeftTab)}
                    {...scanHandlers(k)}
                  />
                  <canvas ref={(el) => { colorbarRefs.current[`small-${k}`] = el; }} width={COLORBAR_W} height={smallH}
                    aria-label={`${panel.label} colorbar`} style={{ width: COLORBAR_W, height: smallH, display: "block" }} />
                </Stack>
              </Box>
            ))}
          </Stack>
        </Box>
      )}
    </Box>
  );
}

// A notebook saved with save_state=False keeps a static picture instead of the images. Reopened
// without a kernel, the widget shows that picture; live, it hides the picture's separate output.
function ShowStrainRoot() {
  const model = useModel();
  const rootRef = React.useRef<HTMLDivElement>(null);
  const [transformBytes] = useModelState<DataView>("transform_bytes");
  const [preview] = useModelState<string>("_static_fallback_image");
  const [previewMime] = useModelState<string>("_static_fallback_mime");
  const hasPixels = extractBytes(transformBytes).length > 0;
  useHideStaticFallback(model, rootRef, hasPixels);
  if (!hasPixels && preview) {
    return (
      <div ref={rootRef} style={{ fontFamily: WIDGET_UI_FONT, padding: 8 }}>
        <img src={`data:${previewMime || "image/png"};base64,${preview}`} alt="Saved ShowStrain view" style={{ maxWidth: "100%", display: "block" }} />
        <div style={{ fontSize: 12, opacity: 0.7 }}>Saved preview · rerun the cell for interaction</div>
      </div>
    );
  }
  return (
    <div ref={rootRef}>
      <ShowStrainWidget />
    </div>
  );
}

export default { render: createRender(ShowStrainRoot) };
