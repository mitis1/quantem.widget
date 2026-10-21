/**
 * ShowStrain: pick lattice vectors on the mean cepstral transform of a 4D-STEM
 * scan, tune the preprocessing and multi-peak refinement, and view the strain maps.
 *
 * Python (quantem's StrainMapAutocorrelation) owns every scientific result. This
 * view draws the float32 images it receives and turns interactions into trait
 * commits: a slider release, a dropped vector, a switch. While a pointer or a
 * slider moves, only browser-local previews change (the dragged vector and the
 * lattice it predicts, the peak candidates, the masked detector pixels), drawn
 * from refs once per animation frame; nothing reaches the kernel until release.
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
import Slider, { type SliderProps } from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import Typography from "@mui/material/Typography";
import { useTheme, type ThemeColors } from "../theme";
import { downloadBlob, extractBytes, extractFloat32, preserveRestoredWidgetModelsOnSave } from "../format";
import { useHideStaticFallback } from "../staticFallback";
import { COLORMAPS } from "../display/colormaps";
import { createColormapEngine, type DisplayColormapEngine } from "../display/cpuColormap";
import { RenderPathBadge } from "../shared/RenderPathBadge";
import { sliderStyles } from "../controlStyles";
import { drawScaleBarInRegion } from "../figure";
import { findPeakCandidates, latticeSites, maskedPixels, type PeakCandidate } from "./peaks";

const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const TRANSFORM_PX = 448;
const DP_PX = 224;
const STRAIN_MAX_H = 300;
const STRAIN_MAX_W = 300;
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
// strain_bytes order (STRAIN_COMPONENTS in showstrain.py), then the fit weight
const STRAIN_PANELS = [
  { key: "e_rr", label: "e_rr", unit: "%" },
  { key: "e_cc", label: "e_cc", unit: "%" },
  { key: "e_rc", label: "e_rc", unit: "%" },
  { key: "phi", label: "phi", unit: "deg" },
  { key: "weight", label: "weight", unit: "" },
] as const;
const SLOT_TRANSFORM = 0;
const SLOT_DP = 1;
const SLOT_STRAIN = 2;  // 2..6

type Vec = [number, number];
type View = { zoom: number; panX: number; panY: number };
type Drag =
  | { mode: "pending"; startX: number; startY: number; startPanX: number; startPanY: number }
  | { mode: "pan"; startX: number; startY: number; startPanX: number; startPanY: number }
  | { mode: "vector"; index: 0 | 1; at: Vec };

const compactButton = { fontSize: 11, py: 0.25, px: 1, minWidth: 0, textTransform: "none" as const };

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

/** A bounded slider row: label, slider and a tabular value, kept together when the row wraps. */
function SliderPair(props: {
  label: string; ariaLabel: string; value: number | number[]; min: number; max: number; step: number;
  format: (v: number | number[]) => string; disabled?: boolean; width?: number; colors: ThemeColors;
  onChange: (v: number | number[]) => void; onCommit: (v: number | number[]) => void; title?: string;
}) {
  const { colors } = props;
  return (
    <Box sx={{ display: "inline-flex", alignItems: "center", gap: "6px", whiteSpace: "nowrap" }} title={props.title}>
      <Typography sx={{ fontSize: 11, color: colors.text }}>{props.label}</Typography>
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
        sx={{ ...sliderStyles.small, width: props.width ?? 110, color: colors.accent }}
      />
      <Typography sx={{ fontSize: 11, fontFamily: "monospace", minWidth: 52, color: colors.text, fontVariantNumeric: "tabular-nums" }}>
        {props.format(props.value)}
      </Typography>
    </Box>
  );
}

function SwitchPair(props: { label: string; checked: boolean; disabled?: boolean; colors: ThemeColors; onChange: (v: boolean) => void; title?: string }) {
  return (
    <Box sx={{ display: "inline-flex", alignItems: "center", gap: "2px", whiteSpace: "nowrap" }} title={props.title}>
      <Typography sx={{ fontSize: 11, color: props.colors.text }}>{props.label}</Typography>
      <Switch
        size="small"
        checked={props.checked}
        disabled={props.disabled}
        slotProps={{ input: { "aria-label": props.label } }}
        onChange={(e) => props.onChange(e.target.checked)}
      />
    </Box>
  );
}

/** Colormap one float32 image on the shared engine (WebGPU, or the matching JS path). */
async function colormapToBitmap(
  engine: DisplayColormapEngine, slot: number, data: Float32Array, width: number, height: number,
  vmin: number, vmax: number, cmap: string,
): Promise<ImageBitmap | null> {
  engine.uploadData(slot, data, width, height);
  const out = await engine.renderSlotsToImageBitmapAsync([slot], [{ vmin, vmax }], false, cmap, COLORMAPS[cmap] ?? COLORMAPS.gray);
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

  const [croppingFactor] = useModelState<number>("cropping_factor");
  const [strainRange, setStrainRange] = useModelState<number[]>("strain_range_percent");
  const [rotationRange, setRotationRange] = useModelState<number[]>("rotation_range_degrees");
  const [referenceRegion, setReferenceRegion] = useModelState<number[]>("reference_region");
  const [calculationMetric, setCalculationMetric] = useModelState<string>("calculation_metric");
  const [referenceVectors] = useModelState<number[][]>("reference_vectors");
  const [referenceMessage] = useModelState<string>("reference_message");
  const [rotationAngle, setRotationAngle] = useModelState<number>("rotation_angle");
  const [maskRange, setMaskRange] = useModelState<number[]>("mask_range");
  const [fitRadius, setFitRadius] = useModelState<number | null>("fit_refine_radius_px");
  const [fitProgress] = useModelState<{ done?: number; total?: number; elapsed_s?: number }>("fit_progress");
  const [peakColor, setPeakColor] = useModelState<string>("peak_color");
  const [peakLineWidth, setPeakLineWidth] = useModelState<number>("peak_line_width");
  const [peakSize, setPeakSize] = useModelState<number>("peak_size");

  const [dpBytes] = useModelState<DataView>("dp_bytes");
  const [maskBytes] = useModelState<DataView>("mask_bytes");
  const [transformBytes] = useModelState<DataView>("transform_bytes");
  const [transformVmax] = useModelState<number>("transform_vmax");
  const [g1Refined] = useModelState<number[]>("g1_refined");
  const [g2Refined] = useModelState<number[]>("g2_refined");
  const [nPeaks] = useModelState<number>("n_peaks");
  const [peaksBytes] = useModelState<DataView>("peaks_bytes");
  const [strainBytes] = useModelState<DataView>("strain_bytes");
  const [weightBytes] = useModelState<DataView>("weight_bytes");
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
  const mask = React.useMemo(() => extractFloat32(maskBytes, dpRows * dpCols), [maskBytes, dpRows, dpCols]);
  const peaks = React.useMemo(() => {
    const raw = extractFloat32(peaksBytes, nPeaks * 5);
    if (!raw) return [] as { row: number; col: number; weight: number }[];
    const out = [];
    for (let i = 0; i < nPeaks; i++) out.push({ row: raw[i * 5], col: raw[i * 5 + 1], weight: raw[i * 5 + 2] });
    return out;
  }, [peaksBytes, nPeaks]);
  const strain = React.useMemo(() => extractFloat32(strainBytes, 4 * scanRows * scanCols), [strainBytes, scanRows, scanCols]);
  const weight = React.useMemo(() => extractFloat32(weightBytes, scanRows * scanCols), [weightBytes, scanRows, scanCols]);

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

  // ---------------------------------------------------------------- local (preview) control values
  const [localGamma, setLocalGamma] = React.useState(gamma);
  React.useEffect(() => setLocalGamma(gamma), [gamma]);
  const [localMaskPercentile, setLocalMaskPercentile] = React.useState(maskPercentile);
  React.useEffect(() => setLocalMaskPercentile(maskPercentile), [maskPercentile]);
  const [localEdgeBlend, setLocalEdgeBlend] = React.useState(maskEdgeBlend);
  React.useEffect(() => setLocalEdgeBlend(maskEdgeBlend), [maskEdgeBlend]);
  const [localRadius, setLocalRadius] = React.useState(refineRadius);
  React.useEffect(() => setLocalRadius(refineRadius), [refineRadius]);
  const [localThreshold, setLocalThreshold] = React.useState(thresholdPercentile);
  React.useEffect(() => setLocalThreshold(thresholdPercentile), [thresholdPercentile]);
  const [localSpacing, setLocalSpacing] = React.useState(minPeakSpacing);
  React.useEffect(() => setLocalSpacing(minPeakSpacing), [minPeakSpacing]);
  const [localStrainRange, setLocalStrainRange] = React.useState<number[]>(strainRange);
  React.useEffect(() => setLocalStrainRange(strainRange), [strainRange]);
  const [localRotationRange, setLocalRotationRange] = React.useState<number[]>(rotationRange);
  React.useEffect(() => setLocalRotationRange(rotationRange), [rotationRange]);
  const [localLogOffsetExp, setLocalLogOffsetExp] = React.useState(Math.log10(logOffset));
  React.useEffect(() => setLocalLogOffsetExp(Math.log10(logOffset)), [logOffset]);
  const [localMaskRange, setLocalMaskRange] = React.useState<number[]>(maskRange);
  React.useEffect(() => setLocalMaskRange(maskRange), [maskRange]);
  const [localRotationAngle, setLocalRotationAngle] = React.useState(rotationAngle);
  React.useEffect(() => setLocalRotationAngle(rotationAngle), [rotationAngle]);
  const shownFitRadius = fitRadius ?? refineRadius;
  const [localFitRadius, setLocalFitRadius] = React.useState(shownFitRadius);
  React.useEffect(() => setLocalFitRadius(shownFitRadius), [shownFitRadius]);
  const [localPeakLineWidth, setLocalPeakLineWidth] = React.useState(peakLineWidth);
  React.useEffect(() => setLocalPeakLineWidth(peakLineWidth), [peakLineWidth]);
  const [localPeakSize, setLocalPeakSize] = React.useState(peakSize);
  React.useEffect(() => setLocalPeakSize(peakSize), [peakSize]);
  const [contrast, setContrast] = React.useState(1);

  // Peak candidates the threshold / spacing sliders would give, while they differ from the committed values.
  const peakPreviewActive = refineAllPeaks && (localThreshold !== thresholdPercentile || localSpacing !== minPeakSpacing);
  const peakPreview = React.useMemo<PeakCandidate[] | null>(() => {
    if (!peakPreviewActive || !transform) return null;
    return findPeakCandidates(transform, dpRows, dpCols, localThreshold, localSpacing);
  }, [peakPreviewActive, transform, dpRows, dpCols, localThreshold, localSpacing]);
  const maskPreviewActive = diffractionMask && localMaskPercentile !== maskPercentile;

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

  const baseScale = dpRows > 0 && dpCols > 0 ? TRANSFORM_PX / Math.max(dpRows, dpCols) : 1;
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
      TRANSFORM_PX / 2 + (centerCol + col + 0.5 - dpCols / 2) * s + panX,
      TRANSFORM_PX / 2 + (centerRow + row + 0.5 - dpRows / 2) * s + panY,
    ];
  }, [baseScale, centerRow, centerCol, dpRows, dpCols]);
  const screenToOffset = React.useCallback((x: number, y: number): Vec => {
    const { zoom, panX, panY } = viewRef.current;
    const s = baseScale * zoom;
    return [
      (y - TRANSFORM_PX / 2 - panY) / s + dpRows / 2 - 0.5 - centerRow,
      (x - TRANSFORM_PX / 2 - panX) / s + dpCols / 2 - 0.5 - centerCol,
    ];
  }, [baseScale, centerRow, centerCol, dpRows, dpCols]);

  // Shown vectors: the refined ones once Python returns them, else the picked seeds.
  const shownVectors: (Vec | null)[] = [asVec(g1Refined) ?? asVec(g1), asVec(g2Refined) ?? asVec(g2)];
  const seeds: (Vec | null)[] = [asVec(g1), asVec(g2)];

  drawRef.current = () => {
    const canvas = transformCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, TRANSFORM_PX, TRANSFORM_PX);
    const bitmap = transformBitmapRef.current;
    const { zoom, panX, panY } = viewRef.current;
    const s = baseScale * zoom;
    if (bitmap) {
      ctx.drawImage(bitmap, TRANSFORM_PX / 2 - (dpCols / 2) * s + panX, TRANSFORM_PX / 2 - (dpRows / 2) * s + panY, dpCols * s, dpRows * s);
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
        if (x < -4 || y < -4 || x > TRANSFORM_PX + 4 || y > TRANSFORM_PX + 4) continue;
        ctx.beginPath();
        ctx.moveTo(x - 3, y); ctx.lineTo(x + 3, y);
        ctx.moveTo(x, y - 3); ctx.lineTo(x, y + 3);
        ctx.stroke();
      }
    }
    // committed refined peaks, radius by weight
    if (refineAllPeaks && peaks.length && !peakPreview) {
      const wmax = Math.max(...peaks.map((p) => Math.abs(p.weight)), 1e-30);
      for (const p of peaks) {
        const [x, y] = offsetToScreen(p.row, p.col);
        const radius = localPeakSize * (3 + 6 * Math.sqrt(Math.abs(p.weight) / wmax));
        // dark halo under the colored ring keeps it visible on bright peaks
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
      ctx.font = "bold 11px " + FONT;
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
    const calibrated = realUnits && realUnits !== "1/pixels" && samplingReal?.length === 2;
    drawScaleBarInRegion(ctx, { x: 0, y: 0, width: TRANSFORM_PX, height: TRANSFORM_PX }, zoom, s,
      calibrated ? (samplingReal[0] + samplingReal[1]) / 2 : 1, calibrated ? realUnits : "px");

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
        readout.textContent = `(${row.toFixed(2)}, ${col.toFixed(2)}) px · ${lengthLabel(Math.hypot(row, col))} · I ${Number.isFinite(value) ? value.toPrecision(4) : "-"}`;
      }
    }
  };

  // |r| in transform pixels and, when calibrated, in real-space units (sampling_real per pixel)
  const lengthLabel = (px: number) => {
    const calibrated = realUnits && realUnits !== "1/pixels" && samplingReal?.length === 2;
    const s = calibrated ? (samplingReal[0] + samplingReal[1]) / 2 : 1;
    return calibrated ? `|r| ${px.toFixed(2)} px = ${(px * s).toFixed(3)} ${realUnits}` : `|r| ${px.toFixed(2)} px`;
  };

  // redraw whenever anything the drawing reads changes
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

  // ---------------------------------------------------------------- transform pointer handling
  const pendingClickRef = React.useRef<number | null>(null);
  const cancelPendingClick = () => {
    if (pendingClickRef.current !== null) { window.clearTimeout(pendingClickRef.current); pendingClickRef.current = null; }
  };
  React.useEffect(() => () => cancelPendingClick(), []);

  const canvasPoint = (e: React.PointerEvent | React.WheelEvent | React.MouseEvent): [number, number] => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return [(e.clientX - rect.left) * (TRANSFORM_PX / rect.width), (e.clientY - rect.top) * (TRANSFORM_PX / rect.height)];
  };

  const commitVectors = (next: (Vec | null)[]) => {
    model.set("g1", next[0] ? [round2(next[0][0]), round2(next[0][1])] : []);
    model.set("g2", next[1] ? [round2(next[1][0]), round2(next[1][1])] : []);
    model.save_changes();
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    if (e.detail >= 2) { cancelPendingClick(); dragRef.current = null; return; }
    const [x, y] = canvasPoint(e);
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
    const [x, y] = canvasPoint(e);
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
      const [x, y] = canvasPoint(e);
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
  }, []);
  const handleWheel = (e: React.WheelEvent) => {
    const [x, y] = canvasPoint(e);
    const { zoom, panX, panY } = viewRef.current;
    const next = clamp(zoom * (e.deltaY > 0 ? 0.9 : 1.1), MIN_ZOOM, MAX_ZOOM);
    const px = (x - TRANSFORM_PX / 2 - panX) / zoom;
    const py = (y - TRANSFORM_PX / 2 - panY) / zoom;
    viewRef.current = { zoom: next, panX: x - TRANSFORM_PX / 2 - px * next, panY: y - TRANSFORM_PX / 2 - py * next };
    requestDraw();
  };

  // ---------------------------------------------------------------- mean diffraction pattern + mask
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
  const dpScale = dpRows && dpCols ? DP_PX / Math.max(dpRows, dpCols) : 1;
  React.useLayoutEffect(() => {
    const ctx = dpCanvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, DP_PX, DP_PX);
    const w = dpCols * dpScale;
    const h = dpRows * dpScale;
    if (dpBitmap) ctx.drawImage(dpBitmap, (DP_PX - w) / 2, (DP_PX - h) / 2, w, h);
    if (maskOverlay) ctx.drawImage(maskOverlay, (DP_PX - w) / 2, (DP_PX - h) / 2, w, h);
  }, [dpBitmap, maskOverlay, dpRows, dpCols, dpScale, colors.bg]);

  // ---------------------------------------------------------------- strain maps
  const strainScale = scanRows && scanCols ? Math.min(STRAIN_MAX_H / scanRows, STRAIN_MAX_W / scanCols) : 1;
  const strainW = Math.max(1, Math.round(scanCols * strainScale));
  const strainH = Math.max(1, Math.round(scanRows * strainScale));
  // plot_strain's frame: e_rr/e_cc/e_rc in the scan frame, e_uu/e_vv/e_uv once rotated
  const strainNames = rotationAngle ? ["e_uu", "e_vv", "e_uv", "phi", "weight"] : ["e_rr", "e_cc", "e_rc", "phi", "weight"];
  // plot_strain(mask_range): weight <= low is black, >= high full color, linear between
  const maskOverlayCanvas = React.useMemo(() => {
    if (!weight || !scanRows || !scanCols) return null;
    const [low, high] = localMaskRange;
    if (low <= 0 && high >= 1) return null;
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
  }, [weight, scanRows, scanCols, localMaskRange]);
  const strainRanges = [localStrainRange, localStrainRange, localStrainRange, localRotationRange, [0, 1]];
  const colorbarRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const strainCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const [strainBitmaps, setStrainBitmaps] = React.useState<(ImageBitmap | null)[]>([]);
  const strainDisplay = React.useMemo(() => {
    if (!strain) return null;
    const n = scanRows * scanCols;
    const panels: Float32Array[] = [];
    for (let k = 0; k < 4; k++) {
      const scale = k === 3 ? 180 / Math.PI : 100;  // phi in degrees, the others in percent
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = strain[k * n + i] * scale;
      panels.push(out);
    }
    panels.push(weight ?? new Float32Array(n).fill(1));
    return panels;
  }, [strain, weight, scanRows, scanCols]);
  React.useEffect(() => {
    if (!engine || !strainDisplay) { setStrainBitmaps([]); return; }
    let cancelled = false;
    const ranges = [localStrainRange, localStrainRange, localStrainRange, localRotationRange, [0, 1]];
    void Promise.all(strainDisplay.map((panel, k) => colormapToBitmap(
      engine, SLOT_STRAIN + k, panel, scanCols, scanRows, ranges[k][0], ranges[k][1], k === 4 ? "gray" : STRAIN_CMAP,
    ))).then((bitmaps) => { if (!cancelled) setStrainBitmaps(bitmaps); });
    return () => { cancelled = true; };
  }, [engine, strainDisplay, scanRows, scanCols, localStrainRange, localRotationRange]);
  // Reference region: the committed box (solid) or the box being dragged (dashed), drawn on every
  // panel so all five views of the selection move together. [row0, row1, col0, col1], half-open.
  const regionDragRef = React.useRef<{ start: Vec; end: Vec } | null>(null);
  const strainHoverRef = React.useRef<{ k: number; row: number; col: number } | null>(null);
  const strainReadoutRef = React.useRef<HTMLSpanElement>(null);
  const strainFrameRef = React.useRef<number | null>(null);
  const strainDrawRef = React.useRef<() => void>(() => {});
  const requestStrainDraw = React.useCallback(() => {
    if (strainFrameRef.current !== null) return;
    strainFrameRef.current = window.requestAnimationFrame(() => {
      strainFrameRef.current = null;
      strainDrawRef.current();
    });
  }, []);
  React.useEffect(() => () => { if (strainFrameRef.current !== null) window.cancelAnimationFrame(strainFrameRef.current); }, []);
  const regionFromDrag = (drag: { start: Vec; end: Vec }): number[] => [
    Math.min(drag.start[0], drag.end[0]), Math.max(drag.start[0], drag.end[0]) + 1,
    Math.min(drag.start[1], drag.end[1]), Math.max(drag.start[1], drag.end[1]) + 1,
  ];
  strainDrawRef.current = () => {
    const drag = regionDragRef.current;
    const region = drag ? regionFromDrag(drag) : (referenceRegion?.length === 4 ? referenceRegion : null);
    strainBitmaps.forEach((bitmap, k) => {
      const ctx = strainCanvasRefs.current[k]?.getContext("2d");
      if (!ctx) return;
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, strainW, strainH);
      if (bitmap) ctx.drawImage(bitmap, 0, 0, strainW, strainH);
      if (maskOverlayCanvas && k < 4) ctx.drawImage(maskOverlayCanvas, 0, 0, strainW, strainH);
      if (strainW >= 60) {
        const unit = scanUnits && scanUnits !== "pixels" ? scanUnits : "px";
        drawScaleBarInRegion(ctx, { x: 0, y: 0, width: strainW, height: strainH }, 1, strainScale, scanSampling || 1, unit,
          { targetBarPx: Math.min(60, strainW * 0.35), margin: 8, barThickness: 3, color: "#000", barShadowColor: "#fff", outline: { color: "#fff", width: 3 } });
      }
      if (!region) return;
      const sx = strainW / scanCols;
      const sy = strainH / scanRows;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(drag ? [3, 2] : []);
      ctx.strokeStyle = drag ? PREVIEW_COLOR : "#000";
      ctx.strokeRect(region[2] * sx + 0.75, region[0] * sy + 0.75, (region[3] - region[2]) * sx - 1.5, (region[1] - region[0]) * sy - 1.5);
      ctx.setLineDash([]);
    });
    const out = strainReadoutRef.current;
    const hover = strainHoverRef.current;
    if (!out) return;
    if (drag) {
      const [r0, r1, c0, c1] = regionFromDrag(drag);
      out.textContent = `reference rows ${r0}-${r1 - 1}, cols ${c0}-${c1 - 1} (${(r1 - r0) * (c1 - c0)} positions) · release to apply`;
    } else if (hover && strainDisplay) {
      const panel = { ...STRAIN_PANELS[hover.k], label: strainNames[hover.k] };
      const value = strainDisplay[hover.k][hover.row * scanCols + hover.col];
      const position = scanUnits && scanUnits !== "pixels"
        ? ` = (${(hover.row * scanSampling).toFixed(2)}, ${(hover.col * scanSampling).toFixed(2)}) ${scanUnits}` : "";
      out.textContent = `scan (${hover.row}, ${hover.col})${position} · ${panel.label} ${value.toFixed(hover.k === 4 ? 3 : 4)} ${panel.unit}`;
    } else {
      out.textContent = "";
    }
  };
  React.useLayoutEffect(() => { strainDrawRef.current(); }, [strainBitmaps, strainW, strainH, referenceRegion, maskOverlayCanvas]);
  // one vertical colorbar per panel: its colormap, range and zero
  React.useLayoutEffect(() => {
    colorbarRefs.current.forEach((canvas, k) => {
      const ctx = canvas?.getContext("2d");
      if (!ctx) return;
      const [vmin, vmax] = strainRanges[k];
      const lut = COLORMAPS[k === 4 ? "gray" : STRAIN_CMAP];
      const barW = 8;
      ctx.clearRect(0, 0, COLORBAR_W, strainH);
      for (let y = 0; y < strainH; y++) {
        const entry = Math.round((1 - y / Math.max(strainH - 1, 1)) * 255) * 3;
        ctx.fillStyle = `rgb(${lut[entry]},${lut[entry + 1]},${lut[entry + 2]})`;
        ctx.fillRect(0, y, barW, 1);
      }
      ctx.strokeStyle = colors.border;
      ctx.strokeRect(0.5, 0.5, barW - 1, strainH - 1);
      ctx.fillStyle = colors.text;
      ctx.font = `9px ${FONT}`;
      ctx.textAlign = "left";
      const fmt = (v: number) => (Math.abs(v) >= 10 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2));
      ctx.textBaseline = "top";
      ctx.fillText(fmt(vmax), barW + 3, 0);
      ctx.textBaseline = "bottom";
      ctx.fillText(fmt(vmin), barW + 3, strainH);
      if (vmin < 0 && vmax > 0 && strainH >= 40) {
        const y = (vmax / (vmax - vmin)) * strainH;
        ctx.textBaseline = "middle";
        ctx.fillText("0", barW + 3, y);
        ctx.fillRect(barW - 3, y, 3, 1);
      }
    });
  }, [strainH, colors.text, colors.border, strainBitmaps, ...strainRanges.flat()]);
  const scanPosition = (e: React.PointerEvent): Vec => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return [
      clamp(Math.floor(((e.clientY - rect.top) / rect.height) * scanRows), 0, scanRows - 1),
      clamp(Math.floor(((e.clientX - rect.left) / rect.width) * scanCols), 0, scanCols - 1),
    ];
  };
  const handleStrainDown = (e: React.PointerEvent) => {
    if (e.button !== 0 || locked || fitStale) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    const at = scanPosition(e);
    regionDragRef.current = { start: at, end: at };
    requestStrainDraw();
  };
  const handleStrainMove = (k: number, e: React.PointerEvent) => {
    const [row, col] = scanPosition(e);
    strainHoverRef.current = { k, row, col };
    if (regionDragRef.current) regionDragRef.current.end = [row, col];
    requestStrainDraw();
  };
  const handleStrainUp = () => {
    const drag = regionDragRef.current;
    regionDragRef.current = null;
    // a click without a drag leaves the reference alone
    if (drag && (drag.start[0] !== drag.end[0] || drag.start[1] !== drag.end[1])) setReferenceRegion(regionFromDrag(drag));
    requestStrainDraw();
  };

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
    fontSize: 11, height: 24, bgcolor: colors.controlBg, color: colors.text,
    "& .MuiSvgIcon-root": { color: colors.text },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: colors.accent },
  };
  const menuProps = { PaperProps: { sx: { bgcolor: colors.controlBg, color: colors.text, "& .MuiMenuItem-root": { fontSize: 11 } } } };
  const sectionLabel = (text: string) => (
    <Typography sx={{ fontSize: 11, fontWeight: 600, color: colors.text, minWidth: 64 }}>{text}</Typography>
  );
  const row = { display: "flex", flexWrap: "wrap" as const, alignItems: "center", columnGap: "14px", rowGap: "4px", py: "3px" };
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

  return (
    <Box className="showstrain-root" sx={{ p: "12px", bgcolor: colors.bg, color: colors.text, fontFamily: FONT, width: "100%", boxSizing: "border-box" }}>
      <Stack direction="row" alignItems="center" sx={{ mb: "6px", flexWrap: "wrap", columnGap: "8px" }}>
        <Typography sx={{ fontSize: 13, fontWeight: 600 }}>{title || "ShowStrain"}</Typography>
        <RenderPathBadge colors={colors} />
        <Typography sx={{ fontSize: 10, color: colors.textMuted }}>
          {scanRows}×{scanCols} scan · {dpRows}×{dpCols} {dataDtype} · fit on {device}
        </Typography>
        <Box sx={{ flex: 1 }} />
        <Button size="small" sx={{ ...compactButton, color: colors.accent }} onClick={resetView}>Reset View</Button>
        {exportEnabled && !offline && (
          <Button size="small" sx={{ ...compactButton, color: colors.accent }} onClick={handleExport} title="Standalone HTML: float32 transform and strain maps">
            Export HTML
          </Button>
        )}
      </Stack>

      <Stack direction="row" sx={{ flexWrap: "wrap", gap: "12px", alignItems: "flex-start" }}>
        <Box>
          <Typography sx={{ fontSize: 11, mb: "2px" }}>Mean diffraction (log)</Typography>
          <Box sx={{ border: `1px solid ${colors.border}`, width: DP_PX, height: DP_PX }}>
            <canvas
              ref={dpCanvasRef}
              width={DP_PX}
              height={DP_PX}
              data-quantem-scientific-output={dpBitmap ? "showstrain-mean-diffraction" : undefined}
              style={{ width: DP_PX, height: DP_PX, imageRendering: "pixelated", display: "block" }}
            />
          </Box>
          <Typography sx={{ fontSize: 10, color: colors.textMuted, mt: "2px", maxWidth: DP_PX }}>
            {diffractionMask ? (maskPreviewActive ? "Cyan: pixels the threshold would mask" : "Red: masked detector (feathered)") : "No detector mask"}
          </Typography>
        </Box>

        <Box>
          <Typography sx={{ fontSize: 11, mb: "2px" }}>
            Mean transform ({mode}{mode === "gamma" ? ` ${gamma}` : ""}{hannWindow ? ", Hann" : ""})
          </Typography>
          <Box ref={transformBoxRef} sx={{ border: `1px solid ${colors.border}`, width: TRANSFORM_PX, height: TRANSFORM_PX, position: "relative" }}>
            <canvas
              ref={transformCanvasRef}
              width={TRANSFORM_PX}
              height={TRANSFORM_PX}
              aria-label="Mean transform"
              data-quantem-scientific-output={transform ? "showstrain-transform" : undefined}
              style={{ width: TRANSFORM_PX, height: TRANSFORM_PX, display: "block", cursor: locked ? "grab" : "crosshair", touchAction: "none", imageRendering: "pixelated" }}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={() => { hoverRef.current = null; requestDraw(); }}
              onDoubleClick={(e) => { e.preventDefault(); cancelPendingClick(); resetView(); }}
              onWheel={handleWheel}
            />
            {computing && (
              <Box sx={{ position: "absolute", top: 4, right: 6, fontSize: 10, px: "4px", bgcolor: "rgba(0,0,0,0.6)", color: "#fff" }}>
                computing…
              </Box>
            )}
          </Box>
          <Typography sx={{ fontSize: 10, color: colors.textMuted, mt: "2px", minHeight: 14, maxWidth: TRANSFORM_PX }}>
            <span ref={readoutRef} style={{ color: colors.accent, fontFamily: "monospace" }} />
          </Typography>
          <Typography sx={{ fontSize: 10, color: colors.textMuted, maxWidth: TRANSFORM_PX }}>
            {locked
              ? "Standalone page: zoom, pan and hover work; rerun in Jupyter to change the lattice."
              : !seeds[0] ? "Click a lattice peak to place g1. Scroll to zoom, drag to pan, double-click to reset."
              : !seeds[1] ? "Click a second, non-collinear peak to place g2."
              : "Drag a vector tip to re-seed it; the lattice it predicts follows the pointer."}
          </Typography>
        </Box>

        <Box sx={{ minWidth: 260, maxWidth: 420, fontSize: 11 }}>
          {VECTOR_NAMES.map((name, i) => (
            <Typography key={name} sx={{ fontSize: 11, fontFamily: "monospace", color: seeds[i] ? VECTOR_COLORS[i] : colors.textMuted }}>
              {name} {vectorReadout(i as 0 | 1)}
            </Typography>
          ))}
          {angle !== null && (
            <Typography sx={{ fontSize: 11, fontFamily: "monospace" }}>angle(g1, g2) {angle.toFixed(2)}°</Typography>
          )}
          {refineAllPeaks && (
            <Typography sx={{ fontSize: 11, fontFamily: "monospace", color: peakColor || "#ffd54f" }}>
              {peakPreview ? <span style={{ color: PREVIEW_COLOR }}>{peakPreview.length} candidate peaks (preview)</span> : `${nPeaks} peaks in the lattice fit`}
            </Typography>
          )}
          <Button
            size="small"
            disabled={locked || (!seeds[0] && !seeds[1])}
            sx={{ ...compactButton, color: colors.accent, mt: "4px" }}
            onClick={() => commitVectors([null, null])}
          >
            Clear Vectors
          </Button>
          <Typography
            aria-label="ShowStrain status"
            sx={{ fontSize: 11, mt: "6px", color: isError ? "#d32f2f" : colors.text, wordBreak: "break-word" }}
          >
            {statusMessage}
          </Typography>
          {(warningMessages || []).length > 0 && (
            <Typography sx={{ fontSize: 10, color: colors.textMuted, mt: "2px" }} title={(warningMessages || []).join("\n")}>
              quantem: {(warningMessages || [])[0]}{(warningMessages || []).length > 1 ? ` (+${warningMessages.length - 1} more)` : ""}
            </Typography>
          )}
          {(localExportStatus || exportStatus) && (
            <Typography sx={{ fontSize: 10, color: /^Export failed/.test(exportStatus || "") ? "#d32f2f" : colors.textMuted, mt: "2px" }}>
              {localExportStatus || exportStatus}
            </Typography>
          )}
        </Box>
      </Stack>

      {showControls && (
        <Box sx={{ mt: "8px", borderTop: `1px solid ${colors.border}`, pt: "4px" }}>
          <Box sx={row}>
            {sectionLabel("Preprocess")}
            <Box sx={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
              <Typography sx={{ fontSize: 11 }}>Mode</Typography>
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
              <Typography sx={{ fontSize: 11 }}>Skip</Typography>
              <Select size="small" value={skip} disabled={locked} sx={themedSelect} MenuProps={menuProps}
                inputProps={{ "aria-label": "Skip" }} onChange={(e) => setSkip(Number(e.target.value))}>
                {Array.from(new Set([1, 2, 3, 4, 5, 10, 20, 50, skip])).sort((a, b) => a - b).map((s) => (
                  <MenuItem key={s} value={s}>
                    {s === 1 ? "all" : `1/${s}`} ({(Math.ceil(scanRows / s) * Math.ceil(scanCols / s)).toLocaleString()} patterns)
                  </MenuItem>
                ))}
              </Select>
            </Box>
            <SwitchPair label="Mask" checked={diffractionMask} disabled={locked} colors={colors} onChange={setDiffractionMask}
              title="Mask detector pixels below a percentile of the mean pattern (diffraction_mask)" />
          </Box>
          {diffractionMask && (
            <Box sx={{ ...row, pl: "78px" }}>
              <SliderPair label="Mask below" ariaLabel="Mask threshold percentile" colors={colors} value={localMaskPercentile} min={0} max={99} step={1}
                disabled={locked} format={(v) => `${(v as number).toFixed(0)} %`}
                onChange={(v) => setLocalMaskPercentile(v as number)} onCommit={(v) => setMaskPercentile(v as number)} />
              <SliderPair label="Feather" ariaLabel="Mask edge blend" colors={colors} value={localEdgeBlend} min={1} max={128} step={1}
                disabled={locked} format={(v) => `${(v as number).toFixed(0)} px`}
                onChange={(v) => setLocalEdgeBlend(v as number)} onCommit={(v) => setMaskEdgeBlend(v as number)} />
            </Box>
          )}
          <Box sx={row}>
            {sectionLabel("Lattice")}
            <SwitchPair label="Gaussian" checked={refineGaussian} disabled={locked} colors={colors} onChange={setRefineGaussian}
              title="Refine each peak with an isotropic 2D Gaussian (else parabolic only)" />
            <SliderPair label="Radius" ariaLabel="Refine radius" colors={colors} value={localRadius} min={0.5} max={5} step={0.25}
              disabled={locked} format={(v) => `${(v as number).toFixed(2)} px`}
              onChange={(v) => setLocalRadius(v as number)} onCommit={(v) => setRefineRadius(v as number)} />
            <SwitchPair label="Multi-peak" checked={refineAllPeaks} disabled={locked} colors={colors} onChange={setRefineAllPeaks}
              title="Fit the basis to every detected peak by weighted least squares (refine_all_peaks)" />
            <Typography sx={{ fontSize: 11 }}>Contrast</Typography>
            <Slider size="small" aria-label="Transform contrast" value={contrast} min={0.25} max={8} step={0.25}
              onChange={(_, v) => setContrast(v as number)} sx={{ ...sliderStyles.small, width: 80, color: colors.accent }} />
          </Box>
          {refineAllPeaks && (
            <Box sx={{ ...row, pl: "78px" }}>
              <SliderPair label="Threshold" ariaLabel="Peak threshold percentile" colors={colors} value={localThreshold} min={0.9} max={0.9995} step={0.0005}
                width={150} disabled={locked} format={(v) => (v as number).toFixed(4)}
                title="Keep local maxima above this quantile of the transform (threshold_percentile)"
                onChange={(v) => setLocalThreshold(v as number)} onCommit={(v) => setThresholdPercentile(v as number)} />
              <SliderPair label="Spacing" ariaLabel="Minimum peak spacing" colors={colors} value={localSpacing} min={0} max={40} step={0.5}
                disabled={locked} format={(v) => `${(v as number).toFixed(1)} px`}
                title="Minimum distance between kept peaks, brightest first (min_peak_spacing; 0 = off)"
                onChange={(v) => setLocalSpacing(v as number)} onCommit={(v) => setMinPeakSpacing(v as number)} />
            </Box>
          )}
          {refineAllPeaks && (
            <Box sx={{ ...row, pl: "78px" }}>
              <Box sx={{ display: "inline-flex", alignItems: "center", gap: "6px" }} title="Color of the fitted peak circles (peak_color)">
                <Typography sx={{ fontSize: 11 }}>Peak color</Typography>
                <input type="color" aria-label="Peak color" value={peakColor || "#ffd54f"}
                  onChange={(e) => setPeakColor(e.target.value)}
                  style={{ width: 28, height: 20, padding: 0, border: `1px solid ${colors.border}`, background: "none", cursor: "pointer" }} />
              </Box>
              <SliderPair label="Line" ariaLabel="Peak line width" colors={colors} value={localPeakLineWidth} min={0.5} max={6} step={0.5}
                title="Circle line width in screen px (peak_line_width)" format={(v) => `${(v as number).toFixed(1)} px`}
                onChange={(v) => setLocalPeakLineWidth(v as number)} onCommit={(v) => setPeakLineWidth(v as number)} />
              <SliderPair label="Size" ariaLabel="Peak circle size" colors={colors} value={localPeakSize} min={0.5} max={4} step={0.25}
                title="Circle radius scale; the radius also grows with the peak weight (peak_size)" format={(v) => `${(v as number).toFixed(2)}x`}
                onChange={(v) => setLocalPeakSize(v as number)} onCommit={(v) => setPeakSize(v as number)} />
            </Box>
          )}
          <Box sx={row}>
            {sectionLabel("Fit")}
            <Box sx={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
              <Typography sx={{ fontSize: 11 }}>Method</Typography>
              <Select size="small" value={fitMethod} disabled={locked} sx={themedSelect} MenuProps={menuProps}
                inputProps={{ "aria-label": "Fit method" }} onChange={(e) => setFitMethod(String(e.target.value))}>
                <MenuItem value="gaussian">gaussian</MenuItem>
                <MenuItem value="cft">cft</MenuItem>
              </Select>
            </Box>
            <SliderPair label="Fit radius" ariaLabel="Fit refine radius" colors={colors} value={localFitRadius} min={0.5} max={6} step={0.25}
              disabled={locked} title="fit_lattice_vectors refine_radius_px (gaussian: fit window; cft: maximum peak move)"
              format={(v) => `${(v as number).toFixed(2)} px`}
              onChange={(v) => setLocalFitRadius(v as number)} onCommit={(v) => setFitRadius(v as number)} />
            <SwitchPair label="DFT" checked={refineDft} disabled={locked || fitMethod === "cft"} colors={colors} onChange={setRefineDft}
              title="DFT-upsampled refinement after the Gaussian fit (needs Hann)" />
            <Button
              size="small"
              variant="outlined"
              disabled={locked || computing || !asVec(g1Refined) || !asVec(g2Refined)}
              sx={{ ...compactButton, color: colors.accent, borderColor: colors.border }}
              onClick={() => setFitRequest((fitRequest || 0) + 1)}
            >
              Fit {scanRows * scanCols} Positions
            </Button>
            {fitStale && <Typography sx={{ fontSize: 11, color: colors.accent }}>Parameters changed since the fit · Fit again</Typography>}
          </Box>
          {fitProgress?.total ? (() => {
            const done = fitProgress.done ?? 0;
            const total = fitProgress.total ?? 1;
            const elapsed = fitProgress.elapsed_s ?? 0;
            const left = done > 0 ? (elapsed * (total - done)) / done : NaN;
            const clock = (sec: number) => (sec >= 60 ? `${Math.floor(sec / 60)} min ${Math.round(sec % 60)} s` : `${Math.round(sec)} s`);
            return (
              <Box sx={{ pl: "78px", pr: "8px", py: "2px", maxWidth: 640 }} aria-label="Fit progress">
                <LinearProgress variant="determinate" value={(100 * done) / total} sx={{ height: 6, borderRadius: 0 }} />
                <Typography sx={{ fontSize: 10, color: colors.text, fontVariantNumeric: "tabular-nums", mt: "2px" }}>
                  {done.toLocaleString()} / {total.toLocaleString()} positions · {clock(elapsed)}
                  {Number.isFinite(left) && done < total ? ` · ~${clock(left)} left` : ""} on {device}
                </Typography>
              </Box>
            );
          })() : null}
        </Box>
      )}

      {strainDisplay && (
        <Box sx={{ mt: "8px", borderTop: `1px solid ${colors.border}`, pt: "6px", opacity: fitStale ? 0.45 : 1 }}>
          <Box sx={row}>
            {sectionLabel("Strain")}
            <SliderPair label="Strain ±" ariaLabel="Strain range percent" colors={colors}
              value={Math.max(Math.abs(localStrainRange[0]), Math.abs(localStrainRange[1]))} min={0.05} max={10} step={0.05}
              title="Symmetric color range, so zero strain is white (plot_strain strain_range_percent)"
              format={(v) => `${(v as number).toFixed(2)} %`}
              onChange={(v) => setLocalStrainRange([-(v as number), v as number])} onCommit={(v) => setStrainRange([-(v as number), v as number])} />
            <SliderPair label="Rotation ±" ariaLabel="Rotation range degrees" colors={colors}
              value={Math.max(Math.abs(localRotationRange[0]), Math.abs(localRotationRange[1]))} min={0.05} max={10} step={0.05}
              title="Symmetric color range of phi (plot_strain rotation_range_degrees)"
              format={(v) => `${(v as number).toFixed(2)}°`}
              onChange={(v) => setLocalRotationRange([-(v as number), v as number])} onCommit={(v) => setRotationRange([-(v as number), v as number])} />
            <SliderPair label="Mask" ariaLabel="Mask range" colors={colors} value={localMaskRange} min={0} max={1} step={0.01}
              title="plot_strain mask_range: weight <= low is black, >= high full color"
              format={(v) => `${(v as number[])[0].toFixed(2)}…${(v as number[])[1].toFixed(2)}`}
              onChange={(v) => setLocalMaskRange(v as number[])} onCommit={(v) => setMaskRange(v as number[])} />
            <SliderPair label="Frame" ariaLabel="Strain frame rotation" colors={colors} value={localRotationAngle} min={-90} max={90} step={0.5}
              disabled={locked || fitStale} title="plot_strain rotation_angle: strain tensor in a frame rotated by this angle (e_uu, e_vv, e_uv)"
              format={(v) => `${(v as number).toFixed(1)}°`}
              onChange={(v) => setLocalRotationAngle(v as number)} onCommit={(v) => setRotationAngle(v as number)} />
          </Box>
          <Box sx={row}>
            {sectionLabel("Reference")}
            <Box sx={{ display: "inline-flex", alignItems: "center", gap: "6px" }} title="Statistic of the reference lattice over the region (update_reference calculation_metric)">
              <Typography sx={{ fontSize: 11 }}>Metric</Typography>
              <Select size="small" value={calculationMetric} disabled={locked || fitStale} sx={themedSelect} MenuProps={menuProps}
                inputProps={{ "aria-label": "Reference metric" }} onChange={(e) => setCalculationMetric(String(e.target.value))}>
                <MenuItem value="median">median</MenuItem>
                <MenuItem value="mean">mean</MenuItem>
              </Select>
            </Box>
            <Button size="small" disabled={locked || fitStale || !(referenceRegion?.length === 4)}
              sx={{ ...compactButton, color: colors.accent }} onClick={() => setReferenceRegion([])}>
              Reset Reference
            </Button>
            <Typography sx={{ fontSize: 11, color: colors.text }} aria-label="Reference status">
              {referenceMessage || "Drag a box on a strain map to choose the reference region."}
            </Typography>
            {referenceVectors?.length === 2 && (
              <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: colors.textMuted }} title="Reference vectors in the scan frame (after the detector-to-scan rotation)">
                g1_ref ({referenceVectors[0][0].toFixed(3)}, {referenceVectors[0][1].toFixed(3)}) · g2_ref ({referenceVectors[1][0].toFixed(3)}, {referenceVectors[1][1].toFixed(3)}) px
              </Typography>
            )}
          </Box>
          <Stack direction="row" sx={{ flexWrap: "wrap", gap: "10px", mt: "4px" }}>
            {STRAIN_PANELS.map((panel, k) => (
              <Box key={panel.key}>
                <Typography sx={{ fontSize: 11 }}>{strainNames[k]}{panel.unit ? ` (${panel.unit})` : ""}</Typography>
                <Stack direction="row" sx={{ gap: "3px" }}>
                <canvas
                  ref={(el) => { strainCanvasRefs.current[k] = el; }}
                  width={strainW}
                  height={strainH}
                  data-quantem-scientific-output={strainBitmaps[k] ? `showstrain-${panel.key}` : undefined}
                  aria-label={`${panel.key} map`}
                  style={{ width: strainW, height: strainH, imageRendering: "pixelated", border: `1px solid ${colors.border}`, display: "block", cursor: locked ? "default" : "crosshair", touchAction: "none" }}
                  onPointerDown={handleStrainDown}
                  onPointerMove={(e) => handleStrainMove(k, e)}
                  onPointerUp={handleStrainUp}
                  onPointerLeave={() => { strainHoverRef.current = null; requestStrainDraw(); }}
                />
                <canvas ref={(el) => { colorbarRefs.current[k] = el; }} width={COLORBAR_W} height={strainH}
                  aria-label={`${strainNames[k]} colorbar`} style={{ width: COLORBAR_W, height: strainH, display: "block" }} />
                </Stack>
              </Box>
            ))}
          </Stack>
          <Typography sx={{ fontSize: 10, color: colors.textMuted, mt: "2px", minHeight: 14 }}>
            <span ref={strainReadoutRef} style={{ color: colors.accent, fontFamily: "monospace" }} />
          </Typography>
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
  const [preview] = useModelState<string>("_static_fallback_jpeg");
  const [previewMime] = useModelState<string>("_static_fallback_mime");
  const hasPixels = extractBytes(transformBytes).length > 0;
  useHideStaticFallback(model, rootRef, hasPixels);
  if (!hasPixels && preview) {
    return (
      <div ref={rootRef} style={{ fontFamily: FONT, padding: 8 }}>
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
