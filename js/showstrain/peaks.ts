/**
 * Drag previews for ShowStrain: the peak candidates and the detector mask that
 * quantem's StrainMapAutocorrelation will use once a slider is released.
 *
 * Both mirror the NumPy code they preview (quantem.diffraction
 * strain_autocorrelation `_find_initial_peaks_weights` and `diffraction_mask`) on
 * the float32 copy of the image the browser holds. They are previews only: the
 * committed peaks, weights and lattice vectors always come back from Python.
 */

/** One candidate peak: offset from the transform centre (rows // 2, cols // 2) and its value. */
export type PeakCandidate = { row: number; col: number; value: number };

/**
 * `np.percentile(values, q)` with NumPy's default linear interpolation; q in [0, 100].
 * `float32` repeats NumPy's lerp in float32, as it runs on a float32 array.
 */
export function percentileLinear(values: ArrayLike<number>, q: number, float32 = false): number {
  const sorted = Float64Array.from(values as ArrayLike<number>).sort();
  const n = sorted.length;
  if (n === 0) return NaN;
  const position = (Math.min(100, Math.max(0, q)) / 100) * (n - 1);
  const lower = Math.floor(position);
  const upper = Math.min(n - 1, lower + 1);
  const round = float32 ? Math.fround : (v: number) => v;
  const t = round(position - lower);
  const a = sorted[lower];
  const b = sorted[upper];
  const diff = round(b - a);
  // numpy's _lerp: from the nearer end, so t = 1 returns b exactly
  return t >= 0.5 ? round(b - round(diff * round(1 - t))) : round(a + round(diff * t));
}

/**
 * Peaks `choose_basis_vectors(refine_all_peaks=True)` starts from: pixels strictly
 * greater than their 8 neighbours (periodic, as `np.roll`) and than the
 * `thresholdFraction` quantile of the image, in row-major order; with
 * `minSpacing > 0`, kept greedily from the brightest down when every kept peak
 * is at least `minSpacing` pixels away.
 */
export function findPeakCandidates(
  image: ArrayLike<number>,
  rows: number,
  cols: number,
  thresholdFraction: number,
  minSpacing: number,
): PeakCandidate[] {
  if (rows <= 0 || cols <= 0 || image.length < rows * cols) return [];
  const threshold = percentileLinear(image, thresholdFraction * 100);
  const centerRow = Math.floor(rows / 2);
  const centerCol = Math.floor(cols / 2);
  const found: PeakCandidate[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const value = image[r * cols + c];
      if (!(value > threshold)) continue;
      let isMax = true;
      for (let dr = -1; dr <= 1 && isMax; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          const nr = (r + dr + rows) % rows;
          const nc = (c + dc + cols) % cols;
          if (!(value > image[nr * cols + nc])) { isMax = false; break; }
        }
      }
      if (isMax) found.push({ row: r - centerRow, col: c - centerCol, value });
    }
  }
  if (!(minSpacing > 0)) return found;
  const byValue = found.slice().sort((a, b) => b.value - a.value);
  const kept: PeakCandidate[] = [];
  for (const peak of byValue) {
    if (kept.every((other) => Math.hypot(other.row - peak.row, other.col - peak.col) >= minSpacing)) kept.push(peak);
  }
  return kept;
}

/**
 * Pixels `diffraction_mask(threshold_percentile=percent)` masks out before
 * feathering: the mean pattern below its `percent` percentile, plus the
 * detector border. 1 = masked. Pass the float32 mean of a float32 scan, as
 * NumPy computes the threshold in the array's precision.
 */
export function maskedPixels(meanPattern: ArrayLike<number>, rows: number, cols: number, percent: number): Uint8Array {
  const out = new Uint8Array(rows * cols);
  if (meanPattern.length < rows * cols) return out;
  const threshold = percentileLinear(meanPattern, percent, meanPattern instanceof Float32Array);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const border = r === 0 || c === 0 || r === rows - 1 || c === cols - 1;
      out[r * cols + c] = border || meanPattern[r * cols + c] < threshold ? 1 : 0;
    }
  }
  return out;
}

/**
 * Lattice sites h*g1 + k*g2 (offsets from the centre) inside a half-extent box,
 * excluding the origin: the positions a basis predicts, drawn while it is dragged.
 */
export function latticeSites(
  g1: [number, number],
  g2: [number, number],
  halfRows: number,
  halfCols: number,
  maxIndex = 12,
): [number, number][] {
  const sites: [number, number][] = [];
  for (let h = -maxIndex; h <= maxIndex; h++) {
    for (let k = -maxIndex; k <= maxIndex; k++) {
      if (h === 0 && k === 0) continue;
      const row = h * g1[0] + k * g2[0];
      const col = h * g1[1] + k * g2[1];
      if (Math.abs(row) <= halfRows && Math.abs(col) <= halfCols) sites.push([row, col]);
    }
  }
  return sites;
}
