// @ts-nocheck
import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { findPeakCandidates, latticeSites, maskedPixels, percentileLinear } from "./peaks";

// Python reference: quantem.diffraction's StrainMapAutocorrelation on a synthetic
// disk lattice. The drag previews must pick the same peaks as
// choose_lattice_vector(refine_all_peaks=True) and mask the same detector pixels
// as diffraction_mask, on the float32 image the browser receives. Skips when
// $PYTHON (default "python") lacks the class.
function strainReferenceFixture() {
  const python = process.env.PYTHON || "python";
  const code = String.raw`
import json
import sys
import warnings

import numpy as np

try:
    from quantem.diffraction import StrainMapAutocorrelation
    from quantem.diffraction.strain_autocorrelation import _refine_lattice_vectors
except ImportError:
    sys.exit(1)

rng = np.random.default_rng(1)
det = 48
rr, cc = np.mgrid[:det, :det].astype(float)
dp = np.zeros((det, det))
for h in range(-3, 4):
    for k in range(-3, 4):
        g = np.array([6.0 * h + 1.5 * k, 7.0 * k])
        dp += np.exp(-np.hypot(*g) / 15) * np.exp(-((rr - 24 - g[0]) ** 2 + (cc - 24 - g[1]) ** 2) / 2.0)
data = np.stack([rng.poisson(dp * 300) / 300.0 for _ in range(6)]).reshape(2, 3, det, det).astype(np.float32)
work = StrainMapAutocorrelation.from_array(data)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    work.preprocess(mode="gamma", gamma=0.5, plot_transform=False)
im32 = work.transform.array.astype(np.float32)
cases = []
for p, s in [(0.98, 0.0), (0.99, 0.0), (0.995, 3.0), (0.97, 6.5), (0.9, 10.0)]:
    _, _, pts, _, _ = _refine_lattice_vectors(
        im32.astype(np.float64), g1_rc=np.array([6.0, 0.0]), g2_rc=np.array([1.5, 7.0]),
        refine_gaussian=False, refine_all_peaks=True, threshold_percentile=p, min_peak_spacing=s)
    cases.append({"p": p, "s": s, "peaks": np.round(pts).astype(int).tolist()})
dp_mean = np.mean(data, axis=(0, 1))
masks = []
for q in (30.0, 50.0, 80.0):
    work.diffraction_mask(threshold_percentile=q, edge_blend=3.0, plot_mask=False)
    masks.append({"q": q, "masked": (work.mask_diffraction == 0).astype(int).ravel().tolist()})
print(json.dumps({"rows": det, "cols": det, "im": im32.astype(float).ravel().tolist(), "cases": cases,
                  "dp": dp_mean.astype(float).ravel().tolist(), "masks": masks,
                  "pct": [float(np.percentile(im32.astype(np.float64), q)) for q in (0, 12.5, 50, 99.75, 100)]}))
`;
  try {
    const out = execFileSync(python, ["-c", code], {
      encoding: "utf8", maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "pipe", "ignore"],
    });
    return JSON.parse(out);
  } catch {
    return null;
  }
}

const key = (rows) => rows.map(([r, c]) => `${r},${c}`).sort();
const fixture = strainReferenceFixture();

(fixture ? describe : describe.skip)("ShowStrain drag previews match StrainMapAutocorrelation", () => {
  const im = fixture ? Float32Array.from(fixture.im) : null;

  it("computes np.percentile with linear interpolation", () => {
    [0, 12.5, 50, 99.75, 100].forEach((q, i) => expect(percentileLinear(im, q)).toBeCloseTo(fixture.pct[i], 12));
  });

  for (const ref of fixture ? fixture.cases : []) {
    it(`picks quantem's peak candidates (threshold ${ref.p}, spacing ${ref.s})`, () => {
      const found = findPeakCandidates(im, fixture.rows, fixture.cols, ref.p, ref.s);
      expect(key(found.map((p) => [p.row, p.col]))).toEqual(key(ref.peaks));
    });
  }

  for (const ref of fixture ? fixture.masks : []) {
    it(`masks quantem's detector pixels (below ${ref.q} %)`, () => {
      const dp = Float32Array.from(fixture.dp);  // np.mean of the float32 scan is float32
      expect(Array.from(maskedPixels(dp, fixture.rows, fixture.cols, ref.q))).toEqual(ref.masked);
    });
  }
});

describe("ShowStrain lattice preview", () => {
  it("lists every h*g1 + k*g2 site in the box except the origin", () => {
    const sites = latticeSites([2, 0], [0, 3], 4, 3);
    expect(sites).toHaveLength(5 * 3 - 1);
    expect(sites).toContainEqual([-4, 3]);
    expect(sites).not.toContainEqual([0, 0]);
  });

  it("keeps the brightest peak and drops a closer, dimmer neighbour", () => {
    const image = new Float32Array(25);
    image[1 * 5 + 1] = 5;  // offset (-1, -1)
    image[3 * 5 + 2] = 3;  // offset (1, 0): 2.2 px from the first
    expect(findPeakCandidates(image, 5, 5, 0.5, 0).map((p) => [p.row, p.col])).toEqual([[-1, -1], [1, 0]]);
    expect(findPeakCandidates(image, 5, 5, 0.5, 3).map((p) => [p.row, p.col])).toEqual([[-1, -1]]);
  });
});
