// Colocated image processing utilities and hook logic extracted from page.tsx

export type Match = { x: number; y: number; w: number; h: number; score: number };

// ---------- Utilities ----------
export const loadImage = (src: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });

export const rAF = () => new Promise((r) => requestAnimationFrame(() => r(null)));

export const toGray = (data: Uint8ClampedArray) => {
  const out = new Float32Array(data.length / 4);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    out[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return out;
};

export const drawToSize = (
  img: HTMLImageElement,
  maxW: number
): { canvas: HTMLCanvasElement; scale: number } => {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  const scale = img.width > maxW ? maxW / img.width : 1;
  canvas.width = Math.round(img.width * scale);
  canvas.height = Math.round(img.height * scale);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  return { canvas, scale };
};

export const meanToScore = (mean: number) => 1 - mean / 255; // 0..1

export const similarityMAD = (
  src: Float32Array,
  srcW: number,
  srcX: number,
  srcY: number,
  tpl: Float32Array,
  tplW: number,
  tplH: number,
  sampleStep = 2,
  abortIfBelow = 0.85
) => {
  let acc = 0;
  let count = 0;
  const maxMean = (1 - abortIfBelow) * 255;
  for (let y = 0; y < tplH; y += sampleStep) {
    const srcRow = (srcY + y) * srcW + srcX;
    const tplRow = y * tplW;
    for (let x = 0; x < tplW; x += sampleStep) {
      const s = src[srcRow + x];
      const t = tpl[tplRow + x];
      acc += Math.abs(s - t);
      count++;
    }
    const mean = acc / Math.max(1, count);
    if (mean > maxMean) return meanToScore(mean);
  }
  const mean = acc / Math.max(1, count);
  return meanToScore(mean);
};

export const similarityNCC = (
  src: Float32Array,
  srcW: number,
  srcX: number,
  srcY: number,
  tpl: Float32Array,
  tplW: number,
  tplH: number,
  sampleStep = 2
) => {
  let n = 0;
  let sumS = 0,
    sumT = 0,
    sumS2 = 0,
    sumT2 = 0,
    sumST = 0;
  for (let y = 0; y < tplH; y += sampleStep) {
    const sRow = (srcY + y) * srcW + srcX;
    const tRow = y * tplW;
    for (let x = 0; x < tplW; x += sampleStep) {
      const sv = src[sRow + x];
      const tv = tpl[tRow + x];
      n++;
      sumS += sv;
      sumT += tv;
      sumS2 += sv * sv;
      sumT2 += tv * tv;
      sumST += sv * tv;
    }
  }
  if (n === 0) return 0;
  const meanS = sumS / n;
  const meanT = sumT / n;
  const varS = Math.max(1e-6, sumS2 - n * meanS * meanS);
  const varT = Math.max(1e-6, sumT2 - n * meanT * meanT);
  const cov = sumST - n * meanS * meanT;
  const corr = cov / Math.sqrt(varS * varT);
  return Math.max(0, Math.min(1, (corr + 1) / 2));
};

export const invertGray = (arr: Float32Array) => {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = 255 - arr[i];
  return out;
};

// ---------- Matching ----------
export const findWatermark = async (
  fullImg: HTMLImageElement
): Promise<{ match: Match | null; scaleToFull: number }> => {
  const { canvas: small, scale } = drawToSize(fullImg, 720);
  const smallCtx = small.getContext("2d")!;
  const smallData = smallCtx.getImageData(0, 0, small.width, small.height);
  const smallGray = toGray(smallData.data);
  const tplImg = await loadImage(`${import.meta.env.BASE_URL}xcom_dark.png`);
  const scales = [0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8];
  const sx0 = Math.floor(small.width * 0.02);
  const sx1 = small.width - 4;
  const sy0 = Math.floor(small.height * 0.05);
  const sy1 = Math.floor(small.height * 0.55);
  let best: Match | null = null;
  for (let si = 0; si < scales.length; si++) {
    const s = scales[si];
    const w = Math.max(6, Math.round(tplImg.width * s));
    const h = Math.max(6, Math.round(tplImg.height * s));
    if (w >= small.width || h >= small.height) continue;
    const tCanvas = document.createElement("canvas");
    tCanvas.width = w;
    tCanvas.height = h;
    const tCtx = tCanvas.getContext("2d")!;
    tCtx.drawImage(tplImg, 0, 0, w, h);
    const tData = tCtx.getImageData(0, 0, w, h);
    const tplGray = toGray(tData.data);
    const tplGrayInv = invertGray(tplGray);
    const step = 3;
    for (let y = sy0; y <= sy1 - h; y += step) {
      for (let x = sx0; x <= sx1 - w; x += step) {
        const quickMAD = Math.max(
          similarityMAD(smallGray, small.width, x, y, tplGray, w, h, 3, 0.82),
          similarityMAD(smallGray, small.width, x, y, tplGrayInv, w, h, 3, 0.82)
        );
        if (quickMAD < 0.84) continue;
        const refineDarkMAD = similarityMAD(smallGray, small.width, x, y, tplGray, w, h, 1, 0.88);
        const refineLightMAD = similarityMAD(smallGray, small.width, x, y, tplGrayInv, w, h, 1, 0.88);
        const refineDarkNCC = similarityNCC(smallGray, small.width, x, y, tplGray, w, h, 2);
        const refineLightNCC = similarityNCC(smallGray, small.width, x, y, tplGrayInv, w, h, 2);
        const refine = Math.max(
          0.5 * refineDarkMAD + 0.5 * refineDarkNCC,
          0.5 * refineLightMAD + 0.5 * refineLightNCC
        );
        if (!best || refine > best.score) {
          best = { x, y, w, h, score: refine };
        }
      }
      if (y % 30 === 0) await rAF();
    }
    await rAF();
  }
  if (!best || best.score < 0.88) {
    for (let si = 0; si < scales.length; si++) {
      const s = scales[si];
      const w = Math.max(6, Math.round(tplImg.width * s));
      const h = Math.max(6, Math.round(tplImg.height * s));
      if (w >= small.width || h >= small.height) continue;
      const tCanvas = document.createElement("canvas");
      tCanvas.width = w;
      tCanvas.height = h;
      const tCtx = tCanvas.getContext("2d")!;
      tCtx.drawImage(tplImg, 0, 0, w, h);
      const tData = tCtx.getImageData(0, 0, w, h);
      const tplGray = toGray(tData.data);
      const tplGrayInv = invertGray(tplGray);
      const step = 4;
      for (let y = 0; y <= small.height - h; y += step) {
        for (let x = 0; x <= small.width - w; x += step) {
          const q = Math.max(
            similarityMAD(smallGray, small.width, x, y, tplGray, w, h, 3, 0.8),
            similarityMAD(smallGray, small.width, x, y, tplGrayInv, w, h, 3, 0.8)
          );
          if (q < 0.82) continue;
          const refine = Math.max(
            0.5 * similarityMAD(smallGray, small.width, x, y, tplGray, w, h, 1, 0.86) +
              0.5 * similarityNCC(smallGray, small.width, x, y, tplGray, w, h, 2),
            0.5 * similarityMAD(smallGray, small.width, x, y, tplGrayInv, w, h, 1, 0.86) +
              0.5 * similarityNCC(smallGray, small.width, x, y, tplGrayInv, w, h, 2)
          );
          if (!best || refine > best.score) best = { x, y, w, h, score: refine };
        }
        if (y % 40 === 0) await rAF();
      }
      await rAF();
    }
  }
  return { match: best && best.score > 0.87 ? best : null, scaleToFull: 1 / scale };
};

// ---------- Inpainting ----------
export const stripeVariance = (
  data: Uint8ClampedArray,
  w: number,
  x0: number,
  x1: number,
  y0: number,
  y1: number
) => {
  let n = 0,
    mr = 0,
    mg = 0,
    mb = 0,
    vr = 0,
    vg = 0,
    vb = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const i = (y * w + x) * 4;
      const r = data[i],
        g = data[i + 1],
        b = data[i + 2];
      n++;
      const dr = r - mr;
      mr += dr / n;
      vr += dr * (r - mr);
      const dg = g - mg;
      mg += dg / n;
      vg += dg * (g - mg);
      const db = b - mb;
      mb += db / n;
      vb += db * (b - mb);
    }
  }
  const denom = Math.max(1, n - 1);
  return (vr / denom + vg / denom + vb / denom) / 3;
};

export const inpaintRect = (
  ctx: CanvasRenderingContext2D,
  rect: { x: number; y: number; w: number; h: number }
) => {
  const { x, y, w, h } = rect;
  const img = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const data = img.data;
  const leftX0 = Math.max(0, x - 6);
  const leftX1 = Math.max(0, x - 1);
  const rightX0 = Math.min(ctx.canvas.width - 1, x + w);
  const rightX1 = Math.min(ctx.canvas.width - 1, x + w + 5);
  const y0 = Math.max(0, y - 2);
  const y1 = Math.min(ctx.canvas.height - 1, y + h + 2);
  const leftVar = leftX1 >= leftX0 ? stripeVariance(data, img.width, leftX0, leftX1, y0, y1) : Number.POSITIVE_INFINITY;
  const rightVar = rightX1 >= rightX0 ? stripeVariance(data, img.width, rightX0, rightX1, y0, y1) : Number.POSITIVE_INFINITY;
  const useLeft = leftVar <= rightVar;
  for (let yy = y; yy < y + h; yy++) {
    for (let xx = x; xx < x + w; xx++) {
      const srcX = useLeft ? Math.max(0, x - 1) : Math.min(ctx.canvas.width - 1, x + w);
      const dstIdx = (yy * img.width + xx) * 4;
      const srcIdx = (yy * img.width + srcX) * 4;
      data[dstIdx] = data[srcIdx];
      data[dstIdx + 1] = data[srcIdx + 1];
      data[dstIdx + 2] = data[srcIdx + 2];
    }
  }
  const bx0 = Math.max(1, x - 1),
    by0 = Math.max(1, y - 1),
    bx1 = Math.min(img.width - 2, x + w + 1),
    by1 = Math.min(img.height - 2, y + h + 1);
  const copy = new Uint8ClampedArray(data);
  const k = [
    1, 1, 1,
    1, 2, 1,
    1, 1, 1,
  ];
  const ks = 10;
  for (let yy = by0; yy <= by1; yy++) {
    for (let xx = bx0; xx <= bx1; xx++) {
      let r = 0, g = 0, b = 0, a = 0;
      let ki = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const idx = ((yy + dy) * img.width + (xx + dx)) * 4;
          const kk = k[ki++];
          r += copy[idx] * kk;
          g += copy[idx + 1] * kk;
          b += copy[idx + 2] * kk;
          a += copy[idx + 3] * kk;
        }
      }
      const di = (yy * img.width + xx) * 4;
      data[di] = r / ks;
      data[di + 1] = g / ks;
      data[di + 2] = b / ks;
      data[di + 3] = a / ks;
    }
  }
  ctx.putImageData(img, 0, 0);
};
