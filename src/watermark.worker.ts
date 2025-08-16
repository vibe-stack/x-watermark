/// <reference lib="webworker" />
/*
  Dedicated worker to find the X.com watermark off the main thread.
  Receives: { id: string, type: 'find', imageSrc: string }
  Replies:  { id, ok: true, result: { match, scaleToFull } } or { id, ok: false, error }
*/

export type Match = { x: number; y: number; w: number; h: number; score: number };

// ---------- Utilities (worker-safe) ----------
const toGray = (data: Uint8ClampedArray) => {
  const out = new Float32Array(data.length / 4);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    out[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return out;
};

const meanToScore = (mean: number) => 1 - mean / 255; // 0..1

const similarityMAD = (
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

const similarityNCC = (
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

const invertGray = (arr: Float32Array) => {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = 255 - arr[i];
  return out;
};

const drawToSizeOffscreen = async (
  bitmap: ImageBitmap,
  maxW: number
): Promise<{ canvas: OffscreenCanvas; scale: number }> => {
  const scale = bitmap.width > maxW ? maxW / bitmap.width : 1;
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, w, h);
  return { canvas, scale };
};

// Small helper to yield to the event loop (like rAF)
const tick = () => new Promise((r) => setTimeout(r, 0));

async function findWatermarkWorker(imageSrc: string): Promise<{ match: Match | null; scaleToFull: number }> {
  // Decode full image
  const imgBlob = await (await fetch(imageSrc)).blob();
  const imgBitmap = await createImageBitmap(imgBlob);

  // Downscale for faster search
  const { canvas: small, scale } = await drawToSizeOffscreen(imgBitmap, 720);
  const smallCtx = small.getContext('2d')!;
  const smallData = smallCtx.getImageData(0, 0, small.width, small.height);
  const smallGray = toGray(smallData.data);

  // Load template once per call (fast and cached by browser)
  const tplBlob = await (await fetch('/xcom_dark.png', { cache: 'force-cache' })).blob();
  const tplBitmap = await createImageBitmap(tplBlob);

  const scales = [0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8];
  const sx0 = Math.floor(small.width * 0.02);
  const sx1 = small.width - 4;
  const sy0 = Math.floor(small.height * 0.05);
  const sy1 = Math.floor(small.height * 0.55);
  let best: Match | null = null;

  const makeTplGray = (w: number, h: number) => {
    const c = new OffscreenCanvas(w, h);
    const c2d = c.getContext('2d')!;
    c2d.drawImage(tplBitmap, 0, 0, w, h);
    const tData = c2d.getImageData(0, 0, w, h);
    const tplGray = toGray(tData.data);
    const tplGrayInv = invertGray(tplGray);
    return { tplGray, tplGrayInv };
  };

  // Pass 1: search top half region densely
  for (let si = 0; si < scales.length; si++) {
    const s = scales[si];
    const w = Math.max(6, Math.round(tplBitmap.width * s));
    const h = Math.max(6, Math.round(tplBitmap.height * s));
    if (w >= small.width || h >= small.height) continue;

    const { tplGray, tplGrayInv } = makeTplGray(w, h);
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
      if (y % 30 === 0) await tick();
    }
    await tick();
  }

  // Pass 2: fallback widen search if needed
  if (!best || best.score < 0.88) {
    for (let si = 0; si < scales.length; si++) {
      const s = scales[si];
      const w = Math.max(6, Math.round(tplBitmap.width * s));
      const h = Math.max(6, Math.round(tplBitmap.height * s));
      if (w >= small.width || h >= small.height) continue;

      const { tplGray, tplGrayInv } = makeTplGray(w, h);
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
        if (y % 40 === 0) await tick();
      }
      await tick();
    }
  }

  return { match: best && best.score > 0.87 ? best : null, scaleToFull: 1 / scale };
}

// ---------- RPC wiring ----------
type Req = { id: string; type: 'find'; imageSrc: string };
type Res = { id: string; ok: boolean; result?: { match: Match | null; scaleToFull: number }; error?: string };

self.onmessage = async (evt: MessageEvent<Req>) => {
  const msg = evt.data as Req;
  if (!msg || msg.type !== 'find') return;
  try {
    if (!(self as unknown as { OffscreenCanvas?: unknown }).OffscreenCanvas) throw new Error('OffscreenCanvas not supported');
    const result = await findWatermarkWorker(msg.imageSrc);
    (self as unknown as { postMessage: (m: Res) => void }).postMessage({ id: msg.id, ok: true, result });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    (self as unknown as { postMessage: (m: Res) => void }).postMessage({ id: msg.id, ok: false, error: message });
  }
};
