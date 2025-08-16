import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { loadImage, inpaintRect, toGray, drawToSize } from "./image-processor";
import type { Match } from "./image-processor";
import WatermarkWorker from "./watermark.worker.ts?worker";

// Create a worker lazily when on the client
const createWatermarkWorker = () => new WatermarkWorker();

export default function Home() {
  // UI state
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [showOriginal, setShowOriginal] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);

  // DOM refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const fullCanvasRef = useRef<HTMLCanvasElement>(null); // hidden full-res canvas
  const workerRef = useRef<Worker | null>(null);
  type WorkerResult = { match: Match | null; scaleToFull: number };
  type WorkerRequest =
    | { id: string; type: "find"; imageSrc: string; templateUrl: string }
    | {
        id: string;
        type: "findGray";
        smallGray: Float32Array;
        smallW: number;
        smallH: number;
        tplGray: Float32Array;
        tplW: number;
        tplH: number;
        scaleToFull: number;
      };
  type WorkerFindGrayPayload = Omit<Extract<WorkerRequest, { type: "findGray" }>, "id" | "type">;
  type WorkerResponse = { id: string; ok: boolean; result?: WorkerResult; error?: string };
  const pendingRef = useRef(
    new Map<string, { resolve: (value: WorkerResult) => void; reject: (reason?: unknown) => void; timer?: number }>()
  );

  // Setup/teardown worker once
  useEffect(() => {
    if (typeof window === "undefined") return;
    // Lazy create on first use; but also pre-warm here to avoid lag
    try {
      if (!workerRef.current) workerRef.current = createWatermarkWorker();
    } catch (e) {
      console.error("Failed to create worker:", e);
      workerRef.current = null;
    }

    const w = workerRef.current;
    if (w) {
      w.onmessage = (evt: MessageEvent<WorkerResponse>) => {
        const { id, ok, result, error } = (evt.data || {}) as WorkerResponse;
        if (!id) return;
        const pending = pendingRef.current.get(id);
        if (!pending) return;
        clearTimeout(pending.timer);
        pendingRef.current.delete(id);
        if (ok && result) pending.resolve(result);
        else pending.reject(new Error(error || "Worker error"));
      };
      w.onerror = () => {
        // Fail all pending calls
        for (const [id, p] of pendingRef.current) {
          clearTimeout(p.timer);
          p.reject(new Error("Worker crashed"));
          pendingRef.current.delete(id);
        }
      };
    }

    return () => {
      const w2 = workerRef.current;
      if (w2) {
        try { w2.terminate(); } catch {}
      }
      workerRef.current = null;
    };
  }, []);

  const callFind = useCallback((imageSrc: string): Promise<WorkerResult> => {
    const w = workerRef.current;
    if (!w) return Promise.reject(new Error("Worker unavailable"));
    const id = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
    return new Promise<WorkerResult>((resolve, reject) => {
      const timer = window.setTimeout(() => {
        pendingRef.current.delete(id);
        reject(new Error("Worker timeout"));
      }, 30000);
      pendingRef.current.set(id, { resolve, reject, timer });
      const templateUrl = `${import.meta.env.BASE_URL}xcom_dark.png`;
      const msg: WorkerRequest = { id, type: "find", imageSrc, templateUrl };
      w.postMessage(msg);
    });
  }, []);

  const callFindGray = useCallback(
    (payload: WorkerFindGrayPayload): Promise<WorkerResult> => {
      const w = workerRef.current;
      if (!w) return Promise.reject(new Error("Worker unavailable"));
      const id = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
      return new Promise<WorkerResult>((resolve, reject) => {
        const timer = window.setTimeout(() => {
          pendingRef.current.delete(id);
          reject(new Error("Worker timeout"));
        }, 30000);
        pendingRef.current.set(id, { resolve, reject, timer });
        const msg: WorkerRequest = { id, type: "findGray", ...payload } as WorkerRequest;
        // Transfer underlying ArrayBuffers to avoid cloning cost
        const transfers: Transferable[] = [payload.smallGray.buffer, payload.tplGray.buffer];
        w.postMessage(msg as any, transfers);
      });
    },
    []
  );

  // ---------- UI handlers ----------
  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please select an image file.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      setOriginalImage(reader.result as string);
      setProcessedImage(null);
      setError(null);
      setStatus("");
      setShowOriginal(false);
    };
    reader.readAsDataURL(file);
  }, []);

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please drop an image file.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      setOriginalImage(reader.result as string);
      setProcessedImage(null);
      setError(null);
      setStatus("");
      setShowOriginal(false);
    };
    reader.readAsDataURL(file);
  }, []);

  const process = useCallback(async () => {
    if (!originalImage) return;
    setIsProcessing(true);
    setProcessedImage(null);
    setError(null);
    setStatus("Loading image…");
    try {
      const img = await loadImage(originalImage);

      // Prepare full-res canvas
      const fullCanvas = fullCanvasRef.current ?? document.createElement("canvas");
      const fullCtx = (fullCanvas.getContext("2d") as CanvasRenderingContext2D)!;
      fullCanvas.width = img.width;
      fullCanvas.height = img.height;
      fullCtx.drawImage(img, 0, 0);
      if (!fullCanvasRef.current) fullCanvasRef.current = fullCanvas;

      setStatus("Searching watermark…");
      let match: Match | null = null;
      let scaleToFull = 1;
      // Always use worker. Prefer array path (no OffscreenCanvas requirement); legacy as fallback.
      try {
        // Prepare small grayscale on main thread (fast) and send to worker
        const { canvas: smallCanvas, scale } = drawToSize(img, 720);
        const smallCtx = smallCanvas.getContext("2d")!;
        const smallData = smallCtx.getImageData(0, 0, smallCanvas.width, smallCanvas.height);
        const smallGray = toGray(smallData.data);
        // Template grayscale
        const tplImg = await loadImage(`${import.meta.env.BASE_URL}xcom_dark.png`);
        const tCanvas = document.createElement("canvas");
        tCanvas.width = tplImg.width;
        tCanvas.height = tplImg.height;
        const tCtx = tCanvas.getContext("2d")!;
        tCtx.drawImage(tplImg, 0, 0);
        const tData = tCtx.getImageData(0, 0, tplImg.width, tplImg.height);
        const tplGray = toGray(tData.data);
        const res = await callFindGray({
          smallGray,
          smallW: smallCanvas.width,
          smallH: smallCanvas.height,
          tplGray,
          tplW: tplImg.width,
          tplH: tplImg.height,
          scaleToFull: 1 / scale,
        });
        match = res.match;
        scaleToFull = res.scaleToFull;
      } catch (err) {
        console.warn("findGray failed, falling back to worker decode path:", err);
        const res = await callFind(originalImage);
        match = res.match;
        scaleToFull = res.scaleToFull;
      }
      if (!match) {
        setError("Could not detect the X.com watermark. Make sure the avatar+name area is visible.");
        setIsProcessing(false);
        return;
      }

      // Map rect to full resolution and add a tiny margin
      const rect = {
        x: Math.max(0, Math.round(match.x * scaleToFull) - 2),
        y: Math.max(0, Math.round(match.y * scaleToFull) - 2),
        w: Math.min(fullCanvas.width, Math.round(match.w * scaleToFull) + 4),
        h: Math.min(fullCanvas.height, Math.round(match.h * scaleToFull) + 4),
      };

      setStatus("Removing watermark…");
      inpaintRect(fullCtx, rect);

      const url = fullCanvas.toDataURL("image/png");
      setProcessedImage(url);
      setStatus("Done");
    } catch (e) {
      console.error(e);
      setError("Failed to process the image. Try another screenshot.");
    } finally {
      setIsProcessing(false);
    }
  }, [originalImage, callFind]);

  // Auto start processing when a new original image is set
  useEffect(() => {
    if (originalImage) {
      process();
    }
  }, [originalImage, process]);

  const download = useCallback(() => {
    // Deprecated: kept as a fallback in non-iOS browsers
    if (!processedImage) return;
    const a = document.createElement("a");
    a.href = processedImage;
    a.download = "xcom-unwatermarked.png";
    a.click();
  }, [processedImage]);

  // --- iOS-friendly saving ---
  const isIOS = typeof navigator !== "undefined" && /iP(hone|od|ad)/.test(navigator.userAgent);
  const isSafari = typeof navigator !== "undefined" && /Safari\/.+ Version\//.test(navigator.userAgent) && !/CriOS|FxiOS|EdgiOS/.test(navigator.userAgent);

  const saveToPhotos = useCallback(async () => {
    if (!processedImage) return;
    try {
      // Convert data URL to Blob
      const res = await fetch(processedImage);
      const blob = await res.blob();
      const fileName = "xcom-unwatermarked.png";

      // Prefer the Web Share API with files (iOS share sheet includes "Save Image")
      const file = new File([blob], fileName, { type: blob.type || "image/png" });
      const canShareFiles = typeof navigator !== "undefined" && (navigator as any).canShare?.({ files: [file] });
      if (canShareFiles && (navigator as any).share) {
        try {
          await (navigator as any).share({ files: [file], title: "Unwatermarked image" });
          return;
        } catch (err) {
          // If user cancels or share fails, fall through to next fallback
        }
      }

      // Fallback on iOS Safari: open the image in a new tab; users can long-press and "Save Image"
      if (isIOS && isSafari) {
        const objectUrl = URL.createObjectURL(blob);
        // Opening in a user gesture context should avoid popup blockers
        window.open(objectUrl, "_blank");
        // Revoke shortly after to avoid leaking; Safari keeps resource alive for open tab
        setTimeout(() => URL.revokeObjectURL(objectUrl), 10000);
        return;
      }

      // Desktop and others: standard download
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(objectUrl);
    } catch (e) {
      // Final fallback: try the original data URL download method
      download();
    }
  }, [processedImage, isIOS, isSafari, download]);

  const reset = () => {
    setOriginalImage(null);
    setProcessedImage(null);
    setError(null);
    setStatus("");
    if (fileInputRef.current) fileInputRef.current.value = "";
    setShowOriginal(false);
  };

  // ---------- Shredder Animation Component ----------
  const Shredder = useMemo(() => {
    function Component({ src }: { src: string }) {
      const canvasRef = useRef<HTMLCanvasElement>(null);
      const imgRef = useRef<HTMLImageElement | null>(null);
      const rafRef = useRef<number | null>(null);
      const t0Ref = useRef<number | null>(null);

      useEffect(() => {
        let cancelled = false;
        loadImage(src).then((img) => {
          if (cancelled) return;
          imgRef.current = img;
          start();
        });

        function start() {
          const canvas = canvasRef.current;
          if (!canvas || !imgRef.current) return;
          const ctx = canvas.getContext("2d")!;
          // Reduce DPR for better performance on mobile
          const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
          const { clientWidth: cw, clientHeight: ch } = canvas;
          canvas.width = Math.max(1, Math.floor(cw * dpr));
          canvas.height = Math.max(1, Math.floor(ch * dpr));
          ctx.scale(dpr, dpr);

          // Performance optimization
          ctx.imageSmoothingEnabled = false;

          const img = imgRef.current;
          const scale = Math.min(cw / img.width, ch / img.height);
          const dw = img.width * scale;
          const dh = img.height * scale;
          const dx = (cw - dw) / 2;
          const dy = (ch - dh) / 2;

          const sliceW = 20; // Narrower slices as requested
          const columns = Math.ceil(dw / sliceW);
          const noise: number[] = new Array(columns)
            .fill(0)
            .map((_, i) => (Math.sin(i * 12.9898) * 43758.5453) % 1)
            .map((v) => (v < 0 ? -v : v));

          const speed = Math.max(2, dh / 180);
          const splitAmp = Math.max(8, dw / 80);

          // Magic stars around the shredder line
          const stars: Array<{
            x: number;
            y: number;
            size: number;
            phase: number;
            speed: number;
          }> = Array.from({ length: 8 }, () => ({
            x: Math.random() * cw,
            y: 0,
            size: Math.random() * 3 + 1,
            phase: Math.random() * Math.PI * 2,
            speed: Math.random() * 0.5 + 0.3,
          }));

          let imageOffsetY = -dh; // Start with image above viewport
          const shredderY = ch * 0.4; // Fixed horizontal line position
          const shreddedPieces: Array<{
            x: number;
            y: number;
            w: number;
            h: number;
            vx: number;
            vy: number;
            sourceX: number;
            sourceY: number;
            opacity: number;
          }> = [];

          const render = (ts: number) => {
            if (cancelled) return;
            if (t0Ref.current == null) t0Ref.current = ts;
            const t = (ts - t0Ref.current) / 1000;

            // Clear canvas
            ctx.clearRect(0, 0, cw, ch);

            // Move image downward
            imageOffsetY += speed;
            const currentImageY = dy + imageOffsetY;

            // Draw the intact part of the image (above the shredder line)
            if (currentImageY < shredderY) {
              ctx.save();
              ctx.beginPath();
              ctx.rect(0, 0, cw, shredderY);
              ctx.clip();
              ctx.drawImage(img, dx, currentImageY, dw, dh);
              ctx.restore();
            }

            // Create new shredded pieces (reduced frequency for performance)
            if (currentImageY + dh > shredderY && currentImageY < shredderY + 3 && Math.random() > 0.5) {
              for (let i = 0; i < columns; i += 1) { // Process every slice
                const pieceX = dx + i * sliceW;
                const pieceW = i === columns - 1 ? dw - i * sliceW : sliceW - 1;
                const sourceX = i * sliceW;
                const sourceY = Math.max(0, shredderY - currentImageY);

                if (sourceY < dh) {
                  const jitter = (noise[i] - 0.25) * 2;
                  shreddedPieces.push({
                    x: pieceX,
                    y: shredderY,
                    w: pieceW,
                    h: Math.min(90, dh - sourceY), // Much larger pieces as requested
                    vx: jitter * splitAmp * 0.6,
                    vy: Math.random() * 4 + 1,
                    sourceX,
                    sourceY,
                    opacity: 1,
                  });
                }
              }
            }

            // Update and draw shredded pieces (no piece limit - let them fall naturally)
            for (let i = shreddedPieces.length - 1; i >= 0; i--) {
              const piece = shreddedPieces[i];

              piece.x += piece.vx;
              piece.y += piece.vy;
              piece.vy += 0.4; // gravity
              piece.opacity -= 0.012; // fade out

              if (piece.y > ch + 50 || piece.opacity <= 0) {
                shreddedPieces.splice(i, 1);
                continue;
              }

              ctx.save();
              ctx.globalAlpha = piece.opacity;
              ctx.beginPath();
              ctx.rect(piece.x, piece.y, piece.w, piece.h);
              ctx.clip();

              ctx.drawImage(
                img,
                piece.sourceX, piece.sourceY, piece.w, piece.h,
                piece.x, piece.y, piece.w, piece.h
              );
              ctx.restore();
            }

            // Draw magical shredder line with glow and stars
            const glowIntensity = 0.7 + 0.3 * Math.sin(t * 3);

            // Glow effect
            ctx.save();
            ctx.shadowColor = "rgba(255,255,255,0.8)";
            ctx.shadowBlur = 12 * glowIntensity;
            ctx.fillStyle = `rgba(255,255,255,${0.9 * glowIntensity})`;
            ctx.fillRect(0, shredderY - 1, cw, 2);
            ctx.restore();

            // Update and draw magical stars
            stars.forEach((star, i) => {
              // Move stars along the line with slight wobble
              star.x += star.speed;
              star.y = shredderY + Math.sin(star.phase + t * 2) * 15;
              star.phase += 0.05;

              // Wrap around
              if (star.x > cw + 20) {
                star.x = -20;
                star.y = shredderY + (Math.random() - 0.5) * 30;
              }

              // Draw twinkling star
              const twinkle = 0.3 + 0.7 * Math.abs(Math.sin(star.phase + i));
              ctx.save();
              ctx.globalAlpha = twinkle;
              ctx.fillStyle = "rgba(255,255,255,0.9)";

              // Simple star shape
              const size = star.size * twinkle;
              ctx.beginPath();
              ctx.arc(star.x, star.y, size, 0, Math.PI * 2);
              ctx.fill();

              // Cross sparkle
              ctx.strokeStyle = "rgba(255,255,255,0.6)";
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(star.x - size * 2, star.y);
              ctx.lineTo(star.x + size * 2, star.y);
              ctx.moveTo(star.x, star.y - size * 2);
              ctx.lineTo(star.x, star.y + size * 2);
              ctx.stroke();
              ctx.restore();
            });

            // Reset cycle
            if (currentImageY > ch + dh + 100) {
              imageOffsetY = -dh;
              shreddedPieces.length = 0;
              t0Ref.current = ts;
            }

            rafRef.current = requestAnimationFrame(render);
          };

          rafRef.current = requestAnimationFrame(render);
        }

        return () => {
          cancelled = true;
          if (rafRef.current) cancelAnimationFrame(rafRef.current);
        };
      }, [src]);

      return (
        <canvas
          ref={canvasRef}
          className="w-full h-full block"
          aria-label="Processing animation"
        />
      );
    }
    return Component;
  }, []);

  return (
    <div className="min-h-screen relative text-zinc-200 antialiased">
      {/* background */}
      <div className="pointer-events-none absolute inset-0 bg-zinc-950" />
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage:
            "radial-gradient(600px 600px at 20% 10%, rgba(80,80,120,0.06), transparent 60%), radial-gradient(900px 900px at 85% 75%, rgba(120,70,180,0.05), transparent 60%)",
        }}
      />

      <div className="relative max-w-5xl mx-auto px-4 py-8">
        <header className="mb-8">
          <div className="flex items-center justify-between">
            <h1 className="text-xl sm:text-2xl font-medium text-zinc-100 tracking-tight">
              Remove Watermark
            </h1>
            <div className="flex gap-4 text-sm">
              {processedImage && (
                <button
                  onClick={isIOS && isSafari ? saveToPhotos : download}
                  className="text-zinc-300/80 hover:text-zinc-100"
                >
                  {isIOS && isSafari ? "Save to Photos" : "Download"}
                </button>
              )}
              {(originalImage || processedImage) && (
                <button onClick={reset} className="text-zinc-500 hover:text-zinc-300">
                  Reset
                </button>
              )}
            </div>
          </div>
          <div className="mt-2">
            <a
              href="https://x.com/alightinastorm"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-zinc-500 hover:text-zinc-400 transition-colors"
            >
              made by @alightinastorm
            </a>
          </div>
        </header>

        {/* Dropzone */}
        {!originalImage && (
          <section
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={onDrop}
            className={
              // Outer container: no border/background; lighter padding
              "relative mb-8 p-4 sm:p-6"
            }
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={onFileChange}
              className="hidden"
            />
            <div className="flex flex-col items-center justify-center gap-4 text-center">
              <div
                role="button"
                tabIndex={0}
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    fileInputRef.current?.click();
                  }
                }}
                className={
                  "aspect-[9/16] w-full max-w-xs rounded-xl overflow-hidden border transition-colors cursor-pointer focus:outline-none focus:ring-2 focus:ring-zinc-500/50 " +
                  (isDragOver ? "bg-zinc-900/80 border-zinc-500/80" : "bg-zinc-950/60 border-zinc-800")
                }
                aria-label="Drop image here or click to select"
              >
                <div className="w-full h-full grid place-items-center text-zinc-500">
                  <span className="text-sm">Drop image here or click to select</span>
                </div>
              </div>
              <p className="text-xs text-zinc-500">
                Use an X.com mobile screenshot showing the avatar and name at the top.
              </p>
            </div>
          </section>
        )}

        {error && (
          <div className="mb-6 rounded-lg border border-red-900/40 bg-red-950/40 text-red-200 px-4 py-3">
            {error}
          </div>
        )}

        {/* Stage */}
        {originalImage && (
          <section className="mb-10">
            <div className="relative mx-auto max-w-[28rem] w-full">
              <div className="aspect-[9/16] w-full rounded-xl overflow-hidden border border-zinc-800 bg-zinc-900/60">
                {/* Processing animation or result */}
                {isProcessing && !processedImage ? (
                  <Shredder src={originalImage} />
                ) : (
                  <div className="w-full h-full grid place-items-center">
                    <img
                      src={(showOriginal ? originalImage : processedImage) ?? undefined}
                      alt="preview"
                      className="max-h-full max-w-full object-contain select-none"
                      draggable={false}
                      style={{ WebkitUserSelect: "none", userSelect: "none" }}
                    />
                  </div>
                )}
              </div>

              {/* Status */}
              {status && (
                <p className="mt-3 text-xs text-zinc-400 text-center">{status}</p>
              )}

              {/* Toggle overlay when done */}
              {processedImage && (
                <button
                  onClick={() => setShowOriginal((s) => !s)}
                  className="absolute top-3 right-4 text-xs text-zinc-300/80 hover:text-zinc-100"
                >
                  {showOriginal ? "Show cleaned" : "Show original"}
                </button>
              )}
            </div>
          </section>
        )}

        {/* Hidden canvas for processing */}
        <canvas ref={fullCanvasRef} className="hidden" />
      </div>
    </div>
  );
}
