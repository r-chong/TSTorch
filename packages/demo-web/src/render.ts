import type { EpochSnapshot, Graph } from "./train";
import { GRID_RES } from "./train";

const CANVAS_SIZE = 350;
const POINT_RADIUS = 4;

/** Sigmoid squash for mapping raw logits → [0,1] confidence */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** Interpolate between two [r,g,b] colors */
function lerp3(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

/**
 * Map a confidence value in [0,1] to a diverging blue→white→red palette.
 * 0 = blue (class 0), 0.5 = near-white, 1 = red (class 1).
 */
function confidenceColor(conf: number): [number, number, number] {
  const blue: [number, number, number] = [40, 100, 220];
  const mid: [number, number, number] = [235, 235, 245];
  const red: [number, number, number] = [220, 50, 50];
  if (conf < 0.5) {
    return lerp3(blue, mid, conf * 2);
  }
  return lerp3(mid, red, (conf - 0.5) * 2);
}

/**
 * Render a single frame (epoch snapshot) onto a canvas.
 */
export function renderFrame(
  ctx: CanvasRenderingContext2D,
  snapshot: EpochSnapshot,
  data: Graph
): void {
  const w = CANVAS_SIZE;
  const h = CANVAS_SIZE;
  const cellW = w / GRID_RES;
  const cellH = h / GRID_RES;

  // --- Decision boundary heatmap ---
  const imageData = ctx.createImageData(w, h);
  const pixels = imageData.data;

  for (let r = 0; r < GRID_RES; r++) {
    for (let c = 0; c < GRID_RES; c++) {
      const logit = snapshot.grid[r * GRID_RES + c]!;
      const conf = sigmoid(logit);
      const [cr, cg, cb] = confidenceColor(conf);

      const startY = Math.floor(r * cellH);
      const endY = Math.floor((r + 1) * cellH);
      const startX = Math.floor(c * cellW);
      const endX = Math.floor((c + 1) * cellW);

      for (let py = startY; py < endY && py < h; py++) {
        for (let px = startX; px < endX && px < w; px++) {
          const idx = (py * w + px) * 4;
          pixels[idx] = cr;
          pixels[idx + 1] = cg;
          pixels[idx + 2] = cb;
          pixels[idx + 3] = 255;
        }
      }
    }
  }
  ctx.putImageData(imageData, 0, 0);

  // --- Data points ---
  for (let i = 0; i < data.N; i++) {
    const [x1, x2] = data.X[i]!;
    const y = data.y[i]!;
    const px = x1 * (w - 1);
    const py = x2 * (h - 1);

    ctx.beginPath();
    ctx.arc(px, py, POINT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = y === 1 ? "#ff3355" : "#3388ff";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // --- HUD ---
  ctx.save();
  const hudLines = [
    `Epoch ${snapshot.epoch}`,
    isNaN(snapshot.loss) ? "" : `Loss ${snapshot.loss.toFixed(4)}`,
    `Acc ${(snapshot.accuracy * 100).toFixed(1)}%`,
  ].filter(Boolean);

  ctx.font = "bold 13px monospace";
  const lineH = 17;
  const padding = 6;
  const maxTextW = Math.max(...hudLines.map((l) => ctx.measureText(l).width));
  const boxW = maxTextW + padding * 2;
  const boxH = hudLines.length * lineH + padding * 2;

  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.beginPath();
  ctx.roundRect(6, 6, boxW, boxH, 4);
  ctx.fill();

  ctx.fillStyle = "#fff";
  ctx.textBaseline = "top";
  hudLines.forEach((line, i) => {
    ctx.fillText(line, 6 + padding, 6 + padding + i * lineH);
  });
  ctx.restore();
}

export { CANVAS_SIZE };
