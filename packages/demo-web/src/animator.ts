import type { EpochSnapshot, Graph } from "./train";
import { renderFrame, CANVAS_SIZE } from "./render";

export class Animator {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private snapshots: EpochSnapshot[] = [];
  private data: Graph | null = null;
  private frameIndex = 0;
  private playing = false;
  private rafId = 0;
  private lastFrameTime = 0;
  /** Frames per second (adjustable via speed multiplier) */
  private baseFps = 12;
  private speedMultiplier = 2;
  private onFrameChange?: (index: number) => void;

  constructor(canvas: HTMLCanvasElement, onFrameChange?: (index: number) => void) {
    this.canvas = canvas;
    this.canvas.width = CANVAS_SIZE;
    this.canvas.height = CANVAS_SIZE;
    this.ctx = canvas.getContext("2d")!;
    this.onFrameChange = onFrameChange;
  }

  load(snapshots: EpochSnapshot[], data: Graph) {
    this.stop();
    this.snapshots = snapshots;
    this.data = data;
    this.frameIndex = 0;
    this.drawCurrent();
  }

  get totalFrames(): number {
    return this.snapshots.length;
  }

  get currentFrame(): number {
    return this.frameIndex;
  }

  get currentSnapshot(): EpochSnapshot | undefined {
    return this.snapshots[this.frameIndex];
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  setSpeed(multiplier: number) {
    this.speedMultiplier = multiplier;
  }

  seekTo(index: number) {
    this.frameIndex = Math.max(0, Math.min(index, this.snapshots.length - 1));
    this.drawCurrent();
  }

  play() {
    if (this.playing || this.snapshots.length === 0) return;
    this.playing = true;
    this.lastFrameTime = performance.now();
    this.tick();
  }

  pause() {
    this.playing = false;
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
  }

  stop() {
    this.pause();
    this.frameIndex = 0;
  }

  /** Get canvas for GIF encoding */
  getCanvas(): HTMLCanvasElement {
    return this.canvas;
  }

  /** Render a specific frame index and return the canvas (for GIF capture) */
  renderAtIndex(index: number) {
    this.frameIndex = Math.max(0, Math.min(index, this.snapshots.length - 1));
    this.drawCurrent();
  }

  private drawCurrent() {
    if (!this.data || this.snapshots.length === 0) return;
    const snap = this.snapshots[this.frameIndex];
    if (!snap) return;
    renderFrame(this.ctx, snap, this.data);
    this.onFrameChange?.(this.frameIndex);
  }

  private tick = () => {
    if (!this.playing) return;
    const now = performance.now();
    const interval = 1000 / (this.baseFps * this.speedMultiplier);

    if (now - this.lastFrameTime >= interval) {
      this.lastFrameTime = now;
      this.frameIndex = (this.frameIndex + 1) % this.snapshots.length;
      this.drawCurrent();
    }
    this.rafId = requestAnimationFrame(this.tick);
  };
}
