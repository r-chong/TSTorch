import { GIFEncoder, quantize, applyPalette } from "gifenc";
import type { Animator } from "./animator";

/**
 * Encode all frames of an animator as a looping GIF and trigger a download.
 */
export async function exportGif(
  animator: Animator,
  filename: string,
  onProgress?: (frame: number, total: number) => void
): Promise<void> {
  const canvas = animator.getCanvas();
  const ctx = canvas.getContext("2d")!;
  const { width, height } = canvas;
  const total = animator.totalFrames;

  const wasPlaying = animator.isPlaying;
  if (wasPlaying) animator.pause();

  const gif = GIFEncoder();
  const delay = Math.round(1000 / 12); // ~12 fps → ~83ms per frame

  for (let i = 0; i < total; i++) {
    animator.renderAtIndex(i);
    const imageData = ctx.getImageData(0, 0, width, height);
    const rgba = imageData.data;

    const palette = quantize(rgba, 256, { format: "rgba4444" });
    const index = applyPalette(rgba, palette, "rgba4444");

    gif.writeFrame(index, width, height, {
      palette,
      delay,
      repeat: 0, // loop forever
    });

    onProgress?.(i + 1, total);

    // Yield to keep UI responsive
    if (i % 10 === 0) {
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  gif.finish();

  const blob = new Blob([gif.bytes()], { type: "image/gif" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);

  // Restore playback state
  if (wasPlaying) animator.play();
}
