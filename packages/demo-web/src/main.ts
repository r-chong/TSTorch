import "./style.css";
import { DATASET_CONFIGS, trainDataset, type TrainResult } from "./train";
import { Animator } from "./animator";
import { exportGif } from "./gif";

interface CardState {
  config: (typeof DATASET_CONFIGS)[number];
  animator: Animator;
  result: TrainResult | null;
  card: HTMLElement;
  scrubber: HTMLInputElement;
  statsEl: HTMLElement;
  progressOverlay: HTMLElement;
  progressFill: HTMLElement;
  progressLabel: HTMLElement;
  gifBtn: HTMLButtonElement;
}

const cards: CardState[] = [];

// ---- Build UI ----
const grid = document.getElementById("grid")!;

for (const config of DATASET_CONFIGS) {
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `
    <div class="card-header">
      <span class="card-title">${config.name}</span>
      <span class="card-config">${config.hidden}h · ${config.epochs}ep · lr ${config.lr}</span>
    </div>
    <div class="card-canvas-wrap">
      <canvas></canvas>
      <div class="progress-overlay hidden">
        <div class="progress-bar-track"><div class="progress-bar-fill"></div></div>
        <div class="progress-label">Waiting…</div>
      </div>
    </div>
    <div class="card-footer">
      <span class="card-stats">—</span>
      <input class="card-scrubber" type="range" min="0" max="0" value="0" disabled />
      <button class="btn small gif-btn" disabled>GIF</button>
    </div>
  `;
  grid.appendChild(card);

  const canvas = card.querySelector("canvas")!;
  const scrubber = card.querySelector<HTMLInputElement>(".card-scrubber")!;
  const statsEl = card.querySelector<HTMLElement>(".card-stats")!;
  const progressOverlay = card.querySelector<HTMLElement>(".progress-overlay")!;
  const progressFill = card.querySelector<HTMLElement>(".progress-bar-fill")!;
  const progressLabel = card.querySelector<HTMLElement>(".progress-label")!;
  const gifBtn = card.querySelector<HTMLButtonElement>(".gif-btn")!;

  const state: CardState = {
    config,
    animator: new Animator(canvas, (index) => {
      const snap = state.animator.currentSnapshot;
      if (snap) {
        scrubber.value = String(index);
        statsEl.textContent = isNaN(snap.loss)
          ? `Epoch 0`
          : `Ep ${snap.epoch} · Loss ${snap.loss.toFixed(3)} · ${(snap.accuracy * 100).toFixed(0)}%`;
      }
    }),
    result: null,
    card,
    scrubber,
    statsEl,
    progressOverlay,
    progressFill,
    progressLabel,
    gifBtn,
  };

  scrubber.addEventListener("input", () => {
    state.animator.seekTo(parseInt(scrubber.value, 10));
  });

  gifBtn.addEventListener("click", async () => {
    if (!state.result) return;
    gifBtn.disabled = true;
    gifBtn.textContent = "…";
    await exportGif(
      state.animator,
      `tstorch-${config.name.toLowerCase()}.gif`,
      (f, t) => {
        gifBtn.textContent = `${Math.round((f / t) * 100)}%`;
      }
    );
    gifBtn.textContent = "GIF";
    gifBtn.disabled = false;
  });

  cards.push(state);
}

// ---- Global controls ----
const btnTrain = document.getElementById("btn-train") as HTMLButtonElement;
const btnPlay = document.getElementById("btn-play") as HTMLButtonElement;
const btnPause = document.getElementById("btn-pause") as HTMLButtonElement;
const speedSlider = document.getElementById("speed-slider") as HTMLInputElement;
const speedValue = document.getElementById("speed-value")!;

speedSlider.addEventListener("input", () => {
  const v = parseInt(speedSlider.value, 10);
  speedValue.textContent = `${v}x`;
  for (const c of cards) c.animator.setSpeed(v);
});

btnPlay.addEventListener("click", () => {
  for (const c of cards) c.animator.play();
});

btnPause.addEventListener("click", () => {
  for (const c of cards) c.animator.pause();
});

btnTrain.addEventListener("click", async () => {
  btnTrain.disabled = true;
  btnTrain.textContent = "Training…";
  btnPlay.disabled = true;
  btnPause.disabled = true;

  for (const state of cards) {
    state.progressOverlay.classList.remove("hidden");
    state.progressLabel.textContent = "Queued…";
    state.progressFill.style.width = "0%";
    state.scrubber.disabled = true;
    state.gifBtn.disabled = true;
  }

  for (const state of cards) {
    state.progressLabel.textContent = "Training…";

    const result = await trainDataset(state.config, (epoch, total) => {
      const pct = (epoch / total) * 100;
      state.progressFill.style.width = `${pct}%`;
      state.progressLabel.textContent = `Epoch ${epoch} / ${total}`;
    });

    state.result = result;
    state.animator.load(result.snapshots, result.data);

    state.scrubber.max = String(result.snapshots.length - 1);
    state.scrubber.value = "0";
    state.scrubber.disabled = false;
    state.gifBtn.disabled = false;

    state.progressOverlay.classList.add("hidden");
  }

  btnTrain.textContent = "Retrain All";
  btnTrain.disabled = false;
  btnPlay.disabled = false;
  btnPause.disabled = false;

  // Auto-play all after training
  const speed = parseInt(speedSlider.value, 10);
  for (const c of cards) {
    c.animator.setSpeed(speed);
    c.animator.play();
  }
});
