import { Scalar, Module, Parameter, SGD, datasets } from "tstorch";

export type Point = [number, number];
export type Graph = { N: number; X: Point[]; y: number[] };

export interface EpochSnapshot {
  epoch: number;
  loss: number;
  accuracy: number;
  /** Flattened row-major grid of model outputs (logits) for the decision boundary */
  grid: Float32Array;
}

export interface DatasetConfig {
  name: string;
  hidden: number;
  epochs: number;
  lr: number;
  /** Capture a snapshot every N epochs */
  captureInterval: number;
}

export const GRID_RES = 50;

export const DATASET_CONFIGS: DatasetConfig[] = [
  { name: "Simple", hidden: 4, epochs: 500, lr: 0.5, captureInterval: 5 },
  { name: "Diag", hidden: 4, epochs: 500, lr: 0.5, captureInterval: 5 },
  { name: "Split", hidden: 8, epochs: 500, lr: 0.5, captureInterval: 5 },
  { name: "Xor", hidden: 8, epochs: 500, lr: 0.5, captureInterval: 5 },
  { name: "Circle", hidden: 8, epochs: 1000, lr: 0.5, captureInterval: 10 },
  { name: "Spiral", hidden: 8, epochs: 1000, lr: 0.5, captureInterval: 10 },
];

class Linear extends Module {
  inSize: number;
  outSize: number;
  weights: Parameter<Scalar>[][];
  bias: Parameter<Scalar>[];

  constructor(inSize: number, outSize: number) {
    super();
    this.inSize = inSize;
    this.outSize = outSize;

    this.weights = Array.from({ length: outSize }, () =>
      Array.from(
        { length: inSize },
        () => new Parameter(new Scalar(2 * (Math.random() - 0.5)))
      )
    );
    this.bias = Array.from(
      { length: outSize },
      () => new Parameter(new Scalar(2 * (Math.random() - 0.5)))
    );

    for (let i = 0; i < outSize; i++) {
      (this as any)[`b_${i}`] = this.bias[i];
      for (let j = 0; j < inSize; j++) {
        (this as any)[`w_${i}_${j}`] = this.weights[i]![j];
      }
    }
  }

  forward(inputs: Scalar[]): Scalar[] {
    const outputs: Scalar[] = [];
    for (let i = 0; i < this.outSize; ++i) {
      let result = this.bias[i]!.value.add(0);
      for (let j = 0; j < this.inSize; ++j) {
        result = result.add(this.weights[i]![j]!.value.mul(inputs[j]!));
      }
      outputs.push(result);
    }
    return outputs;
  }
}

class Network extends Module<Parameter<Scalar>> {
  layer1: Linear;
  layer2: Linear;

  constructor(hidden: number) {
    super();
    this.layer1 = new Linear(2, hidden);
    this.layer2 = new Linear(hidden, 1);
  }

  forward(x: [Scalar, Scalar]): Scalar {
    const h = this.layer1.forward(x).map((s) => s.relu());
    return this.layer2.forward(h)[0]!;
  }
}

function evaluateGrid(model: Network): Float32Array {
  const grid = new Float32Array(GRID_RES * GRID_RES);
  for (let r = 0; r < GRID_RES; r++) {
    for (let c = 0; c < GRID_RES; c++) {
      const x1 = c / (GRID_RES - 1);
      const x2 = r / (GRID_RES - 1);
      const out = model.forward([new Scalar(x1), new Scalar(x2)]);
      grid[r * GRID_RES + c] = out.data;
    }
  }
  return grid;
}

export interface TrainResult {
  data: Graph;
  snapshots: EpochSnapshot[];
}

/**
 * Train on a single dataset, yielding to the event loop periodically
 * so the UI stays responsive.
 */
export async function trainDataset(
  config: DatasetConfig,
  onProgress?: (epoch: number, total: number) => void
): Promise<TrainResult> {
  const datasetFn = datasets[config.name];
  if (!datasetFn) throw new Error(`Unknown dataset: ${config.name}`);

  const data = datasetFn(config.epochs > 500 ? 75 : 50) as Graph;
  const model = new Network(config.hidden);
  const optim = new SGD(model.parameters(), config.lr);
  const snapshots: EpochSnapshot[] = [];

  snapshots.push({
    epoch: 0,
    loss: NaN,
    accuracy: 0,
    grid: evaluateGrid(model),
  });

  for (let epoch = 1; epoch <= config.epochs; epoch++) {
    let totalLoss = new Scalar(0);
    let correct = 0;

    optim.zeroGrad();

    for (let i = 0; i < data.N; i++) {
      const [rx1, rx2] = data.X[i]!;
      const y = data.y[i]!;
      const x = model.forward([new Scalar(rx1), new Scalar(rx2)]);

      if ((x.data > 0 ? 1 : 0) === y) correct++;

      let loss: Scalar;
      if (y === 1) {
        loss = x.neg().exp().add(1).log();
      } else {
        loss = x.exp().add(1).log();
      }
      totalLoss = totalLoss.add(loss.div(data.N));
    }

    totalLoss.backward();
    optim.step();

    if (epoch % config.captureInterval === 0 || epoch === config.epochs) {
      snapshots.push({
        epoch,
        loss: totalLoss.data,
        accuracy: correct / data.N,
        grid: evaluateGrid(model),
      });
    }

    if (epoch % 25 === 0) {
      onProgress?.(epoch, config.epochs);
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  return { data, snapshots };
}
