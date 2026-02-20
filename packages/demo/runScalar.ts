import { Scalar, datasets, SGD, Adam, Optimizer, Module, Parameter } from "tstorch";

type Point = [number, number];
type Graph = { N: number; X: Point[]; y: number[] };

function bceWithLogits(logit: Scalar, label: number): Scalar {
  // softplus(logit) - label * logit
  return logit.exp().add(1).log().sub(logit.mul(label));
}

class Network extends Module<Parameter<Scalar>> {
  layers: Linear[];

  constructor(hiddenSizes: number[]) {
    super();
    const sizes = [2, ...hiddenSizes, 1];
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      const layer = new Linear(sizes[i], sizes[i + 1]);
      this.layers.push(layer);
      this[`layer_${i}`] = layer;
    }
  }

  forward(x: [Scalar, Scalar]): Scalar {
    let h: Scalar[] = x;
    for (let i = 0; i < this.layers.length - 1; i++) {
      const out = this.layers[i].forward(h);
      h = out.map(s => s.leakyRelu());
    }
    const out = this.layers[this.layers.length - 1].forward(h);
    return out[0];
  }
}

class Linear extends Module {
  inSize: number;
  outSize: number;
  weights: Parameter<Scalar>[][];
  bias: Parameter<Scalar>[];  
  
  constructor(inSize: number, outSize: number) {
    super();
    this.inSize = inSize;
    this.outSize = outSize;

    const scale = Math.sqrt(2 / inSize);

    this.weights = Array.from({ length: outSize }, () =>
      Array.from({ length: inSize }, () => new Parameter(new Scalar((Math.random() - 0.5) * 2 * scale)))
    );

    this.bias = Array.from({ length: outSize }, () => new Parameter(new Scalar(0)));

    // register weights and bias as parameters in module
    // note that using our proxy autoregisters but not if it's in an array
    for (let i = 0; i < outSize; i++) {
      this[`b_${i}`] = this.bias[i];

      for (let j = 0; j < inSize; j++) {
        this[`w_${i}_${j}`] = this.weights[i][j];
      }
    }
  }

  forward(inputs: Scalar[]): Scalar[] {
    const outputs: Scalar[] = [];

    for (let i = 0; i < this.outSize; ++i) {
      let result = this.bias[i].value.add(0);

      for (let j = 0; j < this.inSize; ++j) {
        result = result.add(this.weights[i][j].value.mul(inputs[j]));
      }
      outputs.push(result);
    }
    
    return outputs;
  }
}

function defaultLogFn(epoch, totalLoss, correct) {
  console.log("Epoch ", epoch, " loss ", totalLoss, " correct ", correct)
}

class ScalarTrain {
  hiddenSizes: number[];
  model: Network;
  learningRate: number;
  maxEpochs: number;

  constructor(hiddenSizes: number[]) {
    this.hiddenSizes = hiddenSizes;
    this.model = new Network(hiddenSizes);
  }

  runOne(x: Point) {
    return this.model.forward([new Scalar(x[0],undefined,"x1"), new Scalar(x[1],undefined,"x2")]);
  }

  train(data: Graph, learningRate: number, maxEpochs: number = 500, logFn=defaultLogFn, useAdam=false) {
    this.learningRate = learningRate;
    this.maxEpochs = maxEpochs;
    this.model = new Network(this.hiddenSizes);
    const params = this.model.parameters();
    const optim = useAdam ? new Adam(params, learningRate) : new SGD(params, learningRate);

    // const losses = [];

    for (let epoch = 1; epoch < this.maxEpochs + 1; ++epoch) {
      let totalLoss = new Scalar(0);
      let correct = 0;

      // zero out gradients each epoch
      optim.zeroGrad();

      // forward pass
      let loss: Scalar;
      for (let i = 0; i < data.N; ++i) {
        const [rx1, rx2] = data.X[i];
        const y = data.y[i];
        const x1 = new Scalar(rx1);
        const x2 = new Scalar(rx2);
        const logit = this.model.forward([x1, x2]);
        const pred = logit.data > 0 ? 1 : 0;
        if (pred === y) correct++;

        loss = bceWithLogits(logit, y);

        loss = loss.div(data.N);
        totalLoss = totalLoss.add(loss);
        // losses.push(totalLoss);
      }

      // backward pass on accumulated loss
      totalLoss.backward();

      // update gradient descent
      optim.step();

      // log every 10th epoch
      if (epoch % 10 == 0 || epoch == maxEpochs) {
        logFn(epoch, totalLoss.data, correct);
      }
    }
  }
}

export default function runScalar() {
  const PTS = 50;

  const data1 = datasets["Simple"](PTS) as Graph;
  const data2 = datasets["Diag"](PTS) as Graph;
  const data3 = datasets["Split"](PTS) as Graph;
  const data4 = datasets["Xor"](PTS) as Graph;
  const data5 = datasets["Circle"](PTS) as Graph;
  const data6 = datasets["Spiral"](PTS) as Graph;

  console.log("=== Simple [4] ===");
  new ScalarTrain([4]).train(data1, 0.5);

  console.log("\n=== Diag [4] ===");
  new ScalarTrain([4]).train(data2, 0.5);

  console.log("\n=== Split [8] ===");
  new ScalarTrain([8]).train(data3, 0.5);

  console.log("\n=== Xor [8] ===");
  new ScalarTrain([8]).train(data4, 0.5);

  console.log("\n=== Circle [8, 8] ===");
  new ScalarTrain([8, 8]).train(data5, 0.5, 1000);

  console.log("\n=== Spiral [16, 16] + Adam ===");
  new ScalarTrain([16, 16]).train(data6, 0.01, 2000, defaultLogFn, true);
}