import { Scalar, datasets, SGD, Module, Parameter, Mul, add } from "tstorch";

type Point = [number, number];
type Graph = { N: number; X: Point[]; y: number[] };

class Network extends Module<Parameter<Scalar>> {
  layer1: Linear;
  layer2: Linear

  constructor(hiddenLayers: number) {
    super();
    // Take 2 inputs - Point (x,y), process through hidden layers, return 1 output
    this.layer1 = new Linear(2, hiddenLayers);
    this.layer2 = new Linear(hiddenLayers, 1);
  }

  forward(x: [Scalar, Scalar]): Scalar {
    const relu1: Scalar[] = [];

    // input point x into layer 1
    const outputs1: Scalar[] = this.layer1.forward(x);

    for (let i = 0; i < this.layer1.outSize; ++i) {
      relu1.push(outputs1[i].relu());
    }

    // return Scalar array of len 1
    const outputs2: Scalar[] = this.layer2.forward(relu1);

    // get final number and convert to probability
    return outputs2[0];
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

    this.weights = Array.from({ length: outSize }, () =>
      Array.from({ length: inSize }, () => new Parameter(new Scalar(2 * (Math.random() - 0.5))))
    );

    this.bias = Array.from({ length: outSize }, () => new Parameter(new Scalar(2 * (Math.random() - 0.5))));

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
  hiddenLayers: number;
  model: Network;
  learningRate: number;
  maxEpochs: number;

  constructor(hiddenLayers: number) {
    this.hiddenLayers = hiddenLayers;
    this.model = new Network(hiddenLayers);
  }

  // for testing, run one forward pass of the network on a single datapoint
  runOne(x: Point) {
    return this.model.forward([new Scalar(x[0],undefined,"x1"), new Scalar(x[1],undefined,"x2")]);
  }

  train(data: Graph, learningRate: number, maxEpochs: number = 500, logFn=defaultLogFn) {
    this.learningRate = learningRate;
    this.maxEpochs = maxEpochs;
    this.model = new Network(this.hiddenLayers);
    const optim = new SGD(this.model.parameters(), learningRate);

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
        const x = this.model.forward([x1, x2]);

        const pred = x.data > 0 ? 1 : 0;
        if (pred === y) correct++;

        if (y == 1) {
          // log(1 + exp(-x))
          loss = x.neg().exp().add(1).log();
        } else {
          // log(1 + exp(x))
          loss = x.exp().add(1).log();
        }

        loss = loss.div(data.N);
        totalLoss = totalLoss.add(loss);
        // losses.push(totalLoss);
      }

      // backward pass on accumulated loss
      totalLoss.backward();

      // update gradient descent
      optim.step();

      // log every 10th epoch
      if (epoch % 500 == 0 || epoch == maxEpochs) {
        logFn(epoch, totalLoss.data, correct);
      }
    }
  }
}

export default function runScalar() {
  const PTS = 50;
  const RATE = 0.5;

  const data1 = datasets["Simple"](PTS) as Graph;
  const data2 = datasets["Diag"](PTS) as Graph;
  const data3 = datasets["Split"](PTS) as Graph;
  const data4 = datasets["Xor"](PTS) as Graph;
  const data5 = datasets["Circle"](PTS) as Graph;
  const data6 = datasets["Spiral"](PTS) as Graph;

  console.log("=== Simple [4] ===");
  new ScalarTrain(4).train(data1, 0.5);

  console.log("\n=== Diag [4] ===");
  new ScalarTrain(4).train(data2, 0.5);

  console.log("\n=== Split [8] ===");
  new ScalarTrain(8).train(data3, 0.5);

  console.log("\n=== Xor [8] ===");
  new ScalarTrain(8).train(data4, 0.5);

  console.log("\n=== Circle [8, 8] ===");
  new ScalarTrain(8).train(data5, 0.5, 1000);

  console.log("\n=== Circle [8, 8] ===");
  new ScalarTrain(8).train(data6, 0.5, 1000);

}