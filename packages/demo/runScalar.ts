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
    return outputs2[0].sigmoid();
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
      let loss = new Scalar(0);
      for (let i = 0; i < data.N; ++i) {
        const [rx1, rx2] = data.X[i];
        const y = data.y[i];
        const x1 = new Scalar(rx1);
        const x2 = new Scalar(rx2);
        const out = this.model.forward([x1, x2]);
        let prob = new Scalar(0);

        if (y == 1) {
          prob = out;
          if (out.data > 0.5) {
            correct += 1;
          }
        } else {
          prob = out.mul(new Scalar(-1)).add(new Scalar(1.0));
          if (out.data < 0.5) {
            correct += 1;
          }
        }

        loss = prob.log().mul(-1).div(data.N);
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
  const HIDDEN = 2;
  const RATE = 0.5;

  const data = datasets["Simple"](PTS) as Graph;
  new ScalarTrain(HIDDEN).train(data, RATE);
}