import { Scalar, datasets } from "tstorch";

type Point = [number, number];
type Graph = { N: number; X: Point[]; y: number[] };

class Network {
  constructor(hiddenLayers: number) {
    // TODO: implement Task 1.5.
    throw new Error("Need to implement for Task 1.5");
  }

  forward(x: [Scalar, Scalar]): Scalar {
    // depends on task 1.5
    throw new Error("Need to implement for Task 1.5");
  }
}

class Linear {
    inSize: number;
    outSize: number;
    weights: Scalar[][];
    bias: Scalar[];  constructor(inSize: number, outSize: number) {
    // we get errors but they should not be here anymore when we're done 1.5
    this.inSize = inSize;
    this.outSize = outSize;

    this.weights = Array.from({ length: inSize }, () =>
      Array.from({ length: outSize }, () => new Scalar(2 * (Math.random() - 0.5)))
    );

    this.bias = Array.from({ length: outSize }, () => new Scalar(2 * (Math.random() - 0.5)));
  }

  forward(inputs: Scalar[]): Scalar[] {
    // TODO: implement Task 1.5
    throw new Error("Need to implement for Task 1.5");
  }
}

class ScalarTrain {
  hiddenLayers: number;
  model: Network;

  constructor(hiddenLayers: number) {
    this.hiddenLayers = hiddenLayers;
    this.model = new Network(hiddenLayers);
  }

  train(data: Graph, learningRate: number, maxEpochs = 500) {
    // This will fail until Network/Linear are implemented.
    throw new Error("Train loop depends on Task 1.5 implementation");
  }
}

export default function runScalar() {
  const PTS = 50;
  const HIDDEN = 2;
  const RATE = 0.5;

  const data = datasets["Simple"](PTS) as Graph;
  new ScalarTrain(HIDDEN).train(data, RATE);
}