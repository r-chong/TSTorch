import { Scalar, datasets, SGD, Optimizer, Parameter } from "tstorch";
import { Module } from "../tstorch/dist/module.js";

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

    return outputs2[0];
  }
}

class Linear extends Module {
  inSize: number;
  outSize: number;
  weights: Scalar[][];
  bias: Scalar[];  
  
  constructor(inSize: number, outSize: number) {
    super();
    this.inSize = inSize;
    this.outSize = outSize;

    this.weights = Array.from({ length: inSize }, () =>
      Array.from({ length: outSize }, () => new Scalar(2 * (Math.random() - 0.5)))
    );

    this.bias = Array.from({ length: outSize }, () => new Scalar(2 * (Math.random() - 0.5)));
  }

  forward(inputs: Scalar[]): Scalar[] {
    const outputs: Scalar[] = [];

    for (let i = 0; i < this.outSize; ++i) {
      let result = this.bias[i];

      for (let j = 0; j < this.inSize; ++j) {
        result.add(this.weights[i][j].mul(inputs[j]));
      }
      outputs.push(result);
    }
    
    return outputs;
  }
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

  train(data: Graph, learningRate: number, maxEpochs: number = 500) {
    // zero out gradients
    this.learningRate = learningRate;
    this.maxEpochs = maxEpochs;
    this.model = new Network(this.hiddenLayers);
    const optim = new SGD(this.model.parameters(), learningRate);
  }
}

export default function runScalar() {
  const PTS = 50;
  const HIDDEN = 2;
  const RATE = 0.5;

  const data = datasets["Simple"](PTS) as Graph;
  new ScalarTrain(HIDDEN).train(data, RATE);
}