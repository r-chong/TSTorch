import { runDispatchParity } from "./dispatch_parity.js";
import { Scalar, datasets } from "tstorch";

console.log(Scalar);

class Network {
    
};

class Linear {

};

class ScalarTrain {
    constructor(hiddenLayers: number) {
        this.hiddenLayers = hiddenLayers;
        this.model = new Network(hiddenLayers)
    }
}

function main() {
    const PTS = 50;
    const HIDDEN = 2;
    const RATE = 0.5;
    const data = datasets["Simple"](PTS);
    const st = new ScalarTrain(HIDDEN).train(data, RATE);
    const result = runDispatchParity();
    console.log(result);
}
