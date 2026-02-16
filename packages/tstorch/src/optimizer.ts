export type ParameterValue = Scalar; // TODO: add tensor

import { Scalar } from "./scalar.js";
import { Parameter } from "./module.js";

export class Optimizer {
    parameters: Parameter<ParameterValue>[];

    constructor(parameters: Parameter<ParameterValue>[]) {
        this.parameters = parameters;
    }
}

export class SGD extends Optimizer {
    lr: number;

    constructor(parameters: Parameter<ParameterValue>[], lr: number = 1.0) {
        super(parameters);
        this.lr = lr;
    }

    zeroGrad() {
        for (let p of this.parameters) {
            if (!p.value || typeof p.value !== 'object') {
                continue;
            }
            if ("derivative" in p.value) { 
                if (p.value.derivative !== null && p.value.derivative !== undefined) {
                    p.value.derivative = null;
                }
            }
            if ("grad" in p.value) { 
                if (p.value.grad !== null && p.value.grad !== undefined) {
                    p.value.grad = null;
                }
            }
        }
    }

    step() {
    for (let p of this.parameters) {
        if (!p.value || typeof p.value !== 'object') {
            continue;
        }

        // Check for derivative (Scalar-like objects)
        if (p.value instanceof Scalar) {
            if (p.value.derivative !== null && p.value.derivative !== undefined) {
                // p.update(Scalar(p.value.data - this.lr * p.value.derivative))
                p.update(new Scalar(p.value.data - this.lr * p.value.derivative));
            }
        }
    }
}
}