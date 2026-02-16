import { Scalar } from "./scalar.js";
import { Parameter } from "./module.js";

export class Optimizer<T> {
    parameters: Parameter<T>[];

    constructor(parameters: Parameter<T>[]) {
        this.parameters = parameters;
    }
}

export class SGD<T> extends Optimizer<T> {
    lr: number;

    constructor(parameters: Parameter<T>[], lr: number = 1.0) {
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
}