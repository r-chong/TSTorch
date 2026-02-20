import { Scalar } from "./scalar.js";
import { Parameter } from "./module.js";
import type { Tensor } from "./tensor.js";

export type ParameterValue = Scalar | Tensor;

export class Optimizer {
    parameters: Parameter<ParameterValue>[];

    constructor(parameters: Parameter<ParameterValue>[]) {
        this.parameters = parameters;
    }

    zeroGrad() {
        for (let p of this.parameters) {
            if (!p.value || typeof p.value !== 'object') {
                continue;
            }
            if ("derivative" in p.value) {
                if (p.value.derivative !== null && p.value.derivative !== undefined) {
                    p.value.derivative = 0;
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

export class SGD extends Optimizer {
    lr: number;

    constructor(parameters: Parameter<Scalar>[], lr: number = 1.0) {
        super(parameters);
        this.lr = lr;
    }

    step() {
        for (let p of this.parameters) {
            if (!p.value || typeof p.value !== 'object') continue;
            if (p.value instanceof Scalar) {
                const grad = p.value.derivative ?? 0;
                p.value.data -= this.lr * grad;
            }
        }
    }
}

export class Adam extends Optimizer {
    lr: number;
    beta1: number;
    beta2: number;
    eps: number;
    t: number = 0;
    private m: Map<Parameter<ParameterValue>, number> = new Map();
    private v: Map<Parameter<ParameterValue>, number> = new Map();

    constructor(
        parameters: Parameter<Scalar>[],
        lr: number = 0.001,
        beta1: number = 0.9,
        beta2: number = 0.999,
        eps: number = 1e-8
    ) {
        super(parameters);
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        for (const p of parameters) {
            this.m.set(p, 0);
            this.v.set(p, 0);
        }
    }

    step() {
        this.t++;
        for (const p of this.parameters) {
            if (!(p.value instanceof Scalar)) continue;
            const grad = p.value.derivative ?? 0;

            let mi = this.m.get(p)!;
            let vi = this.v.get(p)!;

            mi = this.beta1 * mi + (1 - this.beta1) * grad;
            vi = this.beta2 * vi + (1 - this.beta2) * grad * grad;

            this.m.set(p, mi);
            this.v.set(p, vi);

            const mHat = mi / (1 - Math.pow(this.beta1, this.t));
            const vHat = vi / (1 - Math.pow(this.beta2, this.t));

            p.value.data -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
        }
    }
}
