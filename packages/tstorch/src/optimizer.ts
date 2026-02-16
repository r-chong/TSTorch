import { Scalar } from "./scalar.js";
import { Parameter } from "./module.js";

export class Optimizer<T> {
    parameters: Parameter<T>[];

    constructor(parameters: Parameter<T>[]) {
        this.parameters = parameters;
    }
}