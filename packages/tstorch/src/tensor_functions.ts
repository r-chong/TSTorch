import type {
    Shape
} from './tensor_data.js'

import {
    TensorData,
    shapeProduct,
    shapeBroadcast,
    strides,
} from './tensor_data.js';
import { tensorMap, tensorZip, tensorReduce } from './tensor_ops.js';
import * as operators from './operators.js'



function zeros(shape: Shape): TensorData {
    return TensorData.zeros(shape);
}

export function neg(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.neg);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function sigmoid(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.sigmoid);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function relu(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.relu);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function log(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.log);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function exp(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.exp);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function id(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.id);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function inv(a: TensorData): TensorData {
    const out = zeros(a.shape);
    const mapFn = tensorMap(operators.inv);
    mapFn(out.storage, out.shape, out.strides, a.storage, a.shape, a.strides);
    return out;
}

export function add(a: TensorData, b: TensorData): TensorData {
    const outShape = shapeBroadcast(a.shape, b.shape);
    const out = zeros(outShape);
    const zipFn = tensorZip(operators.add);
    zipFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        b.storage, b.shape, b.strides
    );
    return out;
}

export function mul(a: TensorData, b: TensorData): TensorData {
    const outShape = shapeBroadcast(a.shape, b.shape);
    const out = zeros(outShape);
    const zipFn = tensorZip(operators.mul);
    zipFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        b.storage, b.shape, b.strides
    );
    return out;
}

export function lt(a: TensorData, b: TensorData): TensorData {
    const outShape = shapeBroadcast(a.shape, b.shape);
    const out = zeros(outShape);
    const zipFn = tensorZip(operators.lt);
    zipFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        b.storage, b.shape, b.strides
    );
    return out;
}

export function eq(a: TensorData, b: TensorData): TensorData {
    const outShape = shapeBroadcast(a.shape, b.shape);
    const out = zeros(outShape);
    const zipFn = tensorZip(operators.eq);
    zipFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        b.storage, b.shape, b.strides
    );
    return out;
}

export function isClose(a: TensorData, b: TensorData): TensorData {
    const outShape = shapeBroadcast(a.shape, b.shape);
    const out = zeros(outShape);
    const zipFn = tensorZip(operators.isClose);
    zipFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        b.storage, b.shape, b.strides
    );
    return out;
}

export function sum(a: TensorData, dim: number): TensorData {
    const outShape = a.shape.map((s, i) => (i === dim ? 1 : s));
    const out = zeros(outShape);
    const reduceFn = tensorReduce(operators.add);
    reduceFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        dim
    );
    return out;
}

export function prod(a: TensorData, dim: number): TensorData {
    const outShape = a.shape.map((s, i) => (i === dim ? 1 : s));
    const out = zeros(outShape);
    const reduceFn = tensorReduce(operators.mul);
    reduceFn(
        out.storage, out.shape, out.strides,
        a.storage, a.shape, a.strides,
        dim
    );
    return out;
}

export function permute(a: TensorData, order: number[]): TensorData {
    return a.permute(...order);
}

export function view(a: TensorData, shape: Shape): TensorData {
    const expectedStrides = strides(a.shape);
    const isContiguous = a.strides.every((s, i) => s === expectedStrides[i]);
    
    if (!isContiguous) {
        throw new Error (
            'Cannot view a non-contiguous tensor'
        );
    }

    const newSize = shapeProduct(shape);
    if (newSize !== a.size) {
        throw new Error(
            `Size mismatch of tensor and shape [${shape.join(', ')}]`
        );
    }

    return new TensorData(a.storage, shape);
}

export function contiguous(a: TensorData): TensorData {
    const expectedStrides = strides(a.shape);
    const isContiguous = a.strides.every((s, i) => s === expectedStrides[i]);

    if (isContiguous) {
        return a;
    }

    return id(a);
}

export class TensorContext {
    private _savedTensors: Tensor[] = [];

    saveForBackward(...tensors: Tensor[]): void {
        this._savedTensors = tensors;
    }

    get savedTensors(): Tensor[] {
        return this._savedTensors;
    }
}

export class TensorHistory {
    constructor (
        public lastFn: typeof TensorFunction | null = null,
        public ctx: TensorContext | null = null,
        public inputs: Tensor[] = []
    ) {}
}

export abstract class TensorFunction {
    static forward(ctx: TensorContext, ...inputs: Tensor[]): Tensor {
        throw new Error("forward not implemented");
    }

    static backward(ctx: TensorContext, gradOutput: Tensor): Tensor[] {
        throw new Error("backward not implemented");
    }
}