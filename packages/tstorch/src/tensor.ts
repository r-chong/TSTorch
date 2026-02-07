import type {
    Shape,
    Storage
} from './tensor_data.js'

import {
    TensorData,
    shapeProduct,
    strides as computeStrides,
} from './tensor_data.js'
import * as tensorFunctions from './tensor_functions.js'

export class Tensor {
    private _data: TensorData;
    grad: Tensor | null = null;

    constructor(data: TensorData) {
        this._data = data;
    }

    static tensor(values: number | number[])
}

function flattenArray(arr: any): { flat: number[]; inferredShape: number[] } {
    const shape: number[] = [];
    let current: any = arr;
    while (Array.isArray(current)) {
        shape.push(current.length);
        current = current[0];
    }

    const flat: number[] = [];
    flattenRecursive(arr, flat);

    return {flat, inferredShape: shape};
}

function flattenRecursive(arr: any, out: number[]): void {
    if (Array.isArray(arr)) {
        for (const item of arr) {
            flattenRecursive(item, out);
        }
    } else {
        out.push(arr);
    }
}

function buildNestedArray(data: TensorData, dim: number, idx: number[]): any {
    if (dim == data.dims) {
        return data.get(idx);
    }

    const result: any[] = [];
    for (let i = 0; i < data.shape[dim]!; i++) {
        idx[dim] = i;
        result.push(buildNestedArray(data, dim + 1, idx));
    }
    return result;
}