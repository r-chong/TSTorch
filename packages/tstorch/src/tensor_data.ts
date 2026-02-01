export type Shape = readonly number[];

export type Strides = readonly number[];

export type Index = readonly number[];

export type OutIndex = number[];

export type Storage = Float64Array;

export function indexToPosition(idx: Index, strides: Strides): number {
    let position = 0;
    for (let i = 0; i < idx.length; i++) {
        position += idx[i]! * strides[i]!;
    }
    return position;
}

export function toIndex(ordinal: number, shape: Shape, outIndex: OutIndex): void {
    let remaining = ordinal;
    for (let i = shape.length - 1; i >= 0; i--) {
        const dimSize = shape[i]!;
        outIndex[i] = remaining % dimSize;
        remaining = Math.floor(remaining / dimSize);
    }
}

export function shapeProduct(shape: Shape): number {
    let product = 1;
    for (const dim of shape) {
        product *= dim;
    }
    return product;
}

export function strides(shape: Shape): number[] {
    const result: number[] = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        result[i] = stride;
        stride *= shape[i]!;
    }
    return result;
}

export class TensorData {
    readonly storage: Storage;
    readonly shape: Shape;
    readonly strides: Strides;
    readonly size: number;
    readonly dims: number;

    constructor(
        storage: Storage,
        shape: Shape,
        inputStrides?: Strides,
    ) {
        this.storage = storage;
        this.shape = shape;
        this.strides = inputStrides ?? strides(shape);
        this.size = shapeProduct(shape);
        this.dims = shape.length;

        if (this.strides.length !== this.dims) {
            throw new Error(
                `Strides length (${this.strides.length}) must match shape length (${this.dims})`
            );
        }
    }

    static zeros(shape: Shape): TensorData {
        const size = shapeProduct(shape);
        const storage = new Float64Array(size);
        return new TensorData(storage, shape);
    }

    get(idx: Index): number {
        return this.storage[indexToPosition(idx, this.strides)]!;
    }

    set(idx: Index, value: number): void {
        this.storage[indexToPosition(idx, this.strides)] = value;
    }

    permute(...order: number[]): TensorData {
        if (order.length !== this.dims) {
            throw new Error(
                `Permutation length(${order.length}) must match number of dimensions (${this.dims})`
            );
        }

        const seen = new Set<number>();
        for (const i of order) {
            if (i < 0 || i >= this.dims) {
                throw new Error(`Invalid dimension index: ${i}`);
            }
            if (seen.has(i)) {
                throw new Error(`Duplicate dimension in permutation: ${i}`);
            }
            seen.add(i);
        }

        const newShape: number[] = new Array(this.dims);
        const newStrides: number[] = new Array(this.dims);

        for (let i = 0; i < this.dims; i++) {
            newShape[i] = this.shape[order[i]!]!;
            newStrides[i] = this.strides[order[i]!]!;
        }

        return new TensorData(this.storage, newShape, newStrides);
    }

    toString(): string {
        return `TensorData(shape=${JSON.stringify(this.shape)}, strides=${JSON.stringify(this.strides)})`;
    }
}