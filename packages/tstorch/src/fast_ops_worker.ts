import { parentPort, workerData } from 'node:worker_threads';

function toIndex(ordinal: number, shape: number[], outIndex: number[]): void {
    let remaining = ordinal;
    for (let i = shape.length - 1; i >= 0; i--) {
        const dimSize = shape[i]!;
        outIndex[i] = remaining % dimSize;
        remaining = Math.floor(remaining / dimSize);
    }
}

function indexToPosition(idx: number[], strides: number[]): number {
    let position = 0;
    for (let i = 0; i < idx.length; i++) {
        position += idx[i]! * strides[i]!;
    }
    return position;
}

function broadcastIndex(
    bigIndex: number[],
    bigShape: number[],
    shape: number[],
    outIndex: number[],
): void {
    const offset = bigShape.length - shape.length;
    for (let i = 0; i < shape.length; i++) {
        const bigI = i + offset;
        if (shape[i] === 1) {
            outIndex[i] = 0;
        } else {
            outIndex[i] = bigIndex[bigI]!;
        }
    }
}

interface MapTask {
    type: 'map';
    fnSource: string;
    start: number;
    end: number;
    outBuffer: SharedArrayBuffer;
    outShape: number[];
    outStrides: number[];
    inBuffer: SharedArrayBuffer;
    inShape: number[];
    inStrides: number[];
    aligned: boolean;
}

interface ZipTask {
    type: 'zip';
    fnSource: string;
    start: number;
    end: number;
    outBuffer: SharedArrayBuffer;
    outShape: number[];
    outStrides: number[];
    aBuffer: SharedArrayBuffer;
    aShape: number[];
    aStrides: number[];
    bBuffer: SharedArrayBuffer;
    bShape: number[];
    bStrides: number[];
    aligned: boolean;
}

interface ReduceTask {
    type: 'reduce';
    fnSource: string;
    start: number;
    end: number;
    outBuffer: SharedArrayBuffer;
    outShape: number[];
    outStrides: number[];
    inBuffer: SharedArrayBuffer;
    inShape: number[];
    inStrides: number[];
    reduceDim: number;
    reduceDimSize: number;
}

type Task = MapTask | ZipTask | ReduceTask;


const { workerId, syncBuffer } = workerData as {
    workerId: number;
    syncBuffer: SharedArrayBuffer;
};
const syncArray = new Int32Array(syncBuffer);

function handleMap(task: MapTask): void {
    const fn = new Function('return ' + task.fnSource)() as (x: number) => number;
    const outStorage = new Float64Array(task.outBuffer);
    const inStorage = new Float64Array(task.inBuffer);

    if (task.aligned) {
        for (let i = task.start; i < task.end; i++) {
            outStorage[i] = fn(inStorage[i]!);
        }
    } else {
        const outIndex: number[] = new Array(task.outShape.length).fill(0);
        const inIndex: number[] = new Array(task.inShape.length).fill(0);

        for (let ordinal = task.start; ordinal < task.end; ordinal++) {
            toIndex(ordinal, task.outShape, outIndex);
            broadcastIndex(outIndex, task.outShape, task.inShape, inIndex);

            const inPos = indexToPosition(inIndex, task.inStrides);
            const outPos = indexToPosition(outIndex, task.outStrides);
            outStorage[outPos] = fn(inStorage[inPos]!);
        }
    }
}

function handleZip(task: ZipTask): void {
    const fn = new Function('return ' + task.fnSource)() as (
        a: number,
        b: number,
    ) => number;
    const outStorage = new Float64Array(task.outBuffer);
    const aStorage = new Float64Array(task.aBuffer);
    const bStorage = new Float64Array(task.bBuffer);

    if (task.aligned) {
        for (let i = task.start; i < task.end; i++) {
            outStorage[i] = fn(aStorage[i]!, bStorage[i]!);
        }
    } else {
        const outIndex: number[] = new Array(task.outShape.length).fill(0);
        const aIndex: number[] = new Array(task.aShape.length).fill(0);
        const bIndex: number[] = new Array(task.bShape.length).fill(0);

        for (let ordinal = task.start; ordinal < task.end; ordinal++) {
            toIndex(ordinal, task.outShape, outIndex);
            broadcastIndex(outIndex, task.outShape, task.aShape, aIndex);
            broadcastIndex(outIndex, task.outShape, task.bShape, bIndex);

            const aPos = indexToPosition(aIndex, task.aStrides);
            const bPos = indexToPosition(bIndex, task.bStrides);
            const outPos = indexToPosition(outIndex, task.outStrides);
            outStorage[outPos] = fn(aStorage[aPos]!, bStorage[bPos]!);
        }
    }
}

function handleReduce(task: ReduceTask): void {
    const fn = new Function('return ' + task.fnSource)() as (
        acc: number,
        x: number,
    ) => number;
    const outStorage = new Float64Array(task.outBuffer);
    const inStorage = new Float64Array(task.inBuffer);
    const reduceStride = task.inStrides[task.reduceDim]!;

    const outIndex: number[] = new Array(task.outShape.length).fill(0);
    const inIndex: number[] = new Array(task.inShape.length).fill(0);

    for (let ordinal = task.start; ordinal < task.end; ordinal++) {
        toIndex(ordinal, task.outShape, outIndex);
        const outPos = indexToPosition(outIndex, task.outStrides);

        for (let i = 0; i < task.outShape.length; i++) {
            inIndex[i] = outIndex[i]!;
        }
        inIndex[task.reduceDim] = 0;
        let inPos = indexToPosition(inIndex, task.inStrides);

        let acc = inStorage[inPos]!;
        for (let j = 1; j < task.reduceDimSize; j++) {
            inPos += reduceStride;
            acc = fn(acc, inStorage[inPos]!);
        }

        outStorage[outPos] = acc;
    }
}

parentPort!.on('message', (task: Task) => {
    switch (task.type) {
        case 'map':
            handleMap(task);
            break;
        case 'zip':
            handleZip(task);
            break;
        case 'reduce':
            handleReduce(task);
            break;
    }

    Atomics.store(syncArray, workerId, 1);
    Atomics.notify(syncArray, workerId);
    parentPort!.postMessage('done');
});
