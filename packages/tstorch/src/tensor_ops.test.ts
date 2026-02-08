import { test, fc } from '@fast-check/jest';
import { describe, expect } from '@jest/globals';
import { tensorMap, tensorZip, tensorReduce } from './tensor_ops.js';
import { TensorData, strides, shapeProduct } from './tensor_data.js';

// ============================================================
// Arbitraries for property-based testing
// ============================================================

const smallDim = fc.integer({ min: 1, max: 5 });
const shape1D = fc.tuple(smallDim);
const shape2D = fc.tuple(smallDim, smallDim);
const shape3D = fc.tuple(smallDim, smallDim, smallDim);

// ============================================================
// Task 2.3 - tensorMap
// ============================================================

describe("tensorMap", () => {
    test("identity map preserves values", () => {
        const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const output = TensorData.zeros([2, 3]);
        
        const mapFn = tensorMap((x) => x);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);
        
        expect(Array.from(output.storage)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    test("negation map", () => {
        const input = new TensorData(new Float64Array([1, -2, 3, -4]), [2, 2]);
        const output = TensorData.zeros([2, 2]);
        
        const mapFn = tensorMap((x) => -x);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);
        
        expect(Array.from(output.storage)).toEqual([-1, 2, -3, 4]);
    });

    test("square map", () => {
        const input = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
        const output = TensorData.zeros([4]);
        
        const mapFn = tensorMap((x) => x * x);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);
        
        expect(Array.from(output.storage)).toEqual([1, 4, 9, 16]);
    });

    test("map with broadcasting - scalar to 2D", () => {
        // Input is scalar (shape []), output is 2x2
        const input = new TensorData(new Float64Array([5]), []);
        const output = TensorData.zeros([2, 2]);
        
        const mapFn = tensorMap((x) => x * 2);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);
        
        expect(Array.from(output.storage)).toEqual([10, 10, 10, 10]);
    });

    test("map with broadcasting - 1D to 2D", () => {
        // Input [3] broadcasts to [2, 3]
        const input = new TensorData(new Float64Array([1, 2, 3]), [3]);
        const output = TensorData.zeros([2, 3]);
        
        const mapFn = tensorMap((x) => x);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);
        
        expect(Array.from(output.storage)).toEqual([1, 2, 3, 1, 2, 3]);
    });

    test("map with broadcasting - column vector to 2D", () => {
        // Input [2, 1] broadcasts to [2, 3]
        const input = new TensorData(new Float64Array([10, 20]), [2, 1]);
        const output = TensorData.zeros([2, 3]);
        
        const mapFn = tensorMap((x) => x);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);
        
        expect(Array.from(output.storage)).toEqual([10, 10, 10, 20, 20, 20]);
    });

    test.prop([shape2D])("map preserves shape", (shape) => {
        const size = shapeProduct(shape);
        const input = new TensorData(new Float64Array(size).fill(1), shape);
        const output = TensorData.zeros(shape);
        
        const mapFn = tensorMap((x) => x + 1);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);
        
        expect(output.shape).toEqual(shape);
        expect(output.size).toBe(size);
    });

    test("map on non-contiguous tensor (permuted)", () => {
        // Create a 2x3 tensor and permute to 3x2
        const original = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const permuted = original.permute(1, 0); // Now shape [3, 2], strides [1, 3]
        const output = TensorData.zeros([3, 2]);
        
        const mapFn = tensorMap((x) => x * 10);
        mapFn(output.storage, output.shape, output.strides,
              permuted.storage, permuted.shape, permuted.strides);
        
        // Permuted view: [[1,4], [2,5], [3,6]] * 10 = [[10,40], [20,50], [30,60]]
        expect(Array.from(output.storage)).toEqual([10, 40, 20, 50, 30, 60]);
    });
});

// ============================================================
// Task 2.3 - tensorZip
// ============================================================

describe("tensorZip", () => {
    test("element-wise addition", () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const b = new TensorData(new Float64Array([10, 20, 30, 40]), [2, 2]);
        const output = TensorData.zeros([2, 2]);
        
        const zipFn = tensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);
        
        expect(Array.from(output.storage)).toEqual([11, 22, 33, 44]);
    });

    test("element-wise multiplication", () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const b = new TensorData(new Float64Array([2, 3, 4, 5]), [2, 2]);
        const output = TensorData.zeros([2, 2]);
        
        const zipFn = tensorZip((x, y) => x * y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);
        
        expect(Array.from(output.storage)).toEqual([2, 6, 12, 20]);
    });

    test("element-wise subtraction", () => {
        const a = new TensorData(new Float64Array([10, 20, 30, 40]), [4]);
        const b = new TensorData(new Float64Array([1, 2, 3, 4]), [4]);
        const output = TensorData.zeros([4]);
        
        const zipFn = tensorZip((x, y) => x - y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);
        
        expect(Array.from(output.storage)).toEqual([9, 18, 27, 36]);
    });

    test("zip with broadcasting - scalar + 2D", () => {
        const a = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const b = new TensorData(new Float64Array([10]), []);
        const output = TensorData.zeros([2, 2]);
        
        const zipFn = tensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);
        
        expect(Array.from(output.storage)).toEqual([11, 12, 13, 14]);
    });

    test("zip with broadcasting - row + column", () => {
        // [1, 2, 3] shape [1, 3] + [[10], [20]] shape [2, 1] = [[11,12,13], [21,22,23]] shape [2, 3]
        const a = new TensorData(new Float64Array([1, 2, 3]), [1, 3]);
        const b = new TensorData(new Float64Array([10, 20]), [2, 1]);
        const output = TensorData.zeros([2, 3]);
        
        const zipFn = tensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);
        
        expect(Array.from(output.storage)).toEqual([11, 12, 13, 21, 22, 23]);
    });

    test("zip with broadcasting - 1D + 2D", () => {
        // [1, 2, 3] shape [3] + [[10,20,30], [40,50,60]] shape [2, 3]
        const a = new TensorData(new Float64Array([1, 2, 3]), [3]);
        const b = new TensorData(new Float64Array([10, 20, 30, 40, 50, 60]), [2, 3]);
        const output = TensorData.zeros([2, 3]);
        
        const zipFn = tensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);
        
        expect(Array.from(output.storage)).toEqual([11, 22, 33, 41, 52, 63]);
    });

    test("zip comparison - less than", () => {
        const a = new TensorData(new Float64Array([1, 5, 3, 7]), [4]);
        const b = new TensorData(new Float64Array([2, 4, 3, 8]), [4]);
        const output = TensorData.zeros([4]);
        
        const zipFn = tensorZip((x, y) => x < y ? 1 : 0);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);
        
        expect(Array.from(output.storage)).toEqual([1, 0, 0, 1]);
    });

    test.prop([shape2D])("zip addition is commutative", (shape) => {
        const size = shapeProduct(shape);
        const aData = new Float64Array(size);
        const bData = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            aData[i] = i;
            bData[i] = i * 2;
        }
        const a = new TensorData(aData, shape);
        const b = new TensorData(bData, shape);
        const output1 = TensorData.zeros(shape);
        const output2 = TensorData.zeros(shape);
        
        const zipFn = tensorZip((x, y) => x + y);
        zipFn(output1.storage, output1.shape, output1.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);
        zipFn(output2.storage, output2.shape, output2.strides,
              b.storage, b.shape, b.strides,
              a.storage, a.shape, a.strides);
        
        expect(Array.from(output1.storage)).toEqual(Array.from(output2.storage));
    });
});

// ============================================================
// Task 2.3 - tensorReduce
// ============================================================

describe("tensorReduce", () => {
    test("sum reduction along dim 0 - 2D tensor", () => {
        // [[1, 2, 3], [4, 5, 6]] sum along dim 0 = [[5, 7, 9]]
        const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const output = TensorData.zeros([1, 3]);
        
        const reduceFn = tensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);
        
        expect(Array.from(output.storage)).toEqual([5, 7, 9]);
    });

    test("sum reduction along dim 1 - 2D tensor", () => {
        // [[1, 2, 3], [4, 5, 6]] sum along dim 1 = [[6], [15]]
        const input = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const output = TensorData.zeros([2, 1]);
        
        const reduceFn = tensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 1);
        
        expect(Array.from(output.storage)).toEqual([6, 15]);
    });

    test("product reduction along dim 0", () => {
        // [[1, 2], [3, 4]] product along dim 0 = [[3, 8]]
        const input = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const output = TensorData.zeros([1, 2]);
        
        const reduceFn = tensorReduce((acc, x) => acc * x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);
        
        expect(Array.from(output.storage)).toEqual([3, 8]);
    });

    test("product reduction along dim 1", () => {
        // [[1, 2], [3, 4]] product along dim 1 = [[2], [12]]
        const input = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const output = TensorData.zeros([2, 1]);
        
        const reduceFn = tensorReduce((acc, x) => acc * x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 1);
        
        expect(Array.from(output.storage)).toEqual([2, 12]);
    });

    test("max reduction along dim 0", () => {
        // [[1, 5, 3], [4, 2, 6]] max along dim 0 = [[4, 5, 6]]
        const input = new TensorData(new Float64Array([1, 5, 3, 4, 2, 6]), [2, 3]);
        const output = TensorData.zeros([1, 3]);
        
        const reduceFn = tensorReduce((acc, x) => Math.max(acc, x));
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);
        
        expect(Array.from(output.storage)).toEqual([4, 5, 6]);
    });

    test("min reduction along dim 1", () => {
        // [[1, 5, 3], [4, 2, 6]] min along dim 1 = [[1], [2]]
        const input = new TensorData(new Float64Array([1, 5, 3, 4, 2, 6]), [2, 3]);
        const output = TensorData.zeros([2, 1]);
        
        const reduceFn = tensorReduce((acc, x) => Math.min(acc, x));
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 1);
        
        expect(Array.from(output.storage)).toEqual([1, 2]);
    });

    test("reduction on 1D tensor", () => {
        // [1, 2, 3, 4, 5] sum = [15]
        const input = new TensorData(new Float64Array([1, 2, 3, 4, 5]), [5]);
        const output = TensorData.zeros([1]);
        
        const reduceFn = tensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);
        
        expect(Array.from(output.storage)).toEqual([15]);
    });

    test("reduction on 3D tensor along dim 0", () => {
        // Shape [2, 2, 2], reduce along dim 0 -> [1, 2, 2]
        const input = new TensorData(
            new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]), 
            [2, 2, 2]
        );
        const output = TensorData.zeros([1, 2, 2]);
        
        const reduceFn = tensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);
        
        // [[[1,2],[3,4]], [[5,6],[7,8]]] sum dim 0 = [[6,8],[10,12]]
        expect(Array.from(output.storage)).toEqual([6, 8, 10, 12]);
    });

    test("reduction on 3D tensor along dim 1", () => {
        // Shape [2, 2, 2], reduce along dim 1 -> [2, 1, 2]
        const input = new TensorData(
            new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]), 
            [2, 2, 2]
        );
        const output = TensorData.zeros([2, 1, 2]);
        
        const reduceFn = tensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 1);
        
        // [[[1,2],[3,4]], [[5,6],[7,8]]] sum dim 1 = [[[4,6]], [[12,14]]]
        expect(Array.from(output.storage)).toEqual([4, 6, 12, 14]);
    });

    test("reduction on 3D tensor along dim 2", () => {
        // Shape [2, 2, 2], reduce along dim 2 -> [2, 2, 1]
        const input = new TensorData(
            new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]), 
            [2, 2, 2]
        );
        const output = TensorData.zeros([2, 2, 1]);
        
        const reduceFn = tensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 2);
        
        // [[[1,2],[3,4]], [[5,6],[7,8]]] sum dim 2 = [[[3],[7]], [[11],[15]]]
        expect(Array.from(output.storage)).toEqual([3, 7, 11, 15]);
    });

    test("reduction with single element along dimension", () => {
        // Shape [1, 4], reduce along dim 0 -> should just copy
        const input = new TensorData(new Float64Array([1, 2, 3, 4]), [1, 4]);
        const output = TensorData.zeros([1, 4]);
        
        const reduceFn = tensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);
        
        expect(Array.from(output.storage)).toEqual([1, 2, 3, 4]);
    });

    test.prop([
        fc.integer({ min: 2, max: 4 }),
        fc.integer({ min: 2, max: 4 }),
        fc.integer({ min: 0, max: 1 })
    ])("sum reduction equals manual sum", (dim0, dim1, reduceDim) => {
        const shape = [dim0, dim1] as const;
        const size = dim0 * dim1;
        const data = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            data[i] = i + 1;
        }
        const input = new TensorData(data, shape);
        
        const outShape = shape.map((s, i) => i === reduceDim ? 1 : s);
        const output = TensorData.zeros(outShape);
        
        const reduceFn = tensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, reduceDim);
        
        // Verify by manual calculation
        const reduceSize = shape[reduceDim];
        const otherSize = shape[1 - reduceDim];
        
        for (let i = 0; i < otherSize; i++) {
            let expected = 0;
            for (let j = 0; j < reduceSize; j++) {
                const idx = reduceDim === 0 
                    ? [j, i] 
                    : [i, j];
                expected += input.get(idx);
            }
            const outIdx = reduceDim === 0 ? [0, i] : [i, 0];
            expect(output.get(outIdx)).toBe(expected);
        }
    });
});

// ============================================================
// Edge Cases for tensor_ops
// ============================================================

describe("tensorMap edge cases", () => {
    test("map on 0-dimensional tensor (scalar)", () => {
        const input = new TensorData(new Float64Array([5]), []);
        const output = TensorData.zeros([]);
        
        const mapFn = tensorMap((x) => x * 2);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);
        
        expect(output.storage[0]).toBe(10);
    });

    test("map with same input and output shape", () => {
        const input = new TensorData(new Float64Array([1, 2, 3, 4]), [2, 2]);
        const output = TensorData.zeros([2, 2]);
        
        const mapFn = tensorMap((x) => x + 1);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);
        
        expect(Array.from(output.storage)).toEqual([2, 3, 4, 5]);
    });

    test("map on 4D tensor", () => {
        const input = new TensorData(new Float64Array(16).fill(1), [2, 2, 2, 2]);
        const output = TensorData.zeros([2, 2, 2, 2]);
        
        const mapFn = tensorMap((x) => x * 3);
        mapFn(output.storage, output.shape, output.strides,
              input.storage, input.shape, input.strides);
        
        expect(output.storage.every(v => v === 3)).toBe(true);
    });
});

describe("tensorZip edge cases", () => {
    test("zip two scalar tensors", () => {
        const a = new TensorData(new Float64Array([3]), []);
        const b = new TensorData(new Float64Array([4]), []);
        const output = TensorData.zeros([]);
        
        const zipFn = tensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);
        
        expect(output.storage[0]).toBe(7);
    });

    test("zip on non-contiguous tensors", () => {
        const original = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const permuted = original.permute(1, 0); // [3, 2] with strides [1, 3]
        const b = new TensorData(new Float64Array([10, 20, 30, 40, 50, 60]), [3, 2]);
        const output = TensorData.zeros([3, 2]);
        
        const zipFn = tensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              permuted.storage, permuted.shape, permuted.strides,
              b.storage, b.shape, b.strides);
        
        // permuted: [[1,4], [2,5], [3,6]]
        // b: [[10,20], [30,40], [50,60]]
        // result: [[11,24], [32,45], [53,66]]
        expect(Array.from(output.storage)).toEqual([11, 24, 32, 45, 53, 66]);
    });

    test("zip with 3D broadcasting", () => {
        const a = new TensorData(new Float64Array([1, 2]), [2]); // broadcasts to [2, 2, 2]
        const b = new TensorData(new Float64Array([10, 20, 30, 40, 50, 60, 70, 80]), [2, 2, 2]);
        const output = TensorData.zeros([2, 2, 2]);
        
        const zipFn = tensorZip((x, y) => x + y);
        zipFn(output.storage, output.shape, output.strides,
              a.storage, a.shape, a.strides,
              b.storage, b.shape, b.strides);
        
        expect(Array.from(output.storage)).toEqual([11, 22, 31, 42, 51, 62, 71, 82]);
    });
});

describe("tensorReduce edge cases", () => {
    test("reduce on non-contiguous tensor", () => {
        const original = new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
        const permuted = original.permute(1, 0); // [3, 2] with strides [1, 3]
        // permuted: [[1,4], [2,5], [3,6]]
        const output = TensorData.zeros([3, 1]);
        
        const reduceFn = tensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 permuted.storage, permuted.shape, permuted.strides, 1);
        
        // sum along dim 1: [1+4, 2+5, 3+6] = [5, 7, 9]
        expect(Array.from(output.storage)).toEqual([5, 7, 9]);
    });

    test("reduce 4D tensor along middle dimension", () => {
        // Shape [2, 3, 2, 2], reduce along dim 1 -> [2, 1, 2, 2]
        const size = 2 * 3 * 2 * 2;
        const data = new Float64Array(size);
        for (let i = 0; i < size; i++) data[i] = i + 1;
        const input = new TensorData(data, [2, 3, 2, 2]);
        const output = TensorData.zeros([2, 1, 2, 2]);
        
        const reduceFn = tensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 1);
        
        // Verify first element: sum of elements at [0,0,0,0], [0,1,0,0], [0,2,0,0]
        // = 1 + 5 + 9 = 15
        expect(output.get([0, 0, 0, 0])).toBe(15);
    });

    test("reduce with large dimension", () => {
        const input = new TensorData(new Float64Array(100).fill(1), [100]);
        const output = TensorData.zeros([1]);
        
        const reduceFn = tensorReduce((acc, x) => acc + x);
        reduceFn(output.storage, output.shape, output.strides,
                 input.storage, input.shape, input.strides, 0);
        
        expect(output.storage[0]).toBe(100);
    });
});
