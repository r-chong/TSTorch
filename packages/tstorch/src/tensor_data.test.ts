import { test, fc } from '@fast-check/jest';
import { describe, expect } from '@jest/globals';
import {
    indexToPosition,
    toIndex,
    strides,
    shapeProduct,
    TensorData,
    type OutIndex,
} from './tensor_data.js';

// ============================================================
// Arbitraries for property-based testing
// ============================================================

/** Small dimension size to keep tests fast */
const smallDim = fc.integer({ min: 1, max: 5 });

/** 1D shape */
const shape1D = fc.tuple(smallDim);

/** 2D shape */
const shape2D = fc.tuple(smallDim, smallDim);

/** 3D shape */
const shape3D = fc.tuple(smallDim, smallDim, smallDim);

/** Any shape up to 4D */
const anyShape = fc.oneof(
    shape1D,
    shape2D,
    shape3D,
    fc.tuple(smallDim, smallDim, smallDim, smallDim)
);

// ============================================================
// Task 2.1 - Tensor Data Indexing
// ============================================================

describe("strides computation", () => {
    test("strides for 1D tensor", () => {
        expect(strides([5])).toEqual([1]);
    });

    test("strides for 2D tensor", () => {
        expect(strides([2, 3])).toEqual([3, 1]);
    });

    test("strides for 3D tensor", () => {
        expect(strides([2, 3, 4])).toEqual([12, 4, 1]);
    });

    test("last stride is always 1", () => {
        fc.assert(
            fc.property(anyShape, (shape) => {
                const s = strides(shape);
                expect(s[s.length - 1]).toBe(1);
            })
        );
    });

    test.prop([anyShape])("stride[i] = stride[i+1] * shape[i+1]", (shape) => {
        const s = strides(shape);
        for (let i = 0; i < shape.length - 1; i++) {
            expect(s[i]).toBe(s[i + 1]! * shape[i + 1]!);
        }
    });
});

describe("shapeProduct", () => {
    test("product of 1D shape", () => {
        expect(shapeProduct([5])).toBe(5);
    });

    test("product of 2D shape", () => {
        expect(shapeProduct([2, 3])).toBe(6);
    });

    test("product of 3D shape", () => {
        expect(shapeProduct([2, 3, 4])).toBe(24);
    });

    test.prop([anyShape])("product equals multiplication of all dims", (shape) => {
        const expected = shape.reduce((a, b) => a * b, 1);
        expect(shapeProduct(shape)).toBe(expected);
    });
});

describe("indexToPosition", () => {
    test("first index maps to 0", () => {
        fc.assert(
            fc.property(anyShape, (shape) => {
                const s = strides(shape);
                const firstIndex = new Array(shape.length).fill(0);
                expect(indexToPosition(firstIndex, s)).toBe(0);
            })
        );
    });

    test("known 2D case", () => {
        // Shape [2, 3], strides [3, 1]
        // Index [1, 2] => 1*3 + 2*1 = 5
        expect(indexToPosition([1, 2], [3, 1])).toBe(5);
    });

    test("known 3D case", () => {
        // Shape [2, 3, 4], strides [12, 4, 1]
        // Index [1, 2, 3] => 1*12 + 2*4 + 3*1 = 23
        expect(indexToPosition([1, 2, 3], [12, 4, 1])).toBe(23);
    });

    test("incrementing last dimension increments position by 1", () => {
        // For row-major, moving in last dimension should be stride 1
        const shape = [2, 3, 4];
        const s = strides(shape);
        const pos1 = indexToPosition([0, 0, 0], s);
        const pos2 = indexToPosition([0, 0, 1], s);
        expect(pos2 - pos1).toBe(1);
    });
});

describe("toIndex", () => {
    test("ordinal 0 gives all zeros", () => {
        fc.assert(
            fc.property(anyShape, (shape) => {
                const outIndex: OutIndex = new Array(shape.length).fill(0);
                toIndex(0, shape, outIndex);
                expect(outIndex.every(i => i === 0)).toBe(true);
            })
        );
    });

    test("known 2D cases", () => {
        const shape = [2, 3];
        const outIndex: OutIndex = [0, 0];

        toIndex(0, shape, outIndex);
        expect(outIndex).toEqual([0, 0]);

        toIndex(1, shape, outIndex);
        expect(outIndex).toEqual([0, 1]);

        toIndex(2, shape, outIndex);
        expect(outIndex).toEqual([0, 2]);

        toIndex(3, shape, outIndex);
        expect(outIndex).toEqual([1, 0]);

        toIndex(5, shape, outIndex);
        expect(outIndex).toEqual([1, 2]);
    });

    test("known 3D case", () => {
        const shape = [2, 3, 4];
        const outIndex: OutIndex = [0, 0, 0];

        toIndex(23, shape, outIndex);
        expect(outIndex).toEqual([1, 2, 3]);
    });

    test("last ordinal gives max index", () => {
        const shape = [2, 3, 4];
        const outIndex: OutIndex = [0, 0, 0];
        const lastOrdinal = shapeProduct(shape) - 1; // 23

        toIndex(lastOrdinal, shape, outIndex);
        expect(outIndex).toEqual([1, 2, 3]); // shape - 1 in each dim
    });
});

describe("indexToPosition and toIndex roundtrip", () => {
    test.prop([anyShape])("toIndex then indexToPosition returns ordinal", (shape) => {
        const s = strides(shape);
        const size = shapeProduct(shape);
        const outIndex: OutIndex = new Array(shape.length).fill(0);

        for (let ordinal = 0; ordinal < size; ordinal++) {
            toIndex(ordinal, shape, outIndex);
            const position = indexToPosition(outIndex, s);
            expect(position).toBe(ordinal);
        }
    });

    test.prop([anyShape])("every index is visited exactly once", (shape) => {
        const size = shapeProduct(shape);
        const visited = new Set<string>();
        const outIndex: OutIndex = new Array(shape.length).fill(0);

        for (let ordinal = 0; ordinal < size; ordinal++) {
            toIndex(ordinal, shape, outIndex);
            const key = outIndex.join(',');
            expect(visited.has(key)).toBe(false);
            visited.add(key);
        }

        expect(visited.size).toBe(size);
    });
});

describe("TensorData construction", () => {
    test("zeros creates correct shape and size", () => {
        const td = TensorData.zeros([2, 3, 4]);
        expect([...td.shape]).toEqual([2, 3, 4]);
        expect(td.size).toBe(24);
        expect(td.dims).toBe(3);
        expect([...td.strides]).toEqual([12, 4, 1]);
    });

    test("zeros initializes all values to 0", () => {
        const td = TensorData.zeros([2, 3]);
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < 3; j++) {
                expect(td.get([i, j])).toBe(0);
            }
        }
    });

    test("get and set work correctly", () => {
        const td = TensorData.zeros([2, 3]);
        td.set([1, 2], 42);
        expect(td.get([1, 2])).toBe(42);
        expect(td.get([0, 0])).toBe(0); // others unchanged
    });

    test("throws if strides length mismatches shape", () => {
        const storage = new Float64Array(6);
        expect(() => new TensorData(storage, [2, 3], [1])).toThrow();
    });
});

describe("TensorData.permute", () => {
    test("permute 2D tensor (transpose)", () => {
        const td = TensorData.zeros([2, 3]);
        // Fill with distinct values
        td.set([0, 0], 1); td.set([0, 1], 2); td.set([0, 2], 3);
        td.set([1, 0], 4); td.set([1, 1], 5); td.set([1, 2], 6);

        const transposed = td.permute(1, 0);

        // Shape should be swapped
        expect([...transposed.shape]).toEqual([3, 2]);

        // Values should be transposed
        expect(transposed.get([0, 0])).toBe(1);
        expect(transposed.get([0, 1])).toBe(4);
        expect(transposed.get([1, 0])).toBe(2);
        expect(transposed.get([1, 1])).toBe(5);
        expect(transposed.get([2, 0])).toBe(3);
        expect(transposed.get([2, 1])).toBe(6);
    });

    test("permute 3D tensor", () => {
        const td = TensorData.zeros([2, 3, 4]);
        // Set a specific value
        td.set([1, 2, 3], 99);

        const permuted = td.permute(2, 0, 1);
        expect([...permuted.shape]).toEqual([4, 2, 3]);

        // Original [1, 2, 3] maps to [3, 1, 2] after permute(2, 0, 1)
        expect(permuted.get([3, 1, 2])).toBe(99);
    });

    test("permute shares storage (modifications visible)", () => {
        const td = TensorData.zeros([2, 3]);
        td.set([0, 1], 42);

        const permuted = td.permute(1, 0);

        // Modify via permuted view
        permuted.set([1, 0], 99);

        // Original should see the change
        expect(td.get([0, 1])).toBe(99);
    });

    test("identity permute returns same shape", () => {
        const td = TensorData.zeros([2, 3, 4]);
        const same = td.permute(0, 1, 2);
        expect([...same.shape]).toEqual([2, 3, 4]);
        expect([...same.strides]).toEqual([12, 4, 1]);
    });

    test("permute validates order length", () => {
        const td = TensorData.zeros([2, 3]);
        expect(() => td.permute(0)).toThrow(/must match/);
        expect(() => td.permute(0, 1, 2)).toThrow(/must match/);
    });

    test("permute validates dimension indices", () => {
        const td = TensorData.zeros([2, 3]);
        expect(() => td.permute(0, 2)).toThrow(/Invalid dimension/);
        expect(() => td.permute(-1, 0)).toThrow(/Invalid dimension/);
    });

    test("permute rejects duplicate dimensions", () => {
        const td = TensorData.zeros([2, 3]);
        expect(() => td.permute(0, 0)).toThrow(/Duplicate/);
        expect(() => td.permute(1, 1)).toThrow(/Duplicate/);
    });

    test.prop([shape2D])("double transpose returns original layout", (shape) => {
        const td = TensorData.zeros(shape);
        // Fill with ordinal values
        const outIndex: OutIndex = [0, 0];
        for (let i = 0; i < td.size; i++) {
            toIndex(i, shape, outIndex);
            td.set(outIndex, i);
        }

        const transposed = td.permute(1, 0);
        const restored = transposed.permute(1, 0);

        expect([...restored.shape]).toEqual([...shape]);

        // Check all values match
        for (let i = 0; i < td.size; i++) {
            toIndex(i, shape, outIndex);
            expect(restored.get(outIndex)).toBe(td.get(outIndex));
        }
    });

    test.prop([shape3D])("permute then inverse permute restores values", (shape) => {
        const td = TensorData.zeros(shape);
        const outIndex: OutIndex = [0, 0, 0];

        // Fill with ordinal values
        for (let i = 0; i < td.size; i++) {
            toIndex(i, shape, outIndex);
            td.set(outIndex, i);
        }

        // Permute with [2, 0, 1] and inverse [1, 2, 0]
        const permuted = td.permute(2, 0, 1);
        const restored = permuted.permute(1, 2, 0);

        expect([...restored.shape]).toEqual([...shape]);

        // Check all values
        for (let i = 0; i < td.size; i++) {
            toIndex(i, shape, outIndex);
            expect(restored.get(outIndex)).toBe(td.get(outIndex));
        }
    });
});

describe("TensorData.toString", () => {
    test("returns readable format", () => {
        const td = TensorData.zeros([2, 3]);
        expect(td.toString()).toBe('TensorData(shape=[2,3], strides=[3,1])');
    });
});
