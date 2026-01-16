import { Context } from "./autodiff.js";
import {
    ScalarHistory, 
    ScalarFunction,
    Add,
    Log,
    Mul,
    Inv,
    Neg,
    Sigmoid,
    Relu,
    Exp,
    LT,
    EQ,
} from "./scalar_functions.js";

export type ScalarLike = number | Scalar;

let _varCount = 0;

/**
 * Scalar: A number that tracks its computation history.
 * Behaves like a regular number but records operations for autodiff.
 */
export class Scalar {
    readonly data: number;
    readonly history: ScalarHistory | null;
    readonly uniqueId: number;
    readonly name: string;

    derivative: number | null = null; // Filled in during backward pass

    constructor(
        value: number,  
        history: ScalarHistory | null = null,
        name?: string
    ) {
        _varCount++;
        this.uniqueId = _varCount;
        this.data = value;
        this.history = history;
        this.name = name ?? `var${this.uniqueId}`;
    }

    toString(): string {
        return `Scalar(${this.data})`;
    }

    /**
     * Apply a ScalarFunction to the given values.
     * Handles unwrapping Scalars to numbers, calling forward, and wrapping the result.
     */
    private static applyFn(fn: typeof ScalarFunction, ...vals: ScalarLike[]): Scalar {
        const rawVals: number[] = [];
        const scalars: Scalar[] = [];

        for (const v of vals) {
            if (v instanceof Scalar) {
                scalars.push(v);
                rawVals.push(v.data);
            } else {
                scalars.push(new Scalar(v));
                rawVals.push(v);
            }
        }

        const ctx = new Context();

        const result = fn.forward(ctx, ...rawVals);

        const history = new ScalarHistory(fn, ctx, scalars);
        return new Scalar(result, history);
    }

    add(b: ScalarLike): Scalar {
        return Scalar.applyFn(Add, this, b);
    }

    mul(b: ScalarLike): Scalar {
        return Scalar.applyFn(Mul, this, b);
    }

    div(b: ScalarLike): Scalar {
        return Scalar.applyFn(Mul, this, Scalar.applyFn(Inv, b));
    }

    rdiv(b: ScalarLike): Scalar {
        return Scalar.applyFn(Mul, b, Scalar.applyFn(Inv, this));
    }

    sub(b: ScalarLike): Scalar {
        return Scalar.applyFn(Add, this, Scalar.applyFn(Neg, b));
    }

    neg(): Scalar {
        return Scalar.applyFn(Neg, this);
    }

    lt(b: ScalarLike): Scalar {
        return Scalar.applyFn(LT, this, b);
    }

    eq(b: ScalarLike): Scalar {
        return Scalar.applyFn(EQ, this, b);
    }

    gt(b: ScalarLike): Scalar {
        return Scalar.applyFn(LT, b, this);
    }

    log(): Scalar {
        return Scalar.applyFn(Log, this);
    }

    exp(): Scalar {
        return Scalar.applyFn(Exp, this);
    }

    sigmoid(): Scalar {
        return Scalar.applyFn(Sigmoid, this);
    }

    relu(): Scalar {
        return Scalar.applyFn(Relu, this);
    }

    /* 
        we know how the loss changes with respect to final (dOut)
        we compute an iterable of 
        
        {
            [input, dOut * gradient],
            [input, dOut * gradient],
            ...
        }
    */
    chainRule(dOut: any): Iterable<[ScalarLike, any]> {
        const h = this.history;
        if (!h) {
            throw new Error("Missing scalar history");
        }
        if (h && !h.lastFn) {
            throw new Error("Missing lastFn in scalar history");
        }
        if (h && !h.ctx) {
            throw new Error("Missing ctx in scalar historyh");
        }

        const savedValues = h?.ctx?.savedValues;
        const lastFn = h?.lastFn;
        const ctx = h?.ctx;
        const data: (Scalar | any)[][] = [];

        for (const scalar in savedValues) {
            //@ts-ignore as we haven't implemented 1.4 yet
            data.push([scalar, dOut * lastFn?.backward(ctx, savedValues)]);
        }

        // convert to tuple 
        const iterableData: Iterable<[ScalarLike, any]> = 
        data.map((item): [Scalar, any] => [item[0] as Scalar, item[1]])

        // not to be confused with map in `operators.ts`
        const res = new Map(iterableData);

        return res;
    }
}

export { ScalarHistory };
