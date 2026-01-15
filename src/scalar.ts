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
 * 
 * @param value - The value of the scalar
 * @param history - The history of the scalar
 * @param uniqueId - The unique id of the scalar
 * @param name - The name of the scalar
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

    add(b: ScalarLike): Scalar {
        return Add.apply(this, b);
    }

    mul(b: ScalarLike): Scalar {
        return Mul.apply(this, b);
    }

    div(b: ScalarLike): Scalar {
        return Mul.apply(this, Inv.apply(b));
    }

    rdiv(b: ScalarLike): Scalar {
        return Mul.apply(b, Inv.apply(this));
    }

    sub(b: ScalarLike): Scalar {
        return Add.apply(this, Neg.apply(b));
    }

    neg(): Scalar {
        return Neg.apply(this);
    }

    lt(b: ScalarLike): Scalar {
        return LT.apply(this, b);
    }

    eq(b: ScalarLike): Scalar {
        return EQ.apply(this, b);
    }

    gt(b: ScalarLike): Scalar {
        return LT.apply(b, this);
    }

    log(): Scalar {
        return Log.apply(this);
    }

    exp(): Scalar {
        return Exp.apply(this);
    }

    sigmoid(): Scalar {
        return Sigmoid.apply(this);
    }

    relu(): Scalar {
        return Relu.apply(this);
    }

    isLeaf(): boolean {
        return this.history !== null && this.history.lastFn === null;
    }

    isConstant(): boolean {
        return this.history === null;
    }

    get parents(): Scalar[] {
        return this.history?.inputs ?? [];
    }

    accumulateDerivative(d: number): void {
        if (!this.isLeaf()) {
            throw new Error("Cannot accumulate derivative of a non-leaf scalar");
        }
        if (this.derivative === null) {
            this.derivative = 0;
        }
        this.derivative += d;
    }
}

export {ScalarHistory};