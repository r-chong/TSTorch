import {Context} from "./autodiff.js";
import * as operators from "./operators.js";

import type {Scalar, ScalarLike} from "./scalar.js";

/**
 * ScalarHistory stores how a Scalar was created.
 */
export class ScalarHistory {
    constructor(
        public lastFn: typeof ScalarFunction | null = null, // The 'typeof ScalarFunction' is the type of the ScalarFunction not an instance
        public ctx: Context | null = null,
        public inputs: Scalar[] = []
    ) {}
}

/**
 * Base class for all scalar operations.
 * Each operation implements forward and backward
 */
export abstract class ScalarFunction {
    static apply(this: typeof ScalarFunction, ...vals: ScalarLike[]): Scalar {
        // Prevent circular dependencies by importing Scalar here instead of at the top
        const {Scalar} = require("./scalar.js");

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

        const result = this._forward(ctx, ...rawVals);

        const history = new ScalarHistory(this, ctx, scalars);
        return new Scalar(result, history);
    }

    static _forward(ctx: Context, ...inputs: number[]): number {
        return this.forward(ctx, ...inputs);
    }

    static forward(ctx: Context, ...inputs: number[]): number {
        throw new Error("Not implemented");
    }
}

export class Add extends ScalarFunction {
    static forward(ctx: Context, a: number, b: number): number {
        // Don't need to save for backward since df/da = 1 and df/db = 1, values are not used in backward pass
        return operators.add(a, b);
    }
}

export class Log extends ScalarFunction {
    static forward(ctx: Context, a: number): number {
        ctx.saveForBackward(a);
        return operators.log(a);
    }
}

export class Mul extends ScalarFunction {
    static forward(ctx: Context, a: number, b: number): number {
        ctx.saveForBackward(a, b);
        return operators.mul(a, b);
    }
}

export class Inv extends ScalarFunction {
    static forward(ctx: Context, a: number): number {
        ctx.saveForBackward(a);
        return operators.inv(a);
    }
}

export class Neg extends ScalarFunction {
    static forward(ctx: Context, a: number): number {
        return operators.neg(a);
    }
}

export class Sigmoid extends ScalarFunction {
    static forward(ctx: Context, a: number): number {
        const result = operators.sigmoid(a);
        ctx.saveForBackward(result);
        return result;
    }
}

export class Relu extends ScalarFunction {
    static forward(ctx: Context, a: number): number {
        ctx.saveForBackward(a);
        return operators.relu(a);
    }
}

export class Exp extends ScalarFunction {
    static forward(ctx: Context, a: number): number {
        const result = operators.exp(a);
        ctx.saveForBackward(result);
        return result;
    }
}

export class LT extends ScalarFunction {
    static forward(ctx: Context, a: number, b: number): number {
        return operators.lt(a, b);
    }
}

export class EQ extends ScalarFunction {
    static forward(ctx: Context, a: number, b: number): number {
        return operators.eq(a, b);
    }
}

