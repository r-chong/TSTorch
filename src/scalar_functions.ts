import { Context } from "./autodiff.js";
import * as operators from "./operators.js";

/**
 * ScalarHistory stores how a Scalar was created.
 * Note: inputs is typed as any[] to avoid circular dependency with Scalar
 */
export class ScalarHistory {
    constructor(
        public lastFn: typeof ScalarFunction | null = null,
        public ctx: Context | null = null,
        public inputs: any[] = []  // Will be Scalar[] at runtime
    ) {}
}

/**
 * Base class for all scalar operations.
 * Each operation implements forward (and later, backward).
 * 
 * Note: The apply() logic lives in Scalar class to avoid circular dependencies.
 */
export abstract class ScalarFunction {
    static forward(ctx: Context, ...inputs: number[]): number {
        throw new Error("forward not implemented");
    }
}

export class Add extends ScalarFunction {
    static forward(ctx: Context, a: number, b: number): number {
        // Don't need to save for backward since df/da = 1 and df/db = 1
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
