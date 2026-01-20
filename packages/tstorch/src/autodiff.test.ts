import { test, fc } from '@fast-check/jest';
import { centralDifference, Context, topologicalSort, backPropagate } from "./autodiff.js";
import { Scalar } from "./scalar.js"
import { ScalarFunction, ScalarHistory } from "./scalar_functions.js"; 
import * as operators from './operators.js';
const DIGIT_TOLERANCE = 4;

/** floats from set of small finite floats */
const smallFloat = fc.double({ noNaN: true, min: -100, max: 100 });

/** Positive floats for functions like log, sqrt */
const positiveFloat = fc.double({ noNaN: true, min: 0.1, max: 100 });

/** Non-zero floats for division */
const nonZeroFloat = fc.double({ noNaN: true, min: -100, max: 100 }).filter(x => Math.abs(x) > 0.1);

function assertClose(actual: number, expected: number, digits = DIGIT_TOLERANCE) {
  expect(actual).toBeCloseTo(expected, digits);
}

// ============================================================
// Central Difference Tests
// ============================================================

describe("centralDifference", () => {
  describe("single variable functions", () => {
    test("derivative of f(x) = x is 1", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (a: number) => a;
          assertClose(centralDifference(f, [x]), 1);
        })
      );
    });

    test("derivative of f(x) = x^2 is 2x", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (a: number) => a * a;
          assertClose(centralDifference(f, [x]), 2 * x);
        })
      );
    });

    test("derivative of f(x) = x^3 is 3x^2", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (a: number) => a * a * a;
          assertClose(centralDifference(f, [x]), 3 * x * x, 3);
        })
      );
    });

    test("derivative of f(x) = 5x + 3 is 5", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (a: number) => 5 * a + 3;
          assertClose(centralDifference(f, [x]), 5);
        })
      );
    });

    test("derivative of constant f(x) = 7 is 0", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (_a: number) => 7;
          assertClose(centralDifference(f, [x]), 0);
        })
      );
    });

    test("derivative of f(x) = sin(x) is cos(x)", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (a: number) => Math.sin(a);
          assertClose(centralDifference(f, [x]), Math.cos(x));
        })
      );
    });

    test("derivative of f(x) = cos(x) is -sin(x)", () => {
      fc.assert(
        fc.property(smallFloat, (x) => {
          const f = (a: number) => Math.cos(a);
          assertClose(centralDifference(f, [x]), -Math.sin(x));
        })
      );
    });

    test("derivative of f(x) = e^x is e^x", () => {
      // Use smaller range to avoid overflow
      const boundedFloat = fc.double({ noNaN: true, min: -10, max: 10 });
      fc.assert(
        fc.property(boundedFloat, (x) => {
          const f = (a: number) => Math.exp(a);
          assertClose(centralDifference(f, [x]), Math.exp(x));
        })
      );
    });

    test("derivative of f(x) = log(x) is 1/x", () => {
      fc.assert(
        fc.property(positiveFloat, (x) => {
          const f = (a: number) => Math.log(a);
          assertClose(centralDifference(f, [x]), 1 / x);
        })
      );
    });

    test("derivative of f(x) = 1/x is -1/x^2", () => {
      fc.assert(
        fc.property(nonZeroFloat, (x) => {
          const f = (a: number) => 1 / a;
          assertClose(centralDifference(f, [x]), -1 / (x * x), 3);
        })
      );
    });

    test("derivative of f(x) = sqrt(x) is 1/(2*sqrt(x))", () => {
      fc.assert(
        fc.property(positiveFloat, (x) => {
          const f = (a: number) => Math.sqrt(a);
          assertClose(centralDifference(f, [x]), 1 / (2 * Math.sqrt(x)));
        })
      );
    });
  });

  describe("multi-variable functions - partial derivatives", () => {
    test("partial derivative of f(x,y) = x + y with respect to x is 1", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, (x, y) => {
          const f = (a: number, b: number) => a + b;
          assertClose(centralDifference(f, [x, y], 0), 1);  // df/dx
        })
      );
    });

    test("partial derivative of f(x,y) = x + y with respect to y is 1", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, (x, y) => {
          const f = (a: number, b: number) => a + b;
          assertClose(centralDifference(f, [x, y], 1), 1);  // df/dy
        })
      );
    });

    test("partial derivative of f(x,y) = x * y with respect to x is y", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, (x, y) => {
          const f = (a: number, b: number) => a * b;
          assertClose(centralDifference(f, [x, y], 0), y);  // df/dx = y
        })
      );
    });

    test("partial derivative of f(x,y) = x * y with respect to y is x", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, (x, y) => {
          const f = (a: number, b: number) => a * b;
          assertClose(centralDifference(f, [x, y], 1), x);  // df/dy = x
        })
      );
    });

    test("partial derivative of f(x,y) = x^2 + y^2 with respect to x is 2x", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, (x, y) => {
          const f = (a: number, b: number) => a * a + b * b;
          assertClose(centralDifference(f, [x, y], 0), 2 * x);  // df/dx = 2x
        })
      );
    });

    test("partial derivative of f(x,y) = x^2 + y^2 with respect to y is 2y", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, (x, y) => {
          const f = (a: number, b: number) => a * a + b * b;
          assertClose(centralDifference(f, [x, y], 1), 2 * y);  // df/dy = 2y
        })
      );
    });

    test("partial derivative of f(x,y,z) = x*y*z with respect to each variable", () => {
      fc.assert(
        fc.property(smallFloat, smallFloat, smallFloat, (x, y, z) => {
          const f = (a: number, b: number, c: number) => a * b * c;
          assertClose(centralDifference(f, [x, y, z], 0), y * z, 3);  // df/dx = y*z
          assertClose(centralDifference(f, [x, y, z], 1), x * z, 3);  // df/dy = x*z
          assertClose(centralDifference(f, [x, y, z], 2), x * y, 3);  // df/dz = x*y
        })
      );
    });
  });

  describe("epsilon parameter", () => {
    test("smaller epsilon gives more accurate results for smooth functions", () => {
      const x = 2;
      const f = (a: number) => a * a;  // derivative is 2x = 4
      const expected = 4;

      const result1 = centralDifference(f, [x], 0, 1e-3);
      const result2 = centralDifference(f, [x], 0, 1e-6);
      const result3 = centralDifference(f, [x], 0, 1e-9);

      // All should be close, but smaller epsilon generally more accurate
      assertClose(result1, expected, 3);
      assertClose(result2, expected, 6);
      assertClose(result3, expected, 6);  // Very small epsilon may have floating point issues
    });
  });
});

// ============================================================
// Context Tests
// ============================================================

describe("Context", () => {
  test("saveForBackward stores values", () => {
    const ctx = new Context();
    ctx.saveForBackward(1, 2, 3);
    expect(ctx.savedValues).toEqual([1, 2, 3]);
  });

  test("savedValues returns empty array initially", () => {
    const ctx = new Context();
    expect(ctx.savedValues).toEqual([]);
  });

  test("saveForBackward overwrites previous values", () => {
    const ctx = new Context();
    ctx.saveForBackward(1, 2);
    ctx.saveForBackward(5, 6, 7);
    expect(ctx.savedValues).toEqual([5, 6, 7]);
  });

  test("saveForBackward handles single value", () => {
    const ctx = new Context();
    ctx.saveForBackward(42);
    expect(ctx.savedValues).toEqual([42]);
  });

  test("saveForBackward handles no values", () => {
    const ctx = new Context();
    ctx.saveForBackward();
    expect(ctx.savedValues).toEqual([]);
  });
});

// ============================================================
// Autodiff Tests
// ============================================================


export class Function1 extends ScalarFunction {
    static forward(ctx: Context, x: number, y: number): number {
        return operators.add(x, y);
    }

    static backward(ctx: Context, dOut: number): [number, number] {
      return [dOut, dOut];
    }
}

export class Function2 extends ScalarFunction {
    static forward(ctx: Context, x: number, y: number): number {
        ctx.saveForBackward(x, y);
        return x * y + x;
    }

    static backward(ctx: Context, dOut: number): [number, number] {
      const [x, y] = ctx.savedValues;
      return [dOut * (y + 1), dOut * x];
    }
}

// ============================================================
// Chain Rule Tests
// ============================================================

describe("Chain rule", () => {
  test("length of scalar:gradient array is 2", () => {
    const x = new Scalar(0.0);
    const constant = new Scalar(0.0, new ScalarHistory(Function1, new Context(), [x,x]));
    const back = constant.chainRule(5);

    expect(Array.from(back).length).toEqual(2);
  })

  test("derivative correctness", () => {
    const v = new Scalar(0.0, new ScalarHistory());
    const constant = new Scalar(0.0, new ScalarHistory(Function1, new Context(), [v,v]));
    const back = constant.chainRule(5);
    const backArr = Array.from(back);

    expect(backArr.length).toEqual(2);

    const [_, deriv] = backArr[0]!;
    expect(deriv).toEqual(5);
  })

  test("constants ignored, variables get derivatives", () => {
    const v = new Scalar(5);
    const constant = 10;
    const y = Scalar.apply(Function2, constant, v);
    
    const back = y.chainRule(5);
    const backArr = Array.from(back);

    expect(backArr.length).toEqual(2);

    const [variable, deriv] = backArr[1]!;
    expect(variable.name).toEqual(v.name);
    expect(deriv).toEqual(5 * 10);
  })

  test("rule3: constants ignored, variables get derivatives", () => {
    const constant = 10;
    const v = new Scalar(5);

    const y = Scalar.apply(Function2, constant, v);

    const back = y.chainRule(5);
    const backArr = Array.from(back);

    expect(backArr.length).toEqual(2);

    // second input is the variable `v`
    const [variable, deriv] = backArr[1]!;
    expect(variable.name).toEqual(v.name);
    expect(deriv).toEqual(5 * 10);
  });

  test("rule4: two variables get correct derivatives", () => {
    const v1 = new Scalar(5);
    const v2 = new Scalar(10);

    const y = Scalar.apply(Function2, v1, v2);

    const back = y.chainRule(5);
    const backArr = Array.from(back);

    expect(backArr.length).toEqual(2);

    // grad wrt v1 (x) is dOut * (y + 1)
    let [variable, deriv] = backArr[0]!;
    expect(variable.name).toEqual(v1.name);
    expect(deriv).toEqual(5 * (10 + 1));

    // grad wrt v2 (y) is dOut * x
    [variable, deriv] = backArr[1]!;
    expect(variable.name).toEqual(v2.name);
    expect(deriv).toEqual(5 * 5);
  });

})

// ============================================================
// Topological Sort Tests (task1_4)
// ============================================================

describe("topologicalSort", () => {
  test("single leaf node returns just that node", () => {
    const x = new Scalar(5);
    const sorted = topologicalSort(x);
    
    expect(sorted.length).toEqual(1);
    expect(sorted[0]).toBe(x);
  });

  test("simple operation returns output before inputs", () => {
    const x = new Scalar(3);
    const y = new Scalar(4);
    const z = x.mul(y);
    
    const sorted = topologicalSort(z);
    
    // z should come before x and y
    expect(sorted.length).toEqual(3);
    expect(sorted[0]).toBe(z);
    // x and y should both be in the list after z
    expect(sorted.slice(1)).toContain(x);
    expect(sorted.slice(1)).toContain(y);
  });

  test("chain of operations has correct order", () => {
    const x = new Scalar(2);
    const y = x.mul(3);      // y = x * 3
    const z = y.add(1);      // z = y + 1
    
    const sorted = topologicalSort(z);
    
    // z should come first, then y, then leaves
    const zIdx = sorted.indexOf(z);
    const yIdx = sorted.indexOf(y);
    const xIdx = sorted.indexOf(x);
    
    expect(zIdx).toBeLessThan(yIdx);
    expect(yIdx).toBeLessThan(xIdx);
  });

  test("variable used multiple times appears only once", () => {
    const x = new Scalar(3);
    const z = x.mul(x);  // x * x
    
    const sorted = topologicalSort(z);
    
    // x should appear exactly once
    const xCount = sorted.filter(s => s === x).length;
    expect(xCount).toEqual(1);
    expect(sorted.length).toEqual(2);  // z and x
  });

  test("diamond dependency graph", () => {
    const x = new Scalar(2);
    const a = x.mul(3);      // a = x * 3
    const b = x.add(1);      // b = x + 1
    const c = a.mul(b);      // c = a * b
    
    const sorted = topologicalSort(c);
    
    // c should come first
    expect(sorted[0]).toBe(c);
    
    // a and b should come before x
    const aIdx = sorted.indexOf(a);
    const bIdx = sorted.indexOf(b);
    const xIdx = sorted.indexOf(x);
    
    expect(aIdx).toBeLessThan(xIdx);
    expect(bIdx).toBeLessThan(xIdx);
  });
});

// ============================================================
// Backpropagate Tests (task1_4)
// ============================================================

describe("backPropagate", () => {
  test("simple multiplication: z = x * y", () => {
    const x = new Scalar(3);
    const y = new Scalar(4);
    const z = x.mul(y);
    
    backPropagate(z, 1.0);
    
    // dz/dx = y = 4
    expect(x.derivative).toBeCloseTo(4, 5);
    // dz/dy = x = 3
    expect(y.derivative).toBeCloseTo(3, 5);
  });

  test("simple addition: z = x + y", () => {
    const x = new Scalar(3);
    const y = new Scalar(4);
    const z = x.add(y);
    
    backPropagate(z, 1.0);
    
    // dz/dx = 1
    expect(x.derivative).toBeCloseTo(1, 5);
    // dz/dy = 1
    expect(y.derivative).toBeCloseTo(1, 5);
  });

  test("variable used twice: z = x * x (should accumulate)", () => {
    const x = new Scalar(3);
    const z = x.mul(x);
    
    backPropagate(z, 1.0);
    
    // d(x^2)/dx = 2x = 6
    expect(x.derivative).toBeCloseTo(6, 5);
  });

  test("chain: z = (x * y) + x", () => {
    const x = new Scalar(3);
    const y = new Scalar(4);
    const z = x.mul(y).add(x);
    
    backPropagate(z, 1.0);
    
    // z = x*y + x, dz/dx = y + 1 = 5
    expect(x.derivative).toBeCloseTo(5, 5);
    // dz/dy = x = 3
    expect(y.derivative).toBeCloseTo(3, 5);
  });

  test("log function: z = log(x)", () => {
    const x = new Scalar(2);
    const z = x.log();
    
    backPropagate(z, 1.0);
    
    // d(log(x))/dx = 1/x = 0.5
    expect(x.derivative).toBeCloseTo(0.5, 5);
  });

  test("exp function: z = exp(x)", () => {
    const x = new Scalar(1);
    const z = x.exp();
    
    backPropagate(z, 1.0);
    
    // d(e^x)/dx = e^x = e
    expect(x.derivative).toBeCloseTo(Math.E, 5);
  });

  test("sigmoid function: z = sigmoid(x)", () => {
    const x = new Scalar(0);
    const z = x.sigmoid();
    
    backPropagate(z, 1.0);
    
    // sigmoid(0) = 0.5, d(sigmoid)/dx = sigmoid * (1 - sigmoid) = 0.25
    expect(x.derivative).toBeCloseTo(0.25, 5);
  });

  test("relu function: z = relu(x) for positive x", () => {
    const x = new Scalar(5);
    const z = x.relu();
    
    backPropagate(z, 1.0);
    
    // d(relu(x))/dx = 1 for x > 0
    expect(x.derivative).toBeCloseTo(1, 5);
  });

  test("relu function: z = relu(x) for negative x", () => {
    const x = new Scalar(-5);
    const z = x.relu();
    
    backPropagate(z, 1.0);
    
    // d(relu(x))/dx = 0 for x < 0
    expect(x.derivative).toBeCloseTo(0, 5);
  });

  test("negation: z = -x", () => {
    const x = new Scalar(3);
    const z = x.neg();
    
    backPropagate(z, 1.0);
    
    // d(-x)/dx = -1
    expect(x.derivative).toBeCloseTo(-1, 5);
  });

  test("division: z = x / y", () => {
    const x = new Scalar(6);
    const y = new Scalar(2);
    const z = x.div(y);
    
    backPropagate(z, 1.0);
    
    // d(x/y)/dx = 1/y = 0.5
    expect(x.derivative).toBeCloseTo(0.5, 5);
    // d(x/y)/dy = -x/y^2 = -6/4 = -1.5
    expect(y.derivative).toBeCloseTo(-1.5, 5);
  });

  test("complex expression matches numerical gradient", () => {
    // f(x, y) = x * y + x^2
    const f = (xVal: number, yVal: number) => {
      const x = new Scalar(xVal);
      const y = new Scalar(yVal);
      return x.mul(y).add(x.mul(x));
    };
    
    const xVal = 3;
    const yVal = 4;
    
    // Compute analytical gradients
    const x = new Scalar(xVal);
    const y = new Scalar(yVal);
    const z = x.mul(y).add(x.mul(x));
    backPropagate(z, 1.0);
    
    // Compute numerical gradients
    const numericFn = (a: number, b: number) => a * b + a * a;
    const numDx = centralDifference(numericFn, [xVal, yVal], 0);
    const numDy = centralDifference(numericFn, [xVal, yVal], 1);
    
    expect(x.derivative).toBeCloseTo(numDx, 4);
    expect(y.derivative).toBeCloseTo(numDy, 4);
  });

  test("propagates non-1.0 derivative correctly", () => {
    const x = new Scalar(3);
    const y = new Scalar(4);
    const z = x.mul(y);
    
    backPropagate(z, 2.0);  // Start with derivative of 2
    
    // dz/dx = y * 2 = 8
    expect(x.derivative).toBeCloseTo(8, 5);
    // dz/dy = x * 2 = 6
    expect(y.derivative).toBeCloseTo(6, 5);
  });
});

// ============================================================
// Gradient Comparison with Central Difference (task1_4)
// ============================================================

describe("Gradient correctness via central difference", () => {
  test.prop([positiveFloat])("log gradient matches numerical", (xVal) => {
    const x = new Scalar(xVal);
    const z = x.log();
    backPropagate(z, 1.0);
    
    const numerical = centralDifference((a) => Math.log(a), [xVal], 0);
    assertClose(x.derivative!, numerical, 4);
  });

  test.prop([fc.double({ noNaN: true, min: -5, max: 5 })])("exp gradient matches numerical", (xVal) => {
    const x = new Scalar(xVal);
    const z = x.exp();
    backPropagate(z, 1.0);
    
    const numerical = centralDifference((a) => Math.exp(a), [xVal], 0);
    assertClose(x.derivative!, numerical, 4);
  });

  test.prop([smallFloat])("sigmoid gradient matches numerical", (xVal) => {
    const sigmoid = (a: number) => 1 / (1 + Math.exp(-a));
    
    const x = new Scalar(xVal);
    const z = x.sigmoid();
    backPropagate(z, 1.0);
    
    const numerical = centralDifference(sigmoid, [xVal], 0);
    assertClose(x.derivative!, numerical, 4);
  });

  test.prop([smallFloat, smallFloat])("mul gradient matches numerical", (xVal, yVal) => {
    const x = new Scalar(xVal);
    const y = new Scalar(yVal);
    const z = x.mul(y);
    backPropagate(z, 1.0);
    
    const fn = (a: number, b: number) => a * b;
    assertClose(x.derivative!, centralDifference(fn, [xVal, yVal], 0), 4);
    assertClose(y.derivative!, centralDifference(fn, [xVal, yVal], 1), 4);
  });

  test.prop([smallFloat, smallFloat])("add gradient matches numerical", (xVal, yVal) => {
    const x = new Scalar(xVal);
    const y = new Scalar(yVal);
    const z = x.add(y);
    backPropagate(z, 1.0);
    
    const fn = (a: number, b: number) => a + b;
    assertClose(x.derivative!, centralDifference(fn, [xVal, yVal], 0), 4);
    assertClose(y.derivative!, centralDifference(fn, [xVal, yVal], 1), 4);
  });
});
