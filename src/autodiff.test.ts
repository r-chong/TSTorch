import { test, fc } from '@fast-check/jest';
import { centralDifference, Context } from "./autodiff.js";

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
