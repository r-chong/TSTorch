import { test, fc } from '@fast-check/jest';
// import 'jest';
import { mul, add, neg, max, inv } from "./operators.js";

// task 0.1 - basic hypothesis tests
test('All operators return same value as JavaScript version', () => {
    fc.property(fc.float(), fc.float(), (x, y) => {
        expect(mul(x,y)).toBe(x * y);
        expect(add(x,y)).toBe(x + y);
        expect(neg(x)).toBe(-x);
        expect(max(x,y)).toBe(x > y ? x : y);
        if (Math.abs(x) > 1e-5) {
            expect(inv(x)).toBe(1.0 / x);
        }
    })
})