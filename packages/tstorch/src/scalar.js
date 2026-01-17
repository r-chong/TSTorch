"use strict";
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ScalarHistory = exports.Scalar = void 0;
var autodiff_js_1 = require("./autodiff.js");
var operators_js_1 = require("./operators.js");
var scalar_functions_js_1 = require("./scalar_functions.js");
Object.defineProperty(exports, "ScalarHistory", { enumerable: true, get: function () { return scalar_functions_js_1.ScalarHistory; } });
var _varCount = 0;
/**
 * Scalar: A number that tracks its computation history.
 * Behaves like a regular number but records operations for autodiff.
 */
var Scalar = /** @class */ (function () {
    function Scalar(value, history, name) {
        if (history === void 0) { history = null; }
        this.derivative = null; // Filled in during backward pass
        _varCount++;
        this.uniqueId = _varCount;
        this.data = value;
        this.history = history;
        this.name = name !== null && name !== void 0 ? name : "var".concat(this.uniqueId);
    }
    Scalar.prototype.toString = function () {
        return "Scalar(".concat(this.data, ")");
    };
    /**
     * Apply a ScalarFunction to the given values.
     * Handles unwrapping Scalars to numbers, calling forward, and wrapping the result.
     */
    Scalar.apply = function (fn) {
        var vals = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            vals[_i - 1] = arguments[_i];
        }
        var rawVals = [];
        var scalars = [];
        for (var _a = 0, vals_1 = vals; _a < vals_1.length; _a++) {
            var v = vals_1[_a];
            if (v instanceof Scalar) {
                scalars.push(v);
                rawVals.push(v.data);
            }
            else {
                scalars.push(new Scalar(v));
                rawVals.push(v);
            }
        }
        var ctx = new autodiff_js_1.Context();
        var result = fn.forward.apply(fn, __spreadArray([ctx], rawVals, false));
        var history = new scalar_functions_js_1.ScalarHistory(fn, ctx, scalars);
        return new Scalar(result, history);
    };
    Scalar.prototype.add = function (b) {
        return Scalar.apply(scalar_functions_js_1.Add, this, b);
    };
    Scalar.prototype.mul = function (b) {
        return Scalar.apply(scalar_functions_js_1.Mul, this, b);
    };
    Scalar.prototype.div = function (b) {
        return Scalar.apply(scalar_functions_js_1.Mul, this, Scalar.apply(scalar_functions_js_1.Inv, b));
    };
    Scalar.prototype.rdiv = function (b) {
        return Scalar.apply(scalar_functions_js_1.Mul, b, Scalar.apply(scalar_functions_js_1.Inv, this));
    };
    Scalar.prototype.sub = function (b) {
        return Scalar.apply(scalar_functions_js_1.Add, this, Scalar.apply(scalar_functions_js_1.Neg, b));
    };
    Scalar.prototype.neg = function () {
        return Scalar.apply(scalar_functions_js_1.Neg, this);
    };
    Scalar.prototype.lt = function (b) {
        return Scalar.apply(scalar_functions_js_1.LT, this, b);
    };
    Scalar.prototype.eq = function (b) {
        return Scalar.apply(scalar_functions_js_1.EQ, this, b);
    };
    Scalar.prototype.gt = function (b) {
        return Scalar.apply(scalar_functions_js_1.LT, b, this);
    };
    Scalar.prototype.log = function () {
        return Scalar.apply(scalar_functions_js_1.Log, this);
    };
    Scalar.prototype.exp = function () {
        return Scalar.apply(scalar_functions_js_1.Exp, this);
    };
    Scalar.prototype.sigmoid = function () {
        return Scalar.apply(scalar_functions_js_1.Sigmoid, this);
    };
    Scalar.prototype.relu = function () {
        return Scalar.apply(scalar_functions_js_1.Relu, this);
    };
    Scalar.prototype.chainRule = function (dOut) {
        var h = this.history;
        if (!h)
            throw new Error("Missing scalar history");
        if (!h.lastFn)
            throw new Error("Missing lastFn in scalar history");
        if (!h.ctx)
            throw new Error("Missing ctx in scalar history");
        if (!h.inputs)
            throw new Error("Missing inputs in scalar history");
        // @ts-ignore as 1.4 not implemented yet
        var gradients = h.lastFn.backward(h.ctx, dOut);
        var inputs = h.inputs;
        return (0, operators_js_1.zip)(inputs, gradients);
    };
    Scalar.prototype.isLeaf = function () {
        var _a;
        return !((_a = this.history) === null || _a === void 0 ? void 0 : _a.lastFn);
    };
    Scalar.prototype.isConstant = function () {
        return !this.history;
    };
    Object.defineProperty(Scalar.prototype, "parents", {
        get: function () {
            var _a, _b;
            return (_b = (_a = this.history) === null || _a === void 0 ? void 0 : _a.inputs) !== null && _b !== void 0 ? _b : [];
        },
        enumerable: false,
        configurable: true
    });
    Scalar.prototype.accumulateDerivative = function (d) {
        if (!this.isLeaf()) {
            throw new Error("Cannot accumulate derivative of a non-leaf scalar");
        }
        if (this.derivative === null) {
            this.derivative = 0;
        }
        this.derivative += d;
    };
    return Scalar;
}());
exports.Scalar = Scalar;
