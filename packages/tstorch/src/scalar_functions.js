"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.EQ = exports.LT = exports.Exp = exports.Relu = exports.Sigmoid = exports.Neg = exports.Inv = exports.Mul = exports.Log = exports.Add = exports.ScalarFunction = exports.ScalarHistory = void 0;
var operators = require("./operators.js");
/**
 * ScalarHistory stores how a Scalar was created.
 * Note: inputs is typed as any[] to avoid circular dependency with Scalar
 */
var ScalarHistory = /** @class */ (function () {
    function ScalarHistory(lastFn, ctx, inputs // Will be Scalar[] at runtime
    ) {
        if (lastFn === void 0) { lastFn = null; }
        if (ctx === void 0) { ctx = null; }
        if (inputs === void 0) { inputs = []; }
        this.lastFn = lastFn;
        this.ctx = ctx;
        this.inputs = inputs;
    }
    return ScalarHistory;
}());
exports.ScalarHistory = ScalarHistory;
/**
 * Base class for all scalar operations.
 * Each operation implements forward (and later, backward).
 *
 * Note: The apply() logic lives in Scalar class to avoid circular dependencies.
 */
var ScalarFunction = /** @class */ (function () {
    function ScalarFunction() {
    }
    ScalarFunction.forward = function (ctx) {
        var inputs = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            inputs[_i - 1] = arguments[_i];
        }
        throw new Error("forward not implemented");
    };
    ScalarFunction.backward = function (ctx, dOut) {
        throw new Error("backward not implemented");
    };
    return ScalarFunction;
}());
exports.ScalarFunction = ScalarFunction;
var Add = /** @class */ (function (_super) {
    __extends(Add, _super);
    function Add() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Add.forward = function (ctx, a, b) {
        // Don't need to save for backward since df/da = 1 and df/db = 1
        return operators.add(a, b);
    };
    Add.backward = function (ctx, dOut) {
        return [dOut, dOut];
    };
    return Add;
}(ScalarFunction));
exports.Add = Add;
var Log = /** @class */ (function (_super) {
    __extends(Log, _super);
    function Log() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Log.forward = function (ctx, a) {
        ctx.saveForBackward(a);
        return operators.log(a);
    };
    Log.backward = function (ctx, dOut) {
        var a = ctx.savedValues[0];
        return [dOut * (1 / a)];
    };
    return Log;
}(ScalarFunction));
exports.Log = Log;
var Mul = /** @class */ (function (_super) {
    __extends(Mul, _super);
    function Mul() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Mul.forward = function (ctx, a, b) {
        ctx.saveForBackward(a, b);
        return operators.mul(a, b);
    };
    Mul.backward = function (ctx, dOut) {
        var _a = ctx.savedValues, a = _a[0], b = _a[1];
        return [dOut * b, dOut * a];
    };
    return Mul;
}(ScalarFunction));
exports.Mul = Mul;
var Inv = /** @class */ (function (_super) {
    __extends(Inv, _super);
    function Inv() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Inv.forward = function (ctx, a) {
        ctx.saveForBackward(a);
        return operators.inv(a);
    };
    Inv.backward = function (ctx, dOut) {
        var a = ctx.savedValues[0];
        return [dOut * (-1 / Math.pow(a, 2))];
    };
    return Inv;
}(ScalarFunction));
exports.Inv = Inv;
var Neg = /** @class */ (function (_super) {
    __extends(Neg, _super);
    function Neg() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Neg.forward = function (ctx, a) {
        return operators.neg(a);
    };
    Neg.backward = function (ctx, dOut) {
        return [dOut * (-1)];
    };
    return Neg;
}(ScalarFunction));
exports.Neg = Neg;
var Sigmoid = /** @class */ (function (_super) {
    __extends(Sigmoid, _super);
    function Sigmoid() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Sigmoid.forward = function (ctx, a) {
        var result = operators.sigmoid(a);
        ctx.saveForBackward(result);
        return result;
    };
    Sigmoid.backward = function (ctx, dOut) {
        var result = ctx.savedValues[0];
        return [dOut * result * (1 - result)];
    };
    return Sigmoid;
}(ScalarFunction));
exports.Sigmoid = Sigmoid;
var Relu = /** @class */ (function (_super) {
    __extends(Relu, _super);
    function Relu() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Relu.forward = function (ctx, a) {
        ctx.saveForBackward(a);
        return operators.relu(a);
    };
    Relu.backward = function (ctx, dOut) {
        var a = ctx.savedValues[0];
        return [dOut * (a > 0 ? 1 : 0)];
    };
    return Relu;
}(ScalarFunction));
exports.Relu = Relu;
var Exp = /** @class */ (function (_super) {
    __extends(Exp, _super);
    function Exp() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Exp.forward = function (ctx, a) {
        var result = operators.exp(a);
        ctx.saveForBackward(result);
        return result;
    };
    Exp.backward = function (ctx, dOut) {
        var result = ctx.savedValues[0];
        return [dOut * result];
    };
    return Exp;
}(ScalarFunction));
exports.Exp = Exp;
var LT = /** @class */ (function (_super) {
    __extends(LT, _super);
    function LT() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    LT.forward = function (ctx, a, b) {
        return operators.lt(a, b);
    };
    LT.backward = function (ctx, dOut) {
        return [0, 0];
    };
    return LT;
}(ScalarFunction));
exports.LT = LT;
var EQ = /** @class */ (function (_super) {
    __extends(EQ, _super);
    function EQ() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    EQ.forward = function (ctx, a, b) {
        return operators.eq(a, b);
    };
    EQ.backward = function (ctx, dOut) {
        return [0, 0];
    };
    return EQ;
}(ScalarFunction));
exports.EQ = EQ;
