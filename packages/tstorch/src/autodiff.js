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
exports.Context = void 0;
exports.centralDifference = centralDifference;
exports.topologicalSort = topologicalSort;
exports.backPropagate = backPropagate;
function centralDifference(f, vals, arg, epsilon) {
    if (arg === void 0) { arg = 0; }
    if (epsilon === void 0) { epsilon = 1e-6; }
    var valsPlus = __spreadArray([], vals, true);
    valsPlus[arg] = valsPlus[arg] + epsilon;
    var valsMinus = __spreadArray([], vals, true);
    valsMinus[arg] = valsMinus[arg] - epsilon;
    return (f.apply(void 0, valsPlus) - f.apply(void 0, valsMinus)) / (2 * epsilon);
}
var Context = /** @class */ (function () {
    function Context() {
        this._savedValues = [];
    }
    Context.prototype.saveForBackward = function () {
        var values = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            values[_i] = arguments[_i];
        }
        this._savedValues = values;
    };
    Object.defineProperty(Context.prototype, "savedValues", {
        get: function () {
            return this._savedValues;
        },
        enumerable: false,
        configurable: true
    });
    return Context;
}());
exports.Context = Context;
function topologicalSort(scalar) {
    var visited = new Set();
    var sorted = new Array();
    var dfs = function (scalar) {
        if (visited.has(scalar))
            return;
        visited.add(scalar);
        for (var _i = 0, _a = scalar.parents; _i < _a.length; _i++) {
            var parent_1 = _a[_i];
            dfs(parent_1);
        }
        sorted.push(scalar);
    };
    dfs(scalar);
    return sorted.reverse();
}
function backPropagate(scalar, dOut) {
    var _a;
    var sorted = topologicalSort(scalar);
    var derivatives = new Map();
    derivatives.set(scalar, dOut);
    for (var _i = 0, sorted_1 = sorted; _i < sorted_1.length; _i++) {
        var node = sorted_1[_i];
        var d = derivatives.get(node);
        if (d === undefined)
            continue;
        if (node.isLeaf()) {
            node.accumulateDerivative(d);
        }
        else {
            for (var _b = 0, _c = node.chainRule(d); _b < _c.length; _b++) {
                var _d = _c[_b], parent_2 = _d[0], grad = _d[1];
                derivatives.set(parent_2, ((_a = derivatives.get(parent_2)) !== null && _a !== void 0 ? _a : 0) + grad);
            }
        }
    }
}
