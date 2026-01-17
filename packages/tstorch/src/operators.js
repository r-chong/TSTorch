"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.mul = mul;
exports.id = id;
exports.add = add;
exports.neg = neg;
exports.lt = lt;
exports.eq = eq;
exports.max = max;
exports.isClose = isClose;
exports.sigmoid = sigmoid;
exports.relu = relu;
exports.log = log;
exports.exp = exp;
exports.logBack = logBack;
exports.inv = inv;
exports.invBack = invBack;
exports.reluBack = reluBack;
exports.map = map;
exports.negList = negList;
exports.zipWith = zipWith;
exports.zip = zip;
exports.addLists = addLists;
exports.reduce = reduce;
exports.sum = sum;
exports.prod = prod;
function mul(x, y) {
    return x * y;
}
function id(x) {
    return x;
}
function add(x, y) {
    return x + y;
}
function neg(x) {
    return -x;
}
function lt(x, y) {
    return x < y ? 1 : 0;
}
function eq(x, y) {
    return x === y ? 1 : 0;
}
function max(x, y) {
    return x > y ? x : y;
}
function isClose(x, y) {
    // "$f(x) = |x - y| < 1e-2$"
    // assumed that this meant, return 1.0 if true else 0.0
    return Math.abs(x - y) < 1e-2 ? 1 : 0;
}
function sigmoid(x) {
    if (x >= 0) {
        return (1 / (1 + Math.exp(-x)));
    }
    else {
        return (Math.exp(x) / (1 + Math.exp(x)));
    }
}
function relu(x) {
    return x > 0 ? x : 0;
}
var EPS = 1e-6;
function log(x) {
    return Math.log(x);
}
function exp(x) {
    return Math.exp(x);
}
function logBack(x, d) {
    // since Math.log is the natural log, derivative is 1/x
    return d * (1 / x);
}
function inv(x) {
    // guard this?
    return 1 / x;
}
function invBack(x, d) {
    return -d / (x * x);
}
function reluBack(x, d) {
    // Although relu_back is d * relu(x), JavaScript differentiates -0 and 0, so we instead do:
    return x > 0 ? d : 0;
}
function map(fn) {
    return function (ls) { return ls.map(function (num) { return fn(num); }); };
}
function negList(ls) {
    return map(neg)(ls);
}
function zipWith(fn) {
    return function (ls1, ls2) { return ls1.map(function (num, idx) { return fn(ls1[idx], ls2[idx]); }); };
}
// generic zipping function
function zip(arr1, arr2) {
    return arr1.map(function (x, i) { return [x, arr2[i]]; });
}
function addLists(ls1, ls2) {
    return zipWith(add)(ls1, ls2);
}
function reduce(fn, start) {
    return function (ls) {
        var res = start;
        ls.forEach(function (num, idx) { return res = fn(num, res); });
        return res;
    };
}
function sum(ls) {
    return reduce(add, 0)(ls);
}
function prod(ls) {
    return reduce(mul, 1)(ls);
}
