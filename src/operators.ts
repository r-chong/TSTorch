function mul(x: number, y: number):number {
    return x * y;
}

function id(x: number):number {
    return x;
}

function add(x: number, y: number):number {
    return x + y;
}

function neg(x: number):number {
    return -x;
}

function lt(x: number, y: number):number {
    return x < y? 1 : 0;
}

function eq(x: number, y: number):number {
    return x === y ? 1 : 0;
}

function max(x: number, y: number):number {
    return x > y ? x : y;
}

function isClose(x: number, y:number):number {
    // "$f(x) = |x - y| < 1e-2$"
    // assumed that this meant, return 1.0 if true else 0.0
    return Math.abs(x - y) < 1e-2 ? 1 : 0;
}

function sigmoid(x: number):number {
    if (x >= 0) {
        return (1 / (1 + Math.exp(-x)));
    } else {
        return (Math.exp(x) / (1 + Math.exp(-x)));
    }
}

function relu(x: number):number {
    return x > 0 ? x : 0;
}

const EPS = 1e-6

function log(x: number):number {
    // engineering choice - never return ln(0) which is undefined
    return Math.log(x + EPS); 
}

function exp(x: number):number {
    return Math.exp(x);
}

function logBack(x: number, d: number):number {
    // r"If $f = log$ as above, compute $d \times f'(x)$"
    // What does this mean?
}

function inv(x: number):number {
    // guard this?
    return 1 / x;
}

function invBack(x: number, d: number):number {
    // r"If $f(x) = 1/x$ compute $d \times f'(x)$"
}

function reluBack(x: number, d: number):number {
    // r"If $f = relu$ compute $d \times f'(x)$"
}

type MapFunction = (ls: number[]) => number[];


function map(fn: (num: number) => number):MapFunction {
    return (ls: number[])=>ls.map(num => fn(num));
}

function negList(ls: number[]): number[] {
    return map(neg)(ls);
}

type ZipWithFunction = (ls1: number[], ls2: number[]) => number[];

function zipWith(fn: (num1: number, num2: number) => number):ZipWithFunction {
    return (ls1: number[], ls2: number[]) => ls1.map((num, idx) => fn(ls1[idx]!, ls2[idx]!));
}

function addLists(ls1: number[], ls2: number[]): number[] {
    return zipWith(add)(ls1, ls2);
}

type ReduceType = (ls: number[]) => number;

function reduce(fn: (num1: number, num2: number) => number, start: number):ReduceType {
    return (ls:number[]) => {
        let res: number = start;

        ls.forEach((num, idx) => res = fn(num, res));
        return res;
    }
}

function sum(ls: number[]):number {
    return reduce(add, 0)(ls);
}

function prod(ls: number[]):number {
    return reduce(mul, 1)(ls);
}