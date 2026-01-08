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


