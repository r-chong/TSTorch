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