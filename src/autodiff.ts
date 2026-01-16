import { Scalar } from "./scalar.js";

export function centralDifference(
    f: (...args: number[]) => number,
    vals: number[],
    arg: number = 0,
    epsilon: number = 1e-6
): number {
    const valsPlus = [...vals];
    valsPlus[arg] = valsPlus[arg]! + epsilon;
    
    const valsMinus = [...vals];
    valsMinus[arg] = valsMinus[arg]! - epsilon;

    return (f(...valsPlus) - f(...valsMinus)) / (2 * epsilon);
}

export class Context {
    private _savedValues: number[] = [];

    saveForBackward( ...values: number[]): void {
        this._savedValues = values;
    }

    get savedValues(): number[] {
        return this._savedValues;
    }
}

export function topologicalSort(scalar: Scalar): Scalar[] {
    const visited = new Set<Scalar>();
    const sorted = new Array<Scalar>();

    const dfs: (scalar: Scalar) => void = (scalar) => {
        if (visited.has(scalar)) return;
        visited.add(scalar);
        for (const parent of scalar.parents) {
            dfs(parent);
        }
        sorted.push(scalar);
    };
    dfs(scalar);
    return sorted.reverse();
}

export function backPropagate(scalar: Scalar, dOut: number): void {
    const sorted = topologicalSort(scalar);
    const derivatives: Map<Scalar, number> = new Map();

    derivatives.set(scalar, dOut);

    for (const node of sorted) {
        const d = derivatives.get(node);
        if (d === undefined) continue;

        if (node.isLeaf()) {
            node.accumulateDerivative(d);
        } else {
            for (const [parent, grad] of node.chainRule(d)) {
                derivatives.set(parent, (derivatives.get(parent) ?? 0) + grad);
            }
        }
    }
}