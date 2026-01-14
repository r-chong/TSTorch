function centralDifference(
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