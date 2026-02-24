import { runDispatchParity } from "./dispatch_parity.js";
import runScalar from "./runScalar.js";

runScalar();
const result = runDispatchParity();
console.log(result);