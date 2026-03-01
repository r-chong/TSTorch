import { fastMatrixMultiply } from './fast_ops.js';
import { TensorData, shapeProduct } from './tensor_data.js';
import { Tensor } from './tensor.js';

const input1 = new Tensor(new TensorData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]));
const input2 = new Tensor(new TensorData(new Float64Array([3, 1, 2, 4, 7, 6]), [2, 4]));


test("1", () => {
    expect(fastMatrixMultiply(input1, input2)).toEqual(1)
})
