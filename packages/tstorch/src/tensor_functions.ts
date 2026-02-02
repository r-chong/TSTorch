import type {
    Shape
} from './tensor_data.js'

import {
    TensorData,
    shapeProduct,
    shapeBroadcast,
    strides,
} from './tensor_data.js';

function zeros(shape: Shape): TensorData {
    return TensorData.zeros(shape);
}

export function neg(a: TensorData): TensorData {
    
}