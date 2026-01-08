TSTorch is a TypeScript implementation of PyTorch, intended as a working library and educational resource.

TSTorch exposes the core execution mechanisms behind modern deep learning systems: autograd, graph capture, kernel fusion, and GPU memory planning, using WebGPU as the primary execution target.

> Why not just use Tensorflow.js?

We want to:
- Be WebGPU-first by design
- Expose autograd and graph execution internals
- Support graph capture and compiler-style optimizations
- Make kernel fusion and memory planning observable

In short, TensorFlow.js is a product.
TSTorch is a systems-level exploration of how such products are built.

# Who is this for?

Engineers interested in ML infrastructure and compilers

Developers building ML-powered web applications who want deeper control

Students learning how frameworks like PyTorch and Torch 2 actually work under the hood