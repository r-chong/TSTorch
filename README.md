# What is TSTorch?

TSTorch is a WebGPU-first runtime for predictable local inference.

It is a tool to execute models directly on the user’s device and expose what actually happens during execution: scheduling, memory allocation, graph capture, and GPU dispatch behavior. The goal is to make model behavior observable and predictable under real hardware constraints such as latency, memory pressure, and power limits.

Instead of treating execution as an internal implementation detail, TSTorch treats it as part of the public API.

# What problem this solves

Many modern AI systems fail not because the model is wrong, but because runtime behavior becomes unpredictable on constrained devices.

TSTorch allows you to run a real model locally and see what the runtime actually did on that specific device:

* How many GPU dispatches occurred
* What memory peaks were reached
* What operations were fused
* Where latency actually came from
* Whether the model is viable on-device

This makes TSTorch closer to an execution microscope than a traditional framework.

The intended workflow:
1. Import a trained or fine-tuned model made using a normal framework (PyTorch, etc.) via ONNX
2. Run model locally with TSTorch
3. Inspect latency, memory usage, and execution behavior
4. Decide whether edge deployment is viable

# Steps to use

important: run `pnpm --filter tstorch` before your npm install and run commands

# Run demo
1. Install everything `pnpm install`
2. Compile TSTorch library `pnpm --filter tstorch run build` 
3. Add TSTorch library to demo `pnpm add tstorch --filter demo`
4. Run `pnpm --filter demo run start`
