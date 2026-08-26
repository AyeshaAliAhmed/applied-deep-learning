# TensorFlow Playground: Checkerboard Experiment

A short exploration of neural network basics using the TensorFlow Playground.
The task: classify a checkerboard pattern using only x1 and x2 as inputs.

## What I tried

Tested different network sizes with ReLU activation to find the smallest
architecture that actually works. Spoiler: you need at least 2 hidden layers.
One layer isn't enough because the checkerboard is an XOR-type problem.
The class depends on the *joint sign* of x1 and x2, not either one alone.

**Minimum working setup:** 2 hidden layers, 4 neurons each.

## Interesting things I noticed

Running the same settings multiple times gives different results every
time: different loss curves, different decision boundaries, occasionally a
complete failure. This is because weights are randomly initialised, so
gradient descent starts from a different spot on the loss surface each run.
It's not a bug, it's just how non-convex optimisation works.

## Feature engineering

Adding x1·x2 as an input feature makes the problem almost trivially easy.
Even a single neuron can solve it. That product directly encodes which
quadrant a point is in, which is exactly what the checkerboard boundary is.

Good feature design > bigger model.

## Tools

- [TensorFlow Playground](https://playground.tensorflow.org)
