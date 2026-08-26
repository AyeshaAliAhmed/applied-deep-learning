# Wine Quality Prediction: Neural Network from Scratch

## About the Project
This project is about predicting the quality of white wine (a score from 1 to 10)
based on 11 physical measurements like acidity, sugar, and alcohol content.

The goal was to build a neural network completely from scratch using only NumPy,
no PyTorch, no TensorFlow. Every single step: forward pass, backpropagation,
and gradient descent was implemented manually.

This was part of my Applied Deep Learning course (Exercise 5.1).

## Dataset
The [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
contains 4898 white wines with 11 features each:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

**Target:** Quality score (integer, rated by human experts)

Split: 3000 wines for training, ~1898 for testing.

## Network Architecture
Input (11 features) -> Hidden Layer (100 nodes, ReLU) -> Output (1 quality score)
x W, b Wp, bp

- **W** shape: (100, 11), weights from input to hidden layer
- **b** shape: (100,), biases for hidden layer
- **Wp** shape: (100,), weights from hidden to output layer
- **bp** shape: (1,), single output bias

## What I Learned
- Preprocessing: why normalizing input data matters (my loss went from 1.4 million to 0.55 just by adding normalization!)
- He initialization: how to initialize weights properly for ReLU networks
- Forward pass: computing predictions step by step: `z = W·x + b -> h = ReLU(z) -> ŷ = Wp·h + bp`
- Backpropagation: deriving and implementing gradients manually using the chain rule
- Stochastic Gradient Descent (SGD): updating weights one example at a time
- Learning rate sensitivity: how too large a learning rate causes gradient explosion
- Evaluation: interpreting MSE loss and correlation coefficient

## Results

| Metric | Value |
|--------|-------|
| Best test loss (MSE) | 0.55 |
| Correlation coefficient | 0.55 |

The network predicts wine quality within ~0.74 points on average.
Wines that are genuinely better quality tend to get higher predicted scores,
which means the network learned something real, not just memorized the data.

## Key Challenges
- **Gradient explosion:** increasing hidden nodes without reducing the learning rate
  caused the loss to blow up to 10 trillion. Fixed by reducing `learning_rate` from
  `0.001` to `0.0001`.
- **Forgetting to normalize:** raw wine features have very different scales
  (e.g. density is 0.998 vs total sulfur dioxide is 114). Without normalization,
  training was completely unstable.

## Libraries Used
- Python
- NumPy
- Matplotlib

## Possible Improvements
- Use Adam optimizer instead of plain SGD
- Add more hidden layers
- Train for more epochs (50 to 100)
- Try PyTorch/TensorFlow for comparison
