# Neurons, Layers, and the Forward Pass

## The Single Neuron

A neuron does one thing: takes a list of numbers in, does a weighted sum, adds a bias, and outputs a single number.

```text
inputs:   x1 = 2.0,  x2 = 3.0
weights:  w1 = 0.5,  w2 = -1.0
bias:     b  = 1.0

z = (x1 * w1) + (x2 * w2) + b
  = (2.0 * 0.5) + (3.0 * -1.0) + 1.0
  = 1.0 - 3.0 + 1.0
  = -1.0
```

Then an **activation function** is applied to z (more on these in file 04). For now, think of it as a gate that decides whether to "pass" the signal.

**What are weights and bias?**
- Weights = how much each input matters. A weight of 0 means "ignore this input."
- Bias = a baseline shift. Without it, the neuron can only output 0 when all inputs are 0.
- Both are just numbers the network *learns* during training.

---

## A Layer = Many Neurons in Parallel

If a single neuron produces one output number, a layer of N neurons produces N output numbers — each neuron runs the same formula with its own set of weights.

```text
Input: [2.0, 3.0]   (2 features)

Neuron 1: w=[0.5, -1.0], b=1.0  → output: -1.0
Neuron 2: w=[0.2,  0.8], b=0.5  → output: 0.2*2.0 + 0.8*3.0 + 0.5 = 3.3
Neuron 3: w=[-0.3, 0.4], b=0.0  → output = -0.3*2.0 + 0.4*3.0 + 0.0 = 0.6

Layer output: [-1.0, 3.3, 0.6]   (3 features → expanded to 3 outputs)
```

In matrix form: `output = X * W + b`
where W is a matrix (one column per neuron), b is a bias vector.

---

## A Network = Stack of Layers

Each layer's output becomes the next layer's input.

```text
Raw input (e.g., pixel values, word IDs)
    │
    ▼
[Layer 1]  — finds low-level patterns (edges, letter shapes)
    │
    ▼
[Layer 2]  — combines patterns into mid-level features (corners, syllables)
    │
    ▼
[Layer 3]  — combines mid-level into high-level concepts (faces, words)
    │
    ▼
Output (a number, a class label, a vector, etc.)
```

The "depth" of a network = how many layers it has. Deep networks can represent more complex functions — but are harder to train (why residual connections exist, which you've already seen in ResNet).

---

## The Forward Pass

**Forward pass** = data flowing left to right (input → output). No learning happens here. It's just computation.

Toy example: a 2-layer network predicting if a review is positive (1) or negative (0).

```text
Input: "great product" → [0.9, 0.1]   (some embedding, simplified to 2 numbers)

Layer 1 (2 → 4 neurons):
  output = [-0.3, 1.2, 0.8, -1.1]

Activation (ReLU, zeros out negatives):
  output = [0.0, 1.2, 0.8, 0.0]

Layer 2 (4 → 1 neuron, the output):
  output = 0.85

Activation (sigmoid, squashes to 0–1):
  final  = 0.85  → "85% likely positive"
```

---

## Key Points

| Term | What it is |
|------|-----------|
| **Neuron** | One weighted sum + bias + activation |
| **Layer** | Many neurons running in parallel on the same input |
| **Weight** | How much a neuron cares about each input — learned during training |
| **Bias** | A constant offset — also learned |
| **Forward pass** | Input flowing through all layers to produce an output |
| **Depth** | Number of layers — deeper = can learn more complex patterns |

The forward pass answers: "given these weights, what does the network *predict*?"
Training answers: "how should we change the weights to predict *better*?" → that's file 03 (backprop).
