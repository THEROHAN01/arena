# Learning Path: From First Principles to Our Solution

Each stage depends on the one before it. Do not skip.

---

## Stage 1 — Math Foundations

*Everything in ML is math. These 5 concepts appear everywhere.*

### 1.1 — 2D Coordinate Systems

A screen is a grid. Every pixel has an (x, y) address. x goes left to right, y goes
top to bottom (opposite of math class). A bounding box `[x1, y1, x2, y2]` is just two
corners of a rectangle on this grid. The center is `((x1+x2)/2, (y1+y2)/2)`. That's
it — all our data lives here.

```
  (0,0) ─────────────────────────► x
    │
    │      [x1,y1]───────┐
    │        │           │
    │        │  person   │
    │        └───────[x2,y2]
    │
    ▼ y
```

### 1.2 — Vectors and Rate of Change

If a pedestrian's center is at pixel (100, 200) in frame 0 and (108, 202) in frame 1,
their velocity vector is (8, 2) pixels per frame. That's all velocity is — how much
something moved per unit of time. Acceleration is how much velocity itself changes per
frame. These two concepts are the foundation of all trajectory work.

```
  frame 0:  center = (100, 200)
  frame 1:  center = (108, 202)
  velocity  = (108-100, 202-200) = (8, 2) px/frame

  frame 1:  velocity = (8,  2)
  frame 2:  velocity = (5,  2)
  accel     = (5-8, 2-2) = (-3, 0)  ← they slowed down horizontally
```

### 1.3 — What Probability Actually Means

P(crossing) = 0.7 means: across many similar situations, the pedestrian crossed 70%
of the time. It is NOT certainty. A model that outputs 0.99 when the true rate is 0.7
is confident but wrong — and BCE punishes it for this specifically.

### 1.4 — Logarithms

You need ln (natural log) only to understand BCE. Key facts:

```
  ln(1)    =  0
  ln(0.5)  ≈ -0.69
  ln(0.1)  ≈ -2.30
  ln(0.01) ≈ -4.60
  ln(→0)   → -∞
```

As probability approaches zero, ln approaches negative infinity. This is why BCE
explodes when you are confidently wrong — you are taking the log of a near-zero number.

### 1.5 — Mean and Variance

Mean = average. Variance = how spread out values are. You will use these constantly:
mean ADE, mean BCE, mean velocity. Nothing more needed here.

---

## Stage 2 — What a Model Is

*Before learning any specific model, understand what ML is solving.*

### 2.1 — The Core Idea

A model is a function: `f(input) → output`. You adjust the function's parameters until
its outputs are close to the true answers on your training data. That's it. Every model
in ML — XGBoost, LSTM, GPT — is doing exactly this.

### 2.2 — Loss Functions

A loss function measures how wrong your model is. Lower = better. The training process
is: compute loss → nudge parameters to reduce it → repeat. Two loss functions matter
for us:

```
  MSE / Huber  →  regression   (where will the person be?)

    MSE   = mean( (predicted - true)² )
    Huber = like MSE but less sensitive to large errors (outliers)


  BCE          →  classification   (will they cross?)

    BCE = -mean( y·ln(p) + (1-y)·ln(1-p) )

    where y = 1 if actually crossed, 0 if not
          p = your predicted probability

    Example:
      y=1, p=0.9  →  BCE = -ln(0.9)  ≈  0.10   (confident, correct → low loss)
      y=1, p=0.1  →  BCE = -ln(0.1)  ≈  2.30   (confident, WRONG  → high loss)
```

### 2.3 — Train / Dev / Eval Split

```
  Train   →  data the model learns from            (~29k windows, train.parquet)
  Dev     →  data you use to check progress         (~6k windows,  dev.parquet)
             you have the true labels here
  Eval    →  data the hiring company grades on      (never distributed to you)
             your final leaderboard score comes from here
```

The split in this problem is by video — the model has never seen the intersection or
the pedestrian in Dev or Eval. A model that memorizes "this intersection has many
jaywalkers" gets zero free points.

### 2.4 — Overfitting

A model that memorizes training data but fails on new data.

```
  Healthy:   train loss low,  dev loss low        ← generalizing well
  Overfit:   train loss low,  dev loss HIGH       ← memorized, not learned
```

Fixes: less model complexity, dropout, early stopping (stop training when dev loss
stops improving).

---

## Stage 3 — Gradient Boosted Trees

*Understand this first, then you will clearly see why sequences need something different.*

### 3.1 — Decision Tree

A series of if/else questions on your features that ends in a prediction:

```
  velocity_x > 3?
      YES → position_x < 0.3?
                YES → P(cross) = 0.72
                NO  → P(cross) = 0.31
      NO  → aspect_ratio > 2.1?
                YES → P(cross) = 0.18
                NO  → P(cross) = 0.06
```

Each split is chosen to reduce the loss the most. Simple and interpretable, but one
tree is too weak for a complex problem.

### 3.2 — Gradient Boosting (XGBoost)

Build many small decision trees sequentially. Each tree corrects the errors of all
previous trees:

```
  Tree 1:  makes predictions from scratch
  Tree 2:  learns to fix Tree 1's errors
  Tree 3:  learns to fix Tree 1+2's errors
  ...
  Tree 300: combined, they are very strong

  Final prediction = sum of all 300 trees' outputs
```

The key limitation for our problem: it needs flat, hand-crafted input. You cannot give
it 16 frames of sequence — you have to manually compute "average velocity over last 4
frames," "aspect ratio change," etc. It has no concept of time or order.

### 3.3 — Class Imbalance

7% of our windows are `will_cross=True`. XGBoost will naturally bias toward predicting
"no cross" because that is right 93% of the time.

```
  Naive model: always predict "no cross"
    Accuracy = 93%   ← looks good
    Usefulness = 0   ← catches zero crossings

  Fix: scale_pos_weight = n_negative / n_positive ≈ 13
       tells XGBoost to treat each crossing example as if it appeared 13 times
```

---

## Stage 4 — Why Sequences Need Different Treatment

*The conceptual bridge between Stage 3 and neural networks.*

### 4.1 — Tabular vs Sequential Data

```
  Tabular (XGBoost's world):
    Row 1: [feature_a, feature_b, feature_c] → label
    Row 2: [feature_a, feature_b, feature_c] → label
    Order of rows does not matter. Each row is independent.

  Sequential (our world):
    frame_0 → frame_1 → frame_2 → ... → frame_15 → prediction
    Order matters completely. frame_15 only makes sense after frame_14.
```

"Person was still for 14 frames, then suddenly moved left in frame 15" is a pattern
that requires reading all 16 frames in order. You cannot capture this by averaging.

### 4.2 — The Limit of Hand-Crafted Features

When you compute "mean velocity over last 4 frames" you have already thrown away
the temporal pattern:

```
  Pattern A:  velocities = [1, 2, 3, 4]    mean = 2.5  ← accelerating
  Pattern B:  velocities = [4, 3, 2, 1]    mean = 2.5  ← decelerating
  Pattern C:  velocities = [1, 4, 1, 4]    mean = 2.5  ← oscillating

  XGBoost sees the same number (2.5) for all three.
  A sequence model sees three completely different patterns.
```

A sequence model reads the raw frames and figures out which patterns matter. You do not
have to decide upfront.

---

## Stage 5 — Neural Networks

*The building block for everything in Stage 6.*

### 5.1 — A Single Neuron

Takes N inputs, multiplies each by a weight, sums them, applies an activation:

```
  inputs:   x1=0.5,  x2=0.3,  x3=0.8
  weights:  w1=1.2,  w2=-0.4, w3=0.9
  bias:     b=0.1

  raw sum = (0.5×1.2) + (0.3×-0.4) + (0.8×0.9) + 0.1
          = 0.6 - 0.12 + 0.72 + 0.1
          = 1.30

  output  = activation(1.30)
```

The weights are what training adjusts — billions of these weights across a large model.

### 5.2 — Activation Functions

Without activations, stacking layers is still just one linear operation. Non-linearity
lets the network learn complex shapes.

```
  ReLU(x)    = max(0, x)           used inside the network (fast, simple)

    x = -2  →  ReLU = 0
    x =  0  →  ReLU = 0
    x =  3  →  ReLU = 3


  Sigmoid(x) = 1 / (1 + e^-x)      maps any number to (0, 1)
                                    used as the final layer for probabilities

    x = -4  →  Sigmoid ≈ 0.02
    x =  0  →  Sigmoid = 0.50
    x =  4  →  Sigmoid ≈ 0.98
```

### 5.3 — A Linear Layer

A layer of N neurons all looking at the same input. Takes a vector of size A, outputs
a vector of size B. Written as `nn.Linear(A, B)` in PyTorch.

```
  input  [6]  →  Linear(6, 128)  →  output [128]
```

Each of the 128 output values is a weighted sum of all 6 inputs (plus bias). The layer
has 6×128 + 128 = 896 learnable parameters.

### 5.4 — Forward Pass

Data flows through layers left to right, each transforming it:

```
  raw input [6]
      │
  Linear(6, 128)   →  [128]
      │
  ReLU             →  [128]  (negative values zeroed out)
      │
  Linear(128, 64)  →  [64]
      │
  ReLU             →  [64]
      │
  Linear(64, 1)    →  [1]
      │
  Sigmoid          →  probability in (0, 1)
```

### 5.5 — Backpropagation (conceptual)

After a forward pass computes the loss, backpropagation traces backwards through the
network computing how much each weight contributed to the error. Gradient descent then
nudges each weight slightly in the direction that reduces loss. Repeat for every batch.

You do not need to derive this. PyTorch does it automatically when you call
`loss.backward()`.

### 5.6 — Adam Optimizer

The standard optimizer for neural networks. Adapts the learning rate per parameter
automatically — parameters that get large gradients get smaller updates, and vice
versa. Use `lr=3e-4` as a starting point.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
```

---

## Stage 6 — Sequence Models

*The heart of what differentiates our solution from the baseline.*

### 6.1 — Recurrent Neural Networks (RNN)

A network that processes a sequence step by step, maintaining a hidden state — a
fixed-size vector that summarizes "what I have seen so far":

```
  h_0 = zeros(128)           ← start with empty memory

  h_1 = f(h_0, frame_0)     ← read frame 0, update memory
  h_2 = f(h_1, frame_1)     ← read frame 1, update memory
  h_3 = f(h_2, frame_2)
  ...
  h_16 = f(h_15, frame_15)  ← full sequence compressed into 128 numbers

  prediction = head(h_16)
```

The function `f` (with its weights) is what training learns. The hidden state carries
information forward across time.

### 6.2 — The Vanishing Gradient Problem

Plain RNNs struggle to remember things from early in the sequence. Gradients flowing
backward through 16 time steps shrink exponentially — by the time you adjust weights
for frame 0 based on what happened at frame 15, the signal is nearly zero. LSTM and
GRU were invented to fix this.

### 6.3 — LSTM

Adds a separate "cell state" — a long-term memory channel that runs alongside the
hidden state. Uses three learned gates to control what to write, read, and erase from
this memory:

```
  forget gate  →  what to erase from memory
  input gate   →  what new information to write
  output gate  →  what to read out as the hidden state
```

Solves vanishing gradients. More parameters than a plain RNN.

### 6.4 — GRU (what we use)

A simplified LSTM — two gates instead of three, fewer parameters, trains faster,
similar performance in practice. For our 16-step sequence, GRU is sufficient.

```
  reset gate   →  how much of the past to forget
  update gate  →  how much to update the hidden state

  h_t = update * h_(t-1)  +  (1 - update) * candidate_h
        ↑ keep old memory     ↑ blend in new information
```

GRU in our problem:

```
  frame_0  [6] ──► GRU cell ──► h_1  [128]
                       ▲
  frame_1  [6] ──► GRU cell ──► h_2  [128]
                       ▲
  frame_2  [6] ──► GRU cell ──► h_3  [128]
                       ▲
      ...
                       ▲
  frame_15 [6] ──► GRU cell ──► h_16 [128]
                                   │
                             two output heads
```

The 6 input features per frame: `cx_norm, cy_norm, w_norm, h_norm, ego_speed, ego_yaw`

### 6.5 — Why the Hidden State Works

By frame 15, `h_16` implicitly encodes: was the person accelerating? Were they moving
laterally? Was the vehicle decelerating? These are exactly the signals needed for both
intent and trajectory. The model learns to encode what matters — you do not specify it.

---

## Stage 7 — The Kalman Filter

*A classical alternative to learned trajectory — used in Phase 1 before the GRU.*

### 7.1 — State Estimation

The Kalman filter maintains a belief over a hidden state. In our case:
`[cx, cy, vx, vy]` — position and velocity of the pedestrian center. At each frame
it does two steps:

```
  Predict:  project state forward using physics
              new_cx = cx + vx * dt
              new_cy = cy + vy * dt
              (uncertainty grows — we are less sure about the future)

  Update:   correct prediction using the observed bbox center
              pull the estimate toward what we actually saw
              (uncertainty shrinks — new evidence arrived)
```

### 7.2 — Why It Beats Constant Velocity

Constant velocity: `position = last_position + last_velocity * n_frames`. No
correction, no acceleration, compounds error at every step.

Kalman filter: fits velocity AND acceleration to the full 16-frame history, propagates
uncertainty, and produces a realistic forecast. Handles deceleration and noisy bbox
detections naturally.

```
  Horizon   Constant velocity   Kalman (estimated)
  +0.5 s        7.9 px              ~6 px
  +1.0 s       18.7 px             ~12 px
  +1.5 s       37.4 px             ~22 px
  +2.0 s       61.1 px             ~35 px   ← biggest gain here
```

---

## Stage 8 — Multi-Task Learning

*Why one model doing both jobs is better than two separate models.*

### 8.1 — Shared Representations

When you train one GRU to predict both intent AND trajectory, the hidden state must
become useful for both tasks. This forces the encoder to learn richer representations
of pedestrian motion than it would if trained on either task alone.

### 8.2 — Joint Loss

```
  total_loss = α * BCE_loss(predicted_intent, true_intent)
             + β * Huber_loss(predicted_trajectory, true_trajectory)

  Both losses flow backward through the same GRU encoder weights.
  α=1.0, β=0.1 is a reasonable starting point — tune on dev score.
```

### 8.3 — Why It Helps Here Specifically

A pedestrian about to cross will also move toward the road. The trajectory of a
crossing pedestrian looks different from a non-crossing one. Training jointly means the
model learns this coupling — intent informs trajectory and trajectory informs intent.

```
  Crossing pedestrian:      Non-crossing pedestrian:
    intent  → HIGH            intent  → LOW
    traj    → lateral move    traj    → parallel to road / stationary

  Shared encoder learns this coupling automatically.
```

---

## Full Dependency Map

```
  Stage 1: Math
  (coordinates, vectors, probability, log, mean)
        │
  Stage 2: What a Model Is
  (loss functions, train/dev/eval split, overfitting)
        │
        ├───────────────────────────────────────────────┐
        │                                               │
  Stage 3: Gradient Boosted Trees            Stage 4: Why Sequences Differ
  (decision tree, XGBoost, class imbalance)  (tabular vs sequential,
        │                                    limits of hand-crafted features)
        │                                               │
        └──────────────────────┬────────────────────────┘
                               │
                       Stage 5: Neural Networks
                       (neuron, layers, activations,
                        forward pass, backprop, Adam)
                               │
                  ┌────────────┴────────────┐
                  │                         │
           Stage 6: GRU               Stage 7: Kalman Filter
           (RNN → LSTM → GRU,          (state estimation,
            hidden state as memory)     predict + update cycle)
                  │
           Stage 8: Multi-Task Learning
           (shared encoder, joint loss, intent-trajectory coupling)
                  │
                  ▼
           Our Solution: predict.py
           (GRU encoder → intent head + trajectory head)
```

---

## Honest Time Estimate

| Stage | What It Is            | Time to Understand |
|-------|-----------------------|--------------------|
| 1     | Math                  | 2–3 hours          |
| 2     | What a model is       | 2 hours            |
| 3     | XGBoost               | 3 hours            |
| 4     | Why sequences differ  | 1 hour             |
| 5     | Neural networks       | 4–6 hours          |
| 6     | GRU / LSTM            | 4–6 hours          |
| 7     | Kalman filter         | 2–3 hours          |
| 8     | Multi-task learning   | 1–2 hours          |

Stages 1–4 = understand the baseline completely.
Stages 5–8 = understand what we are building.

The single most important stage is **Stage 6 (GRU)**. Spend the most time here.
Ideally implement a tiny GRU from scratch in NumPy before using PyTorch's built-in
version — once you feel the hidden state updating step by step, the whole architecture
clicks into place.
