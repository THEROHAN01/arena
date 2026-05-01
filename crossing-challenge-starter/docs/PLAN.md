# Crossing Challenge — Implementation Plan

## The Goal

Beat the repo baseline (Eval score **0.74**) by as much as possible.

The score is a composite of two normalized terms — lower is better, 1.0 means you tied
doing nothing, 0.0 means perfect:

```
score = 0.5 * (BCE / 0.2488)  +  0.5 * (mean_pixel_ADE / 49.80)
```

The baseline's two weaknesses are well-defined and independently attackable:

| Term         | Baseline on Eval | Problem                                               |
|--------------|------------------|-------------------------------------------------------|
| Intent (BCE) | contributes ~0.37| XGBoost on 20 static features, ignores sequence       |
| Trajectory   | contributes ~0.37| Constant velocity — ADE grows to 61 px at 2 s        |

---

## Project Setup

### Prerequisites

- Python 3.11 (matches the Dockerfile; 3.10+ should also work)
- No GPU required for Phase 1 and Phase 2a (Kalman filter)
- GPU (or free Kaggle/Colab) required for Phase 2b and 3 (LSTM/GRU)

### First-time setup

```
cd crossing-challenge-starter

# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify the data is present (already in repo — no download needed)
ls data/
#   build_tracklets.py  build_windows.py  dev.parquet  schema.md  train.parquet

# 4. Train the baseline model (generates model.pkl)
python baseline.py
#   Expected: ~5 s, Dev log-loss ~0.18

# 5. Score the baseline locally
python grade.py
#   Expected output: Score: ~0.83  (intent_term ~0.72, traj_term ~0.93)
#   Note: Dev score is ~0.83, Eval score is ~0.74 (Dev has higher positive rate)

# 6. Run contract tests
python -m pytest tests/ -v
#   All tests should pass before any submission

# 7. (Optional) Verify Docker build
docker build -t my-crossing .
docker run --rm -v $(pwd)/data:/work my-crossing /work/dev.parquet /work/preds.csv
```

### Key files

```
crossing-challenge-starter/
├── predict.py          <-- MAIN FILE TO EDIT (inference entry point)
├── baseline.py         <-- training script (replace or extend this)
├── grade.py            <-- scoring harness (do not modify)
├── model.pkl           <-- trained weights (overwrite with your model)
├── requirements.txt    <-- add new deps here
├── Dockerfile          <-- update if you add heavy deps (torch etc.)
├── data/
│   ├── train.parquet   <-- ~29k training windows
│   ├── dev.parquet     <-- ~6k dev windows (self-grade here)
│   └── schema.md       <-- column reference
├── tests/
│   └── test_predict.py <-- shape/contract tests (must stay green)
└── docs/
    └── PLAN.md         <-- this file
```

---

## Data Shape

Each row is one prediction window: 16 frames of history at 15 Hz (~1.07 s of past).

```
  Input (per request)
  ┌───────────────────────────────────────────────────────────────────┐
  │  bbox_history    [16 × 4]   [x1,y1,x2,y2] pixels, oldest→current │
  │  ego_speed_hist  [16]       m/s, zeros if ego_available=False      │
  │  ego_yaw_hist    [16]       rad/s, zeros if ego_available=False    │
  │  frame_w, frame_h           always 1920×1080                       │
  │  time_of_day, weather, location   optional strings                 │
  └───────────────────────────────────────────────────────────────────┘
                                │
                           predict()
                                │
  ┌───────────────────────────────────────────────────────────────────┐
  │  intent         float [0,1]   P(will cross in next 2 s)           │
  │  bbox_500ms     [4]           predicted bbox at +0.5 s            │
  │  bbox_1000ms    [4]           predicted bbox at +1.0 s            │
  │  bbox_1500ms    [4]           predicted bbox at +1.5 s            │
  │  bbox_2000ms    [4]           predicted bbox at +2.0 s            │
  └───────────────────────────────────────────────────────────────────┘
```

Class imbalance: ~7% of windows are `will_cross_2s = True`. Design for it — BCE punishes
overconfident minority-class predictions harshly.

---

## Implementation Plan

### Phase 1 — Better features + Kalman trajectory (CPU, no GPU)

**Target score: ~0.65**  
**Time estimate: 2–3 hours**

```
  Baseline predict.py
        │
        ├─ Intent branch
        │     XGBoost (20 features)
        │         ↓ replace/augment
        │     XGBoost / LightGBM (enriched features)
        │         + bbox area rate of change       (approaching vs. receding)
        │         + lateral velocity (toward frame center)
        │         + acceleration features (2nd derivative of cx/cy)
        │         + deceleration signal (pedestrian slowed/stopped)
        │         + bbox shrink/grow ratio over full 16-frame window
        │         + scale_pos_weight to handle 7% class imbalance
        │
        └─ Trajectory branch
              Constant velocity  →  Kalman filter
                  State:  [cx, cy, vx, vy, ax, ay]
                  Observation: [cx, cy] from bbox center
                  Predict 8, 15, 23, 30 frames ahead at 15 Hz
```

**Why Kalman?**
The baseline's ADE explodes at long horizons because it assumes constant velocity.
A Kalman filter with acceleration state models:
- Pedestrian deceleration before a crossing decision
- Turning motion
- "Saw the vehicle and froze" cases

It is:
- CPU fast (microseconds per prediction)
- No training data required (parameter estimation from the 16 frames themselves)
- Interpretable and debuggable

**New feature ideas for intent classifier:**

| Feature                          | Signal                                      |
|----------------------------------|---------------------------------------------|
| bbox area rate of change         | Growing = ped walking toward camera         |
| normalized lateral vx            | Moving toward curb / road centerline        |
| vx/vy acceleration (last 4 frames)| Sudden motion change often precedes crossing|
| heading angle change             | Turning to face the road                   |
| bbox aspect ratio trend          | Standing tall vs. walking posture           |
| ego_speed last - ego_speed mean  | Vehicle decelerating near intersection      |
| frame position (cx/frame_w)      | Proximity to road edge (left/right quarter) |

---

### Phase 2 — Small GRU trajectory model

**Target score: ~0.55**  
**Time estimate: 4–6 hours (needs GPU for training, CPU for inference)**

Replace the Kalman filter trajectory with a learned GRU model.

```
  Input sequence (16 time steps, 15 Hz)
  ┌────────────────────────────────────────────────────┐
  │  t=0  [cx_norm, cy_norm, w_norm, h_norm, spd, yaw] │
  │  t=1  [cx_norm, cy_norm, w_norm, h_norm, spd, yaw] │
  │  ...                                               │
  │  t=15 [cx_norm, cy_norm, w_norm, h_norm, spd, yaw] │
  └────────────────────────────────────────────────────┘
              │
         GRU encoder
         (hidden_size=64, num_layers=2)
              │
         last hidden state  [64]
              │
         Linear head  →  [4 × 2]
                          ↕
              (cx, cy) for each of 4 future horizons
                          ↕
         expand back to [x1,y1,x2,y2] using last-known bbox size
```

**Training details:**
- Loss: Huber loss on predicted (cx, cy) vs. true bbox centers
- Normalize inputs: divide cx/cy by frame dimensions, w/h by frame dims
- Batch size: 256, Adam optimizer, lr=1e-3, cosine decay
- Augmentation: horizontal flip (mirror cx, flip ego yaw sign)
- Train on train.parquet (~29k windows), validate on dev.parquet
- Training time: ~15 min on a T4 GPU (Kaggle/Colab free tier is sufficient)
- Inference time: ~1 ms per request on CPU after training

---

### Phase 3 — End-to-end multi-task LSTM/GRU (the differentiator)

**Target score: ~0.45–0.50**  
**Time estimate: 6–8 hours**

Train a single model that produces both outputs jointly.
Shared encoder learns features that are useful for both intent and trajectory.

```
  Input sequence  [16, 6]
  (cx_n, cy_n, w_n, h_n, ego_spd, ego_yaw)
        │
        ▼
  ┌─────────────────────────────────────┐
  │   GRU Encoder                       │
  │   input_size=6                      │
  │   hidden_size=128                   │
  │   num_layers=2                      │
  │   dropout=0.2                       │
  └─────────────────────────────────────┘
        │
        │  hidden state  [128]
        │
        ├──────────────────────────────────────────────
        │                                             │
        ▼                                             ▼
  ┌─────────────────┐                    ┌────────────────────────┐
  │  Intent Head    │                    │  Trajectory Head       │
  │                 │                    │                        │
  │  Linear(128,64) │                    │  Linear(128,256)       │
  │  ReLU           │                    │  ReLU                  │
  │  Dropout(0.2)   │                    │  Linear(256, 4×2=8)    │
  │  Linear(64, 1)  │                    │  → (cx,cy) × 4 horizons│
  │  Sigmoid        │                    └────────────────────────┘
  │  → intent prob  │
  └─────────────────┘
        │                                             │
        ▼                                             ▼
   BCE loss                                    Huber loss
   (weighted for                               (on pixel centers)
    class imbalance)
        │                                             │
        └──────────────┬──────────────────────────────┘
                       ▼
          Joint loss = α * BCE_loss + β * Huber_loss
          (α=1.0, β=0.1 to start; tune on dev score)
```

**Why joint training helps:**
A pedestrian slowing down and looking left is both more likely to cross AND will follow
a different trajectory. The shared encoder learns this coupling — the intent signal
informs where the pedestrian goes, and the trajectory signal informs intent confidence.

**Training details:**
- Dataset: train.parquet (29k windows) + optional dev split for early stopping
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
- Scheduler: OneCycleLR
- Class imbalance: `pos_weight = (n_neg / n_pos)` in BCEWithLogitsLoss
- Augmentation:
  - Horizontal bbox flip (mirror x coords, negate ego yaw)
  - Temporal jitter: randomly trim last 0–3 frames from history
- Save checkpoint with best dev score, not best loss

**PyTorch dependency (add to requirements.txt for training):**
```
torch>=2.1,<3
```
For inference-only Docker image, torch can be CPU-only (`torch --index-url https://download.pytorch.org/whl/cpu`).

---

## Score Progression (Expected)

```
  1.00 ─── zero-work floor (class prior + zero velocity)
            │
  0.83 ─── baseline on Dev
  0.74 ─── baseline on Eval  ← current bar to beat
            │
  0.65 ─── Phase 1: better features + Kalman trajectory
            │
  0.55 ─── Phase 2: learned GRU trajectory
            │
  0.45 ─── Phase 3: end-to-end multi-task model
            │
  0.00 ─── perfect (unreachable in practice)
```

All Dev score estimates. Expect ~0.05–0.10 gap vs. Eval (Dev has slightly higher
positive rate than Eval).

---

## ADE Breakdown (Trajectory term)

Baseline ADE per horizon, and what each phase targets:

```
  Horizon   Baseline   Kalman (est.)   GRU (est.)   Multi-task (est.)
  +0.5 s     7.9 px       6 px           4 px           3 px
  +1.0 s    18.7 px      12 px           8 px           6 px
  +1.5 s    37.4 px      22 px          14 px          10 px
  +2.0 s    61.1 px      35 px          22 px          16 px
  mean      31.3 px      19 px          12 px           9 px
```

The 2-second horizon is where the most points are. Constant velocity compounds its
error linearly — any model that captures deceleration or turning wins big here.

---

## Submission Checklist

Before sending the repo link:

- [ ] `python -m pytest tests/ -v` — all tests green
- [ ] `python grade.py` — score better than 0.74 (Dev better than 0.83)
- [ ] `docker build -t my-crossing .` — builds without errors, image ≤ 2 GB
- [ ] `docker run --rm --network=none -v $(pwd)/data:/work my-crossing /work/dev.parquet /work/preds.csv` — runs offline, no network calls
- [ ] `model.pkl` (or equivalent weights file) committed to the repo
- [ ] `README.md` updated with approach, experiments, ablation table, and next steps
- [ ] `CLAUDE.md` or `AGENTS.md` included as required by the challenge rules
- [ ] Git log shows real iteration (not a single "finished" commit)
- [ ] Repo is public (or `gobblecube-hiring` added as collaborator)

---

## README Content for Maximum Impact

The hiring team reads README almost as heavily as the score. Structure it as:

```
## Approach
  What model, what features, what loss function and why.

## Experiments
  Table showing Dev score at each iteration:
    | Experiment                          | Dev Score |
    |-------------------------------------|-----------|
    | Baseline (XGBoost + const velocity) |   0.83    |
    | + enriched features                 |   0.??    |
    | + Kalman trajectory                 |   0.??    |
    | + GRU trajectory                    |   0.??    |
    | Multi-task GRU (final)              |   0.??    |

## What Didn't Work
  E.g. "Tried adding weather one-hot — no measurable improvement on Dev.
  Likely because the dataset has too few rainy windows to learn from."

## What I Would Try Next
  E.g. "Social force model to condition trajectory on ego-vehicle position.
  Optical flow features from raw video if licensed data were available."
```

---

## Key Constraints (never violate)

- No external API calls at inference time — container runs with `--network=none`
- Docker image ≤ 2 GB
- 4 GB RAM / 4 CPUs / 30-minute wall-clock budget at scoring
- Inference must be row-by-row via `predict(request: dict) -> dict`
- Do not re-extract tracklets from JAAD/PIE raw video to look up eval pedestrians
- No hardcoded per-request predictions (grader fuzzes requests)
