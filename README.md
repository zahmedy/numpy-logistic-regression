# NumPy Logistic Regression

Binary logistic regression rebuilt from first principles with nothing but NumPy. The repo is meant to be teaching-friendly: every step of the forward pass, cross-entropy loss, backpropagation, and gradient descent weight updates is visible and tweakable.

## Highlights
- Lightweight `LogisticRegression` + `StandardScaler` implemented in `src/model.py`
- Two runnable notebooks:
  - `notebooks/training.ipynb`: synthetic 2D blobs to show the full training loop and loss curve
  - `notebooks/TitanicSurvaival.ipynb`: preprocess the Titanic survival data and train/evaluate the custom model
- Plain NumPy math with optional L2 regularization (backward compatible with a `l2` parameter)

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. The Titanic CSV is already included in `data/titanic.csv`. If you want to re-fetch it later, use the OpenML `titanic` dataset (version 1).

## Run the notebooks
- `notebooks/training.ipynb`: walk through blob generation, training, and loss visualization.
- `notebooks/TitanicSurvaival.ipynb`: cleans the Titanic data (one-hot encodes categoricals, scales features), trains the NumPy logistic regression, and prints train/test accuracy plus the strongest weight signals.

## Using the model in code
```python
import numpy as np
from src.model import LogisticRegression, StandardScaler

X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

model = LogisticRegression(n_features=X.shape[1], lr=0.05, reg_lambda=0.001)

for epoch in range(300):
    _, y_hat = model.forward(X_scaled)
    dW, dB = model.backward(X_scaled, y, y_hat)
    model.update(dW, dB)
```

## How the training loop works
- **Forward**: logits `z = Xw + b`, probabilities `sigmoid(z)`
- **Loss**: binary cross-entropy + optional L2
- **Backward**: gradients for weights/bias from prediction error
- **Update**: gradient descent using the chosen learning rate

## Project structure
```
numpy-logistic-regression/
├── data/
│   └── titanic.csv        # cached Titanic survival dataset
├── src/
│   └── model.py           # LogisticRegression + StandardScaler
├── notebooks/
│   ├── training.ipynb     # synthetic 2D demo
│   └── TitanicSurvaival.ipynb  # Titanic preprocessing + training
└── README.md
```

## Notes and next steps
- The model is binary-only; extending to softmax for multiclass is a natural follow-up.
- Add unit tests around the training loop and gradient correctness.
- Wrap the class for pip installation once the API stabilizes.
