# NumPy Logistic Regression

Rebuilding binary logistic regression from scratch with nothing but NumPy. The goal of the project is to keep the math and code transparent so you can see every step: forward pass, cross-entropy loss, backpropagation, and gradient descent weight updates.

## What you can do
- Study a minimal, dependency-light implementation of logistic regression
- Step through the full training loop to understand how gradients move weights
- Run the included notebook to visualize loss curves and decision boundaries
- Reuse the class in `src/model.py` for quick binary classification experiments

## Quickstart
1. Create and activate a virtual environment.
2. Install the only dependency:
   ```
   pip install -r requirements.txt
   ```
3. Open `notebooks/training.ipynb` to train and visualize the model step-by-step.

## Using the model in code
The class exposes the core primitives you need for a training loop. A minimal example:

```python
import numpy as np
from src.model import LogisticRegression

X = np.random.randn(200, 2)           # features
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)  # labels

model = LogisticRegression(n_features=X.shape[1], lr=0.1)

for epoch in range(500):
    _, y_hat = model.forward(X)
    loss = model.loss(y, y_hat)
    dW, dB = model.backward(X, y, y_hat)
    model.update(dW, dB)
    if epoch % 100 == 0:
        print(f\"epoch {epoch}: loss={loss:.4f}\")
```

## How the training loop works
- **Forward pass**: compute linear logits `z = Xw + b`, then apply the sigmoid to map to probabilities.
- **Loss**: binary cross-entropy `L = -[y log(y_hat) + (1 - y) log(1 - y_hat)]`.
- **Backward pass**: gradients flow from loss → sigmoid → linear layer → weights and bias.
- **Update**: gradient descent nudges parameters in the direction that lowers the loss.

```mermaid
flowchart TD
  A[Start Epoch] --> B[Forward Pass: z = Xw + b; ŷ = sigmoid|z|]
  B --> C[Compute Loss: binary cross-entropy]
  C --> D[Backward Pass: gradients dW and dB]
  D --> E[Update Weights: gradient descent]
  E --> F[Track loss / metrics]
  F --> G[Next Epoch]
```

## Project structure
```
numpy-logistic-regression/
├── src/
│   └── model.py           # LogisticRegression class (forward, loss, backward, update)
├── notebooks/
│   └── training.ipynb     # Interactive walkthrough and visualization
└── README.md
```

## Roadmap ideas
- Add simple train/eval utilities and accuracy reporting
- Extend to multiclass softmax regression
- Package for pip installation once the public API settles
