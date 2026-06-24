"""Train an ANFIS to approximate a nonlinear function.

    python examples/anfis.py
"""

import numpy as np

import fuzzytool as fz


def main() -> None:
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)

    model = fz.ANFIS(n_inputs=1, n_mf=6).fit(x[:, None], y, epochs=100)
    pred = model.predict(x[:, None])
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))

    print(model)
    print(f"RMSE: epoch 1 = {model.history_[0]:.4f}  ->  final = {rmse:.4f}")


if __name__ == "__main__":
    main()
