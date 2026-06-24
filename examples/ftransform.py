"""Denoise a signal with the F-transform (direct then inverse).

    python examples/ftransform.py
"""

import numpy as np

import fuzzytool as fz


def main() -> None:
    rng = np.random.default_rng(1)
    x = np.linspace(0, 2 * np.pi, 400)
    clean = np.sin(x)
    noisy = clean + rng.normal(0, 0.3, size=x.shape)

    ft = fz.FTransform(0, 2 * np.pi, n_basis=12).fit(x, noisy)
    recon = ft.smooth(x)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    print(ft)
    print(f"{len(ft.components_)} components compress {len(x)} samples")
    print(f"RMSE vs clean signal: noisy = {rmse(noisy, clean):.4f}  ->  "
          f"reconstructed = {rmse(recon, clean):.4f}")


if __name__ == "__main__":
    main()
