"""Fuzzy clustering on synthetic blobs (FCM / Gustafson-Kessel / PCM).

    python examples/clustering.py
"""

import fuzzytool as fz
from fuzzytool import cluster
from fuzzytool.datasets import make_blobs


def main() -> None:
    X = make_blobs(centers=((0, 0), (6, 6), (0, 6)), seed=0)

    for name, fn in [
        ("fuzzy c-means", fz.fuzzy_cmeans),
        ("Gustafson-Kessel", fz.gustafson_kessel),
        ("possibilistic c-means", fz.possibilistic_cmeans),
    ]:
        r = fn(X, c=3, seed=0)
        pc = cluster.partition_coefficient(r.u)
        xb = cluster.xie_beni(X, r.centers, r.u)
        centers = ", ".join(f"({a:.1f}, {b:.1f})" for a, b in r.centers)
        print(f"{name:22s} | iters={r.n_iter:3d} | PC={pc:.3f} | XB={xb:.3f}")
        print(f"{'':22s}   centers: {centers}")


if __name__ == "__main__":
    main()
