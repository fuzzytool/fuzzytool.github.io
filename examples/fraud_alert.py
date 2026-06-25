"""Card-fraud alert score (Mamdani inference).

A payment monitor turns how far a charge deviates from the customer's usual
spend (`amount_anomaly`, in standard deviations) and how many cards swipes
happened in the last hour (`velocity`) into a `fraud_risk` score (0-100).

    python examples/fraud_alert.py
"""

import fuzzytool as fz


def main() -> None:
    amount_anomaly = fz.Variable("amount_anomaly", (0, 8))
    amount_anomaly["normal"] = fz.trap(0, 0, 1, 2.5)
    amount_anomaly["elevated"] = fz.tri(2, 3.5, 5)
    amount_anomaly["extreme"] = fz.trap(4.5, 6, 8, 8)

    velocity = fz.Variable("velocity", (0, 30))
    velocity["low"] = fz.trap(0, 0, 3, 6)
    velocity["moderate"] = fz.tri(4, 9, 14)
    velocity["burst"] = fz.trap(12, 20, 30, 30)

    fraud_risk = fz.Variable("fraud_risk", (0, 100))
    fraud_risk["low"] = fz.tri(0, 12, 35)
    fraud_risk["medium"] = fz.tri(30, 50, 70)
    fraud_risk["high"] = fz.tri(65, 85, 100)

    sys = fz.Mamdani(defuzz="centroid")
    sys.rule(amount_anomaly["extreme"] | velocity["burst"], fraud_risk["high"])  # |=OR  &=AND  ~=NOT
    sys.rule(amount_anomaly["elevated"] & velocity["moderate"], fraud_risk["medium"])
    sys.rule(amount_anomaly["normal"] & velocity["low"], fraud_risk["low"])

    for a, v in [(0.5, 2), (3.5, 9), (6.5, 22)]:
        print(f"anomaly={a}σ, velocity={v}/h  ->  fraud risk = {sys(amount_anomaly=a, velocity=v):.1f}/100")


if __name__ == "__main__":
    main()
