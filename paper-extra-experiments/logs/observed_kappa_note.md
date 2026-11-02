# Observed-kappa interpretation

The observed dynamic range in the CIFAR stress runs is a realized diagnostic of the
trained positive matrices after the chosen logit scale, clipping, and temperature. It
is not the small illustrative kappa value used in the conservative lookup table.

Final available median kappa_obs across Sinkhorn policies in this run: 12739.1.

There is no contradiction if kappa_obs is much larger than the lookup-table range.
The lookup table is a conservative design example; the actual controller records the
measured marginal error epsilon_c, the product certificate, and the exact propagated
backward gain. Large observed kappa mainly says that the kappa-only worst-case Hilbert
bound will be conservative, so matrix-adaptive and measured diagnostics are the right
quantities to report next to the design rule.
