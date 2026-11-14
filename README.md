# make.fun polymarket-faa-attribution-02 entry

Forecast the Yes price of every Polymarket FAA outage attribution market
(Iran / foreign actor / hack / AI sandbox escape) at every minute of the
4 hours after each checkpoint, as five quantiles (0.05, 0.25, 0.5, 0.75,
0.95) per market per minute.

Model (Node 22, stdlib only, no network at runtime): per-market EWMA
volatility (detrended by the family's log-decay rate) + point-mass
mixture over a Student-t(3) jump distribution + adaptive decay of the
center toward 0 + h^0.41 horizon scaling + level vol floor.

Local validation: practice (NVIDIA CEO event cases) pinball loss 0.292
vs house persistence baseline 0.364; on a real FAA crash window
(checkpoint 2026-11-13T22:15Z) 0.881 vs baseline 0.877.
