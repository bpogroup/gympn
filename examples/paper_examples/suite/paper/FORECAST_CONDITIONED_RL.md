# Forecast-conditioned DRL under demand shift (future direction)

Status: idea, recorded 2026-10-27. Nothing built yet.

## Problem

A policy trained on an environment with a fixed arrival rate (as in the
current `bpic15` env: Poisson arrivals at a constant rate per municipality)
learns habits that are right only at that rate. Real processes are not
stationary: demand varies by day, weekday, season and trend, and shifts when
policies or the organisation change. Such a policy can then be worse than a
simple rate-agnostic heuristic (`bpic15_heuristic` never assumed a rate, so it
is a strong baseline under shift).

## Idea

Give the policy an estimate of the future as part of its state, for example a
forecast of the arrival intensity per municipality for today or the next hours,
**together with its uncertainty** (quantiles or an error scale), and train it
over many futures (randomised or data-calibrated demand), with **realistic
forecast errors at training time**.

The policy then learns the best action *given the forecast and its typical
error*, instead of the best action *if the forecast were true*:

- **vs predict-then-optimize (PTO):** PTO plugs a point forecast into an
  optimiser and acts as if it were correct, so its quality hinges on forecast
  accuracy. A forecast-conditioned policy learns how much to trust the
  forecast; with poor forecasts it falls back toward a robust policy.
- **vs a fixed-rate policy:** it generalises across demand levels because it
  saw them in training and is told which one it faces.

## Related work (from memory; verify before citing)

- Contextual MDPs; domain randomisation (Tobin et al. 2017).
- Policies conditioned on an estimate of the environment: UP-OSI (Yu et al.
  2017), Rapid Motor Adaptation (Kumar et al. 2021), meta-RL with inferred
  context (PEARL, Rakelly et al. 2019).
- Imagination-augmented agents (I2A, Weber et al. 2017): the policy reads a
  summary of imagined futures from a learned model and learns to discount them
  when the model is wrong. Close in spirit, but about model rollouts, not
  exogenous forecasts.
- Forecasts as RL state in operations: energy (prices, generation) and
  inventory (demand) control.
- Decision-focused learning / smart predict-then-optimize (Elmachtoub & Grigas):
  the opposite fix, training the predictor for decision quality.
- BPM: prescriptive process monitoring with predictions (and their reliability)
  as RL input (Metzger et al., BPM 2020, "Triggering proactive business process
  adaptations via online reinforcement learning"); concept drift in process
  mining.
- Gap to check: forecast-conditioned DRL for **resource allocation** in process
  control under drifting demand, with forecast-error-aware training, evaluated
  on real-data-calibrated non-stationarity.

## Caveat (what a forecast can and cannot add)

A forecast computed from the current state alone adds no information: a large
enough policy could compute it itself. It can still help by (a) easier
learning, since it is trained with dense, low-noise supervised targets (the
auxiliary-task effect, relevant here because RL from sparse rewards struggles
to learn "what the queues will look like later today"), and (b) real new
information, if the forecaster sees what the state does not (calendar,
history, external signals). The demand-shift setting is the clean case of (b).

## Testbed: BPIC 2015 with non-stationary demand

The BPIC 2015 logs have real case dates per municipality, so the arrival
intensity can be calibrated over time (day-to-day variation, weekly pattern,
trend). First step: **measure how much it actually varies** (by day, week,
year). If it varies little, add a synthetic shift stress test.

Logs: Zenodo record 16268743 (Collection_Event_Logs.zip; 4TU was down on
2026-10-21). No longer on disk as of 2026-10-27. Calibration script:
`suite/scripts/bpic15_calibrate.py <xes folder>`.

A-E PN encoding (fits the existing env):
- a hidden token per municipality with today's intensity, drawn from the
  calibrated distribution, read by the arrival transition;
- an observable "forecast" place holding the forecast (true intensity + a
  calibrated error, or quantiles), seen by the policy like the clock node
  (`GymProblem.clock_node`).

## Experiment

| Policy | Trained on | Sees forecast |
|---|---|---|
| A | fixed rate (today's setup) | no |
| B | randomised / calibrated rates | no |
| C | randomised / calibrated rates | yes, with training-time forecast errors |
| Baselines | heuristic (rate-agnostic); PTO (planner or heuristic acting on the forecast as truth) | |

Evaluate across (1) demand shift (inside and outside the training range) and
(2) forecast quality (error from zero to large). Expected:
- A breaks under shift; B is robust but generic;
- C is best when forecasts are informative and degrades gracefully toward B as
  forecasts worsen, while PTO degrades sharply. That curve is the central figure.

Possible cheap first test: train a forecaster on recorded days, add its
outputs to the observation, and check whether decision-relevant quantities
become more recognisable (the precise-departure AUC, ~0.74 without) and the
critic improves (the clock-node probe setup).

## Building blocks already in place (2026-10-27, uncommitted at time of writing)

- `GymProblem.clock_node` (dedicated clock node; the template for a forecast node),
  `merge_options`, `obs_clock`, `obs_token_age`.
- `run_bpm.py clock=node merge=1`; imitation start in that format
  (`results/bpic15_imitation/bpic15_imitate_mrg_clkn.pth`).
- RL fine-tuning from imitation needs `plr=3e-5` (3e-4 blows up: KL 0.3-1.5).
- Critic probe (`clock_probe.py` pattern): time of day explains 72% of the
  heuristic's state value; a clock more than halves the critic's error.
