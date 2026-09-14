r"""Is the N=2 cgae_cflow deficit just an EFFECTIVE-LAMBDA effect?

On a chain the per-step backward decay is:

    cgae_flow :  lam * w(d->s) * rho      (w < 1 under fan-in)
    cgae_cflow:  lam * 1       * rho      (what = w/R = 1 at a single successor)

At ncopies N=2 the mean edge weight is w = 0.896 and 91.3% of decisions have a
single successor, so cgae_flow's EFFECTIVE lambda is about

    0.95 * 0.896 = 0.851

against cgae_cflow's full 0.95. If the N=2 gap (cgae_flow 0.837 vs cgae_cflow
0.401, 4W/14L, p=0.0043) is really about how far credit propagates rather than
about the normalization itself, then cgae_cflow run at a LOWER lambda should
recover cgae_flow's performance -- and the right "fix" would be a horizon knob,
not a new weighting rule (cgae_cap).

This sweeps cgae_cflow over lambda at N=2, 20 paired CRN seeds, into a separate
output dir per lambda so nothing collides with the lam=0.95 cells already on
disk.

  lam ~ 0.85 recovering ~0.837  -> effective-horizon explanation; cgae_cap is
                                   solving the wrong problem and lambda is the
                                   real knob.
  lam sweep flat near 0.40      -> structural; the normalization itself is the
                                   problem and cgae_cap is the right fix.

Run: python run_lam_probe.py <lam> [workers]
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4

import run_ncopies_three_way_crn as R  # noqa: E402

_orig_cfg = R.stoch_config


def _patched_cfg():
    c = _orig_cfg()
    c.lam = LAM
    return c


# cfg travels to the spawned workers inside the pickled payload, so patching the
# factory here is enough -- the workers do not re-read METHODS/OUTDIR (they act
# on the explicit method in their payload), and CAUSAL already lists cgae_cflow.
R.stoch_config = _patched_cfg
R.METHODS = ("cgae_cflow",)
R.NS = [2]
R.SEEDS = 20
R.OUTDIR = "suite_results_lam_probe_%s" % str(LAM).replace(".", "")

if __name__ == '__main__':
    print("[lam-probe] cgae_cflow @ lam=%.2f, N=2, 20 seeds -> %s"
          % (LAM, R.OUTDIR), flush=True)
    R.main(WORKERS)
