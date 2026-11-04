"""Rebuild the cell records of a run_bpm.py run whose training finished but whose
cells were never written (e.g. an exception after training_run), from the train
folders: the learning curves from the TensorBoard event file and the held-out
scores from final_policy.pth / best_policy.pth. Then rewrites summary.json.

usage: python scripts/rebuild_cells.py <the same arguments as the run_bpm.py run>
       [check=1]   rebuild into <OUTDIR>/cells_rebuilt and compare with existing cells

The arguments select the same OUTDIR, env, observation and certificate options
as the run, so the held-out scoring matches run_bpm.py's own.
"""
import json
import os
import sys
import time
from pathlib import Path

CHECK = any(a == "check=1" for a in sys.argv[1:])
sys.argv = [a for a in sys.argv if a != "check=1"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bpm as rb  # noqa: E402  (parses sys.argv at import)
from common import extract_metrics  # noqa: E402
from heldout import machine_info  # noqa: E402


def history_from_events(folder):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ev = sorted(Path(folder).glob("events.out.tfevents.*"))
    if not ev:
        return None, None
    acc = EventAccumulator(str(ev[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    hist, t = {}, []
    for tag in acc.Tags()["scalars"]:
        pts = acc.Scalars(tag)
        hist[tag] = [p.value for p in pts]
        t += [p.wall_time for p in pts]
    return hist, ((max(t) - min(t)) / 60.0 if t else None)


def main():
    import torch
    torch.set_num_threads(max(1, int(rb.THREADS or 1)))   # threads=k as in the training workers
    out = rb.OUTDIR
    cells = out / ("cells_rebuilt" if CHECK else "cells")
    cells.mkdir(parents=True, exist_ok=True)
    logdir = str(out / "train")
    bp = out / f"baselines_N{rb.N}.json"
    baselines = json.loads(bp.read_text())
    paired = rb._paired_anchors()
    todo = [(m, s) for s in range(rb.SEEDS) for m in rb.METHODS
            if (CHECK or not (cells / f"N{rb.N}__{m}__s{s}.json").exists())
            and os.path.exists(os.path.join(logdir, f"{m}__s{s}", "final_policy.pth"))]
    print(f"[{rb.TAG}] rebuilding {len(todo)} cells into {cells}", flush=True)
    for m, s in todo:
        t0 = time.time()
        name = f"{m}__s{s}"
        hist, minutes = history_from_events(os.path.join(logdir, name))
        rec = extract_metrics(hist or {}, rb.EPOCHS, rb.TEST_FREQ)
        rec["heldout"] = rb._heldout_scores(m, logdir, name)
        rec.update({"env": rb.ENV, "N": rb.N, "method": m, "seed": s,
                    "epochs": rb.EPOCHS, "episodes": rb.EPISODES, "eval_seed": rb.EVAL_SEED,
                    "baselines": baselines, "minutes": minutes,
                    "update_device": "cuda" if rb.GPU else "cpu",
                    "machine": machine_info(workers=None, threads=1), "rebuilt": True})
        f = cells / f"N{rb.N}__{m}__s{s}.json"
        f.write_text(json.dumps(rec))
        msg = f"[{rb.TAG}] {name}: greedy_final={rec['greedy_final']} ({(time.time()-t0)/60:.1f} min)"
        if CHECK:
            ref = out / "cells" / f.name
            if ref.exists():
                j = json.loads(ref.read_text())
                import numpy as np
                msg += (f" | saved greedy_final={j.get('greedy_final')} ent={j.get('entropy_final'):.4f}/{rec['entropy_final']:.4f}"
                        f" heldout final {np.mean(j['heldout']['final']):.3f}/{np.mean(rec['heldout']['final']):.3f}"
                        f" best {np.mean(j['heldout']['best']):.3f}/{np.mean(rec['heldout']['best']):.3f}")
        print(msg, flush=True)
    if not CHECK:
        rb.summary(out, baselines, paired)


if __name__ == "__main__":
    main()
