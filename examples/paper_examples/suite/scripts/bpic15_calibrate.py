r"""Calibrate the BPIC 2015 environment (envs/bpic15_env.py) from the five
municipality logs.

Data: BPI Challenge 2015, municipalities 1-5 (van Dongen, 4TU.ResearchData,
doi:10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1; mirrored in the event-log
collection on Zenodo, record 16268743). Pass the folder that holds
BPI_Challenge_2015_Municipality_<m>.xes.

What is read off the data, per roster period (a window in which a given
employee works in two municipalities in the same weeks):
  * task groups: every event is mapped to intake (01_HOOFD_0xx/1xx), assessment
    (01_HOOFD_2xx-4xx and the side procedures) or decision (01_HOOFD_5xx-8xx,
    05_EIND, 06_VD, 07_OPS);
  * core staff: employees with at least 5% of a municipality's events in the
    period; an employee FITS a task group if at least 25% of their events in
    that municipality belong to it;
  * load: events per working day divided by the core headcount (a shared
    employee counts half in each municipality), relative to the mean.
What is NOT in the data and is assumed by the environment: service times (46%
of the events are stamped at midnight and two thirds are recorded in batches),
the start-of-day backlog, and the length of a day.

Run: python scripts/bpic15_calibrate.py <xes folder>
Prints the ROSTERS dict that envs/bpic15_env.py holds.
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

NS = '{http://www.xes-standard.org/}'
PERIODS = {  # roster period -> (start, end, shared employee and the two municipalities)
    "2014": ("2014-05-01", "2015-03-01", "560532", (2, 5)),
    "2012": ("2012-10-01", "2013-04-01", "560752", (4, 5)),
}
FIT, CORE = 0.25, 0.05
GROUPS = ("I", "A", "D")


def read(folder):
    rows = []
    for m in range(1, 6):
        path = os.path.join(folder, f"BPI_Challenge_2015_Municipality_{m}.xes")
        cur = None
        for action, el in ET.iterparse(path, events=("start", "end")):
            tag = el.tag.replace(NS, "")
            if action == "start":
                if tag == "event":
                    cur = {"mun": m}
                continue
            if cur is not None and tag in ("string", "date"):
                k = el.get("key")
                if k in ("org:resource", "concept:name", "time:timestamp"):
                    cur[k] = el.get("value")
            elif tag == "event":
                rows.append(cur); cur = None; el.clear()
            elif tag == "trace":
                el.clear()
    ev = pd.DataFrame(rows)
    ev["ts"] = pd.to_datetime(ev["time:timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    ev = ev[ev.ts.dt.year >= 2010].copy()
    ev["res"] = ev["org:resource"].astype(str)
    name = ev["concept:name"].astype(str)
    hoofd = name.str.extract(r"^01_HOOFD_(\d)")[0]

    def group(n, h):
        if isinstance(h, str):
            return "I" if h in "01" else ("A" if h in "234" else "D")
        return "D" if n.startswith(("05_EIND", "06_VD", "07_OPS")) else "A"
    ev["grp"] = [group(n, h) for n, h in zip(name, hoofd)]
    ev["day"] = ev.ts.dt.normalize()
    return ev


def calibrate(ev):
    rosters = {}
    for period, (a, b, shared, pair) in PERIODS.items():
        x = ev[(ev.ts >= a) & (ev.ts < b)]
        staff, load, groups = {}, {}, {}
        for m, y in x.groupby("mun"):
            vc = y.res.value_counts()
            core = [r for r in vc.index if vc[r] / len(y) >= CORE and r != shared]
            aff = y.groupby("res").grp.value_counts(normalize=True).unstack(fill_value=0)
            staff[m] = [(r, "".join(g for g in GROUPS if aff.loc[r].get(g, 0) >= FIT)) for r in core]
            heads = len(core) + (0.5 if m in pair else 0)
            load[m] = len(y) / y.day.nunique() / heads
            groups[m] = {g: round(float(v), 2) for g, v in y.grp.value_counts(normalize=True).items()}
        mean = np.mean(list(load.values()))
        sh = x[(x.res == shared) & x.mun.isin(pair)]
        sh_aff = sh.grp.value_counts(normalize=True)
        rosters[period] = {
            "staff": staff,
            "load": {m: round(float(load[m] / mean), 2) for m in load},
            "shared": (shared, "".join(g for g in GROUPS if sh_aff.get(g, 0) >= FIT), pair),
            "groups": groups,
        }
    return rosters


if __name__ == "__main__":
    import pprint
    pprint.pprint(calibrate(read(sys.argv[1])), width=110, sort_dicts=False)
