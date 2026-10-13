"""Parse the BPI Challenge 2012 log into a flat event table, without pm4py:
stream the XES with iterparse.

Input: ../paper/BPI_Challenge_2012.xes (public, 4TU.ResearchData
doi:10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f; not versioned).
Output: ../results/bpi2012_events.csv.gz (not versioned), one row per event with
case, case attributes (REG_DATE, AMOUNT_REQ), activity, lifecycle, resource,
timestamp. Run once; the simulation model is fitted from this table
(bpi2012_fit.py).
"""
import csv
import gzip
import os
import xml.etree.ElementTree as ET

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(SUITE, "paper", "BPI_Challenge_2012.xes")
OUT = os.path.join(SUITE, "results", "bpi2012_events.csv.gz")
NS = "{http://www.xes-standard.org/}"


def _attrs(elem):
    return {c.get("key"): c.get("value") for c in elem if c.get("key") is not None}


def main():
    n_traces = n_events = 0
    with gzip.open(OUT, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "reg_date", "amount_req", "activity", "lifecycle", "resource", "timestamp"])
        case = None
        for ev, elem in ET.iterparse(SRC, events=("start", "end")):
            tag = elem.tag.replace(NS, "")
            if ev == "start" and tag == "trace":
                case = {}
            elif ev == "end" and tag == "event" and case is not None:
                a = _attrs(elem)
                w.writerow([case.get("concept:name"), case.get("REG_DATE"), case.get("AMOUNT_REQ"),
                            a.get("concept:name"), a.get("lifecycle:transition"),
                            a.get("org:resource", ""), a.get("time:timestamp")])
                n_events += 1
                elem.clear()
            elif ev == "end" and tag == "trace":
                n_traces += 1
                case = None
                elem.clear()
            elif ev == "end" and case is not None and elem.get("key") in ("concept:name", "REG_DATE", "AMOUNT_REQ") \
                    and tag in ("string", "date") and not case.get(elem.get("key")):
                # trace-level attribute (event attributes are read in the event branch)
                case[elem.get("key")] = elem.get("value")
    print(f"{n_traces} traces, {n_events} events -> {OUT}")


if __name__ == "__main__":
    main()
