"""Direction B — structural conflict-graph extractor (AEPN_NATIVE_LEARNING.md §4).

Observational lineage *estimates from traces* which rewards are causally
related. The net **structurally guarantees** it: two transitions that share no
input place can never compete for the same token, and two transitions whose
place-neighbourhoods are disjoint can never interact at all — their firings
commute, exactly, data-independently. This module reads those two relations
straight off the static PN definition (``GymProblem`` / simpn ``SimProblem``),
with no simulation, no trace, and no token-value semantics.

Two relations are computed, because they answer two different questions:

* **Conflict** (competition for tokens): transitions ``t1``, ``t2`` conflict iff
  they share an *input* place (``•t1 ∩ •t2 ≠ ∅``). This is the classic PN
  structural-conflict relation. Connected components of the conflict graph
  ("conflict clusters") are the sets of transitions that can ever contend for
  the same tokens — the genuine *decision* couplings. Action-vs-action conflict
  edges are exactly the choices the agent actually makes.

* **Coupling** (Mazurkiewicz dependence): transitions ``t1``, ``t2`` are coupled
  iff their place-neighbourhoods overlap at all,
  ``(•t1 ∪ t1•) ∩ (•t2 ∪ t2•) ≠ ∅``. Connected components of the coupling graph
  are the **structurally-independent subnets**: reward streams in different
  components never touch a common place, so credit need not — and should not —
  be shared across them. This is the object Direction B factors value/advantage
  over, and the one CRN provably cannot deliver (CRN cancels noise *shared* by
  two branches; parallel activity in a disjoint subnet is uncancelled — cf.
  cfpl in CAUSAL_LINEAGE_RETHINK §7.2b).

Soundness caveat for *coloured* nets: guards and markings can make two
structurally-conflicting transitions independent in a given marking, so the
structural relations are a **sound over-approximation** — structural
independence ⇒ true independence (safe to exploit), never the converse. The
decomposition this yields is therefore always correct, at worst too coarse
(missed separability), never wrong.

The global clock place (``SimVarTime``) is excluded by default: every timed
transition reads it, so including it collapses the whole net into one component.
It is a construction-level coupling, not a resource coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:  # simpn is a hard dependency of gympn; the try only guards import-time tools
    from simpn.simulator import SimVar, SimVarTime, SimVarQueue
except Exception:  # pragma: no cover - defensive
    SimVar = SimVarTime = SimVarQueue = ()  # type: ignore


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TransitionInfo:
    """A transition reduced to its structural footprint (place ids only)."""
    id: str
    kind: str  # 'action' (agent decision) | 'event' (evolution)
    inputs: frozenset  # ids of input places (excluded places removed)
    outputs: frozenset  # ids of output places (excluded places removed)

    @property
    def neighbourhood(self) -> frozenset:
        return self.inputs | self.outputs


@dataclass
class StructuralAnalysis:
    """Result of :func:`analyze`. All ids are strings (place/transition ``_id``)."""
    transitions: List[TransitionInfo]
    place_ids: List[str]
    excluded_place_ids: List[str]

    # conflict = share an input place (competition for tokens)
    conflict_edges: List[Tuple[str, str]]
    conflict_clusters: List[List[str]]  # connected components of the conflict graph

    # coupling = share any place (Mazurkiewicz dependence) -> independent subnets
    coupling_components: List[List[str]]

    # place -> transitions, for places touched by >1 transition
    shared_input_places: Dict[str, List[str]]  # a place >1 transition CONSUMES
    coupling_places: Dict[str, List[str]]  # a place >1 transition touches (in or out)

    warnings: List[str] = field(default_factory=list)

    # ----- derived metrics -------------------------------------------------- #
    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    @property
    def n_actions(self) -> int:
        return sum(1 for t in self.transitions if t.kind == "action")

    @property
    def n_components(self) -> int:
        return len(self.coupling_components)

    @property
    def component_sizes(self) -> List[int]:
        return sorted((len(c) for c in self.coupling_components), reverse=True)

    @property
    def largest_component_fraction(self) -> float:
        """Share of transitions in the single biggest independent subnet.

        This is *the* granularity number for the Direction B probe: near 1.0
        means the net is one coupled blob (decomposition buys nothing — the E1
        falsifier); well below 1.0 means genuine parallel structure to exploit.
        """
        if not self.transitions:
            return 0.0
        return max((len(c) for c in self.coupling_components), default=0) / self.n_transitions

    @property
    def action_conflict_edges(self) -> List[Tuple[str, str]]:
        """Conflict edges between two ACTION transitions — the real decisions."""
        actions = {t.id for t in self.transitions if t.kind == "action"}
        return [(a, b) for (a, b) in self.conflict_edges if a in actions and b in actions]


# --------------------------------------------------------------------------- #
# Union-find
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self, items: Sequence[str]):
        self._parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def components(self) -> List[List[str]]:
        groups: Dict[str, List[str]] = {}
        for x in self._parent:
            groups.setdefault(self.find(x), []).append(x)
        return [sorted(g) for g in groups.values()]


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _place_id(p) -> Optional[str]:
    return getattr(p, "_id", None)


def _is_clock(p) -> bool:
    if SimVarTime and isinstance(p, SimVarTime):
        return True
    time_id = getattr(SimVarTime, "TIME_ID", "time") if SimVarTime else "time"
    return _place_id(p) == time_id


def extract_transitions(
    pn,
    exclude_place_ids: Sequence[str] = (),
    include_clock: bool = False,
) -> Tuple[List[TransitionInfo], List[str], List[str], List[str]]:
    """Reduce a ``GymProblem`` to structural transition footprints.

    Returns ``(transitions, place_ids, excluded_place_ids, warnings)``.

    ``pn.actions`` are agent-controlled decision transitions; ``pn.events`` are
    evolutions. Both are included as transitions; ``kind`` records which.
    """
    warnings: List[str] = []
    excluded: Set[str] = set(exclude_place_ids)

    # Collect places and decide exclusions.
    all_places = list(getattr(pn, "places", []) or [])
    for p in all_places:
        pid = _place_id(p)
        if pid is None:
            continue
        if not include_clock and _is_clock(p):
            excluded.add(pid)
        if SimVarQueue and isinstance(p, SimVarQueue):
            warnings.append(f"place '{pid}' is a SimVarQueue (queue view); kept — "
                            "exclude it explicitly if it double-counts a base place")
    kept_place_ids = [pid for p in all_places
                      if (pid := _place_id(p)) is not None and pid not in excluded]

    def footprint(simvars) -> frozenset:
        ids = set()
        for p in simvars or []:
            pid = _place_id(p)
            if pid is not None and pid not in excluded:
                ids.add(pid)
        return frozenset(ids)

    transitions: List[TransitionInfo] = []
    seen: Set[str] = set()
    for kind, coll in (("action", getattr(pn, "actions", []) or []),
                       ("event", getattr(pn, "events", []) or [])):
        for t in coll:
            tid = getattr(t, "_id", None) or getattr(t, "name", None)
            if tid is None:
                continue
            tid = str(tid)
            if tid in seen:
                # gympn keeps actions separate from events, but a defensive
                # dedup guards against a transition appearing in both lists.
                continue
            seen.add(tid)
            transitions.append(TransitionInfo(
                id=tid, kind=kind,
                inputs=footprint(getattr(t, "incoming", [])),
                outputs=footprint(getattr(t, "outgoing", [])),
            ))
    return transitions, kept_place_ids, sorted(excluded), warnings


def conflicted_transition_ids(pn) -> set:
    """The set of ACTION transition ids that appear in some action-vs-action
    structural conflict (compete for a shared input place) — the
    foreclosure-CAPABLE decisions.

    This is the router for `lrq2c` (LINEAGE_SPARSE_CORRECTION.md §3): a decision
    is foreclosure-suspect only if its action can contend with another action for
    a resource. On DISJOINT envs this set is EMPTY → the sparse counterfactual
    correction never fires → pure lrq2 (grid stays perfect). On contested envs
    (s1: start1/start2 share the employee pool) it flags exactly those
    decisions. Reads only the PN formalism (shared input places) — no
    token-value semantics."""
    a = analyze(pn)
    ids = set()
    for x, y in a.action_conflict_edges:
        ids.add(x)
        ids.add(y)
    return ids


def analyze(
    pn,
    exclude_place_ids: Sequence[str] = (),
    include_clock: bool = False,
) -> StructuralAnalysis:
    """Extract the conflict and coupling structure of a PN.

    :param pn: a built ``GymProblem`` (or any simpn ``SimProblem`` with
        ``places`` and ``events``; ``actions`` optional).
    :param exclude_place_ids: place ids to treat as non-coupling (e.g. a global
        resource you deliberately want to ignore for the decomposition).
    :param include_clock: keep the ``SimVarTime`` clock place (default drops it).
    """
    transitions, place_ids, excluded, warnings = extract_transitions(
        pn, exclude_place_ids, include_clock)
    tids = [t.id for t in transitions]

    # place -> transitions that CONSUME it, and that TOUCH it
    consumers: Dict[str, List[str]] = {}
    touchers: Dict[str, List[str]] = {}
    for t in transitions:
        for pid in t.inputs:
            consumers.setdefault(pid, []).append(t.id)
        for pid in t.neighbourhood:
            touchers.setdefault(pid, []).append(t.id)

    # Conflict graph: an edge for every pair consuming a common input place.
    conflict_uf = _UnionFind(tids)
    conflict_edge_set: Set[Tuple[str, str]] = set()
    for pid, ts in consumers.items():
        if len(ts) < 2:
            continue
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = sorted((ts[i], ts[j]))
                conflict_edge_set.add((a, b))
                conflict_uf.union(a, b)

    # Coupling graph: an edge for every pair touching a common place (in or out).
    coupling_uf = _UnionFind(tids)
    for pid, ts in touchers.items():
        if len(ts) < 2:
            continue
        first = ts[0]
        for other in ts[1:]:
            coupling_uf.union(first, other)

    shared_input_places = {p: sorted(ts) for p, ts in consumers.items() if len(ts) > 1}
    coupling_places = {p: sorted(ts) for p, ts in touchers.items() if len(ts) > 1}

    # conflict clusters = components of the conflict graph, keeping only the
    # multi-transition ones (singletons are transitions with no competitor).
    conflict_clusters = [c for c in conflict_uf.components() if len(c) > 1]

    return StructuralAnalysis(
        transitions=transitions,
        place_ids=place_ids,
        excluded_place_ids=excluded,
        conflict_edges=sorted(conflict_edge_set),
        conflict_clusters=sorted(conflict_clusters, key=len, reverse=True),
        coupling_components=sorted(coupling_uf.components(), key=len, reverse=True),
        shared_input_places=shared_input_places,
        coupling_places=coupling_places,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_report(a: StructuralAnalysis, name: str = "") -> str:
    """Human-readable granularity report for the Direction B probe."""
    lines: List[str] = []
    head = f"Structural analysis{f' - {name}' if name else ''}"
    lines.append(head)
    lines.append("=" * len(head))
    lines.append(f"transitions : {a.n_transitions}  ({a.n_actions} actions, "
                 f"{a.n_transitions - a.n_actions} evolutions)")
    lines.append(f"places      : {len(a.place_ids)} kept"
                 + (f", excluded {a.excluded_place_ids}" if a.excluded_place_ids else ""))
    lines.append("")

    # Decomposition granularity
    lines.append("Independent subnets (coupling components):")
    lines.append(f"  count            : {a.n_components}")
    lines.append(f"  sizes            : {a.component_sizes}")
    lines.append(f"  largest fraction : {a.largest_component_fraction:.2f}  "
                 + ("(one coupled blob - decomposition buys little)"
                    if a.largest_component_fraction > 0.95
                    else "(genuine parallel structure)"))
    for i, comp in enumerate(a.coupling_components):
        acts = [t.id for t in a.transitions if t.id in set(comp) and t.kind == "action"]
        lines.append(f"    subnet {i}: {len(comp)} transitions"
                     + (f", actions={acts}" if acts else ", (no actions)"))
    lines.append("")

    # Decisions
    lines.append("Structural conflicts (competition for input tokens):")
    ace = a.action_conflict_edges
    lines.append(f"  action-vs-action conflict edges : {len(ace)}")
    for (x, y) in ace:
        shared = sorted(p for p, ts in a.shared_input_places.items()
                        if x in ts and y in ts)
        lines.append(f"    {x}  <->  {y}   over {shared}")
    other = [e for e in a.conflict_edges if e not in set(ace)]
    if other:
        lines.append(f"  other conflict edges            : {len(other)}")
    if a.shared_input_places:
        lines.append("  contested input places:")
        for p, ts in sorted(a.shared_input_places.items()):
            lines.append(f"    '{p}' consumed by {ts}")
    lines.append("")

    if a.warnings:
        lines.append("Warnings:")
        for w in a.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - smoke/demo on the E1 chain net
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "examples", "paper_examples", "suite"))
    from e1_chain_env import make_e1_chain  # type: ignore

    pn = make_e1_chain(causal_rl=False, allow_postpone=False)
    print(format_report(analyze(pn), name="E1 chain"))