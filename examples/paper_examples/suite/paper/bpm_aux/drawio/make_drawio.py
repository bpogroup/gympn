"""Experiment figures of the BPM paper, drawn as A-E PNs in the style of the
earlier A-E PN papers (draw.io: places as circles with token dots and
{attrs}@time annotations, transitions as bars tagged A or E, arc
inscriptions, guard/reward boxes, the network tag).

The figures are printed at the LNCS text width (about 12 cm), so the fonts are
large relative to the drawing: about 1000 units of width carry 14-16 pt text,
which prints at roughly 5 pt. To keep the nets that compact, a transition
without a box has guard None and reward r = 0; the definitions box of every
figure says so, and also defines the functions and delays used on the arcs. The
initial marking is shown in full. The nets follow suite/envs/insurer_env.py.

Writes one .drawio file per experiment next to this script; export.sh
renders them to PDF and PNG with the draw.io desktop app.
"""
import math
import os
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
INK = "#1a1a1a"
FONT = "fontFamily=Helvetica;"
NOTES = False     # prose notes and callouts: off, their content is in the captions
COLORS = {"claims": ("#EEF4FB", "#7FA7D9"), "underwriting": ("#EEF7EC", "#86BF7E"),
          "complaints": ("#FDF3E6", "#E3A85B")}
LABEL_COLORS = {"claims": "#4a6f9e", "underwriting": "#4f8a47", "complaints": "#b9742a"}
# BPI Challenge 2015: the municipalities that share an employee as claims/complaints, the others green
for _k, _like in (("m2", "claims"), ("m5", "complaints"), ("m1", "underwriting"), ("m3", "underwriting"),
                  ("m4", "underwriting")):
    COLORS[_k], LABEL_COLORS[_k] = COLORS[_like], LABEL_COLORS[_like]
# font sizes (drawing units); the figures are scaled to the text width
FL, FA, FARC, FB, FN, FF, FT = 16, 14, 14, 14, 14, 16, 12   # label, annotation, arc, box, note, frame, bar
R = 22            # place radius
BW, BH = 12, 46   # transition bar


def q(s):
    return escape(s, {'"': "&quot;"})


def tw(s, size):
    """Estimated width of a text in drawing units (Helvetica)."""
    plain = s.replace("<sub>", "").replace("</sub>", "").replace("<b>", "").replace("</b>", "")
    return len(plain) * size * 0.55


class Diagram:
    def __init__(self):
        self.cells = []
        self.n = 0

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def vertex(self, value, style, x, y, w, h):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{q(value)}" style="{style}{FONT}" parent="1" vertex="1">'
                          f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry" />'
                          f'</mxCell>')
        return i

    # ------------------------------------------------------------ A-E PN
    def place(self, x, y, label=None, lpos="below", tokens=(), r=R):
        """A place centred at (x, y); tokens = [(annotation, (dx, dy)[, anchor])],
        the annotation's anchor point relative to the place centre (anchor
        defaults to right of the point for dx < 0, left otherwise)."""
        pid = self.vertex("", f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffffff;strokeColor={INK};"
                              "strokeWidth=1.5;", x - r, y - r, 2 * r, 2 * r)
        if label:
            g = FL * 0.75
            dx, dy, anchor = {"below": (0, r + g, "center"), "above": (0, -r - g, "center"),
                              "left": (-r - 5, 0, "right"), "right": (r + 5, 0, "left"),
                              "ne": (r - 2, -r - 6, "left"), "nw": (-r + 2, -r - 6, "right"),
                              "sw": (-r + 4, r + g, "right"), "se": (r - 4, r + g, "left")}[lpos]
            self.text(label, x + dx, y + dy, anchor)
        offs = {1: [(0, 0)], 2: [(-6, -6), (6, 6)], 3: [(-8, -5), (7, -7), (0, 8)],
                4: [(-7, -7), (7, -7), (-7, 7), (7, 7)],
                6: [(-11, -6), (0, -6), (11, -6), (-11, 6), (0, 6), (11, 6)]}.get(len(tokens), [])
        for tok, (ox, oy) in zip(tokens, offs):
            ann, (ax, ay) = tok[0], tok[1]
            anchor = tok[2] if len(tok) > 2 else ("right" if ax < 0 else "left")
            self.vertex("", "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#330000;strokeColor=#330000;",
                        x + ox - 5, y + oy - 5, 10, 10)
            tx, ty = x + ax, y + ay
            x0, w = self.text(ann, tx, ty, anchor, size=FA)
            px, py = x + ox, y + oy            # leader to the nearest point of the text
            nx = min(max(px, x0 + 3), x0 + w - 3)
            ny = min(max(py, ty - FA * 0.4), ty + FA * 0.4)
            self.leader((px, py), (nx, ny))
        return pid, (x, y)

    def leader(self, p, q_):
        i = self._id()
        self.cells.append(
            f'<mxCell id="{i}" style="endArrow=none;dashed=1;html=1;dashPattern=1 2;strokeWidth=1.2;rounded=0;'
            f'strokeColor=#555555;" parent="1" edge="1"><mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{p[0]:.1f}" y="{p[1]:.1f}" as="sourcePoint" />'
            f'<mxPoint x="{q_[0]:.1f}" y="{q_[1]:.1f}" as="targetPoint" /></mxGeometry></mxCell>')

    def trans(self, x, y, kind, label=None, lpos="below", dashed=False):
        style = (f"rounded=0;whiteSpace=wrap;html=1;fontSize={FT};strokeColor={INK};strokeWidth=1.5;"
                 "fillColor=#ffffff;spacing=0;" + ("dashed=1;dashPattern=3 2;" if dashed else ""))
        tid = self.vertex(kind, style, x - BW / 2, y - BH / 2, BW, BH)
        if label:
            g = BH / 2 + FL * 0.75
            dx, dy, anchor = {"below": (0, g, "center"), "above": (0, -g, "center"),
                              "left": (-BW / 2 - 4, 0, "right"), "right": (BW / 2 + 4, 0, "left"),
                              "se": (BW / 2 + 1, g, "left"), "sw": (-BW / 2 - 1, g, "right")}[lpos]
            self.text(label, x + dx, y + dy, anchor)
        return tid, (x, y)

    def text(self, value, x, y, anchor="center", size=FL, bold=False, w=None, color=INK):
        w = w or max(16, int(tw(value, size)) + 8)
        h = size + 8
        x0 = {"center": x - w / 2, "left": x, "right": x - w}[anchor]
        style = (f"text;html=1;align={'center' if anchor == 'center' else anchor};verticalAlign=middle;"
                 f"resizable=0;points=[];strokeColor=none;fillColor=none;fontSize={size};fontColor={color};"
                 "spacing=0;" + ("fontStyle=1;" if bold else ""))
        self.vertex(value, style, x0, y - h / 2, w, h)
        return x0, w

    def arc(self, src, tgt, label="", points=(), exit=None, entry=None, lab_pos=0.0, lab_off=(0, 0),
            straight=False, dashed=False, color=INK):
        i = self._id()
        st = ("edgeStyle=none;" if straight else "edgeStyle=orthogonalEdgeStyle;orthogonalLoop=1;")
        st += (f"rounded=0;jettySize=auto;html=1;fontSize={FARC};endArrow=block;endFill=1;endSize=7;"
               f"strokeColor={color};strokeWidth=1.3;jumpStyle=arc;jumpSize=9;")
        if exit:
            st += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
        if entry:
            st += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        if dashed:
            st += "dashed=1;"
        pts = "".join(f'<mxPoint x="{px:.1f}" y="{py:.1f}" />' for px, py in points)
        pts = f'<Array as="points">{pts}</Array>' if points else ""
        self.cells.append(f'<mxCell id="{i}" style="{st}" parent="1" source="{src[0]}" target="{tgt[0]}" '
                          f'edge="1"><mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')
        if label:
            j = self._id()
            self.cells.append(
                f'<mxCell id="{j}" value="{q(label)}" style="edgeLabel;html=1;align=center;verticalAlign=middle;'
                f'resizable=0;points=[];fontSize={FARC};labelBackgroundColor=#ffffff;{FONT}" parent="{i}" '
                f'vertex="1" connectable="0"><mxGeometry x="{lab_pos}" relative="1" as="geometry">'
                f'<mxPoint x="{lab_off[0]}" y="{lab_off[1]}" as="offset" /></mxGeometry></mxCell>')
        return i

    def box(self, x, y, guard="None", reward="r = 0", w=200, attach=None, lines=None):
        """Guard/reward box with its top-left corner at (x, y), joined to `attach`
        by a dotted leader to the nearest point of the box. Rows grow with their
        text; `lines` = (guard lines, reward lines) overrides the estimate."""
        yy = y
        for k, (b, t) in enumerate((("Guard Function:", guard), ("Reward Function:", reward))):
            n = lines[k] if lines else max(1, math.ceil((tw(b, FB) * 1.08 + tw(" " + t, FB)) / (w - 12)))
            h = FB * 1.3 * n + 8
            self.vertex(f"<b>{b}</b> {q(t)}",
                        "text;strokeColor=#666666;fillColor=#ffffff;align=left;verticalAlign=middle;spacingLeft=5;"
                        f"spacingRight=4;overflow=hidden;whiteSpace=wrap;html=1;fontSize={FB};", x, yy, w, h)
            yy += h
        if attach:
            ax, ay = attach
            nx = min(max(ax, x), x + w)
            ny = min(max(ay, y), yy)
            self.leader(attach, (nx, ny))
        return yy

    def note(self, x, y, w, html, size=FN, lines=None):
        if not NOTES:
            return y
        lines = lines or html.count("<br>") + 1
        h = lines * size * 1.3 + 14
        self.vertex(html, f"text;strokeColor=#666666;fillColor=#ffffff;align=left;verticalAlign=top;spacingLeft=7;"
                          f"spacingRight=7;spacingTop=4;whiteSpace=wrap;html=1;fontSize={size};",
                    x, y, w, h)
        return y + h

    def frame(self, x, y, w, h, label=None, kind=None, dashed=False):
        fill, stroke = COLORS.get(kind, ("none", "#555555"))
        style = (f"rounded=1;arcSize={max(1, int(1200 / max(w, h)))};whiteSpace=wrap;html=1;fillColor={fill};"
                 f"strokeColor={stroke};strokeWidth=1.5;" + ("dashed=1;dashPattern=6 4;" if dashed else ""))
        self.vertex("", style, x, y, w, h)
        if label:
            self.text(label, x + 10, y + FF * 0.95, "left", size=FF, bold=True,
                      color=LABEL_COLORS.get(kind, INK))

    def tag(self, x, y):
        self.vertex(r"$$(E, \tau=0, \rho=0)$$", "text;html=1;strokeColor=none;fillColor=none;align=left;"
                    "verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=20;", x, y, 240, 34)

    def save(self, name):
        # frames first, so they are drawn behind the net
        cells = sorted(self.cells, key=lambda c: 0 if "rounded=1;arcSize" in c else 1)
        xml = ('<mxfile host="Electron"><diagram id="d" name="Page-1"><mxGraphModel grid="0" gridSize="10" '
               'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" math="1" '
               'shadow="0"><root><mxCell id="0" /><mxCell id="1" parent="0" />\n' + "\n".join(cells) +
               '\n</root></mxGraphModel></diagram></mxfile>')
        with open(os.path.join(HERE, name + ".drawio"), "w", encoding="utf-8") as f:
            f.write(xml)


# ------------------------------------------------------------------ processes
TITLES = {"claims": "Claims", "underwriting": "Underwriting", "complaints": "Complaints"}
RES = {"claims": "Clerks", "underwriting": "Underwriters", "complaints": "Clerks"}
ACTIONS = {"claims": ("Approve", "Investigate"), "complaints": ("Resolve", "Check")}
WAITING = {"claims": ["{risk=0, bad=0}@0", "{risk=1, bad=0}@0", "{risk=2, bad=1}@0"],
           "complaints": ["{risk=0, bad=0, reworked=0}@0", "{risk=1, bad=0, reworked=0}@0",
                          "{risk=2, bad=1, reworked=0}@0"],
           "underwriting": ["{task_type=0}@0", "{task_type=1}@0"]}
ARRIVAL = {"claims": "{risk=0, bad=0}@0", "complaints": "{risk=0, bad=0, reworked=0}@0",
           "underwriting": "{task_type=0}@0"}
DONE_REWARD = {"claims": "r = 1 − X.bad if approved, 1 if investigated",
               "underwriting": "r = 1",
               "complaints": "r = 1 − X.bad if resolved, 1 if checked"}
RES_TOKENS = {"claims": ["{skill=0}@0", "{skill=1}@0"], "underwriting": ["{code=0}@0", "{code=1}@0"],
              "complaints": ["{skill=0}@0", "{skill=1}@0"]}
# x positions within a row
XA, XE, XW, XT, XB, XD = 40, 150, 300, 480, 640, 765
YR = -155          # resource place, relative to the row's centre line
TOP = YR - 75      # frame top, relative to the row's centre line
LEFT = -55         # frame left
ROW_W = 1068       # frame width
DONE_W = 215       # Done reward box width


def process_row(d, y, kind, res_tokens=None, res_lpos="above"):
    """One process, left to right, centred on height y. Returns (resource place, bottom)."""
    arr = d.place(XA, y, "Arrival", lpos="left",
                  tokens=[(ARRIVAL[kind], (-62, R + 30), "left")])
    ev = d.trans(XE, y, "E", "Arrive")
    d.arc(arr, ev, "X", exit=(1, 0.75), entry=(0, 0.75))
    d.arc(ev, arr, "X@+1", exit=(0, 0.25), entry=(1, 0.25))
    wt_tokens = WAITING[kind]
    n = len(wt_tokens)
    anns = [(-40 + 9 * (n - 1 - k), -40 - 19 * (n - 1 - k)) for k in range(n)]
    wt = d.place(XW, y, "Waiting", lpos="sw" if kind == "complaints" else "below",
                 tokens=list(zip(wt_tokens, anns)))
    d.arc(ev, wt, "new()", exit=(1, 0.5), entry=(0, 0.5))
    rt = res_tokens or RES_TOKENS[kind]
    res = d.place(XT, y + YR, RES[kind], lpos=res_lpos,
                  tokens=list(zip(rt, [(-32, -13), (-32, 13)] if len(rt) == 2 else [(-32, 0)])))
    busy = d.place(XB, y, "Busy")
    if kind == "underwriting":
        a = d.trans(XT, y, "A", "Assign")
        d.arc(wt, a, "X")
        d.arc(res, a, "Y")
        d.arc(a, busy, "(X;Y)@+d(X,Y)", lab_off=(0, -14))
        low = y + BH / 2 + FL * 1.3
    else:
        n1, n2 = ACTIONS[kind]
        a1 = d.trans(XT, y - 58, "A")
        d.text(n1, XT - BW / 2 - 4, y - 58 - BH / 2 - 4, "right")
        a2 = d.trans(XT, y + 58, "A", n2)
        d.arc(wt, a1, "X", straight=True, entry=(0, 0.5), lab_off=(-6, -9))
        d.arc(wt, a2, "X", straight=True, entry=(0, 0.5), lab_off=(-6, 9))
        d.arc(res, a1, "Y", exit=(0.5, 1), entry=(0.5, 0))
        d.arc(res, a2, "Y", exit=(1, 0.5), entry=(0.5, 0),
              points=[(XT + 42, y + YR), (XT + 42, y + 14), (XT, y + 14)], lab_pos=-0.35)
        d.arc(a1, busy, "(X;Y)@+d<sub>f</sub>", straight=True, exit=(1, 0.5), lab_off=(10, -11))
        d.arc(a2, busy, "(X;Y)@+d<sub>s</sub>(Y)", straight=True, exit=(1, 0.5), lab_off=(14, 12))
        low = y + 58 + BH / 2 + FL * 1.3
    done = d.trans(XD, y, "E", "Done", lpos="sw")
    d.arc(busy, done, "(X;Y)")
    d.arc(done, res, "Y", exit=(0.5, 0), entry=(1, 0.5), points=[(XD, y + YR)])
    by = d.box(XD + 10, y + BH / 2 + 8, "None", DONE_REWARD[kind], w=DONE_W, attach=(XD + BW / 2, y + BH / 2 - 4),
               lines=(1, 1 if kind == "underwriting" else 2))
    bottom = max(low, by, y + R + 30 + FA) + 14
    if kind == "complaints":
        d.arc(done, wt, "X'@+2 if resolved and X.bad", exit=(0.5, 1), entry=(0.5, 1),
              points=[(XD, bottom), (XW, bottom)], lab_pos=0.15)
        bottom += 16
    return res, bottom


DEFAULTS = "A transition without a box has guard None and reward r = 0."


def defs_insurer(d, x, y, w, slow=False, alloc=False):
    rows = [
        "<b>Definitions.</b> " + DEFAULTS,
        "new(): a fresh case. Claims and complaints: risk ~ U{0,1,2}, bad ~ Bernoulli(p<sub>risk</sub>) with "
        "p = (0.1, 0.5, 0.9), and reworked = 0 for complaints; bad is not observable. "
        "Underwriting: task_type ~ U{0,1}.",
        "d<sub>f</sub> = 1 + U{0,1} (approve, resolve); d<sub>s</sub>(Y) = 4 + U{0,1} for a normal clerk "
        "(skill = 0) and 2 + U{0,1} for an expert (skill = 1) (investigate, check).",
        f"d(X,Y) = 1 + U{{0,1}} if X.task_type = Y.code, else {7 if slow else 3} + U{{0,1}}.",
        "X' = X with reworked = 1 and bad = 0: a wrongly resolved complaint reopens after 2 time units.",
    ]
    if alloc:
        rows.append("S: an employee of the staff pool. To claims or complaints, it joins the clerks with its skill "
                    "(expert: skill = 1); to underwriting, it becomes an underwriter for task type c ({code=c}).")
    # each row wraps at the box width
    lines = sum(max(1, math.ceil(tw(r_, FN) / (w - 20))) for r_ in rows)
    return d.note(x, y, w, "<br>".join(rows), lines=lines)


ROW_GAP = 22


def stack_rows(d, kinds, title, res_tokens=None, res_lpos="above"):
    """Process rows one below the other, each in its component frame. Returns
    (resource places, row centre lines, bottom of the last frame)."""
    res, ys = {}, {}
    y = 0
    for k, kind in enumerate(kinds):
        top = y + TOP
        res[kind], bottom = process_row(d, y, kind, (res_tokens or {}).get(kind), res_lpos)
        d.frame(LEFT, top, ROW_W, bottom - top, title(k, kind), kind=kind)
        ys[kind] = y
        y = bottom + ROW_GAP - TOP
    return res, ys, bottom


def fig_insurer():
    d = Diagram()
    d.tag(LEFT - 5, TOP - 44)
    _, _, bottom = stack_rows(d, ("claims", "underwriting", "complaints"),
                              lambda k, kind: TITLES[kind])
    defs_insurer(d, LEFT, bottom + ROW_GAP, ROW_W)
    d.save("fig_insurer")


def fig_alloc():
    """Start-of-shift allocation. A Staff pool feeds the three resource
    places; while Staff is marked the region is one component, afterwards three."""
    d = Diagram()
    toks = {"claims": ["{skill=0}@0"], "underwriting": ["{code=0}@0"], "complaints": ["{skill=0}@0"]}
    res, ys, bottom = stack_rows(d, ("claims", "underwriting", "complaints"),
                                 lambda k, kind: TITLES[kind],
                                 toks, res_lpos="nw")
    AX, SX = LEFT + ROW_W + 75, LEFT + ROW_W + 245
    ry = {k: ys[k] + YR for k in ys}
    sy = (ry["claims"] + ry["complaints"]) / 2
    staff = d.place(SX, sy, tokens=[("{skill=1}@0", (34, -19)), ("{skill=1}@0", (38, 0)),
                                    ("{skill=0}@0", (34, 19))])
    d.text("Staff", SX + 20, sy - R - FL * 0.7)
    # (label, process, output inscription, transition height, route into the resource place)
    alloc = [("To claims", "claims", "{skill=S.skill}", ry["claims"] - 42, "top"),
             ("To UW, code 0", "underwriting", "{code=0}", ry["underwriting"] - 42, "top"),
             ("To UW, code 1", "underwriting", "{code=1}", ry["underwriting"] + 62, "below"),
             ("To complaints", "complaints", "{skill=S.skill}", ry["complaints"] - 42, "top")]
    for name, kind, out, ty, route in alloc:
        t = d.trans(AX, ty, "A", name, lpos="sw" if kind == "claims" else "below")
        d.arc(staff, t, "S", straight=True, entry=(1, 0.5), lab_pos=0.35)
        if route == "top":
            d.arc(t, res[kind], out, exit=(0, 0.5), entry=(0.5, 0), points=[(XT, ty)], lab_pos=0.3)
        else:
            d.arc(t, res[kind], out, exit=(0, 0.5), entry=(0.85, 0.85), points=[(XT + 16, ty)], lab_pos=0.3)
    top = TOP - 56
    right = SX + 38 + tw("{skill=1}@0", FA) + 18
    d.frame(LEFT - 15, top, right - LEFT + 15, bottom + 15 - top, "Region",
            dashed=True)
    d.tag(LEFT - 20, top - 44)
    defs_insurer(d, LEFT - 15, bottom + 15 + ROW_GAP, right - LEFT + 15, alloc=True)
    d.save("fig_alloc")


def fig_slow():
    """The slow-mismatch insurer: underwriting in full with its wait
    pseudo-binding; claims and complaints as in the insurer figure, each with its own wait."""
    d = Diagram()
    d.tag(LEFT - 5, TOP - 44)
    _, bottom = process_row(d, 0, "underwriting")
    W = ROW_W
    wx, wy = LEFT + W - 30, YR + 5
    d.trans(wx, wy, "A", dashed=True)
    d.text("wait<sub>2</sub>", wx - BW / 2 - 6, wy, "right")
    nx = XD + 22
    d.note(nx, wy + BH / 2 + 12, LEFT + W - 12 - nx, "<b>Selecting wait<sub>2</sub>:</b> underwriting offers "
                                                   "no binding until its own Arrive or Done fires.", lines=3)
    d.frame(LEFT, TOP, W, bottom - TOP, "Underwriting", kind="underwriting")
    by = bottom + ROW_GAP
    half = (W - ROW_GAP) / 2
    for k, (kind, wn, n) in enumerate((("claims", "wait<sub>1</sub>", 1), ("complaints", "wait<sub>3</sub>", 3))):
        x = LEFT + k * (half + ROW_GAP)
        d.frame(x, by, half, 74, TITLES[kind], kind=kind)
        d.trans(x + half - 24, by + 40, "A", dashed=True)
        d.text(wn, x + half - 36, by + 40, "right")
    defs_insurer(d, LEFT, by + 74 + ROW_GAP, W, slow=True)
    d.save("fig_slow")


# ------------------------------------------------------------- observation
ON = 13   # node font


def node(d, x, y, typ, attr, shade=None, r=54):
    lab = f"<b>{q(typ)}</b><br>{q(attr).replace(chr(10), '<br>')}" if attr else f"<b>{q(typ)}</b>"
    fill = {"fwd": "#9FB7D9", "whole": "#DCE6F3", None: "#ffffff"}[shade]
    st = (f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fontSize={ON};fillColor={fill};strokeColor={INK};"
          "strokeWidth=1.4;spacing=2;")
    return d.vertex(lab, st, x - r, y - r, 2 * r, 2 * r), (x, y)


def fig_observation():
    """The observation of one region at the first allocation decision.
    Token nodes are typed by their place, binding nodes are A_Transition, event
    nodes E_Transition; edges follow the arcs. Dark: what can reach the probability
    of 'To UW, code 1' if messages only follow the arcs; light: what the
    component-scoped observation adds (explained in the caption)."""
    d = Diagram()
    C = [0, 140, 280, 420, 560, 700]
    rows = {"claims": 0, "underwriting": 250, "complaints": 500}
    staff = node(d, C[0], 250, "Staff", "skill=1", "fwd")
    al = {"claims": node(d, C[1], 0, "A_Transition", "To claims", "whole"),
          "uw0": node(d, C[1], 188, "A_Transition", "To UW, code 0", "whole"),
          "uw1": node(d, C[1], 312, "A_Transition", "To UW, code 1", "fwd"),
          "complaints": node(d, C[1], 500, "A_Transition", "To complaints", "whole")}
    res = {"claims": node(d, C[2], 0, "Clerks (cl.)", "skill=0", "whole"),
           "underwriting": node(d, C[2], 250, "Underwriters", "code=0", "whole"),
           "complaints": node(d, C[2], 500, "Clerks (co.)", "skill=0", "whole")}
    for k in ("claims", "uw0", "uw1", "complaints"):
        d.arc(staff, al[k], straight=True)
    d.arc(al["claims"], res["claims"], straight=True)
    d.arc(al["uw0"], res["underwriting"], straight=True)
    d.arc(al["uw1"], res["underwriting"], straight=True)
    d.arc(al["complaints"], res["complaints"], straight=True)
    acts = {"claims": ("Approve", "Investigate"), "complaints": ("Resolve", "Check")}
    waits = {"claims": "risk=2", "complaints": "risk=2,\nreworked=0"}
    for kind, y in rows.items():
        if kind == "underwriting":
            b0 = node(d, C[3], y - 62, "A_Transition", "Assign", "whole")
            b1 = node(d, C[3], y + 62, "A_Transition", "Assign", "whole")
            w0 = node(d, C[4], y - 62, "Waiting (uw.)", "task_type=0", "whole")
            w1 = node(d, C[4], y + 62, "Waiting (uw.)", "task_type=1", "whole")
            for b, w in ((b0, w0), (b1, w1)):
                d.arc(res[kind], b, straight=True)
                d.arc(w, b, straight=True)
            ws = [w0, w1]
        else:
            b0 = node(d, C[3], y - 62, "A_Transition", acts[kind][0], "whole")
            b1 = node(d, C[3], y + 62, "A_Transition", acts[kind][1], "whole")
            w = node(d, C[4], y, f"Waiting ({'cl' if kind == 'claims' else 'co'}.)", waits[kind], "whole")
            for b in (b0, b1):
                d.arc(res[kind], b, straight=True)
                d.arc(w, b, straight=True)
            ws = [w]
        ea = node(d, C[5], y, "E_Transition", "Arrive", "whole")
        for w in ws:
            d.arc(ea, w, straight=True)
        ed = node(d, C[2], y - 124 if kind != "complaints" else y + 124, "E_Transition", "Done", "whole")
        d.arc(ed, res[kind], straight=True)
    # legend: the caption explains it in full
    ly = 690
    d.vertex("", "rounded=0;whiteSpace=wrap;html=1;fillColor=#9FB7D9;strokeColor=#1a1a1a;", -50, ly - 9, 26, 18)
    d.text("reaches <i>To UW, code 1</i> along the arcs", -16, ly, "left", size=FA)
    d.vertex("", "rounded=0;whiteSpace=wrap;html=1;fillColor=#DCE6F3;strokeColor=#1a1a1a;", 330, ly - 9, 26, 18)
    d.text("added by the component-scoped observation", 364, ly, "left", size=FA)
    d.save("fig_observation")


# ------------------------------------------------------- BPI Challenge 2015
# Staff of the 2014 roster (suite/envs/bpic15_env.py): the task groups each employee fits.
BPIC_STAFF = {2: ["AD", "A", "AD", "I"], 5: ["A", "AD", "A", "A", "AD", "IAD"]}


def _fits(groups):
    return "{" + ", ".join(f"f{g}={int(g in groups)}" for g in "IAD") + "}@0"


def municipality_row(d, y, m):
    """One municipality, drawn like an insurer process: the three task groups
    are one Start/End pair, with the task group as the token color `stage`
    (behaviourally the same as the implementation's places per task group).
    Returns (staff place, bottom)."""
    n = len(BPIC_STAFF[m])
    arr = d.place(XA, y, "Arrival", lpos="left", tokens=[("{stage=I}@0", (-62, R + 30), "left")])
    ev = d.trans(XE, y, "E", "Arrive")
    d.arc(arr, ev, "X", exit=(1, 0.75), entry=(0, 0.75))
    d.arc(ev, arr, "X@+e", exit=(0, 0.25), entry=(1, 0.25))
    wt = d.place(XW, y, "Waiting", lpos="above")
    d.arc(ev, wt, "X", exit=(1, 0.5), entry=(0, 0.5))
    yb = y + 125
    back = d.place(XA, yb, "Backlog", lpos="left",
                   tokens=[(f"{{stage=I}}@0 ×{n}, {{stage=A}}@0 ×{2 * n}, {{stage=D}}@0 ×{n}", (-62, R + 22), "left")])
    op = d.trans(XE, yb, "E", "Open", lpos="above")
    d.arc(back, op, "X")
    d.arc(op, wt, "o(X)", exit=(1, 0.5), entry=(0.25, 1), points=[(XW - R / 2, yb)], lab_pos=-0.2)
    staff = d.place(XT, y + YR, tokens=list(zip(
        [_fits(f) for f in BPIC_STAFF[m]],
        [(-R - 18, (k - (n - 1) / 2) * 19) for k in range(n)])))
    d.text("Staff", XT + R + 4, y + YR + R + 6, "left")
    st = d.trans(XT, y, "A", "Start")
    busy = d.place(XB, y, "Busy")
    d.arc(wt, st, "X")
    d.arc(staff, st, "Y")
    d.arc(st, busy, "(X;Y)@+d(X,Y)", lab_off=(0, -14))
    end = d.trans(XD, y, "E", "End", lpos="sw")
    d.arc(busy, end, "(X;Y)")
    d.arc(end, staff, "Y", exit=(0.5, 0), entry=(1, 0.5), points=[(XD, y + YR)])
    by = d.box(XD + 10, y + BH / 2 + 8, "None", "r = 1 if X.stage = D", w=DONE_W, attach=(XD + BW / 2, y + BH / 2 - 4),
               lines=(1, 2))
    bottom = max(by, yb + R + 22 + FA) + 14
    d.arc(end, wt, "next(X) if X.stage ≠ D", exit=(0.5, 1), entry=(0.75, 1),
          points=[(XD, bottom), (XW + R / 2, bottom)], lab_pos=0.15)
    return staff, bottom + 16


def fig_bpic15():
    """The BPI Challenge 2015 municipalities (suite/envs/bpic15_env.py, roster
    2014): municipalities 2 and 5 in full with the pool of the employee they
    share, municipalities 1, 3, 4 collapsed. Static certificate {1}, {2, 5},
    {3}, {4}; once the pool is empty, one component per municipality."""
    d = Diagram()
    TOPM = YR - 75                             # frame top, relative to a row's centre line
    res, ys, y = {}, {}, 0
    for k, m in enumerate((2, 5)):
        top = y + TOPM
        res[m], bottom = municipality_row(d, y, m)
        d.frame(LEFT, top, ROW_W, bottom - top, f"Municipality {m}", kind=f"m{m}")
        ys[m] = y
        y = bottom + ROW_GAP - TOPM
    # the pool of the shared employee, right of the two rows
    AX, SX = LEFT + ROW_W + 75, LEFT + ROW_W + 215
    ry = {m: ys[m] + YR for m in ys}
    sy = (ry[2] + ry[5]) / 2
    pool = d.place(SX, sy, tokens=[(_fits("IAD"), (R + 10, 0), "left")])
    d.text("Pool", SX, sy + R + FL * 0.75)
    for m in (2, 5):
        ty = ry[m] - 42
        t = d.trans(AX, ty, "A", f"To M{m}", lpos="below")
        d.arc(pool, t, "S", straight=True, entry=(1, 0.5), lab_pos=0.35)
        d.arc(t, res[m], "S", exit=(0, 0.5), entry=(0.5, 0), points=[(XT, ty)], lab_pos=0.3)
    top = TOPM - 56
    right = SX + R + 10 + tw(_fits("IAD"), FA) + 18
    d.frame(LEFT - 15, top, right - LEFT + 15, bottom + 15 - top,
            "Municipalities 2 and 5", dashed=True)
    d.tag(LEFT - 20, top - 44)
    # municipalities 1, 3, 4
    W = right - LEFT + 15
    RY, RH = bottom + 15 + ROW_GAP, 74
    third = (W - 2 * ROW_GAP) / 3
    for k, m in enumerate((1, 3, 4)):
        x = LEFT - 15 + k * (third + ROW_GAP)
        d.frame(x, RY, third, RH, f"Municipality {m}", kind=f"m{m}")
    rows = ["<b>Definitions.</b> " + DEFAULTS + " Task groups: I (intake), A (assessment), D (decision).",
            "e: an exponential delay with the arrival rate of the municipality. o(X): with probability 1/2, a "
            "permit {stage=X.stage}, else no token (the backlog at the start of the day). "
            "next(X): X with the next task group (I to A, A to D).",
            "d(X,Y) = b + U{0,1} if Y fits X.stage (f<sub>X.stage</sub> = 1), else 2b + U{0,1}, with b = 2, 4, 2 "
            "for I, A, D.",
            "Staff and Pool tokens: the task groups f<sub>I</sub>, f<sub>A</sub>, f<sub>D</sub> that an employee "
            "fits. S: the shared employee; sent to a municipality, it joins its staff for the rest of the day."]
    lines = sum(max(1, math.ceil(tw(r_, FN) / (W - 20))) for r_ in rows)
    d.note(LEFT - 15, RY + RH + ROW_GAP, W, "<br>".join(rows), lines=lines)
    d.save("fig_bpic15")


# ------------------------------------------------------------ postponement
# Three executions of the same situation on a time line: the network tag on
# top, one lane per component below. At tau = 0, claims has a case and a free
# clerk; underwriting has a task of type 1, a free underwriter of code 0, and
# the underwriter of code 1 returns (Done) at tau = 2; a claim arrives at tau = 1.
PX = {0: 150, 1: 470, 2: 790}   # x of tau = 0, 1, 2
PW_A = 150                       # width of a decision round (zero model time)
PLANE_H = 52
PTAG_H = 22
GREY = "#e8e8e8"


def decision(d, x, y, label, dashed=False):
    w = tw(label, FA) + 14
    st = (f"rounded=1;arcSize=30;whiteSpace=wrap;html=1;fontSize={FA};strokeColor={INK};strokeWidth=1.3;"
          "fillColor=#ffffff;spacing=1;" + ("dashed=1;dashPattern=3 2;" if dashed else ""))
    d.vertex(label, st, x, y - 13, w, 26)
    return x + w


def frozen(d, x0, x1, y, label):
    d.vertex(label, f"rounded=0;whiteSpace=wrap;html=1;fontSize={FA};fontColor=#555555;strokeColor=#888888;"
                    f"strokeWidth=1;dashed=1;dashPattern=2 2;fillColor={GREY};", x0, y - 17, x1 - x0, 34)


def postpone_panel(d, yp, title, rounds, claims, uw, note):
    """One execution. rounds: the tau at which a decision round takes place;
    claims / uw: callables that draw the lane content at (lane centre y)."""
    left, right = PX[0] - 118, PX[2] + PW_A + 40
    d.text(title, left + 2, yp + 10, "left", size=FF, bold=True)
    # network tag band
    ty = yp + 34
    d.text("tag", left + 2, ty + PTAG_H / 2, "left", size=FA)
    x = PX[0]
    for k, tau in enumerate((0, 1, 2)):
        if tau in rounds:
            d.vertex("A", f"rounded=0;whiteSpace=wrap;html=1;fontSize={FT};strokeColor={INK};fillColor=#ffffff;",
                     x, ty, PW_A, PTAG_H)
            x += PW_A
        nxt = PX[tau + 1] if tau < 2 else right - 4
        if tau < 2:
            d.vertex("E", f"rounded=0;whiteSpace=wrap;html=1;fontSize={FT};strokeColor={INK};fillColor={GREY};",
                     x, ty, nxt - x, PTAG_H)
        x = nxt
    # lanes
    for k, (kind, draw) in enumerate((("claims", claims), ("underwriting", uw))):
        top = ty + PTAG_H + 10 + k * (PLANE_H + 8)
        d.frame(left, top, right - left, PLANE_H, kind=kind)
        d.text(TITLES[kind], left + 8, top + PLANE_H / 2, "left", size=FA, bold=True,
               color=LABEL_COLORS[kind])
        draw(top + PLANE_H / 2)
    # time axis
    ay = ty + PTAG_H + 10 + 2 * (PLANE_H + 8) + 8
    i = d._id()
    d.cells.append(f'<mxCell id="{i}" style="endArrow=block;endFill=1;endSize=6;html=1;strokeColor={INK};'
                   f'strokeWidth=1.2;" parent="1" edge="1"><mxGeometry relative="1" as="geometry">'
                   f'<mxPoint x="{PX[0] - 10}" y="{ay}" as="sourcePoint" />'
                   f'<mxPoint x="{right}" y="{ay}" as="targetPoint" /></mxGeometry></mxCell>')
    for tau in (0, 1, 2):
        d.leader((PX[tau], ay - 5), (PX[tau], ay + 5))
        d.text(f"τ = {tau}", PX[tau], ay + 14, size=FA)
    d.note(left, ay + 30, right - left, note, lines=1)
    return ay + 30 + FN * 1.3 + 14


def evo(d, x, y, label):
    d.trans(x, y, "E", dashed=False)
    d.text(label, x - BW / 2 - 4, y, "right", size=FA)


def wait(d, x, y, label):
    d.trans(x, y, "A", dashed=True)
    d.text(label, x + BW / 2 + 4, y, "left", size=FA)
    return x + BW / 2 + 4 + tw(label, FA) + 6


def fig_postpone():
    """Three executions of the same situation: the firing rule, global postponement,
    and component-scoped postponement. Gray spans: no binding of the component is
    offered; dashed pills: the postpone pseudo-binding."""
    d = Diagram()
    GR = "#777777"
    x0, x1, x2 = 230, 510, 790
    ticks = [(x0, "τ = 0"), (x1, "τ = 1"), (x2, "τ = 2")]
    lanes = (("claims", "Claims"), ("underwriting", "Underwriting"))
    yp = 0
    # (a) the firing rule: a binding must be selected while one is enabled
    tops, yp = _credit_panel(d, yp, "(a) Firing rule of the A-E PN", lanes, ticks, "")
    yc, yu = tops["Claims"], tops["Underwriting"]
    _pill(d, x0 + 46, yc, "Approve")
    evo(d, x1, yc, "Arrive")
    _pill(d, x1 + 60, yc, "Approve")
    _pill(d, x0 + 70, yu, "Assign code 0")
    evo(d, x2, yu, "Done")
    # (b) global postponement: wait freezes the whole net
    tops, yp = _credit_panel(d, yp + 16, "(b) Global postponement", lanes, ticks, "")
    yc, yu = tops["Claims"], tops["Underwriting"]
    _span(d, x0 + 4, x1 - 22, yc, CLANE - 8, GR)
    evo(d, x1, yc, "Arrive")
    _pill(d, x1 + 60, yc, "Approve")
    _pill(d, x1 + 152, yc, "Approve")
    _pill(d, x0 + 34, yu, "wait", dashed=True)
    _span(d, x0 + 70, x1 - 22, yu, CLANE - 8, GR)
    _pill(d, x1 + 34, yu, "wait", dashed=True)
    _span(d, x1 + 70, x2 - 22, yu, CLANE - 8, GR)
    evo(d, x2, yu, "Done")
    _pill(d, x2 + 80, yu, "Assign code 1")
    # (c) component-scoped postponement: wait_2 freezes underwriting only
    tops, yp = _credit_panel(d, yp + 16, "(c) Component-scoped postponement", lanes, ticks, "")
    yc, yu = tops["Claims"], tops["Underwriting"]
    _pill(d, x0 + 46, yc, "Approve")
    evo(d, x1, yc, "Arrive")
    _pill(d, x1 + 60, yc, "Approve")
    _pill(d, x0 + 36, yu, "wait<sub>2</sub>", dashed=True)
    _span(d, x0 + 74, x2 - 22, yu, CLANE - 8, GR)
    evo(d, x2, yu, "Done")
    _pill(d, x2 + 80, yu, "Assign code 1")
    d.save("fig_postpone")


# ------------------------------------------------------------------- A-E PN
def fig_aepn():
    """One process of the insurer (underwriting) with callouts that explain the
    elements of an A-E PN: tokens with a color and a time, action and evolution
    transitions, bindings, the network tag, and the reward function."""
    d = Diagram()
    d.tag(LEFT - 5, TOP - 44)
    _, bottom = process_row(d, 0, "underwriting")
    W = ROW_W
    cw = 250
    def callout(x, y, html, px, py, lines):
        if not NOTES:
            return y
        yy = d.note(x, y, cw, html, lines=lines)
        nx = min(max(px, x), x + cw)
        ny = min(max(py, y), yy)
        d.leader((px, py), (nx, ny))
        return yy
    # above the net: the network tag
    callout(LEFT + W - cw - 12, TOP - 118,
            "<b>Network tag</b> (E, τ = 0, ρ = 0): which transitions may fire, the clock, "
            "and the reward accumulated since the last decision.", LEFT + 125, TOP - 36, 3)
    # below the net, two rows
    y1, y2 = bottom + 10, bottom + 95
    callout(LEFT + 8, y1,
            "<b>Token</b> {task_type=0}@0: a case with its attributes (color) and the time from which "
            "it can be consumed.", XA - 10, R + 32, 3)
    callout(LEFT + 8 + cw + 20, y1,
            "<b>Evolution transition</b> (E), e.g. Arrive: fires on its own, as soon as the tokens it "
            "consumes have reached their time.", XE, BH / 2 + 2, 4)
    callout(LEFT + 8 + 2 * (cw + 20), y1,
            "<b>Action transition</b> (A), e.g. Assign: the decision maker selects one enabled "
            "<b>binding</b> (X, Y), i.e., a case and an underwriter, and it fires.", XT, BH / 2 + 2, 4)
    callout(LEFT + 8 + 3 * (cw + 20), y1,
            "<b>Arc inscription</b> (X;Y)@+d(X,Y): the tokens produced, available d(X,Y) time "
            "units later.", XB - 55, -12, 3)
    low = (y1 + 4 * FN * 1.3 + 14 + 12) if NOTES else bottom + 12
    d.frame(LEFT, TOP, W, low - TOP, kind=None)
    d.note(LEFT, low + ROW_GAP, W,
           "<b>Definitions.</b> " + DEFAULTS + " Underwriting: task_type ~ U{0,1}; "
           "d(X,Y) = 1 + U{0,1} if X.task_type = Y.code, else 3 + U{0,1}.", lines=2)
    d.save("fig_aepn")


# ------------------------------------------------------------------ overview
def _block(d, x, y, w, h, html, fill="#ffffff", stroke=INK, size=FA, bold_first=False, dashed=False):
    st = (f"rounded=1;arcSize=8;whiteSpace=wrap;html=1;fontSize={size};strokeColor={stroke};strokeWidth=1.3;"
          f"fillColor={fill};align=left;verticalAlign=middle;spacingLeft=8;spacingRight=6;"
          + ("dashed=1;dashPattern=4 3;" if dashed else ""))
    return (d.vertex(html, st, x, y, w, h), (x + w / 2, y + h / 2))


def _arrow(d, src, tgt, label="", exit=(1, 0.5), entry=(0, 0.5), points=()):
    d.arc(src, tgt, label, points=points, exit=exit, entry=entry, straight=not points)


def _line(d, p, q_, color=INK, dashed=False, arrow=False, width=1.3):
    i = d._id()
    st = (f"endArrow={'block' if arrow else 'none'};endFill=1;endSize=5;html=1;strokeColor={color};"
          f"strokeWidth={width};" + ("dashed=1;dashPattern=3 2;" if dashed else ""))
    d.cells.append(f'<mxCell id="{i}" style="{st}" parent="1" edge="1"><mxGeometry relative="1" as="geometry">'
                   f'<mxPoint x="{p[0]:.1f}" y="{p[1]:.1f}" as="sourcePoint" />'
                   f'<mxPoint x="{q_[0]:.1f}" y="{q_[1]:.1f}" as="targetPoint" /></mxGeometry></mxCell>')


def _rect(d, x, y, w, h, fill, stroke, dashed=False, arc=12):
    return d.vertex("", f"rounded=1;arcSize={arc};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                        f"strokeWidth=1.3;" + ("dashed=1;dashPattern=4 3;" if dashed else ""), x, y, w, h)


KINDS = ("claims", "underwriting", "complaints")
GREY_BAR = ("#e4e4e4", "#777777")


def _blocks(d, x, y, w, h, gap=8, labels=False, strong=None):
    """Three process blocks side by side in their colors; strong: index of the block
    drawn with a solid fill (the component that a use acts on)."""
    bw = (w - 2 * gap) / 3
    for k, kind in enumerate(KINDS):
        fill, stroke = COLORS[kind]
        if strong is not None and k != strong:
            fill = "#ffffff"
        if strong == k:
            fill = stroke
        _rect(d, x + k * (bw + gap), y, bw, h, fill, stroke, arc=18)
        if labels:
            d.text(TITLES[kind], x + k * (bw + gap) + bw / 2, y + h / 2, size=9, bold=True,
                   color=LABEL_COLORS[kind])


def _staff_bar(d, x, y, w, label=True):
    _rect(d, x, y, w, 22, GREY_BAR[0], GREY_BAR[1], arc=30)
    if label:
        d.text("shared staff", x + w / 2, y + 11, size=11, color="#555555")


def _step(d, x, y, w, h, html, touched):
    st = (f"rounded=1;arcSize=40;whiteSpace=wrap;html=1;fontSize=12;strokeColor={'#555555' if touched else INK};"
          f"strokeWidth=1.3;fillColor={'#ffffff' if not touched else '#f2f2f2'};spacing=1;align=center;"
          + ("dashed=1;dashPattern=4 3;" if touched else ""))
    return (d.vertex(html, st, x - w / 2, y - h / 2, w, h), (x, y))


def _cycle(d, x, y, w, h, title, steps):
    """Steps on a ring, clockwise from the top, joined by arrows; a touched step
    (uses the certificate) is dashed, with the use in italics on a second line."""
    d.frame(x, y, w, h, label=title, kind=None)
    cx, cy, rx, ry = x + w / 2, y + h / 2 + 14, w / 2 - 70, h / 2 - 50
    ids = []
    n = len(steps)
    for k, (verb, use) in enumerate(steps):
        ang = -math.pi / 2 + 2 * math.pi * k / n
        px, py = cx + rx * math.cos(ang), cy + ry * math.sin(ang)
        html = f"<b>{verb}</b>" + (f"<br><i>{use}</i>" if use else "")
        ids.append(_step(d, px, py, 118 if use else 72, 40 if use else 26, html, bool(use)))
    for k in range(n):
        d.arc(ids[k], ids[(k + 1) % n], straight=True)
    return ids


def fig_overview():
    """The approach: the organization, the certificate it implies (one component
    while the staff is shared, three once it is allocated), and the two learning
    cycles, with the steps that the certificate changes dashed."""
    d = Diagram()
    H = 250
    # --- organization
    OX, OW = 0, 250
    d.frame(OX, 0, OW, H, label="Organization", kind=None)
    _staff_bar(d, OX + 22, 58, OW - 44)
    for k in range(3):
        bx = OX + 22 + k * ((OW - 44 - 16) / 3 + 8) + (OW - 44 - 16) / 6
        _line(d, (bx, 80), (bx, 110), width=1.1)
    _blocks(d, OX + 22, 110, OW - 44, 100, labels=True)
    # --- certificate
    CX, CW = 270, 210
    d.frame(CX, 0, CW, H, label="Certificate", kind=None)
    _rect(d, CX + 16, 44, CW - 32, 86, "none", "#555555", dashed=True, arc=8)
    _staff_bar(d, CX + 28, 54, CW - 56, label=False)
    _blocks(d, CX + 28, 84, CW - 56, 36)
    d.text("one component", CX + CW / 2, 143, size=11, color="#555555")
    _line(d, (CX + CW / 2, 152), (CX + CW / 2, 182), arrow=True, width=1.4)
    d.text("staff allocated", CX + CW / 2 + 8, 167, "left", size=11, color="#555555")
    _blocks(d, CX + 28, 188, CW - 56, 36)
    d.text("three components", CX + CW / 2, 238, size=11, color="#555555")
    _line(d, (OX + OW, H / 2), (CX, H / 2), arrow=True, width=1.4)
    # --- the two learning cycles, in one frame
    LX, W1, W2 = 520, 390, 300
    d.frame(LX, 0, W1 + W2 + 46, H, label="Learning", kind=None)
    _cycle(d, LX + 14, 30, W1, H - 44, "Model-free RL",
           [("simulate", None), ("observe", "own component"), ("decide", "wait per component"),
            ("credit", "own rewards"), ("update", None)])
    _cycle(d, LX + W1 + 32, 30, W2, H - 44, "Simulation-based learning",
           [("simulate", "component-days"), ("label", "restricted lookahead"), ("train", "local loss"),
            ("base policy", None)])
    _line(d, (CX + CW, H / 2), (LX, H / 2), arrow=True, width=1.4)
    d.save("fig_overview")


# ------------------------------------------------------------------- credit
CX_ = {0: 210, 1: 400, 2: 590, 3: 780, 4: 970}   # x of tau = 0..4
CLANE = 50


def _pill(d, x, y, label, fill="#ffffff", dashed=False):
    w = tw(label, FA) + 14
    st = (f"rounded=1;arcSize=30;whiteSpace=wrap;html=1;fontSize={FA};strokeColor={INK};strokeWidth=1.3;"
          f"fillColor={fill};spacing=1;" + ("dashed=1;dashPattern=3 2;" if dashed else ""))
    return (d.vertex(label, st, x - w / 2, y - 13, w, 26), (x, y))


def _reward(d, x, y, label):
    d.leader((x, y - 9), (x, y + 9))
    d.text(label, x, y + 20, size=12, color="#555555")


def _credit_panel(d, yp, title, lanes, ticks, note):
    left, right = CX_[0] - 200, CX_[4] + 60
    d.text(title, left + 2, yp + 10, "left", size=FF, bold=True)
    tops = {}
    for k, (kind, name) in enumerate(lanes):
        top = yp + 30 + k * (CLANE + 8)
        d.frame(left, top, right - left, CLANE, kind=kind)
        d.text(name, left + 8, top + CLANE / 2, "left", size=FA, bold=True, color=LABEL_COLORS.get(kind, INK))
        tops[name] = top + CLANE / 2
    ay = yp + 30 + len(lanes) * (CLANE + 8) + 22
    i = d._id()
    d.cells.append(f'<mxCell id="{i}" style="endArrow=block;endFill=1;endSize=6;html=1;strokeColor={INK};'
                   f'strokeWidth=1.2;" parent="1" edge="1"><mxGeometry relative="1" as="geometry">'
                   f'<mxPoint x="{CX_[0] - 20}" y="{ay}" as="sourcePoint" />'
                   f'<mxPoint x="{right}" y="{ay}" as="targetPoint" /></mxGeometry></mxCell>')
    for tx, lab in ticks:
        d.leader((tx, ay - 5), (tx, ay + 5))
        d.text(lab, tx, ay + 14, size=FA)
    end = d.note(left, ay + 30, right - left, note, lines=3)
    return tops, end + 14


def _credit_arrow(d, src, tgt, label, color, up=True, dx=0):
    """A curved credit arrow between two pills, labelled."""
    i = d._id()
    sx, sy = src[1]
    tx, ty = tgt[1]
    my = (sy - 34) if up else (sy + 34)
    d.cells.append(
        f'<mxCell id="{i}" style="edgeStyle=none;curved=1;html=1;endArrow=block;endFill=1;endSize=6;'
        f'strokeColor={color};strokeWidth=1.5;dashed=1;dashPattern=5 3;" parent="1" source="{src[0]}" '
        f'target="{tgt[0]}" edge="1"><mxGeometry relative="1" as="geometry">'
        f'<Array as="points"><mxPoint x="{(sx + tx) / 2 + dx:.1f}" y="{my:.1f}" /></Array></mxGeometry></mxCell>')
    j = d._id()
    d.cells.append(
        f'<mxCell id="{j}" value="{q(label)}" style="edgeLabel;html=1;align=center;verticalAlign=middle;'
        f'resizable=0;points=[];fontSize=12;fontColor={color};labelBackgroundColor=#ffffff;{FONT}" '
        f'parent="{i}" vertex="1" connectable="0"><mxGeometry x="0" relative="1" as="geometry">'
        f'<mxPoint x="0" y="{-14 if up else 14}" as="offset" /></mxGeometry></mxCell>')


def _span(d, x0, x1, y, h, color, label=None, above=True):
    """A translucent band on a lane from x0 to x1: what a decision is credited with."""
    d.vertex("", f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor={color};strokeColor={color};"
                 "strokeWidth=1;opacity=35;", x0, y - h / 2, x1 - x0, h)
    if label:
        d.text(label, (x0 + x1) / 2, y - h / 2 - 11 if above else y + h / 2 + 11, size=12, color=color, bold=True)


def _rdot(d, x, y, label="r = 1"):
    d.vertex("", "ellipse;fillColor=#330000;strokeColor=#330000;", x - 4, y - 4, 8, 8)
    d.text(label, x, y + 17, size=11, color="#555555")


def fig_credit():
    """Factored credit on a time line: (a) what the decision of underwriting at
    tau = 0 is credited with under NF-PPO (its own lane, until its own next
    decision) and under PPO (every lane, until the next decision of anyone);
    (b) the hand-over when the region splits."""
    d = Diagram()
    UW, GR = LABEL_COLORS["underwriting"], "#777777"
    # (a)
    tops, yp = _credit_panel(d, 0, "(a) Credit of a decision in underwriting",
                             (("underwriting", "Underwriting"), ("claims", "Claims")),
                             [(CX_[k], f"τ = {k}") for k in range(5)], "")
    yu, yc = tops["Underwriting"], tops["Claims"]
    x0, x1, x3 = CX_[0], CX_[1], CX_[3]
    _span(d, x0, x3, yu, CLANE - 8, UW, "NF-PPO: own rewards until its next decision")
    _span(d, x0, x1, yu, CLANE - 8, GR)
    _span(d, x0, x1, yc, CLANE - 8, GR, "PPO: every reward until the next decision of anyone", above=False)
    _pill(d, x0, yu, "Assign")
    _rdot(d, x1 + 55, yu)
    _pill(d, x3, yu, "Assign")
    _pill(d, x0 + 14, yc, "Approve")
    _rdot(d, x0 + 0.6 * (x1 - x0), yc)
    _pill(d, x1, yc, "Investigate")
    _rdot(d, CX_[2] + 40, yc)
    _pill(d, CX_[4] - 40, yc, "Approve")
    # (b)
    x = CX_[0] - 30
    pills = []
    for lab in ("To claims", "To complaints", "To underwriting"):
        w = tw(lab, FA) + 14
        pills.append((x + w / 2, lab))
        x += w + 8
    xs = x + 4                             # the split: Staff is empty after the third allocation
    step = 150
    tops, _ = _credit_panel(d, yp + 24, "(b) Hand-over when the region splits",
                            ((None, "Region"), ("claims", "Claims"), ("underwriting", "Underwriting"),
                             ("complaints", "Complaints")),
                            [(xs, "τ = 0"), (xs + step, "τ = 1"), (xs + 2 * step, "τ = 2"),
                             (xs + 3 * step, "τ = 3")], "")
    yr = tops["Region"]
    for px, lab in pills:
        _pill(d, px, yr, lab)
    firsts = (("Claims", xs + 0.6 * step), ("Underwriting", xs + 1.7 * step), ("Complaints", xs + 1.1 * step))
    for name, fx in firsts:
        _span(d, xs, fx, tops[name], CLANE - 8, UW)
        _pill(d, fx, tops[name], "first decision")
    top_y, bot_y = yr - CLANE / 2, tops["Complaints"] + CLANE / 2
    _line(d, (xs, top_y), (xs, bot_y), color="#555555", dashed=True, width=1.4)
    d.text("split", xs + 6, top_y + 10, "left", size=12, color="#555555")
    d.text("credited to the last allocation", xs + 14, yr, "left", size=12, color=UW, bold=True)
    d.save("fig_credit")


# --------------------------------------------------------------- collection
def _bars(d, x, y, colors, heights=(6, 11, 8, 5, 9)):
    """The label of a decision: one bar per option, colored by component."""
    for k, (c, h) in enumerate(zip(colors, heights)):
        d.vertex("", f"rounded=0;fillColor={c};strokeColor=none;", x + 7 * k, y + 6 - h, 5, h)


def _fan(d, x, y, x1, ys, color, n=2):
    """Rollouts from a decision at (x, y): dashed lines to the horizon x1 in the lanes ys."""
    for ly in ys:
        for k in range(n):
            _line(d, (x, y), (x1, ly + (k - (n - 1) / 2) * 12), color=color, dashed=True, width=1.1)


def fig_collect():
    """(a) DCL: a decision in a day of the whole net, its rollouts simulate every
    lane, its label ranges over the options of every lane. (b) NF-DCL: the same
    decision in a component-day; rollouts and label within the component."""
    d = Diagram()
    GR, UW = "#777777", LABEL_COLORS["underwriting"]
    CL, CO = LABEL_COLORS["claims"], LABEL_COLORS["complaints"]
    # (a) DCL
    lanes = (("claims", "Claims"), ("underwriting", "Underwriting"), ("complaints", "Complaints"))
    ticks = [(CX_[k], f"τ = {k}") for k in range(4)]
    tops, yp = _credit_panel(d, 0, "(a) DCL: a day of the whole net", lanes, ticks, "")
    yc, yu, yo = tops["Claims"], tops["Underwriting"], tops["Complaints"]
    xd, xh = CX_[1], CX_[3] + 90
    _pill(d, CX_[0] + 40, yc, "Approve")
    _pill(d, CX_[0] + 46, yo, "Resolve")
    for ly in (yc, yu, yo):
        _span(d, xd + 36, xh, ly, CLANE - 8, GR)
    _fan(d, xd + 36, yu, xh, (yc, yu, yo), GR)
    _pill(d, xd, yu, "Assign")
    _bars(d, xd - 88, yu, (CL, CL, UW, UW, CO))
    d.text("label", xd - 74, yu - 20, size=11, color="#555555")
    d.text("rollouts", xh - 30, yu - CLANE / 2 - 10, "right", size=11, color="#555555")
    # (b) NF-DCL
    x = CX_[0] - 30
    pills = []
    for lab in ("To claims", "To complaints", "To underwriting"):
        w = tw(lab, FA) + 14
        pills.append((x + w / 2, lab))
        x += w + 8
    xs, step = x + 4, 150
    lanes_b = ((None, "Region"), ("claims", "Claims"), ("underwriting", "Underwriting"), ("complaints", "Complaints"))
    tops, _ = _credit_panel(d, yp + 20, "(b) NF-DCL: a component-day", lanes_b,
                            [(xs + k * step, f"τ = {k}") for k in range(4)], "")
    yr, yc, yu, yo = tops["Region"], tops["Claims"], tops["Underwriting"], tops["Complaints"]
    for px, lab in pills:
        _pill(d, px, yr, lab)
    _line(d, (xs, yr - CLANE / 2), (xs, yo + CLANE / 2), color="#555555", dashed=True, width=1.4)
    d.text("split", xs + 6, yr - CLANE / 2 + 10, "left", size=12, color="#555555")
    xd, xh = xs + step, CX_[4] + 40
    _pill(d, xs + 70, yc, "Approve")
    _pill(d, xs + 76, yo, "Resolve")
    _span(d, xd + 36, xh, yu, CLANE - 8, UW)
    _fan(d, xd + 36, yu, xh, (yu,), UW, n=3)
    _pill(d, xd, yu, "Assign")
    _bars(d, xd - 76, yu, (UW, UW, UW), heights=(8, 11, 5))
    d.text("label", xd - 68, yu - 20, size=11, color="#555555")
    d.text("rollouts", xh - 30, yu - CLANE / 2 - 10, "right", size=11, color="#555555")
    d.save("fig_collect")


if __name__ == "__main__":
    fig_aepn()
    fig_bpic15()
    fig_insurer()
    fig_alloc()
    fig_slow()
    fig_observation()
    fig_overview()
    fig_credit()
    fig_postpone()
    fig_collect()
