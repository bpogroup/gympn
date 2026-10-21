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
COLORS = {"claims": ("#EEF4FB", "#7FA7D9"), "underwriting": ("#EEF7EC", "#86BF7E"),
          "complaints": ("#FDF3E6", "#E3A85B")}
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
                4: [(-7, -7), (7, -7), (-7, 7), (7, 7)]}.get(len(tokens), [])
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
                      color=("#4a6f9e" if kind == "claims" else "#4f8a47" if kind == "underwriting"
                             else "#b9742a" if kind == "complaints" else INK))

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
                              lambda k, kind: f"Component {k + 1}: {TITLES[kind]}")
    defs_insurer(d, LEFT, bottom + ROW_GAP, ROW_W)
    d.save("fig_insurer")


def fig_alloc():
    """Start-of-shift allocation. A Staff pool feeds the three resource
    places; while Staff is marked the region is one component, afterwards three."""
    d = Diagram()
    toks = {"claims": ["{skill=0}@0"], "underwriting": ["{code=0}@0"], "complaints": ["{skill=0}@0"]}
    res, ys, bottom = stack_rows(d, ("claims", "underwriting", "complaints"),
                                 lambda k, kind: f"Component {k + 1} once Staff is empty: {TITLES[kind]}",
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
    d.frame(LEFT - 15, top, right - LEFT + 15, bottom + 15 - top, "Region: one component while Staff is marked",
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
    d.frame(LEFT, TOP, W, bottom - TOP, "Component 2: Underwriting", kind="underwriting")
    by = bottom + ROW_GAP
    half = (W - ROW_GAP) / 2
    for k, (kind, wn, n) in enumerate((("claims", "wait<sub>1</sub>", 1), ("complaints", "wait<sub>3</sub>", 3))):
        x = LEFT + k * (half + ROW_GAP)
        d.frame(x, by, half, 74, f"Component {n}: {TITLES[kind]}", kind=kind)
        d.trans(x + half - 24, by + 40, "A", dashed=True)
        d.text(wn, x + half - 36, by + 40, "right")
        d.text("as in Fig. 2, with its own wait", x + 12, by + 52, "left", size=FA)
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


if __name__ == "__main__":
    fig_insurer()
    fig_alloc()
    fig_slow()
    fig_observation()
