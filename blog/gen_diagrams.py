#!/usr/bin/env python3
"""Generate Excalidraw scene files for the forward grouped-GEMM optimization blog.

One .excalidraw per optimization. Run:  python blog/gen_diagrams.py
Open any file at https://excalidraw.com (File > Open) and export PNG/SVG.

Hand-authored layout — no excalidraw runtime needed. Pure JSON emission.
"""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- palette (excalidraw default-ish) ----------------------------------
INK = "#1e1e1e"
RED = "#c41e3a"        # AMD-ish accent, matches presentation.md
BLUE = "#1971c2"
GREEN = "#2f9e44"
ORANGE = "#e8590c"
VIOLET = "#6741d9"
GRAY = "#868e96"
LBLUE = "#a5d8ff"
LGREEN = "#b2f2bb"
LRED = "#ffc9c9"
LYELLOW = "#ffec99"
LVIOLET = "#d0bfff"
LGRAY = "#e9ecef"
LORANGE = "#ffd8a8"
WHITE = "#ffffff"
TRANSPARENT = "transparent"

_seed = [1000]
def _nz():
    _seed[0] += 7
    return _seed[0] * 65537 % 2147483647

def _base(t, x, y, w, h, **kw):
    e = {
        "id": f"el{_nz()}",
        "type": t, "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": INK, "backgroundColor": TRANSPARENT,
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": None, "seed": _nz(), "version": 1, "versionNonce": _nz(),
        "isDeleted": False, "boundElements": None, "updated": 1, "link": None,
        "locked": False,
    }
    e.update(kw)
    return e

def rect(x, y, w, h, bg=TRANSPARENT, stroke=INK, sw=2, rounded=True, fill="solid", dash="solid"):
    return _base("rectangle", x, y, w, h, backgroundColor=bg, strokeColor=stroke,
                 strokeWidth=sw, fillStyle=fill, strokeStyle=dash,
                 roundness=({"type": 3} if rounded else None))

def ellipse(x, y, w, h, bg=TRANSPARENT, stroke=INK, sw=2):
    return _base("ellipse", x, y, w, h, backgroundColor=bg, strokeColor=stroke, strokeWidth=sw)

def text(x, y, s, size=20, color=INK, align="left", w=None, family=1):
    # approx width if not given
    cw = size * 0.55
    lines = s.split("\n")
    width = w if w is not None else max(len(l) for l in lines) * cw
    height = len(lines) * size * 1.25
    return _base("text", x, y, width, height, strokeColor=color, text=s,
                 fontSize=size, fontFamily=family, textAlign=align,
                 verticalAlign="top", containerId=None, originalText=s,
                 lineHeight=1.25, baseline=size, roundness=None)

def label(cx, cy, s, size=18, color=INK, family=1, align="center"):
    """Text centered at (cx, cy)."""
    cw = size * 0.55
    lines = s.split("\n")
    width = max(len(l) for l in lines) * cw
    height = len(lines) * size * 1.25
    return _base("text", cx - width / 2, cy - height / 2, width, height,
                 strokeColor=color, text=s, fontSize=size, fontFamily=family,
                 textAlign=align, verticalAlign="middle", containerId=None,
                 originalText=s, lineHeight=1.25, baseline=size, roundness=None)

def arrow(x, y, pts, color=INK, sw=2, end=True, start=False, dash="solid"):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    w = max(xs) - min(xs); h = max(ys) - min(ys)
    return _base("arrow", x, y, w, h, strokeColor=color, strokeWidth=sw,
                 strokeStyle=dash, points=[[float(a), float(b)] for a, b in pts],
                 lastCommittedPoint=None, startBinding=None, endBinding=None,
                 startArrowhead=("arrow" if start else None),
                 endArrowhead=("arrow" if end else None), roundness={"type": 2})

def line(x, y, pts, color=INK, sw=2, dash="solid"):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    w = max(xs) - min(xs); h = max(ys) - min(ys)
    return _base("line", x, y, w, h, strokeColor=color, strokeWidth=sw,
                 strokeStyle=dash, points=[[float(a), float(b)] for a, b in pts],
                 lastCommittedPoint=None, startBinding=None, endBinding=None,
                 startArrowhead=None, endArrowhead=None, roundness={"type": 2})

def box(x, y, w, h, s, bg=WHITE, stroke=INK, size=16, tcolor=INK, sw=2, family=1, dash="solid"):
    return [rect(x, y, w, h, bg=bg, stroke=stroke, sw=sw, dash=dash),
            label(x + w / 2, y + h / 2, s, size=size, color=tcolor, family=family)]

def _xml_escape(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

_FONT = {1: "Segoe UI, Helvetica, Arial, sans-serif",
         2: "Helvetica, Arial, sans-serif",
         3: "Consolas, 'Courier New', monospace"}

def to_svg(elements, pad=24):
    # bounds
    xs, ys, xe, ye = [], [], [], []
    for e in elements:
        if e["type"] in ("arrow", "line"):
            for px, py in e["points"]:
                xs.append(e["x"] + px); ys.append(e["y"] + py)
                xe.append(e["x"] + px); ye.append(e["y"] + py)
        else:
            xs.append(e["x"]); ys.append(e["y"])
            xe.append(e["x"] + e["width"]); ye.append(e["y"] + e["height"])
    minx, miny = min(xs) - pad, min(ys) - pad
    maxx, maxy = max(xe) + pad, max(ye) + pad
    W, H = maxx - minx, maxy - miny

    def dash(e):
        if e.get("strokeStyle") == "dashed":
            d = e["strokeWidth"] * 3
            return ' stroke-dasharray="%d,%d"' % (d, d)
        return ""

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
           f'viewBox="{minx:.1f} {miny:.1f} {W:.1f} {H:.1f}" font-family="{_FONT[1]}">',
           f'<rect x="{minx:.1f}" y="{miny:.1f}" width="{W:.1f}" height="{H:.1f}" fill="#ffffff"/>']

    for e in elements:
        t = e["type"]
        sc = e["strokeColor"]; sw = e["strokeWidth"]
        bg = e["backgroundColor"]
        fill = "none" if bg in (None, "transparent", "") else bg
        if t == "rectangle":
            rx = 12 if e.get("roundness") else 0
            out.append(f'<rect x="{e["x"]:.1f}" y="{e["y"]:.1f}" width="{e["width"]:.1f}" '
                       f'height="{e["height"]:.1f}" rx="{rx}" fill="{fill}" stroke="{sc}" '
                       f'stroke-width="{sw}"{dash(e)}/>')
        elif t == "ellipse":
            out.append(f'<ellipse cx="{e["x"]+e["width"]/2:.1f}" cy="{e["y"]+e["height"]/2:.1f}" '
                       f'rx="{e["width"]/2:.1f}" ry="{e["height"]/2:.1f}" fill="{fill}" '
                       f'stroke="{sc}" stroke-width="{sw}"{dash(e)}/>')
        elif t in ("line", "arrow"):
            pts = " ".join(f"{e['x']+px:.1f},{e['y']+py:.1f}" for px, py in e["points"])
            out.append(f'<polyline points="{pts}" fill="none" stroke="{sc}" '
                       f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"{dash(e)}/>')
            if t == "arrow":
                for which, head in (("end", e.get("endArrowhead")), ("start", e.get("startArrowhead"))):
                    if not head:
                        continue
                    if which == "end":
                        (x1, y1), (x0, y0) = e["points"][-2], e["points"][-1]
                    else:
                        (x1, y1), (x0, y0) = e["points"][1], e["points"][0]
                    import math
                    ang = math.atan2(y0 - y1, x0 - x1)
                    L = 11
                    ax, ay = e["x"] + x0, e["y"] + y0
                    for da in (math.radians(28), -math.radians(28)):
                        bx = ax - L * math.cos(ang + da)
                        by = ay - L * math.sin(ang + da)
                        out.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                                   f'stroke="{sc}" stroke-width="{sw}" stroke-linecap="round"/>')
        elif t == "text":
            fs = e["fontSize"]; fam = _FONT.get(e.get("fontFamily", 1))
            anchor = {"left": "start", "center": "middle", "right": "end"}[e["textAlign"]]
            if anchor == "start":
                tx = e["x"]
            elif anchor == "middle":
                tx = e["x"] + e["width"] / 2
            else:
                tx = e["x"] + e["width"]
            lines = e["text"].split("\n")
            for i, ln in enumerate(lines):
                ty = e["y"] + fs * (0.82 + i * e.get("lineHeight", 1.25))
                out.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="{fs}" '
                           f'font-family="{fam}" fill="{e["strokeColor"]}" '
                           f'text-anchor="{anchor}">{_xml_escape(ln)}</text>')
    out.append("</svg>")
    return "\n".join(out)

def save(name, elements, title):
    scene = {
        "type": "excalidraw", "version": 2,
        "source": "grouped-gemms/blog/gen_diagrams.py",
        "elements": elements,
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": {},
    }
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)
    svg_name = name.replace(".excalidraw", ".svg")
    with open(os.path.join(OUT, svg_name), "w") as f:
        f.write(to_svg(elements))
    print(f"wrote {name} + {svg_name}  ({len(elements)} elements)  — {title}")

def header(s, x=40, y=20, color=RED):
    return text(x, y, s, size=28, color=color, family=1)

def caption(s, x, y, color=GRAY, size=14):
    return text(x, y, s, size=size, color=color, family=1)


# =======================================================================
# 1. Routing build: host op-chain  ->  single fused kernel
# =======================================================================
def d1():
    E = []
    E.append(header("Optimization 1 — Sync-free routing build (_expt_data_kernel)"))
    E.append(caption("group_end_offsets  →  hist · offs · pad_sum · packed (block<<16|expt) map", 40, 60))

    # BEFORE — host chain
    E.append(text(60, 110, "BEFORE  ·  host-side op chain", size=18, color=RED))
    ops = ["cat", "diff", "cumsum", "arange", "searchsorted", "clamp", "shift", "where"]
    x = 60; y = 150
    for i, op in enumerate(ops):
        E += box(x, y, 110, 44, op, bg=LRED, size=15, family=3)
        if i < len(ops) - 1:
            E.append(arrow(x + 110, y + 22, [[0, 0], [22, 0]], color=GRAY))
        x += 132
        if (i + 1) % 4 == 0:
            x = 60; y += 90
            if i < len(ops) - 1:
                E.append(arrow(60 + 56, y - 46, [[0, 0], [0, 22]], color=GRAY))
    E.append(text(60, 330, "≈ 30–40 µs host-visible launch overhead · 8 kernel launches · CPU↔GPU syncs", size=14, color=RED))

    # divider
    E.append(line(40, 380, [[0, 0], [1080, 0]], color=LGRAY, sw=2, dash="dashed"))

    # AFTER — one kernel
    E.append(text(60, 410, "AFTER  ·  one Triton launch", size=18, color=GREEN))
    E += box(60, 450, 230, 120,
             "_expt_data_kernel\n\nstatic_range(E)\nprefix-sum in regs",
             bg=LGREEN, size=14, family=3)
    E.append(arrow(290, 510, [[0, 0], [60, 0]], color=INK))
    outs = ["ExptHist (E,)", "ExptOffs (E,)", "ExptOffsSum (0-d)", "ExptData packed map"]
    oy = 410
    for o in outs:
        E += box(370, oy, 250, 34, o, bg=LBLUE, size=14, family=3)
        oy += 44
    E.append(text(60, 590, "≈ 2–3 µs · single launch · no host sync · torch.compile-clean", size=14, color=GREEN))

    # win callout
    E += box(720, 450, 320, 110, "≈ 10–15× lower\nrouting overhead", bg=LYELLOW, stroke=ORANGE, size=22)
    save("01-routing-build.excalidraw", E, "routing build")


# =======================================================================
# 2. Packed expert -> tile map
# =======================================================================
def d2():
    E = []
    E.append(header("Optimization 2 — Packed expert→tile routing (one int32 per tile)"))
    E.append(caption("Each program reads one int32, decodes its expert + block, or skips a -1 pad tile.", 40, 60))

    # the packed int32 layout
    E.append(text(60, 110, "ExptData[pid_m]  —  packed int32", size=18, color=BLUE))
    E += box(60, 150, 300, 56, "block_id  <<  16", bg=LVIOLET, size=18, family=3)
    E += box(360, 150, 300, 56, "expt_id  &  0xFFFF", bg=LBLUE, size=18, family=3)
    E += box(680, 150, 180, 56, "-1  =  pad → return", bg=LRED, size=15, family=3)

    # jagged experts -> tiles
    E.append(text(60, 260, "Expert-sorted M (jagged) chopped into BLOCK_M tiles", size=18, color=INK))
    experts = [("E0", LBLUE, 3), ("E1", LGREEN, 2), ("E2", LYELLOW, 4), ("E3", LORANGE, 1)]
    x = 60; y = 300; tw = 70
    pid = 0
    for name, col, ntiles in experts:
        for b in range(ntiles):
            E += box(x, y, tw, 50, f"{name}\nblk{b}", bg=col, size=12)
            E.append(label(x + tw / 2, y + 66, f"pid {pid}", size=11, color=GRAY))
            x += tw + 6
            pid += 1
        x += 18  # gap between experts
    # pad tile
    E += box(x, y, tw, 50, "PAD\n-1", bg=LRED, size=12)
    E.append(label(x + tw / 2, y + 66, f"pid {pid}", size=11, color=GRAY))

    E.append(text(60, 410, "Mapping is built once by Opt-1; the GEMM kernel does a single tl.load + bit ops — no\nbinary search, no per-tile expert scan inside the hot loop.", size=14, color=GRAY))

    E += box(60, 480, 900, 60,
             "expt_data = tl.load(ExptData + pid_m);  if expt_data == -1: return\n"
             "expt_id = expt_data & 0xFFFF;  block_id = expt_data >> 16",
             bg=LGRAY, size=15, family=3)
    save("02-packed-expert-map.excalidraw", E, "packed expert map")


# =======================================================================
# 3. GROUP_M tile reordering for L2 reuse
# =======================================================================
def d3():
    E = []
    E.append(header("Optimization 3 — GROUP_M tile ordering for L2 reuse"))
    E.append(caption("Reorder program IDs so concurrently-running tiles share the same A rows / B columns in L2.", 40, 60))

    cell = 46
    cols = 6; rows = 6

    def grid(ox, oy, order_fn, colorfn):
        for r in range(rows):
            for c in range(cols):
                idx = order_fn(r, c)
                E.append(rect(ox + c * cell, oy + r * cell, cell - 4, cell - 4,
                              bg=colorfn(idx), stroke=INK, sw=1.5))
                E.append(label(ox + c * cell + (cell - 4) / 2, oy + r * cell + (cell - 4) / 2,
                               str(idx), size=12))

    # naive row-major: consecutive pids sweep a full N row -> reuse only A row
    E.append(text(80, 110, "Naive: pid = row-major  (GROUP_M = 1)", size=16, color=RED))
    def rowmajor(r, c): return r * cols + c
    def heat_row(idx): return LRED if idx < cols else WHITE
    grid(80, 145, rowmajor, heat_row)
    E.append(text(80, 470, "First " + str(cols) + " tiles span one B-column strip;\nlittle overlap in working set → L2 thrash.", size=13, color=GRAY))

    # grouped: super-block of GROUP_M rows traversed column-first
    E.append(text(520, 110, "GROUP_M = 3  super-block, column-first", size=16, color=GREEN))
    GM = 3
    def grouped(r, c):
        group = r // GM
        first = group * GM * cols
        local_rows = GM
        # column-first within group
        return first + c * local_rows + (r % GM)
    def heat_group(idx): return LGREEN if idx < GM * cols else WHITE
    grid(520, 145, grouped, heat_group)
    E.append(text(520, 470, "First " + str(GM * cols) + " tiles form a " + str(GM) + "×" + str(cols) +
                  " block → A rows AND B cols\nre-hit from L2 across the block.", size=13, color=GRAY))

    E += box(360, 560, 420, 50, "_xcd_swizzle → _pid_grid(GROUP_M)  reshapes the launch grid",
             bg=LGRAY, size=14, family=3)
    save("03-group-m-l2-reuse.excalidraw", E, "GROUP_M L2 reuse")


# =======================================================================
# 4. XCD swizzle
# =======================================================================
def d4():
    E = []
    E.append(header("Optimization 4 — XCD swizzle (MI350 8-XCD scheduling)"))
    E.append(caption("MI355X = 8 XCDs; the HW round-robins pids across XCDs. Pre-swizzle so each XCD gets a contiguous tile range.", 40, 70))

    # hardware round-robin
    E.append(text(60, 120, "HW dispatch: pid % 8 → XCD   (round-robin)", size=16, color=RED))
    xcds = 8
    cw = 116
    for x in range(xcds):
        E += box(60 + x * cw, 160, cw - 10, 44, f"XCD {x}", bg=LGRAY, size=14, family=3)
    # show pids landing scattered
    for x in range(xcds):
        E.append(label(60 + x * cw + (cw - 10) / 2, 230,
                       f"{x},{x+8},{x+16}", size=11, color=GRAY, family=3))
    E.append(text(60, 270, "Without swizzle: tiles touching the same expert/B-column scatter across all 8 XCDs →\nno per-XCD L2 locality.", size=13, color=GRAY))

    E.append(arrow(540, 320, [[0, 0], [0, 50]], color=INK, sw=3))
    E.append(label(680, 345, "_xcd_swizzle(pid, domain, 8)", size=14, color=VIOLET, family=3))

    # after: contiguous ranges per XCD
    E.append(text(60, 400, "After swizzle: contiguous tile block per XCD", size=16, color=GREEN))
    cols = ["#ffc9c9", "#a5d8ff", "#b2f2bb", "#ffec99", "#d0bfff", "#ffd8a8", "#c5f6fa", "#eebefa"]
    for x in range(xcds):
        E += box(60 + x * cw, 440, cw - 10, 44, f"XCD {x}", bg=cols[x], size=14, family=3)
        lo = x * 3
        E.append(label(60 + x * cw + (cw - 10) / 2, 510,
                       f"{lo}…{lo+2}", size=11, color=GRAY, family=3))
    E.append(text(60, 545, "Each XCD owns a packed pid range → tiles sharing operands co-locate on one XCD's L2 slice.", size=13, color=GREEN))
    save("04-xcd-swizzle.excalidraw", E, "XCD swizzle")


# =======================================================================
# 5. CDNA4 pre-shuffled MX scale layout
# =======================================================================
def d5():
    E = []
    E.append(header("Optimization 5 — CDNA4-native pre-shuffled MX scale layout"))
    E.append(caption("Feed v_mfma_scale_f32_16x16x128 its scales in the layout it consumes natively — kill the lowering's permute chain.", 40, 70))

    # BEFORE
    E.append(text(60, 120, "BEFORE  ·  #blocked → #linear1 lowering", size=16, color=RED))
    E += box(60, 160, 200, 50, "scales (M, K/32)\nrow-major", bg=WHITE, size=13, family=3)
    E.append(arrow(260, 185, [[0, 0], [40, 0]], color=GRAY))
    chain = ["6 × ds_read_u8", "3 × v_perm_b32", "register shuffle"]
    x = 320
    for c in chain:
        E += box(x, 160, 150, 50, c, bg=LRED, size=12, family=3)
        E.append(arrow(x + 150, 185, [[0, 0], [26, 0]], color=GRAY))
        x += 176
    E += box(x, 160, 150, 50, "MFMA scale\noperand", bg=LGRAY, size=12, family=3)
    E.append(text(60, 230, "Per K-iteration, per scale tensor — pure address-shuffle overhead inside the hot loop.", size=13, color=GRAY))

    E.append(line(40, 280, [[0, 0], [1080, 0]], color=LGRAY, dash="dashed"))

    # AFTER
    E.append(text(60, 310, "AFTER  ·  host pre-shuffle once + no-op in kernel", size=16, color=GREEN))
    E += box(60, 350, 230, 60, "_shuffle_*_scales_cdna4\n(host, once at launch)", bg=LGREEN, size=12, family=3)
    E.append(arrow(290, 380, [[0, 0], [40, 0]], color=INK))
    E += box(340, 350, 230, 60, "scales already in\nMFMA-native order", bg=LBLUE, size=13, family=3)
    E.append(arrow(570, 380, [[0, 0], [40, 0]], color=INK))
    E += box(620, 350, 230, 60, "1 coalesced load →\ntl.dot_scaled", bg=LBLUE, size=13, family=3)
    E.append(text(60, 430, "In-kernel _unswizzle_*_cdna4 is a pure tl.reshape/permute on registers → compiles to v_perm_b32 = 0.", size=13, color=GREEN))

    E += box(60, 480, 270, 90, "nonkdim 16  vs  32\nselectable per shape\n(nk32 wins 27/36)", bg=LYELLOW, stroke=ORANGE, size=14)
    E += box(360, 480, 360, 90,
             "Gated: BLOCK_K ≥ 256  &  K%256==0\n&  N%32==0  &  M%32==0\n(use_cdna4_scale)",
             bg=LGRAY, size=13, family=3)
    E += box(750, 480, 300, 90, "v_perm_b32 count = 0\nconfirmed in AMDGCN", bg=LGREEN, stroke=GREEN, size=16)
    save("05-cdna4-scale-shuffle.excalidraw", E, "CDNA4 scale shuffle")


# =======================================================================
# 6. dot_scaled K-loop + EVEN_K peel
# =======================================================================
def d6():
    E = []
    E.append(header("Optimization 6 — Direct dot_scaled K-loop + EVEN_K peel"))
    E.append(caption("Consume e8m0 scales inside MFMA (no dequant pass); peel the ragged tail so the main loop is branch-free.", 40, 70))

    # main loop box
    E.append(text(60, 120, "Main loop  ·  num_k_iter (full BLOCK_K tiles)", size=16, color=BLUE))
    lx, ly, lw, lh = 60, 155, 560, 210
    E.append(rect(lx, ly, lw, lh, bg="#f8f9fa", stroke=BLUE, sw=2))
    E += box(lx + 20, ly + 24, 150, 44, "load X tile", bg=LBLUE, size=13, family=3)
    E += box(lx + 20, ly + 84, 150, 44, "load W tile", bg=LBLUE, size=13, family=3)
    E += box(lx + 200, ly + 24, 160, 44, "load X/W scales", bg=LGREEN, size=12, family=3)
    E += box(lx + 200, ly + 84, 160, 104, "tl.dot_scaled\ne4m3 × e4m3\nacc += , fast_math", bg=LYELLOW, size=13, family=3)
    E += box(lx + 390, ly + 24, 150, 44, "advance ptrs\n+= BLOCK_K", bg=LGRAY, size=12, family=3)
    E.append(arrow(lx + 360, ly + 70, [[0, 0], [-100, 60]], color=GRAY))
    # loop-back arrow
    E.append(arrow(lx + lw / 2, ly + lh, [[0, 0], [0, 30], [-lw/2 + 20, 30], [-lw/2 + 20, -lh + 10]],
                   color=BLUE, dash="dashed"))

    # peel
    E.append(text(700, 120, "EVEN_K peel (tail)", size=16, color=ORANGE))
    E += box(700, 160, 320, 70, "if K % BLOCK_K == 0 → EVEN_K=True\nmain loop runs all iters, no masking", bg=LGREEN, size=13, family=3)
    E += box(700, 245, 320, 90, "else → run num_k_iter-1 unmasked,\nthen ONE masked tail iteration\n(offs_k < MASK_K_LIMIT)", bg=LORANGE, size=13, family=3)
    E.append(text(700, 350, "→ no per-iteration K-mask compare in the\n  common (EVEN_K) path; bounds check paid once.", size=13, color=GRAY))

    E += box(60, 400, 960, 56,
             "acc = tl.dot_scaled(x, x_scales, \"e4m3\", w, w_scales, \"e4m3\", acc=acc, fast_math=True)",
             bg=LGRAY, size=15, family=3)
    E.append(text(60, 475, "Scales ride straight into the MFMA scale operand — no separate fp8→bf16 dequantize + multiply pass over the tile.", size=13, color=GREEN))
    save("06-dot-scaled-kloop.excalidraw", E, "dot_scaled K-loop")


# =======================================================================
# 7. Per-shape autotuning
# =======================================================================
def d7():
    E = []
    E.append(header("Optimization 7 — Per-shape autotuning (8-GPU parallel sweep)"))
    E.append(caption("576 configs × 36 Llama4 shapes, dispatched one-shape-per-GPU across 8 MI355X → baked _BEST_CFGS table.", 40, 70))

    # search space
    E.append(text(60, 120, "Search space (per shape)", size=16, color=VIOLET))
    space = [
        "BLOCK_M ∈ {64,128,256}",
        "BLOCK_N ∈ {128,256}",
        "BLOCK_K ∈ {128,256}",
        "GROUP_M ∈ {1,4,8}",
        "num_warps ∈ {4,8}",
        "num_stages ∈ {1,2}",
        "waves_per_eu ∈ {0,2}",
        "nonkdim ∈ {16,32}",
    ]
    y = 160
    for s in space:
        E += box(60, y, 250, 34, s, bg=LVIOLET, size=13, family=3)
        y += 42

    # arrow to sweep
    E.append(arrow(320, 300, [[0, 0], [70, 0]], color=INK, sw=3))

    # 8 GPUs
    E.append(text(410, 120, "ProcessPoolExecutor(8)", size=16, color=BLUE))
    for i in range(8):
        gx = 410 + (i % 4) * 90
        gy = 160 + (i // 4) * 80
        E += box(gx, gy, 80, 64, f"GPU {i}", bg=LBLUE, size=13, family=3)
    E.append(text(410, 330, "CUDA/HIP_VISIBLE_DEVICES=i\nPYTHONPATH set per worker", size=12, color=GRAY, family=3))

    E.append(arrow(790, 300, [[0, 0], [70, 0]], color=INK, sw=3))

    # best cfgs table
    E.append(text(880, 120, "_BEST_CFGS[(E,N,K)]", size=16, color=GREEN))
    E += box(880, 160, 260, 150,
             "36 tuned entries\n+ small/large-K\nfallback heuristic\n\nlooked up at launch\nby _pick_config",
             bg=LGREEN, size=13, family=3)

    # cache hints + result
    E += box(60, 470, 480, 70,
             "Per-shape cache hints baked in:\n12× X evict_first · 4× W .cg  (biggest +10% on (1,2048,2048))",
             bg=LYELLOW, stroke=ORANGE, size=14)
    E += box(580, 470, 560, 70,
             "Geomean 1.389× → 1.487× vs bf16\n(+7.0% from tuning · ~13–18 min wall per full sweep)",
             bg=LGREEN, stroke=GREEN, size=16)
    save("07-per-shape-autotune.excalidraw", E, "per-shape autotune")


if __name__ == "__main__":
    d1(); d2(); d3(); d4(); d5(); d6(); d7()
    print("done.")
