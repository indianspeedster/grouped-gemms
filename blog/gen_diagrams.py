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
# FOUNDATIONS
# =======================================================================

# --- 00  MXFP8 block layout --------------------------------------------
def d_mxblock():
    E = []
    E.append(header("MXFP8 — 32 elements share one scale"))
    E.append(caption("Microscaling FP8: an e4m3 value per element + one e8m0 power-of-two scale per 32-element block.", 40, 60))

    # a (M,K) row split into K/32 blocks
    E.append(text(60, 110, "One row of the activation tensor  (K elements)", size=16, color=BLUE))
    cell = 24
    x0 = 60; y0 = 150
    for b in range(2):                       # two example blocks of 32
        bx = x0 + b * (32 * cell + 40)
        for i in range(32):
            E.append(rect(bx + i * cell, y0, cell - 3, cell - 3, bg=LBLUE, sw=1))
        E.append(line(bx, y0 - 8, [[0, 0], [32 * cell - 3, 0]], color=GRAY))
        E.append(label(bx + 16 * cell, y0 - 22, "32 × e4m3 (1 byte each)", size=13, color=GRAY))
        # the shared scale byte
        E += box(bx + 11 * cell, y0 + 50, 200, 40, "1 × e8m0 scale", bg=LYELLOW, stroke=ORANGE, size=14, family=3)
        E.append(arrow(bx + 16 * cell, y0 + 30, [[0, 0], [0, 18]], color=ORANGE))
    E.append(text(60, 270, "→ block_size = 32   (offsets fed to the kernel must be multiples of 32)", size=14, color=GRAY))

    # tensor decomposition
    E.append(text(60, 330, "Whole tensor decomposes into two arrays:", size=16, color=INK))
    E += box(60, 370, 250, 80, "data  (M, K)\nfloat8_e4m3fn", bg=LBLUE, size=15, family=3)
    E.append(text(330, 400, "+", size=28, color=INK))
    E += box(380, 370, 250, 80, "scales  (M, K/32)\ne8m0  (uint8 view)", bg=LYELLOW, size=15, family=3)
    E.append(text(60, 470, "e4m3 = 1 sign · 4 exp · 3 mantissa, max 448.   e8m0 = 8-bit exponent only (a pure 2^k scale).", size=13, color=GRAY))
    save("00-mxfp8-block.excalidraw", E, "mxfp8 block layout")


# --- 01  e8m0 scale calculation ----------------------------------------
def d_scalecalc():
    E = []
    E.append(header("How the e8m0 scale is computed (FLOOR mode)"))
    E.append(caption("torchao to_mx FLOOR mode — extract the power-of-two exponent straight from the fp32 bit pattern.", 40, 60))

    steps = [
        ("block of 32\nhp values", LBLUE),
        ("amax =\nmax|x|", LBLUE),
        ("e = floor(log2(amax))\nvia (bits>>23 & 0xFF) - 127", LVIOLET),
        ("e8m0 = e - 8\n(8 = floor(log2(448)))", LVIOLET),
        ("store u8 =\nclamp(e8m0 + 127, 0, 254)", LYELLOW),
    ]
    x = 60; y = 130
    for i, (s, c) in enumerate(steps):
        E += box(x, y, 200, 70, s, bg=c, size=13, family=3)
        if i < len(steps) - 1:
            E.append(arrow(x + 200, y + 35, [[0, 0], [22, 0]], color=INK))
        x += 222
        if (i + 1) % 3 == 0:
            x = 60; y += 130
            E.append(arrow(160, y - 60, [[0, 0], [0, 22]], color=INK))

    E.append(line(40, 400, [[0, 0], [1060, 0]], color=LGRAY, dash="dashed"))
    E.append(text(60, 420, "Dequant / quantize back:", size=16, color=GREEN))
    E += box(60, 455, 300, 56, "scale_f32 = 2^(u8 - 127)", bg=LGREEN, size=15, family=3)
    E.append(arrow(360, 483, [[0, 0], [30, 0]], color=INK))
    E += box(400, 455, 420, 56, "q = clamp(x / scale_f32, ±448) → e4m3", bg=LGREEN, size=14, family=3)
    E.append(text(60, 525, "Power-of-two scale → exact, cheap (a bit-shift on the exponent); no per-element fp multiply to undo.", size=13, color=GRAY))
    save("01-scale-calc.excalidraw", E, "e8m0 scale calc")


# --- 02  grouped GEMM concept ------------------------------------------
def d_grouped():
    E = []
    E.append(header("What a grouped GEMM computes"))
    E.append(caption("MoE routes a jagged number of tokens to each expert: out[g] = A[group g] @ B[g]^T, per expert.", 40, 60))

    col_top = 150
    experts = [("E0", LBLUE, 70), ("E1", LGREEN, 40), ("E2", LYELLOW, 95), ("E3", LORANGE, 35)]
    col_bot = col_top + sum(h + 4 for _, _, h in experts)   # ≈ 406

    # A jagged
    E.append(text(70, 118, "A  (M, K)  — expert-sorted rows", size=15, color=BLUE))
    y = col_top
    for name, col, h in experts:
        E.append(rect(70, y, 150, h, bg=col, sw=1.5))
        E.append(label(145, y + h / 2, name, size=13))
        y += h + 4
    # offsets annotation in the gap, vertically centered on the column
    E.append(line(232, col_top, [[0, 0], [0, col_bot - col_top - 4]], color=GRAY))
    E.append(label(326, (col_top + col_bot) / 2, "offsets\n[70,110,\n205,240]", size=12, color=GRAY, family=3))

    # × B per expert
    E.append(label(326, col_top - 22, "×", size=26, color=INK))
    E.append(text(420, 118, "B  (E, N, K)", size=15, color=VIOLET))
    for i in range(4):
        E += box(420 + i * 70, col_top, 60, 60, f"B{i}", bg=LVIOLET, size=13)
    E.append(text(420, col_top + 70, "one weight matrix\nper expert", size=12, color=GRAY))

    # = output
    E.append(label(745, (col_top + col_bot) / 2, "=", size=28, color=INK))
    E.append(text(790, 118, "out  (M, N)", size=15, color=GREEN))
    y = col_top
    for name, col, h in experts:
        E.append(rect(790, y, 170, h, bg=col, sw=1.5))
        E.append(label(875, y + h / 2, f"{name} @ B{name[1]}^T", size=11))
        y += h + 4

    by = col_bot + 28
    E.append(text(70, by, "The row-count per expert is data-dependent (jagged) and only known at runtime — so the kernel", size=14, color=GRAY))
    E.append(text(70, by + 22, "must discover, per output tile, which expert it belongs to. That routing is the first thing a naive version gets wrong.", size=14, color=GRAY))
    save("02-grouped-gemm.excalidraw", E, "grouped gemm concept")


# --- 03  naive kernel ---------------------------------------------------
def d_naive():
    E = []
    E.append(header("The naive baseline (L0)"))
    E.append(caption("Correct MXFP8 grouped GEMM, zero AMD tuning. One output tile per program. Measured: 914 TFLOPS geomean.", 40, 60))

    E += box(60, 110, 300, 60, "host torch routing\ncat·diff·cumsum·searchsorted…", bg=LRED, size=13, family=3)
    E += box(60, 195, 300, 50, "GROUP_M=1 · XCD=1\n(row-major tiles)", bg=LRED, size=13, family=3)
    E += box(60, 270, 300, 50, "plain scales\n(row-major, re-shuffled in-kernel)", bg=LRED, size=12, family=3)
    E += box(60, 345, 300, 50, "small 64×128 tile · 1 stage", bg=LRED, size=13, family=3)

    # kernel inner
    E.append(text(440, 110, "per-program inner loop", size=16, color=BLUE))
    lx, ly = 440, 145
    E.append(rect(lx, ly, 420, 200, bg="#f8f9fa", stroke=BLUE))
    E += box(lx + 20, ly + 24, 180, 40, "load X, W tile", bg=LBLUE, size=13, family=3)
    E += box(lx + 20, ly + 78, 180, 40, "load + unshuffle scales", bg=LYELLOW, size=12, family=3)
    E += box(lx + 220, ly + 24, 170, 94, "acc +=\ntl.dot_scaled\n(e4m3 × e4m3)", bg=LGREEN, size=13, family=3)
    E += box(lx + 20, ly + 132, 370, 40, "loop K  →  write acc to out", bg=LGRAY, size=13, family=3)

    E += box(440, 380, 420, 60, "Result: 914 TFLOPS geomean — the scale-load shuffle and\ntiny tiles leave the MFMA units starved.", bg=LYELLOW, stroke=ORANGE, size=14)
    E.append(text(60, 460, "Every box on the left is an opportunity. The rest of the post lights them up one rung at a time.", size=13, color=GRAY))
    save("03-naive-kernel.excalidraw", E, "naive kernel")


# =======================================================================
# OPTIMIZATIONS (ladder rungs)
# =======================================================================

# --- 04  fused routing  (L1) -------------------------------------------
def d_routing():
    E = []
    E.append(header("L1 — Sync-free routing build  (+14% vs naive)"))
    E.append(caption("Replace the host op-chain with one Triton launch. Biggest win on small / many-expert shapes.", 40, 60))

    E.append(text(60, 105, "BEFORE · host op chain", size=16, color=RED))
    ops = ["cat", "diff", "cumsum", "arange", "searchsorted", "clamp", "shift", "where"]
    x = 60; y = 140
    for i, op in enumerate(ops):
        E += box(x, y, 110, 40, op, bg=LRED, size=14, family=3)
        if i < len(ops) - 1 and (i + 1) % 4 != 0:
            E.append(arrow(x + 110, y + 20, [[0, 0], [20, 0]], color=GRAY))
        x += 130
        if (i + 1) % 4 == 0:
            x = 60; y += 70
    E.append(text(60, 285, "≈ 30–40 µs · 8 launches · host↔device syncs · breaks torch.compile", size=13, color=RED))

    E.append(line(40, 320, [[0, 0], [1020, 0]], color=LGRAY, dash="dashed"))
    E.append(text(60, 340, "AFTER · one kernel", size=16, color=GREEN))
    E += box(60, 375, 240, 90, "_expt_data_kernel\nstatic_range(E)\nprefix-sum in registers", bg=LGREEN, size=13, family=3)
    E.append(arrow(300, 420, [[0, 0], [50, 0]], color=INK))
    outs = ["ExptHist", "ExptOffs", "ExptOffsSum", "ExptData (packed map →)"]
    oy = 360
    for o in outs:
        E += box(360, oy, 240, 30, o, bg=LBLUE, size=13, family=3)
        oy += 38
    E += box(650, 375, 380, 90, "≈ 2–3 µs · one launch · no sync\n+14% TF geomean; up to +35% TF\non small 2048×2048 shapes", bg=LYELLOW, stroke=ORANGE, size=15)
    save("04-fused-routing.excalidraw", E, "fused routing")


# --- 05  packed expert->tile map ---------------------------------------
def d_packed():
    E = []
    E.append(header("L1 detail — Packed expert→tile map"))
    E.append(caption("The router emits one int32 per tile; the GEMM decodes its expert + row-block with a load and two bit-ops.", 40, 60))

    E += box(60, 120, 300, 54, "block_id  <<  16", bg=LVIOLET, size=17, family=3)
    E += box(360, 120, 300, 54, "expt_id  &  0xFFFF", bg=LBLUE, size=17, family=3)
    E += box(680, 120, 200, 54, "-1 = pad → return", bg=LRED, size=14, family=3)

    E.append(text(60, 210, "Expert-sorted M chopped into BLOCK_M tiles, coloured by owning expert:", size=15, color=INK))
    experts = [("E0", LBLUE, 3), ("E1", LGREEN, 2), ("E2", LYELLOW, 4), ("E3", LORANGE, 1)]
    x = 60; y = 250; tw = 72; pid = 0
    for name, col, nt in experts:
        for b in range(nt):
            E += box(x, y, tw, 48, f"{name}\nblk{b}", bg=col, size=12)
            E.append(label(x + tw / 2, y + 62, f"pid{pid}", size=10, color=GRAY))
            x += tw + 6; pid += 1
        x += 16
    E += box(x, y, tw, 48, "PAD\n-1", bg=LRED, size=12)

    E += box(60, 360, 900, 56,
             "expt_data = tl.load(ExptData + pid_m)\n"
             "if expt_data == -1: return    # else: expt_id = expt_data & 0xFFFF ; block_id = expt_data >> 16",
             bg=LGRAY, size=14, family=3)
    E.append(text(60, 435, "No binary search, no per-tile expert scan in the hot loop — one coalesced int32 read.", size=13, color=GRAY))
    save("05-packed-map.excalidraw", E, "packed expert map")


# --- 06  tiles + pipeline  (L2) ----------------------------------------
def d_tiles():
    E = []
    E.append(header("L2 — Bigger tiles + software pipeline"))
    E.append(caption("256×128×256 tiles and num_stages=2. Helps compute-bound large shapes; HURTS small ones (see L5).", 40, 60))

    # tile size compare
    E.append(text(60, 110, "Tile footprint", size=16, color=BLUE))
    E.append(rect(60, 145, 60, 120, bg=LRED, sw=1.5)); E.append(label(90, 280, "64×128", size=12, color=GRAY))
    E.append(rect(160, 145, 120, 120, bg=LGREEN, sw=1.5)); E.append(label(220, 280, "256×128", size=12, color=GRAY))
    E.append(text(60, 305, "more MFMA work per program →\nbetter arithmetic intensity", size=12, color=GRAY))

    # pipeline timeline
    E.append(text(420, 110, "num_stages=2: overlap load(K+1) with compute(K)", size=15, color=GREEN))
    E.append(text(420, 150, "1 stage (serial):", size=13, color=RED))
    seq = [("ld0", LBLUE), ("mma0", LGREEN), ("ld1", LBLUE), ("mma1", LGREEN), ("ld2", LBLUE), ("mma2", LGREEN)]
    x = 420
    for s, c in seq:
        E += box(x, 175, 64, 36, s, bg=c, size=12, family=3); x += 66
    E.append(text(420, 235, "2 stages (pipelined):", size=13, color=GREEN))
    # loads row
    lds = ["ld0", "ld1", "ld2"]
    for i, s in enumerate(lds):
        E += box(420 + i * 132, 262, 64, 32, s, bg=LBLUE, size=12, family=3)
    mms = ["mma0", "mma1", "mma2"]
    for i, s in enumerate(mms):
        E += box(486 + i * 132, 300, 64, 32, s, bg=LGREEN, size=12, family=3)
    E.append(text(420, 345, "loads of the next K-tile run under the current MFMA → fewer stalls.", size=12, color=GRAY))

    E += box(60, 400, 970, 56,
             "Measured: large square (1·8192·8192) 1022→1189 TF (+16%); but small (8·2048·2048) 946→753 TF (−20%).\n"
             "A single fixed tile can't be right for every shape — this is exactly what L5 (per-shape autotune) fixes.",
             bg=LYELLOW, stroke=ORANGE, size=13, family=1)
    save("06-tiles-pipeline.excalidraw", E, "tiles + pipeline")


# --- 07  GROUP_M  (L3 part 1) ------------------------------------------
def d_groupm():
    E = []
    E.append(header("L3a — GROUP_M tile ordering for L2 reuse"))
    E.append(caption("Reorder program IDs into super-blocks so concurrent tiles share A rows / B columns in L2.", 40, 60))
    cell = 46; cols = 6; rows = 6

    def grid(ox, oy, order_fn, colorfn):
        for r in range(rows):
            for c in range(cols):
                idx = order_fn(r, c)
                E.append(rect(ox + c * cell, oy + r * cell, cell - 4, cell - 4, bg=colorfn(idx), sw=1.5))
                E.append(label(ox + c * cell + (cell - 4) / 2, oy + r * cell + (cell - 4) / 2, str(idx), size=12))

    E.append(text(80, 110, "GROUP_M=1 (row-major)", size=15, color=RED))
    grid(80, 140, lambda r, c: r * cols + c, lambda i: LRED if i < cols else WHITE)
    E.append(text(80, 430, "first 6 pids = one thin B strip", size=12, color=GRAY))

    E.append(text(520, 110, "GROUP_M=3 super-block", size=15, color=GREEN))
    GM = 3
    def grouped(r, c):
        g = r // GM; return g * GM * cols + c * GM + (r % GM)
    grid(520, 140, grouped, lambda i: LGREEN if i < GM * cols else WHITE)
    E.append(text(520, 430, "first 18 pids = compact 3×6 reuse block", size=12, color=GRAY))
    save("07-group-m.excalidraw", E, "GROUP_M reuse")


# --- 08  XCD swizzle  (L3 part 2) --------------------------------------
def d_xcd():
    E = []
    E.append(header("L3b — XCD swizzle (8-XCD scheduling)"))
    E.append(caption("MI355X = 8 XCDs, each with its own L2 slice. HW dispatches pid % 8. Pre-swizzle for contiguous per-XCD ranges.", 40, 60))
    xcds = 8; cw = 116
    E.append(text(60, 115, "Without swizzle: pid % 8 scatters a reuse block across all 8 XCDs", size=15, color=RED))
    for x in range(xcds):
        E += box(60 + x * cw, 150, cw - 10, 42, f"XCD {x}", bg=LGRAY, size=13, family=3)
        E.append(label(60 + x * cw + (cw - 10) / 2, 215, f"{x},{x+8},{x+16}", size=11, color=GRAY, family=3))
    E.append(arrow(520, 250, [[0, 0], [0, 40]], color=VIOLET, sw=3))
    E.append(label(700, 270, "_xcd_swizzle(pid, domain, 8)", size=14, color=VIOLET, family=3))
    E.append(text(60, 315, "After swizzle: each XCD owns a contiguous tile range → its L2 slice sees the reuse", size=15, color=GREEN))
    cols = ["#ffc9c9", "#a5d8ff", "#b2f2bb", "#ffec99", "#d0bfff", "#ffd8a8", "#c5f6fa", "#eebefa"]
    for x in range(xcds):
        E += box(60 + x * cw, 350, cw - 10, 42, f"XCD {x}", bg=cols[x], size=13, family=3)
        lo = x * 3
        E.append(label(60 + x * cw + (cw - 10) / 2, 415, f"{lo}…{lo+2}", size=11, color=GRAY, family=3))
    E.append(text(60, 450, "L3 (GROUP_M + XCD together): modest on these large shapes; the locality story is mostly amortized once tiles are big.", size=12, color=GRAY))
    save("08-xcd-swizzle.excalidraw", E, "XCD swizzle")


# --- 09  CDNA4 scale shuffle  (L4) -------------------------------------
def d_cdna4():
    E = []
    E.append(header("L4 — CDNA4-native scale layout  (the big AMD win)"))
    E.append(caption("Feed v_mfma_scale_f32_16x16x128 its scales pre-shuffled → kill the per-K-iter permute chain. 993 → 1473 TF (+48%).", 40, 60))

    E.append(text(60, 110, "BEFORE · #blocked → #linear1 lowering (every K iteration)", size=15, color=RED))
    E += box(60, 145, 180, 48, "scales (M, K/32)\nrow-major", bg=WHITE, size=12, family=3)
    E.append(arrow(240, 169, [[0, 0], [26, 0]], color=GRAY))
    chain = ["6 × ds_read_u8", "3 × v_perm_b32"]
    x = 280
    for c in chain:
        E += box(x, 145, 150, 48, c, bg=LRED, size=12, family=3)
        E.append(arrow(x + 150, 169, [[0, 0], [22, 0]], color=GRAY)); x += 172
    E += box(x, 145, 150, 48, "MFMA scale\noperand", bg=LGRAY, size=12, family=3)

    E.append(line(40, 225, [[0, 0], [1040, 0]], color=LGRAY, dash="dashed"))
    E.append(text(60, 245, "AFTER · host pre-shuffle once, in-kernel reshape folds away", size=15, color=GREEN))
    E += box(60, 280, 220, 56, "_shuffle_*_scales_cdna4\n(host, once)", bg=LGREEN, size=12, family=3)
    E.append(arrow(280, 308, [[0, 0], [34, 0]], color=INK))
    E += box(324, 280, 240, 56, "scales already in\nMFMA-native order", bg=LBLUE, size=13, family=3)
    E.append(arrow(564, 308, [[0, 0], [34, 0]], color=INK))
    E += box(608, 280, 230, 56, "1 coalesced load →\ntl.dot_scaled", bg=LBLUE, size=13, family=3)

    E += box(60, 370, 300, 80, "in-kernel _unswizzle is a pure\ntl.reshape/permute → compiles to\nv_perm_b32 count = 0 (checked in ISA)", bg=LGREEN, stroke=GREEN, size=12, family=1)
    E += box(390, 370, 300, 80, "gate: BLOCK_K≥256 & K%256==0\n& N%32==0 & M%32==0\n(needs the L2 big-tile rung first)", bg=LGRAY, size=12, family=3)
    E += box(720, 370, 320, 80, "nonkdim 16 vs 32 selectable\nper shape (nk32 wins 27/36)\nDOMINANT optimization", bg=LYELLOW, stroke=ORANGE, size=14)
    save("09-cdna4-scales.excalidraw", E, "CDNA4 scale shuffle")


# --- 10  dot_scaled K-loop + EVEN_K ------------------------------------
def d_kloop():
    E = []
    E.append(header("Inside the inner loop — dot_scaled + EVEN_K peel"))
    E.append(caption("Scales ride into the MFMA (no dequant pass); the ragged K tail is peeled so the main loop is branch-free.", 40, 60))
    lx, ly, lw, lh = 60, 130, 540, 200
    E.append(text(60, 105, "Main loop · full BLOCK_K tiles", size=15, color=BLUE))
    E.append(rect(lx, ly, lw, lh, bg="#f8f9fa", stroke=BLUE))
    E += box(lx + 20, ly + 22, 150, 40, "load X tile", bg=LBLUE, size=13, family=3)
    E += box(lx + 20, ly + 78, 150, 40, "load W tile", bg=LBLUE, size=13, family=3)
    E += box(lx + 190, ly + 22, 160, 40, "load X/W scales", bg=LYELLOW, size=12, family=3)
    E += box(lx + 190, ly + 78, 160, 96, "tl.dot_scaled\ne4m3 × e4m3\nacc += , fast_math", bg=LGREEN, size=13, family=3)
    E += box(lx + 370, ly + 22, 150, 40, "advance ptrs\n+= BLOCK_K", bg=LGRAY, size=12, family=3)
    E.append(arrow(lx + lw / 2, ly + lh, [[0, 0], [0, 26], [-lw / 2 + 20, 26], [-lw / 2 + 20, -lh + 8]], color=BLUE, dash="dashed"))

    E.append(text(660, 105, "EVEN_K peel", size=15, color=ORANGE))
    E += box(660, 130, 330, 64, "K % BLOCK_K == 0 → main loop runs\nall iters, no per-iter K-mask", bg=LGREEN, size=13, family=3)
    E += box(660, 206, 330, 80, "else → run num_k_iter−1 unmasked,\nthen ONE masked tail iteration\n(offs_k < MASK_K_LIMIT)", bg=LORANGE, size=13, family=3)

    E += box(60, 360, 930, 50,
             'acc = tl.dot_scaled(x, x_scales, "e4m3", w, w_scales, "e4m3", acc=acc, fast_math=True)',
             bg=LGRAY, size=14, family=3)
    save("10-dot-scaled-kloop.excalidraw", E, "dot_scaled K-loop")


# --- 11  ladder results bar chart --------------------------------------
def d_ladder():
    E = []
    E.append(header("The ladder — measured geomean TFLOPS"))
    E.append(caption("6 representative Llama4 shapes on MI355X. Each rung adds one optimization; bars show MXFP8 throughput.", 40, 60))

    bars = [
        ("L0\nnaive", 914, LRED),
        ("L1\n+routing", 1040, LBLUE),
        ("L2\n+tiles", 954, LBLUE),
        ("L3\n+sched", 993, LBLUE),
        ("L4\n+cdna4\nscales", 1473, LGREEN),
        ("L5\n+autotune", 1585, LGREEN),
    ]
    maxtf = 1585.0
    base_y = 470; scale = 380.0 / maxtf; x0 = 130; bw = 120; gap = 40
    span = len(bars) * (bw + gap) + 10
    # axes
    E.append(line(x0 - 30, base_y, [[0, 0], [span, 0]], color=INK))
    E.append(line(x0 - 30, base_y, [[0, 0], [0, -400]], color=INK))
    E.append(label(x0 - 65, base_y - 200, "TFLOPS", size=13, color=GRAY))
    # naive baseline reference line
    ry = base_y - bars[0][1] * scale
    E.append(line(x0 - 30, ry, [[0, 0], [span, 0]], color=RED, dash="dashed"))
    E.append(label(x0 + 55, ry - 13, "naive = 914 TF", size=12, color=RED))
    for i, (name, val, col) in enumerate(bars):
        bx = x0 + i * (bw + gap)
        h = val * scale
        E.append(rect(bx, base_y - h, bw, h, bg=col, sw=1.5))
        E.append(label(bx + bw / 2, base_y - h - 30, f"{val} TF", size=15, color=INK))
        E.append(label(bx + bw / 2, base_y - h - 12, f"{val/bars[0][1]:.2f}× naive", size=11, color=GRAY))
        E.append(label(bx + bw / 2, base_y + 26, name, size=13))
    E.append(text(130, 540,
                  "Routing helps small shapes; tiles/scheduling are a wash at a fixed config; the CDNA4 scale layout (L4)",
                  size=13, color=GRAY))
    E.append(text(130, 562,
                  "is the dominant win (+48% TF); per-shape autotuning (L5) adds another +8% → 1.74× the naive throughput.",
                  size=13, color=GRAY))
    save("11-ladder-results.excalidraw", E, "ladder results")


# --- 12  per-shape autotune  (L5) --------------------------------------
def d_autotune():
    E = []
    E.append(header("L5 — Per-shape autotuning  (→ 1585 TFLOPS)"))
    E.append(caption("576 configs × 36 shapes, one shape per GPU across 8 MI355X. Frozen into _BEST_CFGS, looked up at launch.", 40, 60))
    space = ["BLOCK_M ∈ {64,128,256}", "BLOCK_N ∈ {128,256}", "BLOCK_K ∈ {128,256}",
             "GROUP_M ∈ {1,4,8}", "num_warps ∈ {4,8}", "num_stages ∈ {1,2}",
             "waves_per_eu ∈ {0,2}", "nonkdim ∈ {16,32}"]
    y = 120
    for s in space:
        E += box(60, y, 250, 32, s, bg=LVIOLET, size=12, family=3); y += 40
    E.append(arrow(320, 270, [[0, 0], [60, 0]], color=INK, sw=3))
    E.append(text(400, 110, "ProcessPoolExecutor(8)", size=15, color=BLUE))
    for i in range(8):
        gx = 400 + (i % 4) * 84; gy = 145 + (i // 4) * 70
        E += box(gx, gy, 76, 58, f"GPU {i}", bg=LBLUE, size=12, family=3)
    E.append(text(400, 290, "CUDA/HIP_VISIBLE_DEVICES=i\n~13–18 min full sweep", size=12, color=GRAY, family=3))
    E.append(arrow(740, 270, [[0, 0], [55, 0]], color=INK, sw=3))
    E += box(810, 130, 250, 150, "_BEST_CFGS[(E,N,K)]\n\n36 tuned entries\n+ small/large-K\nfallback\n\n_pick_config()", bg=LGREEN, size=13, family=3)

    E += box(60, 470, 460, 64, "Fixes the L2 tile mismatch: small shapes get small tiles,\nlarge get 256/256 + nk32 — best of both.", bg=LYELLOW, stroke=ORANGE, size=13)
    E += box(560, 470, 500, 64, "Representative geomean 1473 → 1585 TFLOPS\n(1.61× → 1.74× the naive throughput).", bg=LGREEN, stroke=GREEN, size=14)
    save("12-autotune.excalidraw", E, "per-shape autotune")


if __name__ == "__main__":
    d_mxblock(); d_scalecalc(); d_grouped(); d_naive()
    d_routing(); d_packed(); d_tiles(); d_groupm(); d_xcd()
    d_cdna4(); d_kloop(); d_ladder(); d_autotune()
    print("done.")
