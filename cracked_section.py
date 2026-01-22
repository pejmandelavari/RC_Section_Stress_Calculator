# app.py
# -------------------------------------------------------------------------
# RC Cracked Service Section Solver (N, Mx, My) — SPColumn Text Input
#
# SIGN CONVENTIONS (LOCKED / VERIFIED PATH):
#   - INTERNAL (analysis): Compression is POSITIVE.
#   - Axes: x to the right, y up. Axes pass through concrete centroid.
#   - User convention for moments:
#       Mx > 0 => bottom of section (y < 0) in compression
#       My > 0 => right side of section (x > 0) in compression
#     Implemented by:
#       Mx = - Σ(F * y)
#       My = + Σ(F * x)
#
# OUTPUT DISPLAY OPTION:
#   - You can choose to display steel stress with TENSION POSITIVE (common in practice),
#     while keeping INTERNAL compression-positive sign for equilibrium.
#
# Fixes included:
#   - Stable combo switching with st.session_state (no blank page after switching)
#   - Concrete contour masking (no contour outside concrete / into voids)
#   - Neutral axis clipped to concrete region
#   - PDF now prints ALL rebars (no truncation to 70)
#   - Correct reporting of max steel tension/compression for compression-positive internal sign
#   - Governing combo identification consistent with chosen reporting convention
#
# Units:
#   Geometry x,y in mm; Rebar As in mm²; Ec/Es in MPa (=N/mm²)
#   Loads:
#       Combo, N_kN, Mx_kNm, My_kNm  (Mx/My in kN·m)
# -------------------------------------------------------------------------

import io
import hashlib
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep
import scipy.optimize as opt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader


# ----------------------------
# Utilities
# ----------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def fmt_mpa(v: float) -> str:
    return f"{v:.2f} MPa"


# ----------------------------
# Parse SPColumn section text
# ----------------------------
def parse_spcolumn_section_text(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 0

    def expect(tag: str):
        nonlocal i
        if i >= len(lines) or lines[i].upper() != tag:
            got = lines[i] if i < len(lines) else "EOF"
            raise ValueError(f"Expected '{tag}' but got '{got}'.")
        i += 1

    def read_int():
        nonlocal i
        v = int(float(lines[i]))
        i += 1
        return v

    def read_xy():
        nonlocal i
        a, b = map(float, lines[i].split())
        i += 1
        return (a, b)

    solids, openings, rebars = [], [], []

    expect("SOLIDS")
    n_solids = read_int()
    for _ in range(n_solids):
        n_pts = read_int()
        solids.append([read_xy() for __ in range(n_pts)])

    expect("OPENINGS")
    n_open = read_int()
    for _ in range(n_open):
        n_pts = read_int()
        openings.append([read_xy() for __ in range(n_pts)])

    expect("REINFORCEMENT")
    n_bars = read_int()
    for _ in range(n_bars):
        As, x, y = map(float, lines[i].split())
        i += 1
        rebars.append((As, x, y))

    return solids, openings, rebars


# ----------------------------
# Center geometry at concrete centroid
# ----------------------------
def build_centered_geometry(solids, openings):
    solid_polys = [Polygon(p) for p in solids]
    if any(not p.is_valid for p in solid_polys):
        raise ValueError("One or more SOLIDS polygons are invalid.")

    region = unary_union(solid_polys)

    opening_polys = [Polygon(p) for p in openings]
    for op in opening_polys:
        if not op.is_valid:
            raise ValueError("One or more OPENINGS polygons are invalid.")
        region = region.difference(op)

    if region.is_empty:
        raise ValueError("Concrete region is empty after subtracting openings.")

    cx, cy = region.centroid.x, region.centroid.y

    def shift_poly(poly):
        return [(x - cx, y - cy) for (x, y) in poly]

    solids_c = [shift_poly(p) for p in solids]
    openings_c = [shift_poly(p) for p in openings]

    region_c = unary_union([Polygon(p) for p in solids_c])
    for op in openings_c:
        region_c = region_c.difference(Polygon(op))

    if region_c.is_empty:
        raise ValueError("Centered concrete region is empty (unexpected).")

    return region_c, solids_c, openings_c, (cx, cy)


# ----------------------------
# Cached fiber meshing (grid)
# ----------------------------
@st.cache_data(show_spinner=False)
def build_concrete_fibers_cached(section_hash: str, solids_c, openings_c, mesh_mm: float):
    region = unary_union([Polygon(p) for p in solids_c])
    for op in openings_c:
        region = region.difference(Polygon(op))

    minx, miny, maxx, maxy = region.bounds
    dx = float(mesh_mm)
    if dx <= 0:
        raise ValueError("Mesh size must be > 0.")

    xs = np.arange(minx + dx / 2.0, maxx, dx)
    ys = np.arange(miny + dx / 2.0, maxy, dx)

    fx, fy, fA = [], [], []
    cellA = dx * dx

    for x in xs:
        for y in ys:
            if region.contains(Point(x, y)):
                fx.append(x)
                fy.append(y)
                fA.append(cellA)

    if len(fx) == 0:
        raise ValueError("No fibers created. Reduce mesh size or check geometry.")

    return {
        "x": np.array(fx, dtype=float),
        "y": np.array(fy, dtype=float),
        "A": np.array(fA, dtype=float),
        "bounds": (float(minx), float(miny), float(maxx), float(maxy)),
        "cell_area": float(cellA),
    }


# ----------------------------
# Cracked service model
# eps(x,y)=eps0 - kx*y + ky*x
# Concrete: compression only
# Steel: linear
# Moment convention locked:
#   Mx = - Σ(F*y)
#   My = + Σ(F*x)
# ----------------------------
def eps_xy(eps0, kx, ky, x, y):
    return eps0 - kx * y + ky * x


def section_response(eps0, kx, ky, conc, reb, Ec, Es):
    xc, yc, Ac = conc["x"], conc["y"], conc["A"]
    e_c = eps_xy(eps0, kx, ky, xc, yc)
    comp = e_c > 0.0
    sig_c = np.zeros_like(e_c)
    sig_c[comp] = Ec * e_c[comp]  # MPa (compression +)
    Fc = sig_c * Ac               # N

    As, xs, ys = reb["As"], reb["x"], reb["y"]
    e_s = eps_xy(eps0, kx, ky, xs, ys)
    sig_s = Es * e_s              # MPa (compression +, tension -)
    Fs = sig_s * As               # N

    N = Fc.sum() + Fs.sum()

    # Moment convention locked to match user's description:
    Mx = -((Fc * yc).sum() + (Fs * ys).sum())  # N*mm
    My = +((Fc * xc).sum() + (Fs * xs).sum())  # N*mm

    # Concrete compression area and resultant (useful for verification)
    Acomp = float(Ac[comp].sum()) if comp.any() else 0.0
    Cc = float(Fc.sum())  # N (compression +)

    return {
        "N": float(N),
        "Mx": float(Mx),
        "My": float(My),
        "sig_s": sig_s,
        "sig_c": sig_c,
        "comp_mask": comp,
        "sig_c_max": float(sig_c.max()) if comp.any() else 0.0,
        "Acomp": Acomp,
        "Cc": Cc,
    }


def initial_guess(Nt, conc, reb, Ec, Es):
    Ac = float(np.sum(conc["A"]))
    As = float(np.sum(reb["As"]))
    eps0_0 = Nt / (Ec * Ac + Es * As + 1e-9)
    return np.array([eps0_0, 1e-6, 1e-6], dtype=float)


def solve_combo_strict(Nt, Mxt, Myt, conc, reb, Ec, Es, x0, N_tol_N, M_tol_Nmm):
    Nref = max(abs(Nt), 1e6)
    Mxref = max(abs(Mxt), 1e12)
    Myref = max(abs(Myt), 1e12)

    def R(x):
        r = section_response(x[0], x[1], x[2], conc, reb, Ec, Es)
        return np.array([
            (r["N"] - Nt) / Nref,
            (r["Mx"] - Mxt) / Mxref,
            (r["My"] - Myt) / Myref,
        ], dtype=float)

    lb = np.array([-0.02, -2e-3, -2e-3], dtype=float)
    ub = np.array([ 0.02,  2e-3,  2e-3], dtype=float)

    sol = opt.least_squares(
        R, x0, bounds=(lb, ub),
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
        max_nfev=1200
    )

    x = sol.x.astype(float)
    r = section_response(x[0], x[1], x[2], conc, reb, Ec, Es)

    errN  = r["N"]  - Nt
    errMx = r["Mx"] - Mxt
    errMy = r["My"] - Myt

    ok = (abs(errN) <= N_tol_N) and (abs(errMx) <= M_tol_Nmm) and (abs(errMy) <= M_tol_Nmm)
    res_norm = float(np.linalg.norm(sol.fun))
    return ok, x, r, errN, errMx, errMy, res_norm


# ----------------------------
# Plots
# ----------------------------
def plot_section(solids_c, openings_c, reb, show_ids, rebar_radius_mm):
    fig, ax = plt.subplots()

    for poly in solids_c:
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, linewidth=1)

    for poly in openings_c:
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, linewidth=1)

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    for i, (x, y) in enumerate(zip(reb["x"], reb["y"]), start=1):
        circ = plt.Circle((x, y), rebar_radius_mm, fill=False)
        ax.add_patch(circ)
        if show_ids:
            ax.text(x, y, str(i), fontsize=7, ha="center", va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)  [Right +]")
    ax.set_ylabel("y (mm)  [Up +]")
    ax.set_title("Section (Centered at Concrete Centroid)")
    ax.grid(True, linewidth=0.5)

    return fig


def plot_concrete_contour_and_na(solids_c, openings_c, conc, reb, Ec, eps0, kx, ky, title):
    region = unary_union([Polygon(p) for p in solids_c])
    for op in openings_c:
        region = region.difference(Polygon(op))
    pregion = prep(region)

    xc, yc = conc["x"], conc["y"]
    e_c = eps_xy(eps0, kx, ky, xc, yc)
    comp = e_c > 0.0
    sig_c = np.zeros_like(e_c)
    sig_c[comp] = Ec * e_c[comp]

    fig, ax = plt.subplots()

    for poly in solids_c:
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, linewidth=1)
    for poly in openings_c:
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, linewidth=1)

    ax.scatter(reb["x"], reb["y"], s=18)

    if comp.sum() >= 30:
        xcp = xc[comp]; ycp = yc[comp]; scp = sig_c[comp]
        tri = mtri.Triangulation(xcp, ycp)

        tris = tri.triangles
        cx_t = (xcp[tris[:, 0]] + xcp[tris[:, 1]] + xcp[tris[:, 2]]) / 3.0
        cy_t = (ycp[tris[:, 0]] + ycp[tris[:, 1]] + ycp[tris[:, 2]]) / 3.0
        mask = np.array([not pregion.contains(Point(x, y)) for x, y in zip(cx_t, cy_t)], dtype=bool)
        tri.set_mask(mask)

        cf = ax.tricontourf(tri, scp, levels=12)
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Concrete compressive stress (MPa)  [Compression +]")
    else:
        ax.text(0.02, 0.98, "Compression region too small for contour",
                transform=ax.transAxes, ha="left", va="top")

    # NA: eps=0 => eps0 - kx*y + ky*x = 0
    minx, miny, maxx, maxy = conc["bounds"]
    xs_line = np.linspace(minx, maxx, 500)
    if abs(kx) > 1e-12:
        ys_line = (eps0 + ky * xs_line) / kx
        inside = np.array([pregion.contains(Point(x, y)) for x, y in zip(xs_line, ys_line)], dtype=bool)
        ax.plot(xs_line[inside], ys_line[inside], linewidth=2)
    elif abs(ky) > 1e-12:
        x_na = -eps0 / ky
        ys_line = np.linspace(miny, maxy, 500)
        inside = np.array([pregion.contains(Point(x_na, y)) for y in ys_line], dtype=bool)
        ax.plot([x_na]*inside.sum(), ys_line[inside], linewidth=2)
    else:
        ax.text(0.02, 0.90, "Neutral axis undefined (kx≈0 & ky≈0)",
                transform=ax.transAxes, ha="left", va="top")

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    ax.set_aspect("equal", adjustable="box")
    pad = max(maxx - minx, maxy - miny) * 0.15
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.5)
    return fig


# ----------------------------
# PDF
# ----------------------------
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def _pdf_draw_wrapped(c: canvas.Canvas, x, y, text, width_chars=95, leading=12):
    lines = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
            continue
        lines += textwrap.wrap(para, width=width_chars)
    for ln in lines:
        c.drawString(x, y, ln)
        y -= leading
    return y

def build_pdf_report_bytes(report_title, meta: dict, section_png: bytes, contour_png: bytes,
                           summary_df: pd.DataFrame, bar_df: pd.DataFrame,
                           notes: str, steel_report_convention: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter

    y = H - 0.8*inch
    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.8*inch, y, report_title)
    y -= 0.30*inch

    c.setFont("Helvetica", 10)
    c.drawString(0.8*inch, y, f"Steel stress reporting: {steel_report_convention}")
    y -= 0.22*inch

    for k, v in meta.items():
        c.drawString(0.8*inch, y, f"{k}: {v}")
        y -= 0.18*inch

    if notes.strip():
        y -= 0.08*inch
        c.setFont("Helvetica-Oblique", 9)
        y = _pdf_draw_wrapped(c, 0.8*inch, y, "Notes:\n" + notes, width_chars=100, leading=11)

    y -= 0.10*inch
    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.8*inch, y, "Section Geometry")
    y -= 0.15*inch
    img1 = ImageReader(io.BytesIO(section_png))
    c.drawImage(img1, 0.8*inch, y-3.25*inch, width=6.9*inch, height=3.25*inch, preserveAspectRatio=True, anchor='sw')
    y -= 3.45*inch

    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.8*inch, y, "Concrete Compression Contour + Neutral Axis (Selected Combo)")
    y -= 0.15*inch
    img2 = ImageReader(io.BytesIO(contour_png))
    c.drawImage(img2, 0.8*inch, y-3.25*inch, width=6.9*inch, height=3.25*inch, preserveAspectRatio=True, anchor='sw')

    c.showPage()

    # Summary table (all combos shown in the app; PDF prints first N rows to keep size reasonable)
    y = H - 0.8*inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.8*inch, y, "Summary Results (first 80 rows)")
    y -= 0.25*inch
    c.setFont("Helvetica", 8)

    cols = [
        "Combo", "Converged",
        "sigma_c_max_MPa",
        "steel_max_tension_MPa", "steel_max_compression_MPa",
        "Acomp_mm2",
        "errN_kN", "errMx_kNm", "errMy_kNm"
    ]
    cols = [cc for cc in cols if cc in summary_df.columns]
    dfp = summary_df[cols].copy().head(80)

    header = " | ".join([f"{col:>16s}" for col in cols])
    c.drawString(0.8*inch, y, header[:115])
    y -= 0.18*inch

    for _, rr in dfp.iterrows():
        def v(col):
            val = rr[col]
            if isinstance(val, (float, int, np.floating, np.integer)):
                if col.endswith("_MPa"):
                    return f"{float(val):.2f}"
                if col.startswith("err"):
                    return f"{float(val):.3f}"
                if col.endswith("_mm2"):
                    return f"{float(val):.0f}"
                return f"{float(val):.5g}"
            return str(val)

        line = " | ".join([f"{v(col):>16s}" for col in cols])
        c.drawString(0.8*inch, y, line[:115])
        y -= 0.16*inch
        if y < 1.0*inch:
            c.showPage()
            y = H - 0.8*inch
            c.setFont("Helvetica", 8)

    c.showPage()

    # Rebar table — PRINT ALL REBARS (no truncation)
    y = H - 0.8*inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.8*inch, y, "Rebar Stresses (Selected Combo) — All Rebars")
    y -= 0.25*inch
    c.setFont("Helvetica", 8)

    bar_cols = ["Rebar_ID", "As_mm2", "x_mm", "y_mm", "sigma_s_MPa"]
    bar_cols = [cc for cc in bar_cols if cc in bar_df.columns]
    bdf = bar_df[bar_cols].copy()  # ALL

    header = " | ".join([f"{col:>12s}" for col in bar_cols])
    c.drawString(0.8*inch, y, header)
    y -= 0.18*inch

    for _, rr in bdf.iterrows():
        vals = []
        for col in bar_cols:
            val = rr[col]
            if col == "Rebar_ID":
                vals.append(f"{int(val):d}")
            elif isinstance(val, (float, int, np.floating, np.integer)):
                if col in ("As_mm2", "x_mm", "y_mm"):
                    vals.append(f"{float(val):.1f}")
                elif col == "sigma_s_MPa":
                    vals.append(f"{float(val):.2f}")
                else:
                    vals.append(f"{float(val):.4g}")
            else:
                vals.append(str(val))
        line = " | ".join([f"{vv:>12s}" for vv in vals])
        c.drawString(0.8*inch, y, line)
        y -= 0.16*inch
        if y < 1.0*inch:
            c.showPage()
            y = H - 0.8*inch
            c.setFont("Helvetica", 8)

    c.save()
    buf.seek(0)
    return buf.getvalue()


# ----------------------------
# Streamlit state init
# ----------------------------
if "has_results" not in st.session_state:
    st.session_state.has_results = False
if "run_hash" not in st.session_state:
    st.session_state.run_hash = None


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="RC Cracked Service Section Solver", layout="wide")
st.title("RC Cracked Service Section Solver (Cracked Service) — Steel Sign-Control Fixed")

with st.sidebar:
    st.header("Inputs")
    sec_file = st.file_uploader("SPColumn Section File (.txt)", type=["txt"])
    load_file = st.file_uploader("Load Combos File (.xlsx or .csv)", type=["xlsx", "csv"])

    st.subheader("Material")
    Ec = st.number_input("Concrete modulus Ec (MPa)", value=30000.0, step=1000.0)
    Es = st.number_input("Steel modulus Es (MPa)", value=200000.0, step=5000.0)

    st.subheader("Concrete Fiber Mesh")
    mesh_mm = st.number_input("Mesh size (mm)", value=20.0, min_value=2.0, step=1.0)

    st.subheader("Strict Convergence")
    N_tol_kN = st.number_input("Force tolerance |ΔN| (kN)", value=2.0, min_value=0.1, step=0.5)
    M_tol_kNm = st.number_input("Moment tolerance |ΔMx|,|ΔMy| (kN·m)", value=2.0, min_value=0.1, step=0.5)

    st.subheader("Reporting Convention")
    tension_positive_output = st.checkbox("Display steel stress with Tension POSITIVE", value=True)

    st.subheader("Geometry Options")
    swap_xy = st.checkbox("Swap X and Y", value=False)

    st.subheader("Plot Options")
    show_rebar_ids = st.checkbox("Show rebar IDs", value=False)
    rebar_radius_mm = st.number_input("Rebar display radius (mm)", value=10.0, min_value=1.0, step=1.0)

    st.subheader("Report")
    report_title = st.text_input("PDF report title", value="RC Cracked Service Section Report")
    report_notes = st.text_area("Report notes (optional)", value="")

    run_btn = st.button("Run analysis")


if not sec_file or not load_file:
    st.info("Upload a section file and a load combos file to start.")
    st.stop()

# Hash all run-critical inputs
sec_bytes = sec_file.getvalue()
load_bytes = load_file.getvalue()
run_hash = sha256_bytes(sec_bytes + load_bytes + f"{Ec}-{Es}-{mesh_mm}-{N_tol_kN}-{M_tol_kNm}-{swap_xy}-{tension_positive_output}".encode("utf-8"))

# Parse section
sec_text = sec_bytes.decode("utf-8", errors="ignore")
solids, openings, rebars_list = parse_spcolumn_section_text(sec_text)
region_c, solids_c, openings_c, (cx, cy) = build_centered_geometry(solids, openings)

reb = {
    "As": np.array([r[0] for r in rebars_list], dtype=float),
    "x": np.array([r[1] - cx for r in rebars_list], dtype=float),
    "y": np.array([r[2] - cy for r in rebars_list], dtype=float),
}

conc = build_concrete_fibers_cached(sha256_bytes(sec_bytes), solids_c, openings_c, float(mesh_mm))

if swap_xy:
    reb = {"As": reb["As"], "x": reb["y"].copy(), "y": reb["x"].copy()}
    conc = {"x": conc["y"].copy(), "y": conc["x"].copy(), "A": conc["A"].copy(), "bounds": conc["bounds"], "cell_area": conc["cell_area"]}

# Load combos
if load_file.name.lower().endswith(".csv"):
    combos = pd.read_csv(io.BytesIO(load_bytes))
else:
    combos = pd.read_excel(io.BytesIO(load_bytes))

required_cols = {"Combo", "N_kN", "Mx_kNm", "My_kNm"}
missing = required_cols - set(combos.columns)
if missing:
    st.error(f"Load file missing columns: {sorted(missing)}")
    st.stop()

# Always show section
colA, colB = st.columns([1.2, 0.8])
with colA:
    st.subheader("Section")
    fig_geom = plot_section(solids_c, openings_c, reb, show_rebar_ids, float(rebar_radius_mm))
    st.pyplot(fig_geom, use_container_width=True)

with colB:
    st.subheader("Summary")
    st.write(f"Centroid shift applied: **dx={cx:.2f} mm**, **dy={cy:.2f} mm**")
    st.write(f"Rebars: **{len(rebars_list)}**")
    st.write(f"Concrete fibers: **{len(conc['A'])}**")
    st.write(f"Concrete fiber cell area: **{conc['cell_area']:.1f} mm²**")
    st.write("Internal sign: **Compression (+), Tension (−)**")
    st.write("Moments: **Mx>0 bottom compression**, **My>0 right compression**")

st.subheader("Load Combos Preview")
st.dataframe(combos.head(50), use_container_width=True)

# Control compute
if run_btn:
    st.session_state.has_results = False

need_compute = (not st.session_state.has_results) or (st.session_state.run_hash != run_hash)
if need_compute and not run_btn:
    st.info("Click **Run analysis**. After running, switching combos will update instantly.")
    st.stop()

if need_compute:
    st.subheader("Running analysis...")
    results = []
    x0 = None
    progress = st.progress(0)
    status = st.empty()
    n_rows = len(combos)

    N_tol_N = float(N_tol_kN) * 1000.0
    M_tol_Nmm = float(M_tol_kNm) * 1e6

    for idx, row in combos.iterrows():
        combo = str(row["Combo"])
        Nt  = float(row["N_kN"])   * 1000.0
        Mxt = float(row["Mx_kNm"]) * 1_000_000.0
        Myt = float(row["My_kNm"]) * 1_000_000.0

        if x0 is None:
            x0 = initial_guess(Nt, conc, reb, float(Ec), float(Es))

        ok, x_sol, r, errN, errMx, errMy, res_norm = solve_combo_strict(
            Nt, Mxt, Myt, conc, reb, float(Ec), float(Es),
            x0, N_tol_N, M_tol_Nmm
        )

        sig_s_int = r["sig_s"]  # internal: compression(+), tension(-)
        smax_comp = float(np.max(sig_s_int))      # + MPa (most compression)
        smin_tens = float(np.min(sig_s_int))      # - MPa (most tension signed)
        tens_mag = float(-smin_tens)              # + MPa (tension magnitude)

        Acomp = float(r["Acomp"])  # mm² (approx. from fibers)

        results.append({
            "Combo": combo,
            "Converged": bool(ok),
            "eps0": x_sol[0],
            "kx": x_sol[1],
            "ky": x_sol[2],
            "sigma_c_max_MPa": float(r["sig_c_max"]),
            "Acomp_mm2": Acomp,

            # Store BOTH: tension magnitude and compression (positive)
            "steel_max_tension_MPa": tens_mag,          # positive magnitude
            "steel_max_compression_MPa": smax_comp,     # positive

            # Also store signed most-tension for debugging
            "steel_most_tension_signed_MPa": smin_tens, # negative

            "errN_kN": errN / 1000.0,
            "errMx_kNm": errMx / 1e6,
            "errMy_kNm": errMy / 1e6,
            "scaled_residual_norm": res_norm,
        })

        x0 = x_sol.copy()
        progress.progress(min(1.0, (idx + 1) / max(1, n_rows)))
        status.write(
            f"Solving: {combo} | ΔN={errN/1000:.3f} kN | ΔMx={errMx/1e6:.3f} kN·m | ΔMy={errMy/1e6:.3f} kN·m"
        )

    res_df = pd.DataFrame(results)
    st.session_state.res_df = res_df
    st.session_state.has_results = True
    st.session_state.run_hash = run_hash

res_df = st.session_state.res_df

st.subheader("Results Summary")
st.dataframe(res_df, use_container_width=True)

# Governing combos (consistent)
gov_steel = res_df.loc[res_df["steel_max_tension_MPa"].idxmax()]
gov_conc  = res_df.loc[res_df["sigma_c_max_MPa"].idxmax()]

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### Governing steel tension (magnitude)")
    st.write(f"Combo: **{gov_steel['Combo']}**")
    st.write(f"Max steel tension: **{gov_steel['steel_max_tension_MPa']:.2f} MPa**")
with c2:
    st.markdown("### Governing concrete compression (max fiber)")
    st.write(f"Combo: **{gov_conc['Combo']}**")
    st.write(f"Max concrete compression: **{gov_conc['sigma_c_max_MPa']:.2f} MPa**")
with c3:
    st.markdown("### Concrete compression area (info)")
    st.write("Smaller Acomp does **not** always mean max stress, but it’s useful to compare.")
    st.write(f"Min Acomp: **{res_df['Acomp_mm2'].min():.0f} mm²**")
    st.write(f"Max Acomp: **{res_df['Acomp_mm2'].max():.0f} mm²**")

# Select combo
st.subheader("Selected Combo")
pick = st.selectbox("Select Combo", res_df["Combo"].tolist(), index=0, key="combo_pick")
sel = res_df[res_df["Combo"] == pick].iloc[0]

# Recompute response for selected combo
r_sel = section_response(
    float(sel["eps0"]), float(sel["kx"]), float(sel["ky"]),
    conc, reb, float(Ec), float(Es)
)
sig_s_int = r_sel["sig_s"]
sig_s_rep = (-sig_s_int) if tension_positive_output else sig_s_int

steel_report_label = "Steel stress (MPa)  [Tension +]" if tension_positive_output else "Steel stress (MPa)  [Compression +]"
bar_df = pd.DataFrame({
    "Rebar_ID": np.arange(1, len(reb["As"]) + 1),
    "As_mm2": reb["As"],
    "x_mm": reb["x"],
    "y_mm": reb["y"],
    "sigma_s_MPa": sig_s_rep,
})

st.write(f"Reporting convention: **{steel_report_label}**")
st.dataframe(bar_df, use_container_width=True)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Max concrete compression (selected)", fmt_mpa(float(sel["sigma_c_max_MPa"])))
with m2:
    st.metric("Acomp (selected)", f"{float(sel['Acomp_mm2']):.0f} mm²")
with m3:
    st.metric("Steel max tension (mag, selected)", fmt_mpa(float(sel["steel_max_tension_MPa"])))
with m4:
    st.metric("Steel max compression (selected)", fmt_mpa(float(sel["steel_max_compression_MPa"])))
with m5:
    st.metric("Converged?", str(bool(sel["Converged"])))

st.caption(
    f"Equilibrium errors: ΔN={sel['errN_kN']:.3f} kN, "
    f"ΔMx={sel['errMx_kNm']:.3f} kN·m, ΔMy={sel['errMy_kNm']:.3f} kN·m"
)

# Contour + NA
st.subheader("Concrete Compression Contour + Neutral Axis (Selected Combo)")
fig_contour = plot_concrete_contour_and_na(
    solids_c, openings_c, conc, reb, float(Ec),
    float(sel["eps0"]), float(sel["kx"]), float(sel["ky"]),
    title=f"Combo {sel['Combo']} — Concrete compression (MPa) + Neutral Axis"
)
st.pyplot(fig_contour, use_container_width=True)

# Downloads
st.subheader("Download Outputs")

# Excel
excel_out = io.BytesIO()
with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
    res_df.to_excel(writer, index=False, sheet_name="Summary")
    bar_df.to_excel(writer, index=False, sheet_name="BarStresses_Selected")
excel_out.seek(0)

st.download_button(
    "Download Results (Excel)",
    data=excel_out,
    file_name="rc_cracked_service_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# PDF
section_png = fig_to_png_bytes(fig_geom)
contour_png = fig_to_png_bytes(fig_contour)

meta = {
    "Sign convention (internal)": "Compression (+), Tension (−)",
    "Moments": "Mx>0 bottom compression; My>0 right compression",
    "Ec (MPa)": f"{float(Ec):.1f}",
    "Es (MPa)": f"{float(Es):.1f}",
    "Mesh (mm)": f"{float(mesh_mm):.1f}",
    "Centroid shift dx (mm)": f"{cx:.2f}",
    "Centroid shift dy (mm)": f"{cy:.2f}",
    "Selected combo": str(sel["Combo"]),
    "Selected converged": str(bool(sel["Converged"])),
    "Gov steel tension combo": str(gov_steel["Combo"]),
    "Gov steel tension (MPa)": f"{float(gov_steel['steel_max_tension_MPa']):.2f}",
    "Gov conc compression combo": str(gov_conc["Combo"]),
    "Gov conc compression (MPa)": f"{float(gov_conc['sigma_c_max_MPa']):.2f}",
}

steel_conv = "Tension positive (displayed)" if tension_positive_output else "Compression positive (displayed)"
pdf_bytes = build_pdf_report_bytes(
    report_title=report_title,
    meta=meta,
    section_png=section_png,
    contour_png=contour_png,
    summary_df=res_df,
    bar_df=bar_df,
    notes=report_notes,
    steel_report_convention=steel_conv
)

st.download_button(
    "Download Full PDF Report",
    data=pdf_bytes,
    file_name="rc_section_report.pdf",
    mime="application/pdf",
)

st.caption(
    "Tip for validation:\n"
    "- Set a combo with only +Mx (My≈0). You should see concrete compression at the bottom (y<0).\n"
    "- If you display steel stress with 'Tension Positive', rebars in compression will show negative values."
)
