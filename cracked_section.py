# app.py
# -----------------------------------------------------------------------------
# RC Cracked Service Section Solver (N, Mx, My) — SPColumn Text Input
#
# INTERNAL SIGN CONVENTION (for calculations):
#   - Compression is POSITIVE, tension is NEGATIVE
#
# USER MOMENT CONVENTION (LOCKED as requested):
#   - x right, y up (axes pass through concrete centroid)
#   - Mx > 0 => bottom (y<0) in compression
#   - My > 0 => right  (x>0) in compression
#   Implemented by:
#       Mx = - Σ(F*y)
#       My = + Σ(F*x)
#
# REPORTING:
#   - Steel stress shown as fs (tension positive, compression negative)
#   - Concrete compressive stress shown as fc (compression positive, concrete tension = 0)
#   - Summary uses fs_t_max (maximum tensile steel stress, MPa) and fc_max (MPa)
#
# PDF REPORT:
#   Page 1: Title + Input filenames + Assumptions + Section geometry figure (bigger)
#   Page 2+: Summary table (starts at top of Page 2, yellow highlight for governing)
#   Next page: Governing steel tension combo contour (fc) + neutral axis (fixed size)
#   Next page (if different): Governing concrete compression combo contour (fc) + neutral axis (fixed size)
#
# FIX (IMPORTANT):
#   Residual scaling uses realistic floors:
#     Nref  >= 10 kN
#     Mref  >= 10 kN·m
#
# LATEST UI/PDF FIXES:
#   - Governing contour plots are same size: fixed figsize + fixed ReportLab box
#   - Page-1 section figure is larger to reduce white space
# -----------------------------------------------------------------------------

import io
import hashlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep

import scipy.optimize as opt

from PIL import Image as PILImage  # pillow

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
)
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# Helpers
# =========================
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


def rl_image_fit(png_bytes: bytes, max_w: float, max_h: float) -> Image:
    """ReportLab Image scaled to fit max_w,max_h preserving aspect ratio."""
    im = PILImage.open(io.BytesIO(png_bytes))
    w_px, h_px = im.size
    scale = min(max_w / w_px, max_h / h_px)
    w = w_px * scale
    h = h_px * scale
    return Image(io.BytesIO(png_bytes), width=w, height=h)


def rl_image_fit_box(png_bytes: bytes, box_w: float, box_h: float) -> Image:
    """
    Fit image into a fixed box (same box for all), preserving aspect ratio.
    Using a fixed box ensures two governing plots appear the same size in PDF.
    """
    im = PILImage.open(io.BytesIO(png_bytes))
    w_px, h_px = im.size
    scale = min(box_w / w_px, box_h / h_px)
    w = w_px * scale
    h = h_px * scale
    return Image(io.BytesIO(png_bytes), width=w, height=h)


# =========================
# Parse SPColumn section text
# =========================
Point2D = Tuple[float, float]
RebarRow = Tuple[float, float, float]  # (As, x, y)


def parse_spcolumn_section_text(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 0

    def expect(tag: str):
        nonlocal i
        if i >= len(lines) or lines[i].upper() != tag:
            got = lines[i] if i < len(lines) else "EOF"
            raise ValueError(f"Expected '{tag}' but got '{got}'.")
        i += 1

    def read_int() -> int:
        nonlocal i
        v = int(float(lines[i]))
        i += 1
        return v

    def read_xy() -> Point2D:
        nonlocal i
        a, b = map(float, lines[i].split())
        i += 1
        return (a, b)

    solids: List[List[Point2D]] = []
    openings: List[List[Point2D]] = []
    rebars: List[RebarRow] = []

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


# =========================
# Geometry: center at concrete centroid
# =========================
def build_centered_geometry(solids, openings):
    solid_polys = [Polygon(p) for p in solids]
    if any(not p.is_valid for p in solid_polys):
        raise ValueError("One or more SOLIDS polygons are invalid (self-intersection etc.).")

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


# =========================
# Concrete fibers (grid) — cached
# =========================
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


# =========================
# Strain field and section response
# eps(x,y)=eps0 - kx*y + ky*x
# Concrete: compression only (fc = Ec*eps if eps>0 else 0)
# Steel: fs_int = Es*eps (compression +, tension -)
# Locked moment convention:
#   Mx = - Σ(F*y)
#   My = + Σ(F*x)
# =========================
def eps_xy(eps0, kx, ky, x, y):
    return eps0 - kx * y + ky * x


def section_response(eps0, kx, ky, conc, reb, Ec, Es):
    xc, yc, Ac = conc["x"], conc["y"], conc["A"]

    e_c = eps_xy(eps0, kx, ky, xc, yc)
    comp = e_c > 0.0
    fc = np.zeros_like(e_c)
    fc[comp] = Ec * e_c[comp]  # MPa, compression+

    Fc = fc * Ac  # N (MPa=N/mm2)

    As, xs, ys = reb["As"], reb["x"], reb["y"]
    e_s = eps_xy(eps0, kx, ky, xs, ys)
    fs_int = Es * e_s  # MPa (compression +, tension -)
    Fs = fs_int * As   # N

    N = Fc.sum() + Fs.sum()

    Mx = -((Fc * yc).sum() + (Fs * ys).sum())  # N*mm
    My = +((Fc * xc).sum() + (Fs * xs).sum())  # N*mm

    Acomp = float(Ac[comp].sum()) if comp.any() else 0.0
    fc_max = float(fc.max()) if comp.any() else 0.0

    return {
        "N": float(N),
        "Mx": float(Mx),
        "My": float(My),
        "fs_int": fs_int,
        "fc": fc,
        "comp_mask": comp,
        "fc_max": fc_max,
        "Acomp": Acomp,
    }


def initial_guess(Nt, conc, reb, Ec, Es):
    Ac = float(np.sum(conc["A"]))
    As = float(np.sum(reb["As"]))
    eps0_0 = Nt / (Ec * Ac + Es * As + 1e-9)
    return np.array([eps0_0, 1e-6, 1e-6], dtype=float)


# =========================
# SOLVER (FIXED SCALING)
# =========================
def solve_combo_strict(Nt, Mxt, Myt, conc, reb, Ec, Es, x0, N_tol_N, M_tol_Nmm):
    # IMPORTANT FIX: realistic floors (10 kN, 10 kN·m)
    Nref  = max(abs(Nt),  10e3)   # 10 kN = 10e3 N
    Mxref = max(abs(Mxt), 10e6)   # 10 kN·m = 10e6 N·mm
    Myref = max(abs(Myt), 10e6)

    def R(x):
        r = section_response(x[0], x[1], x[2], conc, reb, Ec, Es)
        return np.array([
            (r["N"]  - Nt)  / Nref,
            (r["Mx"] - Mxt) / Mxref,
            (r["My"] - Myt) / Myref,
        ], dtype=float)

    # bounds (wide)
    lb = np.array([-0.03, -5e-3, -5e-3], dtype=float)
    ub = np.array([ 0.03,  5e-3,  5e-3], dtype=float)

    sol = opt.least_squares(
        R, x0, bounds=(lb, ub),
        loss="linear",
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
        max_nfev=2500
    )

    x = sol.x.astype(float)
    r = section_response(x[0], x[1], x[2], conc, reb, Ec, Es)

    errN  = r["N"]  - Nt
    errMx = r["Mx"] - Mxt
    errMy = r["My"] - Myt

    ok = (abs(errN) <= N_tol_N) and (abs(errMx) <= M_tol_Nmm) and (abs(errMy) <= M_tol_Nmm)
    near = (abs(errN) <= 1.5 * N_tol_N) and (abs(errMx) <= 1.5 * M_tol_Nmm) and (abs(errMy) <= 1.5 * M_tol_Nmm)

    hit_bounds = (
        np.any(np.isclose(x, lb, rtol=0, atol=1e-12)) or
        np.any(np.isclose(x, ub, rtol=0, atol=1e-12))
    )

    if ok:
        status = "OK"
        reason = ""
    elif near:
        status = "NEAR"
        reason = "Within 1.5× tolerances"
    else:
        status = "FAIL"
        if hit_bounds:
            reason = "Hit solver bounds (eps0/kx/ky)"
        else:
            reason = "Residual too high (check loads sign/units/reference)"

    res_norm = float(np.linalg.norm(sol.fun))

    return ok, near, status, reason, x, r, errN, errMx, errMy, res_norm


# =========================
# Plots
# =========================
def plot_section(solids_c, openings_c, reb, show_ids=False, rebar_radius_mm=10.0):
    fig, ax = plt.subplots(figsize=(7.8, 4.6))  # slightly larger for nicer PDF capture
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
    fig.tight_layout()
    return fig


def plot_fc_contour_and_na(solids_c, openings_c, conc, reb, Ec, eps0, kx, ky, title):
    # FIX: fixed figsize so governing plots match size
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    region = unary_union([Polygon(p) for p in solids_c])
    for op in openings_c:
        region = region.difference(Polygon(op))
    pregion = prep(region)

    xc, yc = conc["x"], conc["y"]
    e_c = eps_xy(eps0, kx, ky, xc, yc)
    comp = e_c > 0.0

    fc = np.zeros_like(e_c)
    fc[comp] = Ec * e_c[comp]

    # boundary
    for poly in solids_c:
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, linewidth=1)
    for poly in openings_c:
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, linewidth=1)

    # rebars
    ax.scatter(reb["x"], reb["y"], s=16)

    # contour only within compression region (and clipped to polygon via tri mask)
    if comp.sum() >= 30:
        xcp, ycp, fcp = xc[comp], yc[comp], fc[comp]
        tri = mtri.Triangulation(xcp, ycp)

        tris = tri.triangles
        cx_t = (xcp[tris[:, 0]] + xcp[tris[:, 1]] + xcp[tris[:, 2]]) / 3.0
        cy_t = (ycp[tris[:, 0]] + ycp[tris[:, 1]] + ycp[tris[:, 2]]) / 3.0
        mask = np.array([not pregion.contains(Point(x, y)) for x, y in zip(cx_t, cy_t)], dtype=bool)
        tri.set_mask(mask)

        cf = ax.tricontourf(tri, fcp, levels=12)
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("fc (MPa)  [Compression]")
    else:
        ax.text(0.02, 0.98, "Compression region too small for contour",
                transform=ax.transAxes, ha="left", va="top")

    # Neutral axis: eps=0  => eps0 - kx*y + ky*x = 0
    minx, miny, maxx, maxy = conc["bounds"]
    xs_line = np.linspace(minx, maxx, 900)

    if abs(kx) > 1e-12:
        ys_line = (eps0 + ky * xs_line) / kx
        inside = np.array([pregion.contains(Point(x, y)) for x, y in zip(xs_line, ys_line)], dtype=bool)
        ax.plot(xs_line[inside], ys_line[inside], linewidth=2)
    elif abs(ky) > 1e-12:
        x_na = -eps0 / ky
        ys_line = np.linspace(miny, maxy, 900)
        inside = np.array([pregion.contains(Point(x_na, y)) for y in ys_line], dtype=bool)
        ax.plot([x_na] * inside.sum(), ys_line[inside], linewidth=2)
    else:
        ax.text(0.02, 0.90, "Neutral axis undefined (kx≈0 & ky≈0)",
                transform=ax.transAxes, ha="left", va="top")

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_aspect("equal", adjustable="box")

    # fixed limits (based on region bounds + consistent padding)
    pad = max(maxx - minx, maxy - miny) * 0.15
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    return fig


# =========================
# PDF
# =========================
def build_pdf_report_bytes(
    report_title: str,
    section_filename: str,
    load_filename: str,
    meta_lines: List[str],
    summary_df: pd.DataFrame,
    gov_fs_combo: str,
    gov_fc_combo: str,
    fig_section_png: bytes,
    fig_gov_fs_png: bytes,
    fig_gov_fc_png: bytes,
    gov_fs_stats: dict,
    gov_fc_stats: dict,
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.6 * inch,
        title=report_title,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(report_title, styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>Input files</b>", styles["Heading3"]))
    story.append(Paragraph(f"Section file: <b>{section_filename}</b>", styles["BodyText"]))
    story.append(Paragraph(f"Load file: <b>{load_filename}</b>", styles["BodyText"]))
    story.append(Spacer(1, 0.10 * inch))

    story.append(Paragraph("<b>Analysis assumptions</b>", styles["Heading3"]))
    for ln in meta_lines:
        story.append(Paragraph(f"• {ln}", styles["BodyText"]))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("<b>Section geometry</b>", styles["Heading3"]))
    # FIX: make page-1 section figure larger to reduce empty space
    story.append(rl_image_fit(fig_section_png, max_w=6.8 * inch, max_h=4.6 * inch))
    story.append(PageBreak())

    story.append(Paragraph("Summary (fs and fc by load combination)", styles["Heading2"]))
    story.append(Paragraph("Governing combinations are highlighted in <b>yellow</b>.", styles["BodyText"]))
    story.append(Spacer(1, 0.10 * inch))

    header = list(summary_df.columns)
    data = [header] + summary_df.astype(str).values.tolist()
    tbl = Table(data, repeatRows=1, hAlign="LEFT")

    base_style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    tbl.setStyle(TableStyle(base_style))

    yellow = colors.Color(1.0, 1.0, 0.6)
    for r_i in range(1, len(data)):
        combo_val = data[r_i][0]
        if combo_val == gov_fs_combo or combo_val == gov_fc_combo:
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, r_i), (-1, r_i), yellow),
                ("FONTNAME", (0, r_i), (-1, r_i), "Helvetica-Bold"),
            ]))

    story.append(tbl)
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph(
        f"<b>Governing steel tension combo</b>: {gov_fs_combo} — fs,t,max = {gov_fs_stats['fs_t_max']:.2f} MPa",
        styles["BodyText"]
    ))
    story.append(Paragraph(
        f"<b>Governing concrete compression combo</b>: {gov_fc_combo} — fc,max = {gov_fc_stats['fc_max']:.2f} MPa",
        styles["BodyText"]
    ))

    # Governing plots with fixed box size to enforce same visual size
    BOX_W = 6.8 * inch
    BOX_H = 5.35 * inch

    story.append(PageBreak())
    story.append(Paragraph(f"Governing Steel Tension — Combo: <b>{gov_fs_combo}</b>", styles["Heading2"]))
    story.append(Paragraph(
        f"fs,t,max = <b>{gov_fs_stats['fs_t_max']:.2f} MPa</b> (tension),   "
        f"fc,max = {gov_fs_stats['fc_max']:.2f} MPa,   "
        f"Acomp ≈ {gov_fs_stats['Acomp']:.0f} mm²",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(rl_image_fit_box(fig_gov_fs_png, BOX_W, BOX_H))

    if gov_fc_combo != gov_fs_combo:
        story.append(PageBreak())
        story.append(Paragraph(f"Governing Concrete Compression — Combo: <b>{gov_fc_combo}</b>", styles["Heading2"]))
        story.append(Paragraph(
            f"fc,max = <b>{gov_fc_stats['fc_max']:.2f} MPa</b> (compression),   "
            f"fs,t,max = {gov_fc_stats['fs_t_max']:.2f} MPa,   "
            f"Acomp ≈ {gov_fc_stats['Acomp']:.0f} mm²",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 0.08 * inch))
        story.append(rl_image_fit_box(fig_gov_fc_png, BOX_W, BOX_H))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# =========================
# Streamlit app
# =========================
st.set_page_config(page_title="RC Cracked Section Solver", layout="wide")
st.title("RC Cracked Section Solver")

if "has_results" not in st.session_state:
    st.session_state.has_results = False
if "run_hash" not in st.session_state:
    st.session_state.run_hash = None
if "res_df" not in st.session_state:
    st.session_state.res_df = None
if "fig_geom_png" not in st.session_state:
    st.session_state.fig_geom_png = None
if "gov_figs_cache" not in st.session_state:
    st.session_state.gov_figs_cache = {}

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
    N_tol_kN = st.number_input("|ΔN| tolerance (kN)", value=2.0, min_value=0.1, step=0.5)
    M_tol_kNm = st.number_input("|ΔMx|,|ΔMy| tolerance (kN·m)", value=2.0, min_value=0.1, step=0.5)

    st.subheader("Geometry Options")
    swap_xy = st.checkbox("Swap X and Y", value=False)

    st.subheader("Plot Options")
    show_rebar_ids = st.checkbox("Show rebar IDs", value=False)
    rebar_radius_mm = st.number_input("Rebar display radius (mm)", value=10.0, min_value=1.0, step=1.0)

    st.subheader("Report")
    report_title = st.text_input("PDF report title", value="RC Cracked Section Analysis")

    run_btn = st.button("Run analysis")

if not sec_file or not load_file:
    st.info("Upload a section file and a load combos file to start.")
    st.stop()

section_filename = sec_file.name
load_filename = load_file.name

sec_bytes = sec_file.getvalue()
load_bytes = load_file.getvalue()

run_hash = sha256_bytes(
    sec_bytes
    + load_bytes
    + f"{Ec}-{Es}-{mesh_mm}-{N_tol_kN}-{M_tol_kNm}-{swap_xy}-{show_rebar_ids}-{rebar_radius_mm}".encode("utf-8")
)

# Parse section
sec_text = sec_bytes.decode("utf-8", errors="ignore")
solids, openings, rebars_list = parse_spcolumn_section_text(sec_text)
region_c, solids_c, openings_c, (cx, cy) = build_centered_geometry(solids, openings)

reb = {
    "As": np.array([r[0] for r in rebars_list], dtype=float),
    "x":  np.array([r[1] - cx for r in rebars_list], dtype=float),
    "y":  np.array([r[2] - cy for r in rebars_list], dtype=float),
}

conc = build_concrete_fibers_cached(sha256_bytes(sec_bytes), solids_c, openings_c, float(mesh_mm))

if swap_xy:
    reb = {"As": reb["As"], "x": reb["y"].copy(), "y": reb["x"].copy()}
    conc = {
        "x": conc["y"].copy(),
        "y": conc["x"].copy(),
        "A": conc["A"].copy(),
        "bounds": conc["bounds"],
        "cell_area": conc["cell_area"],
    }

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

# Section plot
cA, cB = st.columns([1.25, 0.75])
with cA:
    st.subheader("Section")
    fig_geom = plot_section(solids_c, openings_c, reb, show_rebar_ids, float(rebar_radius_mm))
    st.pyplot(fig_geom, use_container_width=True)
    st.session_state.fig_geom_png = fig_to_png_bytes(fig_geom)

with cB:
    st.subheader("Info")
    st.write(f"Section file: **{section_filename}**")
    st.write(f"Load file: **{load_filename}**")
    st.write(f"Centroid shift: dx={cx:.2f} mm, dy={cy:.2f} mm")
    st.write(f"Rebars: **{len(rebars_list)}**")
    st.write(f"Fibers: **{len(conc['A'])}**")
    st.write("Internal sign: Compression(+), Tension(−)")
    st.write("Moments: Mx>0 bottom compression, My>0 right compression")

st.subheader("Load Combos Preview")
st.dataframe(combos.head(80), use_container_width=True)

if run_btn:
    st.session_state.has_results = False

need_compute = (not st.session_state.has_results) or (st.session_state.run_hash != run_hash)
if need_compute and not run_btn:
    st.info("Click **Run analysis**.")
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
        N_kN = float(row["N_kN"])
        Mx_kNm = float(row["Mx_kNm"])
        My_kNm = float(row["My_kNm"])

        Nt  = N_kN * 1000.0
        Mxt = Mx_kNm * 1_000_000.0
        Myt = My_kNm * 1_000_000.0

        if x0 is None:
            x0 = initial_guess(Nt, conc, reb, float(Ec), float(Es))

        ok, near, stat, reason, x_sol, r, errN, errMx, errMy, res_norm = solve_combo_strict(
            Nt, Mxt, Myt, conc, reb, float(Ec), float(Es),
            x0, N_tol_N, M_tol_Nmm
        )

        # Reporting: tension positive for steel
        fs_signed = -r["fs_int"]  # tension +, compression -
        fs_t_max = float(np.max(np.maximum(fs_signed, 0.0)))
        fs_c_maxmag = float(np.max(np.maximum(-fs_signed, 0.0)))
        fc_max = float(r["fc_max"])

        results.append({
            "Combo": combo,
            "N_kN": N_kN,
            "Mx_kNm": Mx_kNm,
            "My_kNm": My_kNm,
            "Status": stat,
            "Converged": bool(ok),
            "Near": bool(near),
            "FailureReason": reason,
            "eps0": float(x_sol[0]),
            "kx": float(x_sol[1]),
            "ky": float(x_sol[2]),
            "fs_t_max_MPa": fs_t_max,
            "fs_c_max_MPa": fs_c_maxmag,
            "fc_max_MPa": fc_max,
            "Acomp_mm2": float(r["Acomp"]),
            "errN_kN": errN / 1000.0,
            "errMx_kNm": errMx / 1e6,
            "errMy_kNm": errMy / 1e6,
            "scaled_residual_norm": res_norm,
        })

        # warm start next combo
        x0 = x_sol.copy()
        progress.progress(min(1.0, (idx + 1) / max(1, n_rows)))
        status.write(
            f"Solving: {combo} | ΔN={errN/1000:.3f} kN | ΔMx={errMx/1e6:.3f} kN·m | ΔMy={errMy/1e6:.3f} kN·m | {stat}"
        )

    st.session_state.res_df = pd.DataFrame(results)
    st.session_state.has_results = True
    st.session_state.run_hash = run_hash
    st.session_state.gov_figs_cache = {}

res_df = st.session_state.res_df

st.subheader("Results (fs and fc)")
st.dataframe(res_df, use_container_width=True)

# Select governing among OK/NEAR only
okdf = res_df[res_df["Status"].isin(["OK", "NEAR"])].copy()
if len(okdf) == 0:
    st.error("No OK/NEAR results. Check units/signs/reference of loads.")
    st.stop()

gov_fs = okdf.loc[okdf["fs_t_max_MPa"].idxmax()]
gov_fc = okdf.loc[okdf["fc_max_MPa"].idxmax()]
gov_fs_combo = str(gov_fs["Combo"])
gov_fc_combo = str(gov_fc["Combo"])

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Governing Steel Tension")
    st.write(f"Combo: **{gov_fs_combo}**")
    st.write(f"Status: **{gov_fs['Status']}**")
    st.write(f"fs,t,max: **{gov_fs['fs_t_max_MPa']:.2f} MPa**")
with c2:
    st.markdown("### Governing Concrete Compression")
    st.write(f"Combo: **{gov_fc_combo}**")
    st.write(f"Status: **{gov_fc['Status']}**")
    st.write(f"fc,max: **{gov_fc['fc_max_MPa']:.2f} MPa**")

# Governing plots cache
def get_gov_plot(row, tag: str):
    key = f"{tag}:{row['Combo']}:{float(row['eps0']):.6e}:{float(row['kx']):.6e}:{float(row['ky']):.6e}"
    if key in st.session_state.gov_figs_cache:
        return st.session_state.gov_figs_cache[key]

    fig = plot_fc_contour_and_na(
        solids_c, openings_c, conc, reb, float(Ec),
        float(row["eps0"]), float(row["kx"]), float(row["ky"]),
        title=f"{tag} — Combo {row['Combo']} | fc contour + NA"
    )
    png = fig_to_png_bytes(fig)
    stats = {
        "fs_t_max": float(row["fs_t_max_MPa"]),
        "fc_max": float(row["fc_max_MPa"]),
        "Acomp": float(row["Acomp_mm2"]),
    }
    st.session_state.gov_figs_cache[key] = (png, stats)
    return png, stats

gov_fs_png, stats_gov_fs = get_gov_plot(gov_fs, "Governing Steel Tension")
gov_fc_png, stats_gov_fc = get_gov_plot(gov_fc, "Governing Concrete Compression")

st.subheader("Governing Combo Plots (Preview)")
p1, p2 = st.columns(2)
with p1:
    st.image(gov_fs_png, use_container_width=True)
with p2:
    st.image(gov_fc_png, use_container_width=True)

# Summary table for PDF
summary_df = okdf.copy()
summary_df["Governing"] = ""
summary_df.loc[summary_df["Combo"] == gov_fs_combo, "Governing"] += "Steel(fs)"
summary_df.loc[summary_df["Combo"] == gov_fc_combo, "Governing"] += (" & " if gov_fc_combo == gov_fs_combo else "") + "Concrete(fc)"

summary_pdf = summary_df[[
    "Combo", "N_kN", "Mx_kNm", "My_kNm",
    "fs_t_max_MPa", "fc_max_MPa",
    "Status", "Governing"
]].copy()

# Nice formatting
summary_pdf["N_kN"] = summary_pdf["N_kN"].map(lambda v: f"{v:.1f}")
summary_pdf["Mx_kNm"] = summary_pdf["Mx_kNm"].map(lambda v: f"{v:.2f}")
summary_pdf["My_kNm"] = summary_pdf["My_kNm"].map(lambda v: f"{v:.2f}")
summary_pdf["fs_t_max_MPa"] = summary_pdf["fs_t_max_MPa"].map(lambda v: f"{v:.2f}")
summary_pdf["fc_max_MPa"] = summary_pdf["fc_max_MPa"].map(lambda v: f"{v:.2f}")

st.subheader("Download Outputs")

excel_out = io.BytesIO()
with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
    res_df.to_excel(writer, index=False, sheet_name="Results")
excel_out.seek(0)

st.download_button(
    "Download Results (Excel)",
    data=excel_out,
    file_name="rc_cracked_service_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

meta_lines = [
    "Cracked service, linear elastic model.",
    "Plane sections remain plane (linear strain field).",
    "Concrete in tension neglected (fc = 0 for εc ≤ 0).",
    "Concrete in compression: fc = Ec·εc.",
    "Steel stress: fs = Es·εs (reported with tension positive).",
    "Moment convention: Mx>0 bottom in compression; My>0 right in compression.",
]

pdf_bytes = build_pdf_report_bytes(
    report_title=report_title,
    section_filename=section_filename,
    load_filename=load_filename,
    meta_lines=meta_lines,
    summary_df=summary_pdf,
    gov_fs_combo=gov_fs_combo,
    gov_fc_combo=gov_fc_combo,
    fig_section_png=st.session_state.fig_geom_png,
    fig_gov_fs_png=gov_fs_png,
    fig_gov_fc_png=gov_fc_png,
    gov_fs_stats=stats_gov_fs,
    gov_fc_stats=stats_gov_fc,
)

st.download_button(
    "Download Clean PDF Report",
    data=pdf_bytes,
    file_name="rc_section_report_clean.pdf",
    mime="application/pdf",
)
