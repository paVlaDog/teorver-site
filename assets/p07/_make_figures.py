"""Lecture figures for practice 7. Run from repo: python docs/assets/p07/_make_figures.py"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parent
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)
RED = "#c0392b"
BLUE = "#2471a3"
FILL = "#5dade2"
GRAY = "#7f8c8d"


def save(name: str) -> None:
    plt.savefig(OUT / name)
    plt.close()


def fig_cdf_increment() -> None:
    x = np.linspace(-1.2, 4.2, 400)
    # smooth CDF-like: logistic stretched
    F = 1 / (1 + np.exp(-1.6 * (x - 1.4)))
    a, b = 0.4, 2.4
    Fa, Fb = 1 / (1 + np.exp(-1.6 * (a - 1.4))), 1 / (1 + np.exp(-1.6 * (b - 1.4)))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(x, F, color=BLUE, lw=2)
    ax.axvline(a, color=GRAY, ls="--", lw=1)
    ax.axvline(b, color=GRAY, ls="--", lw=1)
    ax.hlines(Fa, -1.2, a, color=GRAY, ls=":", lw=1)
    ax.hlines(Fb, -1.2, b, color=GRAY, ls=":", lw=1)
    ax.annotate(
        "",
        xy=(-0.85, Fb),
        xytext=(-0.85, Fa),
        arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6),
    )
    ax.text(-0.72, (Fa + Fb) / 2, r"$F(b)-F(a)$", color=RED, va="center")
    ax.set_xlim(-1.2, 4.2)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks([a, b])
    ax.set_xticklabels(["$a$", "$b$"])
    ax.set_yticks([0, 1])
    ax.set_ylabel("$F(x)$")
    ax.set_xlabel("$x$")
    save("fig-cdf-increment.png")


def fig_pdf_area() -> None:
    x = np.linspace(-0.5, 5, 400)
    f = np.exp(-((x - 2) ** 2) / 1.4)
    f = f / np.trapezoid(f, x)
    a, b = 1.1, 2.8
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(x, f, color=RED, lw=2)
    mask = (x >= a) & (x <= b)
    ax.fill_between(x[mask], f[mask], color=FILL, alpha=0.55)
    ax.set_xlim(-0.3, 4.6)
    ax.set_ylim(0, f.max() * 1.15)
    ax.set_xticks([a, b])
    ax.set_xticklabels(["$a$", "$b$"])
    ax.set_yticks([])
    ax.set_ylabel("$f(x)$")
    ax.set_xlabel("$x$")
    ax.text((a + b) / 2, f.max() * 0.35, r"$P(a<X<b)$", ha="center", color="#1a5276")
    save("fig-pdf-area.png")


def fig_252() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    xs = [-1.8, -1, -1, 1 / 3, 1 / 3, 1.6]
    ys = [0, 0, 0, 1, 1, 1]
    ax.plot([-1.8, -1], [0, 0], color=BLUE, lw=2)
    ax.plot([-1, 1 / 3], [0, 1], color=BLUE, lw=2)
    ax.plot([1 / 3, 1.6], [1, 1], color=BLUE, lw=2)
    ax.plot([0, 0], [0, 0.75], color=RED, ls="--", lw=1)
    ax.plot([1 / 3, 1 / 3], [0, 1], color=RED, ls="--", lw=1)
    ax.plot([0], [0.75], "o", color=RED)
    ax.plot([1 / 3], [1], "o", color=RED)
    ax.annotate(
        "",
        xy=(0.05, 1),
        xytext=(0.05, 0.75),
        arrowprops=dict(arrowstyle="<->", color=RED, lw=1.4),
    )
    ax.text(0.12, 0.86, r"$1/4$", color=RED)
    ax.set_xlim(-1.9, 1.7)
    ax.set_ylim(-0.08, 1.15)
    ax.set_xticks([-1, 0, 1 / 3])
    ax.set_xticklabels(["$-1$", "$0$", r"$1/3$"])
    ax.set_yticks([0, 1])
    ax.set_ylabel("$F(x)$")
    save("fig-252-F.png")


def fig_253() -> None:
    x = np.linspace(-8, 8, 500)
    F = 0.5 + np.arctan(x) / np.pi
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(x, F, color=BLUE, lw=2)
    ax.axvline(0, color=GRAY, ls="--", lw=1)
    ax.axvline(1, color=GRAY, ls="--", lw=1)
    ax.plot([0], [0.5], "o", color=RED)
    ax.plot([1], [0.75], "o", color=RED)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("$F(x)$")
    ax.set_xlabel("$x$")
    save("fig-253-F.png")


def fig_g_levelset() -> None:
    x = np.linspace(0, 6.2, 500)
    g = 0.35 + 1.15 * np.exp(-((x - 2.4) ** 2) / 1.6)
    y0 = 0.72
    # two roots
    idx = np.where(np.diff(np.sign(g - y0)))[0]
    x1, x2 = x[idx[0]], x[idx[1]]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(x, g, color=RED, lw=2)
    ax.axhline(y0, color=GRAY, lw=1)
    ax.fill_between([x1, x2], 0, y0, color=FILL, alpha=0.35)
    ax.plot([x1, x2], [y0, y0], "o", color=RED)
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 1.65)
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(["$x_1$", "$x_2$"])
    ax.set_yticks([y0])
    ax.set_yticklabels(["$y$"])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$g(x)$")
    ax.text((x1 + x2) / 2, 0.18, r"$\{g(x)\leqslant y\}=[x_1;x_2]$", ha="center")
    save("fig-g-levelset.png")


def fig_sin2() -> None:
    x = np.linspace(0, 2 * np.pi, 800)
    g = np.sin(x) ** 2
    y0 = 0.55
    alpha = np.arcsin(np.sqrt(y0))
    roots = [alpha, np.pi - alpha, np.pi + alpha, 2 * np.pi - alpha]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(x, g, color=RED, lw=2.2)
    ax.axhline(y0, color="#5d6d7e", lw=1)
    for r in roots:
        ax.plot([r, r], [0, y0], color=GRAY, lw=1)
        ax.plot(r, y0, "o", color=RED, ms=7)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 1.12)
    pairs = [
        (roots[0], r"$x_1$"),
        (roots[1], r"$x_2$"),
        (np.pi, r"$\pi$"),
        (roots[2], r"$x_3$"),
        (roots[3], r"$x_4$"),
        (2 * np.pi, r"$2\pi$"),
    ]
    ax.set_xticks([p[0] for p in pairs])
    ax.set_xticklabels([p[1] for p in pairs])
    ax.set_yticks([y0])
    ax.set_yticklabels(["$y$"])
    ax.set_xlabel("$x$")
    ax.set_title(r"$g(x)=\sin^2 x$ на $[0,2\pi]$")
    save("fig-sin2.png")


def fig_mono(kind: str) -> None:
    x = np.linspace(0.2, 5.5, 300)
    if kind == "inc":
        g = 0.35 + 0.45 * (x - 0.2) ** 0.85
        x0 = 2.6
        y0 = 0.35 + 0.45 * (x0 - 0.2) ** 0.85
        shade_from, shade_to = 0.2, x0
        name = "fig-mono-inc.png"
        caption = r"$\{g\leqslant y\}=(-\infty;x]$"
    else:
        g = 3.1 - 0.48 * (x - 0.2) ** 0.85
        x0 = 2.6
        y0 = 3.1 - 0.48 * (x0 - 0.2) ** 0.85
        shade_from, shade_to = x0, 5.5
        name = "fig-mono-dec.png"
        caption = r"$\{g\leqslant y\}=[x;+\infty)$"
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(x, g, color=RED, lw=2)
    ax.axhline(y0, color=GRAY, lw=1)
    ax.fill_betweenx([0, y0], shade_from, shade_to, color=FILL, alpha=0.35)
    ax.plot(x0, y0, "o", color=RED)
    ax.set_xlim(0.15, 5.6)
    ax.set_ylim(0, max(g) * 1.15)
    ax.set_xticks([x0])
    ax.set_xticklabels([r"$g^{-1}(y)$"])
    ax.set_yticks([y0])
    ax.set_yticklabels(["$y$"])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$g(x)$")
    ax.text(0.3, max(g) * 0.92, caption)
    save(name)


def fig_nonmono_pieces() -> None:
    x = np.linspace(0, 10, 900)
    g = 1.15 + 0.95 * np.sin(0.9 * x) * np.exp(-0.03 * (x - 5) ** 2 / 4)
    # split by critical points of sin-like
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    cuts = [0, 1.75, 5.25, 8.7, 10]
    colors = ["#e67e22", "#8e44ad", "#2980b9", "#27ae60"]
    for i in range(4):
        m = (x >= cuts[i]) & (x <= cuts[i + 1])
        ax.plot(x[m], g[m], color=colors[i], lw=2.4)
        xm = (cuts[i] + cuts[i + 1]) / 2
        ax.text(xm, -0.22, rf"$A_{{{i+1}}}$", ha="center")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.45, 2.4)
    ax.set_xticks([])
    ax.set_ylabel("$g(x)$")
    ax.set_xlabel("$x$")
    save("fig-nonmono-pieces.png")


def fig_xsquared() -> None:
    x = np.linspace(-2.4, 2.4, 400)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(x, x**2, color=RED, lw=2)
    y0 = 2.25
    ax.axhline(y0, color=GRAY, lw=1)
    ax.plot([-1.5, 1.5], [y0, y0], "o", color=RED)
    ax.annotate("", xy=(-1.5, y0), xytext=(0, y0 + 0.55),
                arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.annotate("", xy=(1.5, y0), xytext=(0, y0 + 0.55),
                arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.text(-1.85, y0 + 0.15, r"$-\sqrt{y}$")
    ax.text(1.55, y0 + 0.15, r"$+\sqrt{y}$")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5.2)
    ax.set_xticks([0])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$x^2$")
    save("fig-x-squared.png")


def fig_2d_partition() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.3, 5.2)
    ax.axis("off")
    # left blob pieces
    pieces = [
        [(0.3, 3.6), (1.6, 4.6), (2.6, 3.9), (2.0, 2.7), (0.6, 2.8)],
        [(0.5, 1.0), (2.1, 2.5), (3.2, 1.6), (2.4, 0.4), (0.7, 0.35)],
        [(2.2, 2.8), (3.4, 3.7), (4.3, 2.5), (3.3, 1.7)],
        [(3.5, 1.5), (4.6, 2.2), (5.1, 0.9), (3.8, 0.4)],
    ]
    cols = ["#af7ac5", "#5dade2", "#58d68d", "#f5b041"]
    labels = [r"$A_1$", r"$A_2$", r"$A_3$", r"$A_4$"]
    for poly, c, lab in zip(pieces, cols, labels):
        ax.add_patch(Polygon(poly, closed=True, facecolor=c, edgecolor="#2c3e50", alpha=0.85))
        cx = np.mean([p[0] for p in poly])
        cy = np.mean([p[1] for p in poly])
        ax.text(cx, cy, lab, ha="center", va="center")
    ax.add_patch(FancyBboxPatch((6.6, 1.1), 3.0, 2.8, boxstyle="round,pad=0.15",
                                facecolor="#f9e79f", edgecolor="#2c3e50"))
    ax.text(8.1, 2.5, r"$\Omega_{U,V}$", ha="center", va="center", fontsize=14)
    ax.annotate("", xy=(6.55, 2.5), xytext=(5.25, 2.5),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.8))
    ax.text(5.9, 2.85, r"$g$", ha="center")
    ax.text(2.4, 4.85, r"$\Omega_{X,Y}$")
    save("fig-2d-partition.png")


def fig_sin_mono() -> None:
    x = np.linspace(-np.pi / 2, np.pi / 2, 400)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(x, np.sin(x), color=RED, lw=2)
    ax.set_xlim(-np.pi / 2, np.pi / 2)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([-np.pi / 2, 0, np.pi / 2])
    ax.set_xticklabels([r"$-\pi/2$", "$0$", r"$\pi/2$"])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\sin x$")
    save("fig-sin-mono.png")


if __name__ == "__main__":
    fig_cdf_increment()
    fig_pdf_area()
    fig_252()
    fig_253()
    fig_g_levelset()
    fig_sin2()
    fig_mono("inc")
    fig_mono("dec")
    fig_nonmono_pieces()
    fig_xsquared()
    fig_2d_partition()
    fig_sin_mono()
    print("ok", OUT)
