"""Lecture density plots for practice 8. Run: python docs/assets/p08/_make_figures.py"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
COLORS = ["#2471a3", "#c0392b", "#1e8449", "#8e44ad", "#d35400"]
GRAY = "#7f8c8d"


def save(name: str) -> None:
    plt.savefig(OUT / name)
    plt.close()


def panel() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    return fig, ax


def finish(ax: plt.Axes, xmin: float, xmax: float, ymax: float | None = None) -> None:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0, top=ymax)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend(frameon=False, loc="best")


def fig_uniform() -> None:
    _, ax = panel()
    specs = [(0, 1), (0, 2), (-1, 1)]
    xs = [
        np.linspace(-0.4, 1.4, 400),
        np.linspace(-0.4, 2.4, 400),
        np.linspace(-1.6, 1.6, 400),
    ]
    for (a, b), x, c in zip(specs, xs, COLORS):
        f = np.where((x >= a) & (x <= b), 1 / (b - a), 0.0)
        ax.plot(x, f, color=c, lw=2, label=rf"$U({a},{b})$")
    finish(ax, -1.7, 2.5, 1.35)
    save("fig-uniform.png")


def fig_exponential() -> None:
    _, ax = panel()
    x = np.linspace(0, 6, 500)
    for lam, c in zip((0.5, 1, 2), COLORS):
        ax.plot(x, lam * np.exp(-lam * x), color=c, lw=2, label=rf"$\lambda={lam:g}$")
    finish(ax, 0, 6, 2.15)
    save("fig-exp.png")


def fig_normal() -> None:
    _, ax = panel()
    x = np.linspace(-6, 8, 700)

    def pdf(mu: float, sigma: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    specs = [(0, 1, r"$N(0,1)$"), (0, 2, r"$N(0,2^2)$"), (2, 1, r"$N(2,1)$")]
    for (mu, sigma, lab), c in zip(specs, COLORS):
        ax.plot(x, pdf(mu, sigma), color=c, lw=2, label=lab)
    finish(ax, -6, 8, None)
    save("fig-normal.png")


def gamma_pdf(x: np.ndarray, k: float, lam: float) -> np.ndarray:
    coef = (lam**k) / math.gamma(k)
    out = np.zeros_like(x, dtype=float)
    m = x > 0
    out[m] = coef * np.power(x[m], k - 1) * np.exp(-lam * x[m])
    return out


def fig_chisq() -> None:
    _, ax = panel()
    x = np.linspace(0.001, 12, 700)
    for k, c in zip((1, 2, 4, 8), COLORS):
        ax.plot(x, gamma_pdf(x, k / 2, 0.5), color=c, lw=2, label=rf"$k={k}$")
    finish(ax, 0, 12, 0.55)
    save("fig-chisq.png")


def fig_gamma() -> None:
    _, ax = panel()
    x = np.linspace(0.001, 12, 700)
    specs = [(1, 1, r"$k=1,\ \lambda=1$"), (2, 1, r"$k=2,\ \lambda=1$"), (4, 1, r"$k=4,\ \lambda=1$"), (2, 0.5, r"$k=2,\ \lambda=1/2$")]
    for (k, lam, lab), c in zip(specs, COLORS):
        ax.plot(x, gamma_pdf(x, k, lam), color=c, lw=2, label=lab)
    finish(ax, 0, 12, None)
    save("fig-gamma.png")


def fig_cauchy() -> None:
    _, ax = panel()
    x = np.linspace(-12, 12, 900)

    def pdf(scale: float) -> np.ndarray:
        return 1 / (np.pi * scale * (1 + (x / scale) ** 2))

    for scale, c in zip((0.5, 1, 2), COLORS):
        ax.plot(x, pdf(scale), color=c, lw=2, label=rf"$C(0,{scale:g})$")
    finish(ax, -12, 12, None)
    save("fig-cauchy-pdf.png")


def fig_cauchy_ray() -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.axhline(0, color="#2c3e50", lw=1)
    ax.axvline(0, color="#2c3e50", lw=1)
    ax.plot(4, 0, "o", color=COLORS[1], ms=7, zorder=5)
    ax.annotate(r"$A(4;0)$", xy=(4, 0), xytext=(4.15, -1.3), color=COLORS[1], ha="left")
    t = math.atan(2.2 / 4)
    y_hit = 4 * math.tan(t)
    ax.plot([4, 0], [0, y_hit], color=COLORS[0], lw=2)
    ax.plot(0, y_hit, "o", color=COLORS[0], ms=6)
    ax.annotate(r"$y$", xy=(0, y_hit), xytext=(-1.35, y_hit), va="center", color=COLORS[0])
    # angle arc near A, from negative x-axis
    phi = np.linspace(math.pi, math.pi - t, 40)
    ax.plot(4 + 0.85 * np.cos(phi), 0 + 0.85 * np.sin(phi), color=GRAY, lw=1.3)
    ax.text(2.85, 0.28, r"$t$", color=GRAY)
    ax.text(-0.35, -0.45, r"$O$")
    ax.set_xlim(-2.2, 5.6)
    ax.set_ylim(-2.4, 4.2)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xticks([0, 4])
    ax.set_yticks([0])
    save("fig-cauchy-ray.png")


def student_pdf(x: np.ndarray, nu: float) -> np.ndarray:
    c = math.gamma((nu + 1) / 2) / (math.sqrt(nu * math.pi) * math.gamma(nu / 2))
    return c * np.power(1 + x**2 / nu, -(nu + 1) / 2)


def fig_student() -> None:
    _, ax = panel()
    x = np.linspace(-6, 6, 700)
    ax.plot(x, student_pdf(x, 1), color=COLORS[0], lw=2, label=r"$\nu=1$ ($C(0,1)$)")
    ax.plot(x, student_pdf(x, 3), color=COLORS[2], lw=2, label=r"$\nu=3$")
    ax.plot(x, student_pdf(x, 10), color=COLORS[3], lw=2, label=r"$\nu=10$")
    ax.plot(
        x,
        np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi),
        color=COLORS[1],
        lw=2,
        ls="--",
        label=r"$N(0,1)$",
    )
    finish(ax, -6, 6, 0.45)
    save("fig-student.png")


def fisher_pdf(x: np.ndarray, d1: float, d2: float) -> np.ndarray:
    B = math.gamma(d1 / 2) * math.gamma(d2 / 2) / math.gamma((d1 + d2) / 2)
    out = np.zeros_like(x, dtype=float)
    m = x > 0
    xm = x[m]
    out[m] = (
        (1 / B)
        * (d1 / d2) ** (d1 / 2)
        * np.power(xm, d1 / 2 - 1)
        / np.power(1 + (d1 / d2) * xm, (d1 + d2) / 2)
    )
    return out


def fig_fisher() -> None:
    _, ax = panel()
    x = np.linspace(0.001, 5, 700)
    specs = [(1, 1), (2, 5), (5, 2), (10, 10)]
    for (d1, d2), c in zip(specs, COLORS):
        ax.plot(x, fisher_pdf(x, d1, d2), color=c, lw=2, label=rf"$({d1},{d2})$")
    finish(ax, 0, 5, 1.15)
    save("fig-fisher.png")


def fig_lognormal() -> None:
    _, ax = panel()
    x = np.linspace(0.001, 8, 700)

    def pdf(mu: float, sigma: float) -> np.ndarray:
        return np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))

    specs = [(0, 0.5, r"$\mu=0,\ \sigma=0{,}5$"), (0, 1, r"$\mu=0,\ \sigma=1$"), (1, 0.5, r"$\mu=1,\ \sigma=0{,}5$")]
    for (mu, sigma, lab), c in zip(specs, COLORS):
        ax.plot(x, pdf(mu, sigma), color=c, lw=2, label=lab)
    finish(ax, 0, 8, None)
    save("fig-lognormal.png")


if __name__ == "__main__":
    fig_uniform()
    fig_exponential()
    fig_normal()
    fig_chisq()
    fig_gamma()
    fig_cauchy()
    fig_cauchy_ray()
    fig_student()
    fig_fisher()
    fig_lognormal()
    print("ok", OUT)
