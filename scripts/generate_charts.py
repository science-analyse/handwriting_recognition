"""
Codecademy Catalog — Business Insight Charts
Produces 8 charts saved to charts/
"""

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "figure.dpi": 140,
})

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_FILE = BASE / "data" / "codecademy.csv"
CHARTS_DIR = BASE / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ── Brand palette ────────────────────────────────────────────────────────────
BLUE   = "#1F5EFF"
TEAL   = "#00C4B3"
ORANGE = "#FF6B35"
PURPLE = "#7B2FBE"
GREEN  = "#2DC653"
SLATE  = "#64748B"
GOLD   = "#F59E0B"
RED    = "#EF4444"

DIFF_COLORS = {"Beginner": TEAL, "Intermediate": BLUE, "Advanced": PURPLE}
DIFF_ORDER  = ["Beginner", "Intermediate", "Advanced"]

# ── Load data ────────────────────────────────────────────────────────────────
with open(DATA_FILE, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} rows.")


def save(fig: plt.Figure, name: str) -> None:
    path = CHARTS_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1 — Catalog Composition by Content Type and Difficulty
# ═══════════════════════════════════════════════════════════════════════════════
def chart_01_catalog_composition():
    type_labels = {
        "Track": "Course Track",
        "Path": "Learning Path",
        "ExternalCourse": "External Course",
        "ExternalPath": "External Path",
    }
    types = ["Track", "Path", "ExternalCourse", "ExternalPath"]
    td = defaultdict(lambda: Counter())
    for r in rows:
        if r["type"] in types:
            td[r["type"]][r["difficulty"]] += 1

    x = np.arange(len(types))
    width = 0.55
    fig, ax = plt.subplots(figsize=(10, 6))

    bottoms = np.zeros(len(types))
    for diff in DIFF_ORDER:
        vals = np.array([td[t][diff] for t in types])
        bars = ax.bar(x, vals, width, bottom=bottoms,
                      color=DIFF_COLORS[diff], label=diff)
        for bar, val in zip(bars, vals):
            if val > 0:
                cy = bar.get_y() + bar.get_height() / 2
                ax.text(bar.get_x() + bar.get_width() / 2, cy,
                        str(val), ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([type_labels[t] for t in types], fontsize=11)
    ax.set_ylabel("Number of Offerings", fontsize=11)
    ax.set_title("Catalog Composition by Content Type and Difficulty Level",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(title="Difficulty", loc="upper right", framealpha=0.9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    totals = [sum(td[t].values()) for t in types]
    for i, (tot, xp) in enumerate(zip(totals, x)):
        ax.text(xp, tot + 4, f"n={tot}", ha="center",
                fontsize=9, color=SLATE)

    fig.tight_layout()
    save(fig, "chart_01_catalog_composition.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2 — Free vs Pro Tracks by Difficulty
# ═══════════════════════════════════════════════════════════════════════════════
def chart_02_free_vs_pro_by_difficulty():
    tracks = [r for r in rows if r["type"] == "Track"]
    data = {}
    for diff in DIFF_ORDER:
        sub = [r for r in tracks if r["difficulty"] == diff]
        data[diff] = {
            "Pro (Paid)": sum(1 for r in sub if r["pro"] == "True"),
            "Free":       sum(1 for r in sub if r["pro"] == "False"),
        }

    categories = ["Pro (Paid)", "Free"]
    colors     = [ORANGE, TEAL]
    x = np.arange(len(DIFF_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (cat, col) in enumerate(zip(categories, colors)):
        vals = [data[d][cat] for d in DIFF_ORDER]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=cat, color=col)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 3,
                    str(val), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(DIFF_ORDER, fontsize=12)
    ax.set_ylabel("Number of Course Tracks", fontsize=11)
    ax.set_title("Free vs. Pro (Paid) Course Tracks by Difficulty Level",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(framealpha=0.9)

    # Pro penetration % annotation
    for i, diff in enumerate(DIFF_ORDER):
        total = data[diff]["Pro (Paid)"] + data[diff]["Free"]
        pct = data[diff]["Pro (Paid)"] / total * 100
        ax.text(x[i], -22, f"{pct:.0f}% paid",
                ha="center", fontsize=9, color=ORANGE, style="italic")

    fig.tight_layout()
    save(fig, "chart_02_free_vs_pro_by_difficulty.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3 — Course Duration Distribution (all offerings)
# ═══════════════════════════════════════════════════════════════════════════════
def chart_03_duration_distribution():
    bucket_order = ["< 5h", "5–15h", "15–30h", "30–60h", "60h+"]
    bucket_colors = [TEAL, BLUE, ORANGE, PURPLE, RED]

    def bucket(h: float) -> str:
        if h < 5:   return "< 5h"
        if h < 15:  return "5–15h"
        if h < 30:  return "15–30h"
        if h < 60:  return "30–60h"
        return "60h+"

    counts = Counter()
    for r in rows:
        h = r["medianDurationHours"] or r["durationHours"]
        if h:
            counts[bucket(float(h))] += 1

    vals = [counts[b] for b in bucket_order]
    total = sum(vals)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(bucket_order, vals, color=bucket_colors, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 4,
                f"{val}\n({val/total*100:.0f}%)",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Number of Offerings", fontsize=11)
    ax.set_xlabel("Total Learning Time", fontsize=11)
    ax.set_title("Course Duration Distribution Across All Offerings",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylim(0, max(vals) * 1.2)

    fig.tight_layout()
    save(fig, "chart_03_duration_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4 — External Certification Provider Landscape
# ═══════════════════════════════════════════════════════════════════════════════
def chart_04_certification_providers():
    provider_map = {
        "MICROSOFT": "Microsoft",
        "COMPTIA": "CompTIA",
        "AWS": "Amazon AWS",
        "GOOGLE": "Google",
        "ISC2": "ISC²",
        "ORACLE": "Oracle",
        "KUBERNETES": "Kubernetes",
        "ISTQB": "ISTQB",
        "IIBA": "IIBA",
        "HRCI": "HRCI",
        "CISCO": "Cisco",
        "PROJECT_MANAGEMENT_INSTITUTE": "PMI",
        "RED_HAT": "Red Hat",
        "ISACA": "ISACA",
    }

    prov_diff = defaultdict(lambda: Counter())
    for r in rows:
        if r["certificationProvider"]:
            prov_diff[r["certificationProvider"]][r["difficulty"]] += 1

    sorted_provs = sorted(prov_diff.items(),
                          key=lambda x: sum(x[1].values()), reverse=True)

    labels  = [provider_map.get(p, p) for p, _ in sorted_provs]
    y       = np.arange(len(labels))
    width   = 0.55

    fig, ax = plt.subplots(figsize=(10, 7))
    lefts = np.zeros(len(labels))
    for diff in DIFF_ORDER:
        vals = np.array([c[diff] for _, c in sorted_provs])
        bars = ax.barh(y, vals, width, left=lefts,
                       color=DIFF_COLORS[diff], label=diff)
        for bar, val in zip(bars, vals):
            if val > 0:
                cx = bar.get_x() + bar.get_width() / 2
                cy = bar.get_y() + bar.get_height() / 2
                ax.text(cx, cy, str(val),
                        ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        lefts += vals

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Certification Prep Paths", fontsize=11)
    ax.set_title("External Certification Provider Landscape",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(title="Difficulty", loc="lower right", framealpha=0.9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="x", alpha=0.35)
    ax.grid(axis="y", alpha=0)

    totals = [sum(c.values()) for _, c in sorted_provs]
    for i, (tot, yp) in enumerate(zip(totals, y)):
        ax.text(tot + 0.2, yp, str(tot),
                va="center", fontsize=9, color=SLATE)

    fig.tight_layout()
    save(fig, "chart_04_certification_providers.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 5 — Average Learning Duration by Difficulty
# ═══════════════════════════════════════════════════════════════════════════════
def chart_05_avg_duration_by_difficulty():
    diff_dur = defaultdict(list)
    for r in rows:
        h = r["medianDurationHours"] or r["durationHours"]
        if h and r["difficulty"]:
            diff_dur[r["difficulty"]].append(float(h))

    avgs = {d: sum(v) / len(v) for d, v in diff_dur.items()}
    ns   = {d: len(v) for d, v in diff_dur.items()}

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        DIFF_ORDER,
        [avgs[d] for d in DIFF_ORDER],
        color=[DIFF_COLORS[d] for d in DIFF_ORDER],
        width=0.5,
    )
    for bar, diff in zip(bars, DIFF_ORDER):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{avgs[diff]:.1f}h\n(n={ns[diff]})",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Average Duration (hours)", fontsize=11)
    ax.set_title("Average Course Duration by Difficulty Level",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylim(0, max(avgs.values()) * 1.25)

    fig.tight_layout()
    save(fig, "chart_05_avg_duration_by_difficulty.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 6 — Top 12 Longest Career & Learning Programs
# ═══════════════════════════════════════════════════════════════════════════════
def chart_06_top_career_paths():
    candidates = [
        r for r in rows
        if r["medianDurationHours"] and float(r["medianDurationHours"]) > 0
    ]
    top = sorted(candidates,
                 key=lambda r: float(r["medianDurationHours"]),
                 reverse=True)[:12]
    top = list(reversed(top))

    titles = [r["title"][:45] + ("…" if len(r["title"]) > 45 else "")
              for r in top]
    durs   = [float(r["medianDurationHours"]) for r in top]
    colors = [DIFF_COLORS.get(r["difficulty"], SLATE) for r in top]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(titles, durs, color=colors, height=0.6)
    for bar, val, r in zip(bars, durs, top):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}h  [{r['difficulty']}]",
                va="center", fontsize=9.5)

    ax.set_xlabel("Median Completion Time (hours)", fontsize=11)
    ax.set_title("Top 12 Longest Learning Programs by Completion Time",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlim(0, max(durs) * 1.22)
    ax.grid(axis="x", alpha=0.35)
    ax.grid(axis="y", alpha=0)

    # legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=DIFF_COLORS[d], label=d)
                       for d in DIFF_ORDER]
    ax.legend(handles=legend_elements, title="Difficulty",
              loc="lower right", framealpha=0.9)

    fig.tight_layout()
    save(fig, "chart_06_top_career_paths.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 7 — Certificate-Granting Rate by Content Type
# ═══════════════════════════════════════════════════════════════════════════════
def chart_07_certificate_rate_by_type():
    type_labels = {
        "Track": "Course Track",
        "Path": "Learning Path",
        "ExternalCourse": "External Course",
        "ExternalPath": "External Path",
    }
    types = ["Track", "Path", "ExternalCourse", "ExternalPath"]
    cert_count  = Counter()
    total_count = Counter()
    for r in rows:
        if r["type"] in types:
            total_count[r["type"]] += 1
            if r["grantsCertificate"] == "True":
                cert_count[r["type"]] += 1

    pcts = [cert_count[t] / total_count[t] * 100 for t in types]
    labels = [type_labels[t] for t in types]

    fig, ax = plt.subplots(figsize=(9, 6))
    bar_colors = [BLUE, TEAL, ORANGE, PURPLE]
    bars = ax.bar(labels, pcts, color=bar_colors, width=0.5)
    for bar, pct, t in zip(bars, pcts, types):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{pct:.0f}%\n({cert_count[t]}/{total_count[t]})",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Certificate-Granting Rate (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Certificate-Granting Rate by Content Type",
                 fontsize=13, fontweight="bold", pad=14)
    ax.axhline(y=100, color=SLATE, linestyle="--", alpha=0.4, linewidth=1)

    fig.tight_layout()
    save(fig, "chart_07_certificate_rate_by_type.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 8 — Pro Monetisation Rate by Difficulty (Tracks)
# ═══════════════════════════════════════════════════════════════════════════════
def chart_08_pro_monetisation_by_difficulty():
    tracks = [r for r in rows if r["type"] == "Track"]
    data = {}
    for diff in DIFF_ORDER:
        sub  = [r for r in tracks if r["difficulty"] == diff]
        pro  = sum(1 for r in sub if r["pro"] == "True")
        free = sum(1 for r in sub if r["pro"] == "False")
        data[diff] = {"Pro": pro, "Free": free, "total": pro + free}

    x      = np.arange(len(DIFF_ORDER))
    width  = 0.55

    fig, ax = plt.subplots(figsize=(8, 6))
    bottoms = np.zeros(len(DIFF_ORDER))
    for cat, col in [("Free", TEAL), ("Pro", ORANGE)]:
        vals = np.array([data[d][cat] for d in DIFF_ORDER])
        bars = ax.bar(x, vals, width, bottom=bottoms, label=cat, color=col)
        for bar, val in zip(bars, vals):
            if val > 0:
                cy = bar.get_y() + bar.get_height() / 2
                ax.text(bar.get_x() + bar.get_width() / 2, cy,
                        str(val), ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")
        bottoms += vals

    # pro % labels on top
    for i, diff in enumerate(DIFF_ORDER):
        tot = data[diff]["total"]
        pct = data[diff]["Pro"] / tot * 100
        ax.text(x[i], tot + 3, f"{pct:.0f}% paid",
                ha="center", fontsize=9.5, color=ORANGE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(DIFF_ORDER, fontsize=12)
    ax.set_ylabel("Number of Course Tracks", fontsize=11)
    ax.set_title("Monetisation Split — Free vs. Pro Tracks by Difficulty",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    save(fig, "chart_08_pro_monetisation_by_difficulty.png")


# ── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating charts...")
    chart_01_catalog_composition()
    chart_02_free_vs_pro_by_difficulty()
    chart_03_duration_distribution()
    chart_04_certification_providers()
    chart_05_avg_duration_by_difficulty()
    chart_06_top_career_paths()
    chart_07_certificate_rate_by_type()
    chart_08_pro_monetisation_by_difficulty()
    print(f"\nAll charts saved to: {CHARTS_DIR}")
