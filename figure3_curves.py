"""
figure3_curves.py
==================
Reproduit l'esprit de la Figure 3(c) de Mellado et al. (2012).

Contrairement à la version précédente (3 figures par région, avec
axes doubles τ/η + κ), cette version produit DEUX figures seulement,
une par quantité, chacune superposant les TROIS régions
géométriques (concavité, bord, jonction) :

    results/golf_ball/figure3_tau.png    (offset algébrique τ)
    results/golf_ball/figure3_kappa.png  (courbure κ)

η (angle) n'est plus tracé : il sert uniquement, en interne, au
filtre anti-singularité lors de la sélection des points.

Sélection des points (filtre anti-singularité η=90°) :
  - Concavité : 15e-25e percentile de κ (concave modéré)
  - Junction  : 75e-85e percentile de κ (convexe modéré)
  - Edge      : κ proche médiane + grande variance entre échelles
"""

import os
import sys
sys.path.append("src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from notebooks import notebook_exists, load_results


# ─────────────────────── Paramètres généraux ─────────────────────
OBJ_NAME   = "golf_ball"
OUTPUT_DIR = "results"

# Région -> style visuel + étiquette française (cf. figures de référence)
REGIONS = {
    "edge":      {"color": "#F2994A", "label": "Bord"},
    "concavity": {"color": "#1F9D8C", "label": "Concavité"},
    "junction":  {"color": "#7B5FD0", "label": "Jonction"},
}

CONVERGENCE_FRACTION = 0.75   # début de la bande "convergence" (fraction de [t_min, t_max])
LABEL_MIN_GAP_FRAC   = 0.09   # écart vertical minimal entre étiquettes (fraction de l'étendue des données)


# ─────────────────────── Sélection robuste des points ────────────
def select_three_points(KAPPA, ETA_angle):
    """
    Sélection robuste basée sur des percentiles MODÉRÉS, avec filtre
    anti-singularité η=90° (fit GLS instable au fond exact d'une
    concavité isotrope où ∇s_û ≈ 0). η n'est utilisé qu'ici, pour
    filtrer les points sains — il n'apparaît dans aucune figure.
    """
    kappa_small = KAPPA[:, 0]
    valid       = ~np.isnan(kappa_small)

    eta_range      = np.nanmax(ETA_angle, axis=1) - np.nanmin(ETA_angle, axis=1)
    eta_healthy    = eta_range > 5.0
    eta_not_pinned = np.abs(ETA_angle[:, 0] - 90.0) > 2.0

    good_mask = valid & eta_healthy & eta_not_pinned
    print(f"[FIGURE3] Points sains (η non saturé) : "
          f"{int(np.sum(good_mask))}/{len(kappa_small)}")

    good_idx = np.where(good_mask)[0]
    k_good   = kappa_small[good_idx]

    # Concavité modérée
    p15, p25 = np.percentile(k_good, [15, 25])
    cand_c   = good_idx[(kappa_small[good_idx] >= p15) &
                        (kappa_small[good_idx] <= p25)]
    idx_concavity = int(cand_c[len(cand_c) // 2])

    # Junction modérée
    p75, p85 = np.percentile(k_good, [75, 85])
    cand_j   = good_idx[(kappa_small[good_idx] >= p75) &
                        (kappa_small[good_idx] <= p85)]
    idx_junction = int(cand_j[len(cand_j) // 2])

    # Edge
    median_k    = np.median(k_good)
    dist_to_med = np.abs(kappa_small - median_k)
    dist_to_med[~good_mask] = np.inf
    median_mask = dist_to_med < np.percentile(dist_to_med[good_mask], 10)
    kappa_var   = np.nanstd(KAPPA, axis=1)
    score       = np.where(median_mask, kappa_var, -np.inf)
    idx_edge    = int(np.argmax(score))

    return idx_concavity, idx_edge, idx_junction


# ─────────────────────── Style commun aux deux figures ───────────
def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAAAAA")
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.tick_params(colors="#888888", labelsize=10)
    ax.set_xlabel("Échelle  t", fontsize=12, color="#888888")
    ax.axhline(0, color="#CFCFCF", linewidth=1, zorder=0)


def _add_convergence_band(ax, scales):
    t_min, t_max = scales[0], scales[-1]
    conv_start = t_min + CONVERGENCE_FRACTION * (t_max - t_min)
    ax.axvspan(conv_start, t_max, color="#999999", alpha=0.10, zorder=0)
    ax.text((conv_start + t_max) / 2, 0.93, "convergence",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=10,
            color="#999999", style="italic")


def _place_labels(ax, scales, finals, value_fmt, y_data_range):
    """
    finals : liste de tuples (clé_région, valeur_finale)

    Empile les étiquettes par ordre décroissant de valeur, à droite
    du dernier point de chaque courbe, en imposant un écart vertical
    minimal pour éviter tout chevauchement (utile quand les trois
    courbes convergent vers des valeurs très proches).
    """
    t_min, t_max = scales[0], scales[-1]
    x_text = t_max + 0.04 * (t_max - t_min)

    ordered = sorted(finals, key=lambda kv: -kv[1])
    min_gap = LABEL_MIN_GAP_FRAC * y_data_range

    placed_y = []
    for i, (key, val) in enumerate(ordered):
        y = val if i == 0 else min(val, placed_y[-1] - min_gap)
        placed_y.append(y)

    for (key, val), y in zip(ordered, placed_y):
        color = REGIONS[key]["color"]
        label = REGIONS[key]["label"]
        ax.plot(t_max, val, "o", color=color, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.text(x_text, y, f"{label}  {value_fmt.format(val)}",
                color=color, fontsize=11.5, fontweight="bold",
                va="center", ha="left", zorder=6)


# ─────────────────────── Tracé combiné (1 quantité, 3 régions) ───
def plot_combined(scales, curves, title_symbol, title_color,
                  value_fmt, output_path):
    """
    curves : dict {clé_région: array(n_scales,)}  -- ex. TAU[idx,:] ou KAPPA[idx,:]
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for key, vals in curves.items():
        ax.plot(scales, vals, color=REGIONS[key]["color"],
                linewidth=2.6, solid_capstyle="round", zorder=3)

    # Étendue verticale réelle des courbes (toutes échelles, toutes régions)
    all_vals     = np.concatenate(list(curves.values()))
    y_lo, y_hi   = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    y_data_range = (y_hi - y_lo) if (y_hi - y_lo) > 1e-9 else 1.0
    # Marge verticale pour que les étiquettes empilées ne touchent jamais l'axe
    ax.set_ylim(y_lo - 0.20 * y_data_range, y_hi + 0.12 * y_data_range)

    _add_convergence_band(ax, scales)
    _style_axes(ax)

    t_min, t_max = scales[0], scales[-1]
    # Graduations confinées à la plage réelle des données (avant d'élargir
    # xlim pour laisser de la place aux étiquettes à droite)
    xticks = [tk for tk in MaxNLocator(nbins=6).tick_values(t_min, t_max)
              if t_min - 1e-9 <= tk <= t_max + 1e-9]
    ax.set_xticks(xticks)
    ax.set_xlim(t_min - 0.02 * (t_max - t_min), t_max + 0.34 * (t_max - t_min))

    finals = [(key, float(vals[-1])) for key, vals in curves.items()]
    _place_labels(ax, scales, finals, value_fmt, y_data_range)

    ax.text(0.0, 1.04, title_symbol, transform=ax.transAxes,
            fontsize=22, color=title_color, fontweight="bold",
            ha="left", va="bottom")

    # Pas de tight_layout : il recalcule les marges sans tenir compte des
    # étiquettes posées hors de la zone d'axes ; bbox_inches="tight" au
    # moment de la sauvegarde s'en charge correctement.
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"[FIGURE3]  Sauvegardé : {output_path}")


# ─────────────────────── Sauvegarde des valeurs (TXT) ────────────
def save_values_txt(scales, TAU, KAPPA, idx_by_region, output_path):
    with open(output_path, "w") as f:
        f.write("# Valeurs sélectionnées pour la Figure 3 (τ, κ)\n")
        for key, idx in idx_by_region.items():
            f.write(f"# {REGIONS[key]['label']:<10} -> point p{idx:06d}\n")
        f.write("\n")

        header = f"  {'t':>10}"
        for key in idx_by_region:
            label = REGIONS[key]["label"]
            header += f"  {label + '_tau':>16}  {label + '_kappa':>16}"
        f.write(header + "\n")
        f.write("  " + "-" * (len(header) - 2) + "\n")

        for j, t in enumerate(scales):
            row = f"  {t:>10.4f}"
            for key, idx in idx_by_region.items():
                row += f"  {TAU[idx, j]:>+16.6f}  {KAPPA[idx, j]:>+16.4f}"
            f.write(row + "\n")


# ─────────────────────────── Main ─────────────────────────────────
if __name__ == "__main__":

    if not notebook_exists(OBJ_NAME):
        print(f"ERREUR : aucun cache pour '{OBJ_NAME}'.")
        sys.exit(1)

    print(f"\n[FIGURE3] Lecture du cache pour {OBJ_NAME}…")
    results   = load_results(OBJ_NAME)
    scales    = results["scales"]
    TAU       = results["TAU"]
    KAPPA     = results["KAPPA"]
    ETA_angle = results["ETA_angle"]   # utilisé uniquement par le filtre de sélection

    print(f"[FIGURE3] {len(scales)} échelles, t ∈ [{scales[0]:.3f}, {scales[-1]:.3f}]")
    print(f"[FIGURE3] {TAU.shape[0]} points\n")

    idx_conc, idx_edge, idx_junc = select_three_points(KAPPA, ETA_angle)
    idx_by_region = {"concavity": idx_conc, "edge": idx_edge, "junction": idx_junc}

    print("\n[FIGURE3] Points sélectionnés :")
    for key, idx in idx_by_region.items():
        print(f"  {REGIONS[key]['label']:<10} p{idx:06d}  "
              f"κ_s01 = {KAPPA[idx, 0]:+.3f}  η_s01 = {ETA_angle[idx, 0]:.1f}°")

    output_folder = os.path.join(OUTPUT_DIR, OBJ_NAME)
    os.makedirs(output_folder, exist_ok=True)

    # ── Figure τ : les trois régions superposées ──
    tau_curves = {key: TAU[idx, :] for key, idx in idx_by_region.items()}
    plot_combined(
        scales, tau_curves,
        title_symbol="τ", title_color="#D9534F",
        value_fmt="{:+.3f}",
        output_path=os.path.join(output_folder, "figure3_tau.png"),
    )

    # ── Figure κ : les trois régions superposées ──
    kappa_curves = {key: KAPPA[idx, :] for key, idx in idx_by_region.items()}
    plot_combined(
        scales, kappa_curves,
        title_symbol="K", title_color="#178A8A",
        value_fmt="{:+.2f}",
        output_path=os.path.join(output_folder, "figure3_kappa.png"),
    )

    save_values_txt(scales, TAU, KAPPA, idx_by_region,
                    os.path.join(output_folder, "figure3_values.txt"))

    print("\n[FIGURE3] 2 figures (τ, κ) + 1 fichier de valeurs sauvegardés.")