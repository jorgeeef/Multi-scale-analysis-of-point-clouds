"""
figure3_curves.py
==================
Reproduit l'esprit de la Figure 3(c) de Mellado et al. (2012).

Trois figures séparées, une par région géométrique :
    results/golf_ball/figure3_concavity.png
    results/golf_ball/figure3_edge.png
    results/golf_ball/figure3_junction.png

Chaque figure utilise DEUX AXES Y :
    - Axe gauche  : τ (offset) et η (angle en degrés)
    - Axe droit   : κ (courbure)

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

from notebooks import notebook_exists, load_results


# ─────────────────────── Paramètres ─────────────────────────────
OBJ_NAME    = "golf_ball"
OUTPUT_DIR  = "results"


# ─────────────────────── Sélection robuste ──────────────────────
def select_three_points(KAPPA, ETA_angle):
    """
    Sélection robuste basée sur des percentiles MODÉRÉS, avec filtre
    anti-singularité η=90° (fit GLS instable au fond exact d'une
    concavité isotrope où ∇s_û ≈ 0).
    """
    kappa_small = KAPPA[:, 0]
    valid       = ~np.isnan(kappa_small)

    # Filtre anti-singularité η
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


# ─────────────────────── Tracé à 2 axes Y ─────────────────────────
def plot_single_point(scales, tau_vals, eta_vals, kappa_vals,
                      idx, region_label, output_path):
    """
    Trace τ, η, κ pour un point sur un graphique avec DEUX AXES Y :
      - Axe gauche  : τ (offset) + η (angle en degrés)
      - Axe droit   : κ (courbure)
    """
    fig, ax_left = plt.subplots(figsize=(10, 6.5))
    ax_right     = ax_left.twinx()

    # ── Axe gauche : τ et η ──
    line_tau = ax_left.plot(scales, tau_vals,
                            color="#D62728", linewidth=2.4,
                            marker="o", markersize=5,
                            label="τ  (offset algébrique)")

    # η ramené à l'échelle de τ pour le co-tracé : on utilise l'axe
    # gauche mais en normalisant η dans la plage [min(τ), max(τ)]
    # via une transformation linéaire visible dans la légende.
    # POUR ÉVITER LA CONFUSION : on met η sur son propre axe gauche
    # secondaire mais on garde l'apparence d'un seul axe gauche, en
    # affichant les deux directement sur le même axe avec leurs unités.
    # Solution simple : tracer η sur l'axe gauche tel quel (les degrés
    # vont de 0 à ~180, τ va de -0.1 à +0.1 → on tient compte).
    # Pour préserver la lisibilité, on choisit d'avoir τ et η ensemble
    # mais avec leur véritable amplitude — on étiquette clairement.

    # Pour la cohabitation τ + η sur l'axe gauche, on transforme η
    # vers la plage de τ via : η_scaled = η × (τ_range / 180)
    tau_min, tau_max = np.nanmin(tau_vals), np.nanmax(tau_vals)
    tau_span = tau_max - tau_min if tau_max - tau_min > 1e-12 else 1.0
    eta_scaled = (eta_vals / 180.0) * tau_span + tau_min

    line_eta = ax_left.plot(scales, eta_scaled,
                            color="#1F77B4", linewidth=2.4,
                            marker="s", markersize=5,
                            label="η  (angle, ré-échelonné)")

    # Annoter les vraies valeurs de η aux extrémités
    ax_left.annotate(f"η = {eta_vals[0]:.1f}°",
                     xy=(scales[0], eta_scaled[0]),
                     xytext=(8, 8), textcoords="offset points",
                     fontsize=9, color="#1F77B4")
    ax_left.annotate(f"η = {eta_vals[-1]:.1f}°",
                     xy=(scales[-1], eta_scaled[-1]),
                     xytext=(-50, 8), textcoords="offset points",
                     fontsize=9, color="#1F77B4")

    ax_left.set_xlabel(r"Échelle  $t$", fontsize=13)
    ax_left.set_ylabel("τ  (offset algébrique)", fontsize=12, color="#D62728")
    ax_left.tick_params(axis="y", labelcolor="#D62728")
    ax_left.grid(True, linestyle="--", alpha=0.4)
    ax_left.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    # ── Axe droit : κ ──
    line_kappa = ax_right.plot(scales, kappa_vals,
                               color="#2CA02C", linewidth=2.4,
                               marker="^", markersize=5,
                               label="κ  (courbure signée)")
    ax_right.set_ylabel("κ  (courbure signée)",
                        fontsize=12, color="#2CA02C")
    ax_right.tick_params(axis="y", labelcolor="#2CA02C")
    ax_right.axhline(0, color="#2CA02C", linewidth=0.5,
                     alpha=0.3, linestyle=":")

    # ── Titre et légende ──
    fig.suptitle(
        f"Évolution multi-échelle  —  {region_label}\n"
        f"(point p{idx:06d})",
        fontsize=13, y=0.99
    )

    eta_range_str = f"[{np.nanmin(eta_vals):.1f}°, {np.nanmax(eta_vals):.1f}°]"

    # Combiner les légendes des 2 axes
    lines = line_tau + line_eta + line_kappa
    labels = [
        f"τ  ∈ [{tau_min:+.3f}, {tau_max:+.3f}]",
        f"η  ∈ {eta_range_str}  (axe gauche, ré-échelonné depuis [0°,180°])",
        f"κ  ∈ [{np.nanmin(kappa_vals):+.2f}, {np.nanmax(kappa_vals):+.2f}]  (axe droit)",
    ]
    ax_left.legend(lines, labels, loc="best", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE3]  Sauvegardé : {output_path}")


# ─────────────────────── Sauvegarde TXT ─────────────────────────
def save_values_txt(scales, TAU, ETA, KAPPA, idx, region_label, output_path):
    with open(output_path, "w") as f:
        f.write(f"# Point sélectionné : p{idx:06d}\n")
        f.write(f"# Région : {region_label}\n")
        f.write(f"# n_scales : {len(scales)}\n\n")

        f.write(f"  {'t':>10}  {'τ':>12}  {'η°':>10}  {'κ':>12}\n")
        f.write("  " + "-" * 50 + "\n")

        for j, t in enumerate(scales):
            f.write(f"  {t:>10.4f}  {TAU[idx,j]:>+12.6f}  "
                    f"{ETA[idx,j]:>10.4f}  {KAPPA[idx,j]:>+12.4f}\n")


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
    ETA_angle = results["ETA_angle"]

    print(f"[FIGURE3] {len(scales)} échelles, "
          f"t ∈ [{scales[0]:.3f}, {scales[-1]:.3f}]")
    print(f"[FIGURE3] {TAU.shape[0]} points\n")

    idx_conc, idx_edge, idx_junc = select_three_points(KAPPA, ETA_angle)

    print(f"\n[FIGURE3] Points sélectionnés :")
    print(f"  Concavité : p{idx_conc:06d}  "
          f"κ_s01 = {KAPPA[idx_conc,0]:+.3f}  "
          f"η_s01 = {ETA_angle[idx_conc,0]:.1f}°")
    print(f"  Edge      : p{idx_edge:06d}  "
          f"κ_s01 = {KAPPA[idx_edge,0]:+.3f}  "
          f"η_s01 = {ETA_angle[idx_edge,0]:.1f}°")
    print(f"  Junction  : p{idx_junc:06d}  "
          f"κ_s01 = {KAPPA[idx_junc,0]:+.3f}  "
          f"η_s01 = {ETA_angle[idx_junc,0]:.1f}°\n")

    output_folder = os.path.join(OUTPUT_DIR, OBJ_NAME)
    os.makedirs(output_folder, exist_ok=True)

    points = [
        (idx_conc, "Concavité  (fond d'alvéole)",   "concavity"),
        (idx_edge, "Edge  (bord d'alvéole)",        "edge"),
        (idx_junc, "Junction  (entre alvéoles)",    "junction"),
    ]

    for idx, label, fname in points:
        png_path = os.path.join(output_folder, f"figure3_{fname}.png")
        txt_path = os.path.join(output_folder, f"figure3_{fname}.txt")

        plot_single_point(
            scales      = scales,
            tau_vals    = TAU[idx, :],
            eta_vals    = ETA_angle[idx, :],
            kappa_vals  = KAPPA[idx, :],
            idx         = idx,
            region_label= label,
            output_path = png_path,
        )

        save_values_txt(scales, TAU, ETA_angle, KAPPA,
                        idx, label, txt_path)

    print(f"\n[FIGURE3] 3 figures + 3 fichiers de valeurs sauvegardés.")