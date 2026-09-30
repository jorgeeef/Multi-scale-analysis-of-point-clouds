# analyse_tau_v.py
# =========================================================
# Analyse des points à τ extrême à une échelle choisie :
#
# 1. L'utilisateur choisit une échelle parmi les échelles
#    disponibles dans le cache.
#
# 2. Recherche des N_TOP points de τ maximal à cette échelle.
#
# 3. Visualisation 3D du nuage de points :
#       - tous les points : gris
#       - les points sélectionnés : rouge
#
# 4. Tracé, sur un même graphe, des courbes (échelle, ν)
#    de ces points.
#
#    (ν = variation géométrique — Mellado et al. 2012, Eq. 5)
#
# 5. Pour chacun de ces points, régression linéaire de la courbe
#    log-log (log ν, log t) sur toutes les échelles valides.
#
#    La pente obtenue est l'exposant a du modèle :
#
#        ν ~ C · t^a
#
# 6. Sauvegarde de toutes les figures en PNG.
#
# Le script ne recalcule PAS le GLS.
# Il lit uniquement le cache déjà produit dans :
#
#     notebooks/<obj>/
#
# Lancez d'abord main.py pour l'objet choisi si le cache n'existe pas.
#
# =========================================================


import os
import sys

sys.path.append("src")

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from obj_reader import (
    load_obj,
    build_point_cloud_with_normals,
    print_stats,
)

from notebooks import (
    notebook_exists,
    load_results,
)

from geometry import (
    clean_point_cloud,
)


# PARAMÈTRES
N_TOP = 10

DATA_FOLDER = "data"
OUTPUT_DIR = "results"



# SÉLECTION DU FICHIER OBJ
def select_obj_name(data_folder=DATA_FOLDER):
    """
    Liste les fichiers .obj présents dans data/
    et laisse l'utilisateur choisir.
    """

    obj_files = sorted(
        f
        for f in os.listdir(data_folder)
        if f.lower().endswith(".obj")
    )

    if not obj_files:
        raise FileNotFoundError(
            f"Aucun fichier .obj trouvé dans le dossier "
            f"'{data_folder}'."
        )

    print("\nFichiers OBJ disponibles :")

    for i, fname in enumerate(obj_files, start=1):
        print(f"  {i}. {fname}")

    while True:

        try:

            choice = int(
                input(
                    "\nChoisissez un fichier "
                    "(numéro) : "
                )
            )

            if 1 <= choice <= len(obj_files):
                break

            print(
                f"  Numéro invalide "
                f"(1 – {len(obj_files)})."
            )

        except ValueError:

            print(
                "  Veuillez entrer un nombre entier."
            )

    selected = obj_files[choice - 1]

    return os.path.splitext(selected)[0]



# SÉLECTION DE L'ÉCHELLE
def select_scale(scales):
    """
    Affiche toutes les échelles disponibles et laisse
    l'utilisateur choisir l'échelle d'analyse.
    """

    print("\n")
    print("=" * 60)
    print("ÉCHELLES DISPONIBLES")
    print("=" * 60)

    for i, scale in enumerate(scales, start=1):

        print(
            f"  {i:2d}. "
            f"t = {scale:.6f}"
        )

    print("=" * 60)

    while True:

        try:

            choice = int(
                input(
                    f"\nChoisissez une échelle "
                    f"(1-{len(scales)}) : "
                )
            )

            if 1 <= choice <= len(scales):
                break

            print(
                f"  Numéro invalide "
                f"(1 – {len(scales)})."
            )

        except ValueError:

            print(
                "  Veuillez entrer un nombre entier."
            )

    # Conversion numéro utilisateur → indice Python
    selected_scale_idx = choice - 1

    selected_scale = scales[
        selected_scale_idx
    ]

    print(
        f"\n[TAU-V] Échelle sélectionnée : "
        f"t = {selected_scale:.6f} "
        f"(échelle {choice}/{len(scales)})"
    )

    return (
        selected_scale_idx,
        selected_scale
    )


# VISUALISATION OPEN3D

def visualize_selected_points_open3d(
    vertices,
    top_idx,
    obj_name,
    selected_scale
):
    """
    Visualise le nuage de points complet avec :
        - tous les points en gris
        - uniquement les points sélectionnés en rouge
    """

    # NUAGE DE POINTS COMPLET

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        vertices
    )

    # Tous les points sont gris
    pcd.paint_uniform_color(
        [0.7, 0.7, 0.7]
    )

    # POINTS SÉLECTIONNÉS

    selected_points = vertices[top_idx]

    selected_pcd = o3d.geometry.PointCloud()

    selected_pcd.points = o3d.utility.Vector3dVector(
        selected_points
    )

    # Points sélectionnés en rouge
    selected_pcd.paint_uniform_color(
        [1.0, 0.0, 0.0]
    )

    # INFORMATIONS

    print(
        f"\n[OPEN3D] Visualisation : {obj_name}"
    )

    print(
        f"[OPEN3D] Nombre total de points : "
        f"{len(vertices)}"
    )

    print(
        f"[OPEN3D] Points sélectionnés : "
        f"{len(top_idx)}"
    )

    print(
        "[OPEN3D] Tous les points sont gris."
    )

    print(
        f"[OPEN3D] Les {len(top_idx)} points "
        f"sélectionnés à t={selected_scale:.6f} "
        f"sont rouges."
    )

    # AFFICHAGE

    o3d.visualization.draw_geometries(
        [
            pcd,
            selected_pcd
        ],
        window_name=(
            f"{obj_name} - "
            f"{len(top_idx)} points à tau maximal "
            f"- t={selected_scale:.4f}"
        ),
        point_show_normal=False
    )

# POINTS À TAU MAXIMAL À UNE ÉCHELLE DONNÉE

def top_tau_points_at_scale(
    TAU,
    scale_idx,
    n_top=N_TOP
):
    """
    Retourne les indices des n_top points ayant les plus
    grandes valeurs de τ à l'échelle choisie.

    """

    # τ à l'échelle choisie

    tau_at_scale = TAU[:, scale_idx]

    # Points valides

    idx_valid = np.where(
        ~np.isnan(tau_at_scale)
    )[0]

    if len(idx_valid) < n_top:

        raise RuntimeError(
            f"Seulement {len(idx_valid)} points valides "
            f"à l'échelle sélectionnée, impossible "
            f"d'en extraire {n_top}."
        )

    # Tri du plus grand τ au plus petit

    order = np.argsort(
    np.abs(tau_at_scale[idx_valid])
    )[::-1][:n_top]

    top_idx = idx_valid[order]

    return top_idx


# RÉGRESSION LOG-LOG

def loglog_slope(
    scales,
    v,
    min_points=3
):
    """
    Régression linéaire de log(v) en fonction de log(t).

    Modèle :

        v ~ C · t^a

    donc :

        log(v) = a log(t) + log(C)

    La pente a est l'exposant de puissance recherché.

    Les valeurs v <= 0 ne peuvent pas être utilisées
    car log(v) n'est pas défini.
    """

    # Valeurs valides

    valid_mask = (
        ~np.isnan(v)
        &
        (v > 0)
        &
        (scales > 0)
    )

    if np.sum(valid_mask) < min_points:

        return (
            None,
            None,
            valid_mask
        )

    # Passage en logarithme

    log_t = np.log(
        scales[valid_mask]
    )

    log_v = np.log(
        v[valid_mask]
    )

    # Régression linéaire

    slope, intercept = np.polyfit(
        log_t,
        log_v,
        1
    )

    return (
        slope,
        intercept,
        valid_mask
    )


# COURBES DES DESCRIPTEURS

def plot_descriptor_curves(
    scales,
    data,
    top_idx,
    TAU,
    selected_scale_idx,
    selected_scale,
    obj_name,
    output_folder,
    ylabel,
    descriptor_name,
    file_tag
):
    """
    Trace, sur un même graphe, les courbes :

        (échelle, valeur)

    pour les points sélectionnés.

    Le graphe est sauvegardé en PNG.
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    # Palette de couleurs

    cmap = plt.get_cmap("tab20")

    # Courbe de chaque point
    for rank, i in enumerate(top_idx):

        ax.semilogx(
            scales,
            data[i, :],
            marker="o",
            markersize=3,
            color=cmap(rank % 20),
            label=(
                f"p{i:06d} "
                f"(τ={TAU[i, selected_scale_idx]:+.3f})"
            )
        )

    # Ligne verticale correspondant à l'échelle choisie
    ax.axvline(
        selected_scale,
        linestyle="--",
        linewidth=1.2,
        color="black",
        label=(
            f"échelle sélectionnée "
            f"t={selected_scale:.4f}"
        )
    )

    # Axes
    ax.set_xlabel(
        "échelle t"
    )

    ax.set_ylabel(
        ylabel
    )

    # Titre
    ax.set_title(
        f"{obj_name} — {descriptor_name} "
        f"pour les {len(top_idx)} points de τ maximal "
        f"à t={selected_scale:.4f}"
    )

    # Légende
    ax.legend(
        fontsize=6,
        ncol=2,
        loc="best"
    )
    # Grille

    ax.grid(
        True,
        which="both",
        alpha=0.3
    )

    fig.tight_layout()

    # Sauvegarde

    fig_path = os.path.join(
        output_folder,
        f"{obj_name}_top{len(top_idx)}_"
        f"scale{selected_scale_idx + 1}_{file_tag}_curves.png"
    )

    fig.savefig(
        fig_path,
        dpi=150
    )

    plt.close(fig)

    print(
        f"\n[TAU-V] Graphe sauvegardé : "
        f"{fig_path}"
    )

# GRAPHES LOG-LOG INDIVIDUELS

def plot_nu_loglog_per_point(
    scales,
    NU,
    top_idx,
    TAU,
    selected_scale_idx,
    selected_scale,
    obj_name,
    output_folder,
    slopes,
    intercepts,
    valid_masks
):
    """
    Trace un graphe log-log séparé pour chaque point.

    Le graphe contient :

        - les valeurs ν(p,t)
        - la droite de régression
        - la pente a

    Modèle :

        ν ~ C · t^a

    """

    # Un graphe par point

    for rank, i in enumerate(top_idx):

        v = NU[i, :]

        valid_mask = valid_masks[rank]

        slope = slopes[rank]

        intercept = intercepts[rank]

    
        # Création de la figure

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        # Données mesurées
        ax.loglog(
            scales[valid_mask],
            v[valid_mask],
            marker="o",
            markersize=4,
            linestyle="none",
            color="tab:blue",
            label="ν(p,t) mesuré"
        )

        # Droite de régression

        if slope is not None:

            fit_t = scales[
                valid_mask
            ]

            fit_v = (
                np.exp(intercept)
                *
                fit_t ** slope
            )

            ax.plot(
                fit_t,
                fit_v,
                linestyle="--",
                linewidth=1.5,
                color="tab:red",
                label=f"fit : a = {slope:+.4f}"
            )

        # Ligne verticale à l'échelle sélectionnée

        ax.axvline(
            selected_scale,
            linestyle=":",
            linewidth=1.2,
            color="black",
            label=(
                f"échelle sélectionnée "
                f"t={selected_scale:.4f}"
            )
        )

        # Axes

        ax.set_xlabel(
            "échelle t (log)"
        )

        ax.set_ylabel(
            "ν(p,t) (log)"
        )

        # Titre
        ax.set_title(
            f"{obj_name} — p{i:06d} "
            f"(τ={TAU[i, selected_scale_idx]:+.3f} "
            f"à t={selected_scale:.4f})"
        )

        # Légende
        ax.legend(
            fontsize=8,
            loc="best"
        )

        # Grille

        ax.grid(
            True,
            which="both",
            alpha=0.3
        )

        fig.tight_layout()

        # Sauvegarde
        fig_path = os.path.join(
            output_folder,
            f"{obj_name}_p{i:06d}_"
            f"scale{selected_scale_idx + 1}_"
            f"nu_loglog_fit.png"
        )

        fig.savefig(
            fig_path,
            dpi=150
        )

        plt.close(fig)

    print(
        f"\n[TAU-V] {len(top_idx)} graphes log-log "
        f"sauvegardés dans : {output_folder}"
    )


# PROGRAMME PRINCIPAL
def main():

    # 1. CHOIX DU FICHIER OBJ
    obj_name = select_obj_name()

    selected_file = obj_name + ".obj"

    path = os.path.join(
        DATA_FOLDER,
        selected_file
    )

    print(
        f"\n[INFO] Fichier sélectionné : "
        f"{path}"
    )

    # 2. CHARGEMENT + NETTOYAGE
    vertices, faces, obj_normals = load_obj(
        path
    )

    vertices = clean_point_cloud(
        vertices
    )

    # 3. CONSTRUCTION DU NUAGE DE POINTS

    pcd = build_point_cloud_with_normals(
        vertices,
        faces,
        obj_normals
    )

    print_stats(
        vertices,
        faces,
        pcd
    )

    # 4. VÉRIFICATION DU CACHE

    if not notebook_exists(obj_name):

        raise RuntimeError(
            f"Aucun résultat en cache pour "
            f"'{obj_name}'. "
            f"Lancez d'abord main.py pour cet objet "
            f"(calcul GLS)."
        )

    # 5. CHARGEMENT DU CACHE
    print(
        f"\n[NOTEBOOK] Chargement du cache : "
        f"notebooks/{obj_name}"
    )

    results = load_results(
        obj_name
    )

    # Échelles
    scales = np.asarray(
        results["scales"]
    )

    # TAU
    TAU = np.asarray(
        results["TAU"]
    )

    # NU
    NU = np.asarray(
        results["NU"]
    )

    # KAPPA
    KAPPA = np.asarray(
        results["KAPPA"]
    )

    print(
        f"\n[NOTEBOOK] TAU   chargé : "
        f"{TAU.shape}"
    )

    print(
        f"[NOTEBOOK] NU    chargé : "
        f"{NU.shape}"
    )

    print(
        f"[NOTEBOOK] KAPPA chargé : "
        f"{KAPPA.shape}"
    )

    # 6. CHOIX DE L'ÉCHELLE
    selected_scale_idx, selected_scale = select_scale(
        scales
    )

    # 7. RECHERCHE DES POINTS À TAU MAXIMAL
    top_idx = top_tau_points_at_scale(
        TAU=TAU,
        scale_idx=selected_scale_idx,
        n_top=N_TOP
    )

    print(
        f"\n[TAU-V] {N_TOP} points à τ maximal "
        f"(t={selected_scale:.6f}) :"
    )

    for rank, i in enumerate(
        top_idx,
        start=1
    ):

        print(
            f"  {rank:2d}. "
            f"p{i:06d} "
            f"τ = "
            f"{TAU[i, selected_scale_idx]:+.6f}"
        )


        # Coordonnées XYZ du point
        x, y, z = vertices[i]

        print(
            f"       XYZ = "
            f"({x:.6f}, "
            f"{y:.6f}, "
            f"{z:.6f})"
        )

        print(
            f"       OBJ vertex = {i + 1}"
        )

    # 8. VISUALISATION 3D
    visualize_selected_points_open3d(
        vertices=vertices,
        top_idx=top_idx,
        obj_name=obj_name,
        selected_scale=selected_scale
    )

    # 9. DOSSIER DE SORTIE
    output_folder = os.path.join(
        OUTPUT_DIR,
        obj_name,
        "tau_v_analysis"
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    print(
        f"\n[INFO] Dossier de sortie : "
        f"{output_folder}"
    )

    # 10. COURBES ν

    plot_descriptor_curves(
        scales=scales,
        data=NU,
        top_idx=top_idx,
        TAU=TAU,
        selected_scale_idx=selected_scale_idx,
        selected_scale=selected_scale,
        obj_name=obj_name,
        output_folder=output_folder,
        ylabel="ν(p,t) (v)",
        descriptor_name="ν(p,t)",
        file_tag="nu"
    )

    # 11. COURBES τ

    plot_descriptor_curves(
        scales=scales,
        data=TAU,
        top_idx=top_idx,
        TAU=TAU,
        selected_scale_idx=selected_scale_idx,
        selected_scale=selected_scale,
        obj_name=obj_name,
        output_folder=output_folder,
        ylabel="τ(p,t)",
        descriptor_name="τ(p,t)",
        file_tag="tau"
    )

    # 12. COURBES κ

    plot_descriptor_curves(
        scales=scales,
        data=KAPPA,
        top_idx=top_idx,
        TAU=TAU,
        selected_scale_idx=selected_scale_idx,
        selected_scale=selected_scale,
        obj_name=obj_name,
        output_folder=output_folder,
        ylabel="κ(p,t)",
        descriptor_name="κ(p,t)",
        file_tag="kappa"
    )

    # 13. RÉGRESSION LOG-LOG DE ν

    print(
        "\n[TAU-V] Régression log-log de ν(p,t) "
        "(pente = exposant de puissance) :"
    )

    slopes = []
    intercepts = []
    valid_masks = []

    for i in top_idx:

        slope, intercept, valid_mask = loglog_slope(
            scales,
            NU[i, :]
        )

        slopes.append(
            slope
        )

        intercepts.append(
            intercept
        )

        valid_masks.append(
            valid_mask
        )

    # 14. AFFICHAGE DES RÉSULTATS

    print("\n")
    print(
        "=" * 70
    )

    print(
        "RÉSULTATS DE LA RÉGRESSION LOG-LOG"
    )

    print(
        "=" * 70
    )

    header = (
        f"{'rang':>4} "
        f"{'point':>8} "
        f"{'tau(t_sel)':>12} "
        f"{'pente (a)':>12} "
        f"{'n_pts':>6}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    # 15. RAPPORT TEXTE

    report_lines = [

        f"# Analyse tau/nu — {obj_name}\n",

        f"# {N_TOP} points de tau maximal "
        f"a l'echelle selectionnee "
        f"(t={selected_scale:.6f})\n",

        f"# Indice de l'echelle : "
        f"{selected_scale_idx + 1}/{len(scales)}\n",

        f"# Nombre total d'echelles : "
        f"{len(scales)}\n",

        "# v = nu(p,t), variation geometrique "
        "# (Mellado et al. 2012, Eq. 5)\n",

        "# pente a de la regression lineaire "
        "# log(nu) ~ a*log(t) + b\n",

        "# soit nu ~ C * t^a\n",

        "# regression effectuee sur toutes "
        "# les echelles valides du point\n\n",

        header + "\n",

        "-" * len(header) + "\n"
    ]

    # Résultats individuels

    for rank, i in enumerate(
        top_idx,
        start=1
    ):

        slope = slopes[
            rank - 1
        ]

        n_pts = int(
            np.sum(
                valid_masks[
                    rank - 1
                ]
            )
        )

        if slope is not None:

            slope_str = (
                f"{slope:+.6f}"
            )

        else:

            slope_str = "—"

        print(
            f"{rank:4d} "
            f"p{i:06d} "
            f"{TAU[i, selected_scale_idx]:+12.6f} "
            f"{slope_str:>12} "
            f"{n_pts:6d}"
        )

        report_lines.append(
            f"{rank:>4d} "
            f"p{i:06d} "
            f"{TAU[i, selected_scale_idx]:>+12.6f} "
            f"{slope_str:>12} "
            f"{n_pts:>6d}\n"
        )

    # 16. STATISTIQUES SUR LES PENTES
    valid_slopes = np.array(
        [
            s
            for s in slopes
            if s is not None
        ]
    )

    if len(valid_slopes) > 0:

        mean_slope = (
            valid_slopes.mean()
        )

        std_slope = (
            valid_slopes.std()
        )

        min_slope = (
            valid_slopes.min()
        )

        max_slope = (
            valid_slopes.max()
        )

        print(
            "\n[TAU-V] Statistiques sur les pentes:"
        )

        print(
            f"  Moyenne     : "
            f"{mean_slope:+.6f}"
        )

        print(
            f"  Écart-type  : "
            f"{std_slope:.6f}"
        )

        print(
            f"  Minimum     : "
            f"{min_slope:+.6f}"
        )

        print(
            f"  Maximum     : "
            f"{max_slope:+.6f}"
        )

        report_lines.append(
            f"\nPente moyenne : "
            f"{mean_slope:+.6f} "
            f"ecart-type : "
            f"{std_slope:.6f} "
            f"min : "
            f"{min_slope:+.6f} "
            f"max : "
            f"{max_slope:+.6f}\n"
        )

    # 17. SAUVEGARDE DU RAPPORT

    report_path = os.path.join(
        output_folder,
        f"{obj_name}_top{N_TOP}_"
        f"scale{selected_scale_idx + 1}_"
        f"tau_v_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.writelines(
            report_lines
        )

    print(
        f"\n[TAU-V] Rapport texte sauvegardé : "
        f"{report_path}"
    )

    # 18. GRAPHES LOG-LOG INDIVIDUELS
    plot_nu_loglog_per_point(
        scales=scales,
        NU=NU,
        top_idx=top_idx,
        TAU=TAU,
        selected_scale_idx=selected_scale_idx,
        selected_scale=selected_scale,
        obj_name=obj_name,
        output_folder=output_folder,
        slopes=slopes,
        intercepts=intercepts,
        valid_masks=valid_masks
    )

    # 19. INTERPRÉTATION

    print(
        "\n[INTERPRÉTATION]"
    )

    if len(valid_slopes) > 0:

        print(
            f"  Pente log-log moyenne "
            f"(ν ~ C·t^a) sur "
            f"{len(valid_slopes)}/{N_TOP} "
            f"points valides : "
            f"a = {mean_slope:+.4f} "
            f"(écart-type {std_slope:.4f})."
        )

        if mean_slope > 0:

            print(
                "  → a > 0 en moyenne : "
                "ν croît avec l'échelle."
            )

        elif mean_slope < 0:

            print(
                "  → a < 0 en moyenne : "
                "ν décroît avec l'échelle."
            )

        else:

            print(
                "  → a ≈ 0 en moyenne : "
                "ν varie peu avec l'échelle."
            )

    else:

        print(
            "  Aucun point ne dispose d'assez "
            "de valeurs valides pour une "
            "régression log-log."
        )

    # 20. FIN

    print("\n")
    print(
        "=" * 60
    )

    print(
        "ANALYSE TERMINÉE"
    )

    print(
        "=" * 60
    )

    print(
        f"Résultats disponibles dans : "
        f"{output_folder}"
    )

    print(
        "=" * 60
    )

    plt.close("all")


if __name__ == "__main__":
    main()