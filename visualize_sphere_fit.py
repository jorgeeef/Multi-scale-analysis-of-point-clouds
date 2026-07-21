# visualize_sphere_fit.py
# =========================================================
# Illustre l'ajustement de la sphère algébrique GLS: 
# Pour quelques points choisis (concave / convexe / quasi-plan) 
# et quelques échelles (petite / moyenne / grande), montrant le
# voisinage P_t(p) et la sphère ajustée correspondante, et sauvegarde
# les valeurs numériques (τ, η, κ, ϕ, ν, centre, rayon) dans un
# fichier texte. 

import os
import sys
sys.path.append("src")

import numpy as np

from obj_reader import load_obj, build_point_cloud_with_normals
from geometry import (
    clean_point_cloud,
    build_kdtree,
    estimate_mean_spacing,
)
from gls import gls_at_point, sphere_center_radius
from visualization import show_sphere_fit_interactive


OBJ_PATH   = "data/sphere.obj"
OBJ_NAME   = "12140_Skull_v3"
OUTPUT_DIR = "results"


def pick_reference_points(vertices, normals_np, tree, t_ref):
    """
    Choisit 3 points caractéristiques (concave / convexe / quasi-plan)
    en fittant à une échelle de référence t_ref et en triant par κ.
    """
    N = len(vertices)
    sample_idx = np.random.default_rng(0).choice(N, min(N, 1500), replace=False)

    kappas = {}
    for i in sample_idx:
        p = vertices[i]
        idx = tree.query_ball_point(p, r=t_ref)
        if len(idx) < 12:
            continue
        res = gls_at_point(p, vertices[idx], normals_np[idx], t_ref)
        if res:
            kappas[i] = res["kappa"]

    if not kappas:
        raise RuntimeError("Aucun fit valide à l'échelle de référence.")

    idx_sorted = sorted(kappas, key=lambda i: kappas[i])
    idx_concave  = idx_sorted[int(0.02 * len(idx_sorted))]     # κ très négatif
    idx_convex   = idx_sorted[int(0.98 * len(idx_sorted))]     # κ très positif
    idx_flat     = min(kappas, key=lambda i: abs(kappas[i]))   # κ ≈ 0

    return {
        "concave": idx_concave,
        "convexe": idx_convex,
        "quasi_plan": idx_flat,
    }


def main():
    print(f"[SPHERE-FIT] Chargement : {OBJ_PATH}")
    vertices, faces, obj_normals = load_obj(OBJ_PATH) # Charger le modèle du crâne
    vertices = clean_point_cloud(vertices) # Nettoyer le nuage de points

    pcd = build_point_cloud_with_normals(vertices, faces, obj_normals) # Calculer les normales des points.
    normals_np = np.asarray(pcd.normals) 

    tree = build_kdtree(vertices) # Construire la structure de recherche des voisins.
    spacing = estimate_mean_spacing(vertices) # Calculer l'espacement moyen des points.
    print(f"[SPHERE-FIT] {len(vertices)} points, espacement moyen = {spacing:.5f}")

    # 3 échelles : petite / moyenne / grande (facteurs de l'espacement)
    scales = {
        "petite" : 6.0  * spacing,
        "moyenne": 14.0 * spacing,
        "grande" : 30.0 * spacing,
    }
    print(f"[SPHERE-FIT] Échelles testées : "
          f"{ {k: round(v,4) for k,v in scales.items()} }")

    labeled_points = pick_reference_points(vertices, normals_np, tree,
                                           t_ref=scales["petite"]) # Choisir un point concave, convexe et quasi-plan.

    print(f"[SPHERE-FIT] Points choisis : "
          f"{ {k: f'p{i:06d}' for k,i in labeled_points.items()} }")

    # Contexte affiché en gris : le crâne complet (tous les points),
    # pas un sous-échantillon.
    context_points = vertices

    output_folder = os.path.join(OUTPUT_DIR, OBJ_NAME, "sphere_fit")
    os.makedirs(output_folder, exist_ok=True)
    table_path = os.path.join(output_folder, f"{OBJ_NAME}_sphere_fit_values.txt")

    rows = []

    for point_label, i in labeled_points.items():
        p = vertices[i]
        for scale_label, t in scales.items():
            idx = tree.query_ball_point(p, r=t) # Trouver les voisins du point.
            if len(idx) < 12:
                print(f"  [SKIP] {point_label} p{i:06d} @ {scale_label} "
                      f"(t={t:.4f}) : trop peu de voisins ({len(idx)})")
                continue

            nb, nr = vertices[idx], normals_np[idx]
            res = gls_at_point(p, nb, nr, t, with_variation=True)  # Ajuster une sphère avec la méthode GLS.
            if res is None:
                print(f"  [SKIP] {point_label} p{i:06d} @ {scale_label} "
                      f"(t={t:.4f}) : fit échoué")
                continue

            sphere = sphere_center_radius(res["u_hat"]) # Calculer le centre et le rayon de la sphère.
            tag = f"p{i:06d}_{point_label}_{scale_label}_t{t:.4f}"

            if sphere is None:
                print(f"  [PLAT ] {point_label} p{i:06d} @ {scale_label} "
                      f"(t={t:.4f}) : fit quasi-planaire, pas de sphère finie "
                      f"(κ={res['kappa']:+.2e})")
                center_str, radius_str = "n/a (plan)", "n/a (plan)"
            else:
                center, radius = sphere
                center_str = np.array2string(center, precision=4)
                radius_str = f"{radius:.4f}"
                print(f"  [FENETRE] {tag}  (r={radius:.4f}, K={len(nb)}) "
                      f"— fermez la fenêtre pour continuer")
                show_sphere_fit_interactive(
                    p=p, neighbor_points=nb, center=center, radius=radius,
                    tag=tag, context_points=context_points,
                ) # Afficher la sphère ajustée.

            eta_str = np.array2string(res["eta"], precision=4)

            rows.append({
                "point_label": point_label, "idx": i, "scale_label": scale_label,
                "t": t, "K": len(idx),
                "tau": res["tau"], "eta": eta_str, "kappa": res["kappa"],
                "phi": res["phi"], "nu": res["nu"],
                "center": center_str, "radius": radius_str,
            })  # Enregistrer les valeurs calculées.

    with open(table_path, "w", encoding="utf-8") as f: # Sauvegarder les résultats dans un fichier texte.
        f.write(f"# Sphères ajustées GLS — {OBJ_NAME}\n")
        f.write(f"# Points : {', '.join(f'{k}=p{i:06d}' for k,i in labeled_points.items())}\n")
        f.write(f"# Mellado et al. (2012), reproduction style Figures 1/3(a-b)/5\n")
        f.write(f"# tau   : offset algébrique (Eq. 4)\n")
        f.write(f"# eta   : normale du champ ajusté au point p (Eq. 4)\n")
        f.write(f"# kappa : courbure signée (Eq. 4)\n")
        f.write(f"# phi   : fitness, qualité d'alignement du fit (Section 4.1)\n")
        f.write(f"# nu    : variation géométrique ν(p,t) (Section 4.2, Eq. 5)\n\n")
        header = (f"{'point':<12} {'idx':>8} {'echelle':<10} {'t':>10} {'K':>6} "
                  f"{'tau':>12} {'eta':<22} {'kappa':>12} {'phi':>8} {'nu':>12}   "
                  f"{'centre':<28} {'rayon':>10}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in rows:
            f.write(f"{r['point_label']:<12} p{r['idx']:06d} {r['scale_label']:<10} "
                    f"{r['t']:>10.4f} {r['K']:>6d} "
                    f"{r['tau']:>+12.5f} {r['eta']:<22} {r['kappa']:>+12.5f} "
                    f"{r['phi']:>8.4f} {r['nu']:>12.5f}   "
                    f"{r['center']:<28} {r['radius']:>10}\n")

    print(f"\n[SPHERE-FIT] Table de résultats sauvegardée : {table_path}")


if __name__ == "__main__":
    main()
