# main.py
# =================================================================
# PIPELINE PRINCIPAL — GLS Multi-échelle (Mellado et al. 2012)
# =================================================================

import os
import sys
sys.path.append("src")

import numpy as np
import open3d as o3d

from obj_reader import (
    load_obj,
    build_point_cloud_with_normals,
    print_stats,
    visualize_points,
    visualize_normals,
    compare_normals,
    save_pointcloud_ply,         
    save_pointcloud_screenshot,
)
from geometry import (
    clean_point_cloud,
    build_kdtree,
    knn_neighbors,
    multi_scale_neighbors,
    estimate_mean_spacing,
    build_scales_from_spacing,
    compute_validity_mask,
    print_scale_stats,
)
from gls      import gls_at_point
from notebooks import notebook_exists, save_results, load_results


# =================================================================
if __name__ == "__main__":

    # ----------------------------------------------------------
    # STEP 1 — SÉLECTION DU FICHIER OBJ
    # ----------------------------------------------------------
    data_folder = "data"
    obj_files   = sorted(f for f in os.listdir(data_folder)
                         if f.lower().endswith(".obj"))

    if not obj_files:
        raise FileNotFoundError(
            f"Aucun fichier .obj trouvé dans le dossier '{data_folder}'."
        )

    print("\nFichiers OBJ disponibles :")
    for i, fname in enumerate(obj_files, start=1):
        print(f"  {i}. {fname}")

    while True:
        try:
            choice = int(input("\nChoisissez un fichier (numéro) : "))
            if 1 <= choice <= len(obj_files):
                break
            print(f"  Numéro invalide (1 – {len(obj_files)}).")
        except ValueError:
            print("  Veuillez entrer un nombre entier.")

    selected  = obj_files[choice - 1]
    path      = os.path.join(data_folder, selected)
    obj_name  = os.path.splitext(selected)[0]
    print(f"\n[INFO] Fichier sélectionné : {path}")

    # ----------------------------------------------------------
    # STEP 2 — CHARGEMENT + NETTOYAGE  (toujours)
    # ----------------------------------------------------------
    vertices, faces, obj_normals = load_obj(path)
    vertices = clean_point_cloud(vertices)

    # ----------------------------------------------------------
    # STEP 3 — NORMALES + PCD  (toujours)
    # ----------------------------------------------------------
    pcd = build_point_cloud_with_normals(vertices, faces, obj_normals)
    print_stats(vertices, faces, pcd)

    # ----------------------------------------------------------
    # STEP 4 — SAUVEGARDES PCD  (toujours, à chaque exécution)
    # ----------------------------------------------------------
    save_pointcloud_ply(pcd, obj_name)
    save_pointcloud_screenshot(pcd, obj_name, show_normals=False)
    save_pointcloud_screenshot(pcd, obj_name, show_normals=True)

    # ----------------------------------------------------------
    # STEP 5 — VISUALISATION OPEN3D  (toujours)
    # ----------------------------------------------------------
    visualize_points(pcd)
    visualize_normals(pcd)

    # ----------------------------------------------------------
    # STEP 6 — CACHE GLS  (lecture si dispo, sinon calcul)
    # ----------------------------------------------------------
    if notebook_exists(obj_name):

        print(f"[NOTEBOOK] Cache trouvé pour '{obj_name}' → lecture.")
        results   = load_results(obj_name)
        scales    = results["scales"]
        TAU       = results["TAU"]
        KAPPA     = results["KAPPA"]
        ETA_angle = results["ETA_angle"]

        print(f"       scales : {np.round(scales, 4)}")
        print(f"       TAU    : {TAU.shape}")
        print(f"       KAPPA  : {KAPPA.shape}")
        print(f"       ETA°   : {ETA_angle.shape}")

    else:

        print(f"[NOTEBOOK] Aucun cache pour '{obj_name}' → calcul complet.")

        # Comparaison normales seulement au premier calcul
        compare_normals(vertices, faces, obj_normals)

        # KD-TREE + K-NN
        tree = build_kdtree(vertices)
        knn  = knn_neighbors(tree, vertices, k=30)
        print(f"[K-NN] shape : {knn.shape}")

        # Voisinages multi-échelle
        spacing = estimate_mean_spacing(vertices)
        print(f"[SPACING] espacement moyen : {spacing:.6f}")

        scales = build_scales_from_spacing(
            spacing,
            n_scales   = 12,
            factor_min = 5,
            factor_max = 15,
            mode       = "log",
        )
        print(f"[SCALES] {np.round(scales, 4)}")

        neighborhoods_dict = multi_scale_neighbors(vertices, scales)
        masks_dict         = compute_validity_mask(neighborhoods_dict,
                                                   min_neighbors=6)
        neighborhoods = [neighborhoods_dict[t] for t in scales]
        masks         = [masks_dict[t]         for t in scales]

        print_scale_stats(neighborhoods_dict, scales, masks_dict)

        # Fitting GLS
        normals_np = np.asarray(pcd.normals)
        N          = len(vertices)
        S          = len(scales)

        TAU   = np.full((N, S),    np.nan)
        KAPPA = np.full((N, S),    np.nan)
        PHI   = np.full((N, S),    np.nan)
        ETA   = np.full((N, S, 3), np.nan)

        for j, t in enumerate(scales):
            print(f"[GLS] échelle {j+1}/{S}  t={t:.4f}")
            for i, p in enumerate(vertices):
                if not masks[j][i]:
                    continue
                idx    = neighborhoods[j][i]
                result = gls_at_point(p, vertices[idx], normals_np[idx], t)
                if result:
                    TAU[i, j]   = result["tau"]
                    KAPPA[i, j] = result["kappa"]
                    PHI[i, j]   = result["phi"]
                    ETA[i, j]   = result["eta"]

        print("[GLS] Calcul terminé.")

        # Sauvegarde dans notebooks/
        save_results(
            obj_name           = obj_name,
            vertices           = vertices,
            faces              = faces,
            pcd                = pcd,
            spacing            = spacing,
            scales             = scales,
            masks_dict         = masks_dict,
            neighborhoods_dict = neighborhoods_dict,
            TAU                = TAU,
            ETA                = ETA,
            KAPPA              = KAPPA,
            normals_np         = normals_np,
        )