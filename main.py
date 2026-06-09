# main.py
# PIPELINE PRINCIPAL — PRÉPARATION DES DONNÉES (ÉTAPES 1 à 6)
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
    compare_normals
)

from geometry import (
    clean_point_cloud,
    build_kdtree,
    knn_neighbors,
    multi_scale_neighbors,
    estimate_mean_spacing,
    build_scales_from_spacing,
    compute_validity_mask,
    print_scale_stats
)

from gls import gls_at_point


if __name__ == "__main__":


    # STEP 1 — LECTURE DU MODÈLE .OBJ
    #path = os.path.join("data", "12140_Skull_v3.obj")
    data_folder = "data"
    obj_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".obj")]
    if not obj_files:
        raise FileNotFoundError(f"Aucun fichier .obj trouvé dans le dossier '{data_folder}'.")
    print("\nFichiers OBJ disponibles :")
    for i, file in enumerate(obj_files, start=1):
        print(f"{i}. {file}")
    while True:
        try:
            choice = int(input("\nChoisissez un fichier (numéro) : "))
            if 1 <= choice <= len(obj_files):
                break
            else:
                print("Numéro invalide.")
        except ValueError:
            print("Veuillez entrer un nombre.")

    # Construire le chemin du fichier sélectionné
    path = os.path.join(data_folder, obj_files[choice - 1])
    print(f"\nFichier sélectionné : {path}")

    vertices, faces, obj_normals = load_obj(path)

    # STEP 2 — NETTOYAGE DES DONNÉES (suppression NaN / Inf)
    vertices = clean_point_cloud(vertices)

    #  STEP 3 — NORMALES + COMPARAISON + VISUALISATION
    # Priorité : maillage - OBJ - estimation ACP
    pcd = build_point_cloud_with_normals(vertices, faces, obj_normals)
    print_stats(vertices, faces, pcd)

    compare_normals(vertices, faces, obj_normals)

    visualize_points(pcd)     # sans normales
    visualize_normals(pcd)    # avec normales (fenêtre suivante)

    # STEP 4 — KD-TREE
    tree = build_kdtree(vertices)

    # STEP 5 — VOISINAGE K-NN
    knn = knn_neighbors(tree, vertices, k=30)
    print("[K-NN] shape:", knn.shape)   # (N, 30)


    # STEP 6 — VOISINAGES MULTI-ÉCHELLE
    spacing = estimate_mean_spacing(vertices)
    print("[SPACING] mean nearest-neighbor distance:", spacing)

    scales = build_scales_from_spacing(
        spacing,
        n_scales=12,
        factor_min=5,
        factor_max=15,
        mode="log"
    )
    print("[SCALES]", np.round(scales, 4))

    neighborhoods_dict = multi_scale_neighbors(vertices, scales)
    masks_dict         = compute_validity_mask(neighborhoods_dict, min_neighbors=6)

    neighborhoods = [neighborhoods_dict[t] for t in scales]
    masks         = [masks_dict[t]         for t in scales]

    print_scale_stats(neighborhoods_dict, scales, masks_dict)   

    # GLS et descripteur
    normals_np = np.asarray(pcd.normals)

    TAU   = np.full((len(vertices), len(scales)), np.nan)
    KAPPA = np.full((len(vertices), len(scales)), np.nan)
    PHI   = np.full((len(vertices), len(scales)), np.nan)
    ETA   = np.full((len(vertices), len(scales), 3), np.nan)

    for j, t in enumerate(scales):
        print(f"[GLS] échelle {j+1}/{len(scales)}  t={t:.4f}")
        for i, p in enumerate(vertices):
            if not masks[j][i]:
                continue
            idx    = neighborhoods[j][i]
            result = gls_at_point(p, vertices[idx], normals_np[idx], t)
            if result:
                TAU[i, j]      = result["tau"]
                KAPPA[i, j]    = result["kappa"]
                PHI[i, j]      = result["phi"]
                ETA[i, j]      = result["eta"]