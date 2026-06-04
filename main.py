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


if __name__ == "__main__":


    # STEP 1 — LECTURE DU MODÈLE .OBJ
    path = os.path.join("data", "12140_Skull_v3.obj")
    vertices, faces, obj_normals = load_obj(path)

    # STEP 2 — NETTOYAGE DES DONNÉES (suppression NaN / Inf)
    vertices = clean_point_cloud(vertices)

    #  STEP 3 — NORMALES + COMPARAISON + VISUALISATION
    # Priorité : maillage > OBJ > estimation ACP
    pcd = build_point_cloud_with_normals(vertices, faces, obj_normals)
    print_stats(vertices, faces, pcd)

    compare_normals(vertices, faces, obj_normals)

    visualize_points(pcd)     # 1. sans normales
    visualize_normals(pcd)    # 2. avec normales (fenêtre suivante)

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