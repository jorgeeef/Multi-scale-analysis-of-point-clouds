
# main.py
# =================================================================
# PIPELINE PRINCIPAL — ANALYSE MULTI-ÉCHELLE 
#
# Ce script orchestre le pipeline complet de l'analyse de 
# Mellado sur un nuage de points 3D chargé depuis un fichier .obj.
#
#   Étapes :
#   Étape 1 — Chargement du modèle 3D
#   Étape 2 — Préparation géométrique (normales, KD-tree, échelles)
# =================================================================

import os
import sys
sys.path.append("src")    # accès aux modules locaux

import numpy as np
import open3d as o3d


from obj_reader import load_obj, create_point_cloud, create_point_cloud_with_normals, visualize, print_stats
from geometry import clean_point_cloud, estimate_normals, build_kdtree, knn_neighbors, multi_scale_neighbors, estimate_mean_spacing, build_scales_from_spacing, compute_validity_mask
from gls import fit_algebraic_sphere, extract_descriptors, gaussian_weights, compute_fitness
from visualiser_spheres import visualize_sphere_at_point, visualize_sphere_patch




if __name__ == "__main__":
    

    #Étape 1 — chargement du modèle 3D 
    path = os.path.join("data", "12140_Skull_v3.obj") # le nom du fichier obj
    vertices, faces, obj_normals = load_obj(path)
    #print(f"[INFO] OBJ normals found: {len(obj_normals)}")

    # STEP 2 — GEOMETRY PREP
    vertices = clean_point_cloud(vertices) #nettoyage
    pcd = create_point_cloud(vertices) 
    if len(obj_normals) != 0:
        print("[VISU] Normales du fichier OBJ")
        pcd_obj = create_point_cloud_with_normals(
            vertices,
            obj_normals
        )

        o3d.visualization.draw_geometries(
            [pcd_obj],
            point_show_normal=True
        )
    else:
        print("[VISU] Normales estimated")
    
    pcd = estimate_normals(pcd)

    print_stats(vertices, faces, pcd)

    # KD TREE 
    tree = build_kdtree(vertices)

    # TEST k-NN 
    knn = knn_neighbors(tree, vertices, k=30)
    print("[INFO] k-NN shape:", knn.shape)




    spacing = estimate_mean_spacing(vertices)
    print("[INFO] mean spacing =", spacing)

    scales = build_scales_from_spacing(spacing, n_scales=12, factor_min=5, factor_max=15, mode="log")
    scales = scales.tolist()
    neighborhoods = multi_scale_neighbors(vertices, scales)

    masks = compute_validity_mask(neighborhoods, min_neighbors=6)


    # Vérification du pourcentage
    for t in scales:
        valid = np.sum(masks[t])
        print(f"[INFO] t={t:.3f} — points valides: {valid}/{len(vertices)} ({100*valid/len(vertices):.1f}%)")

    # Vérification à la plus petite échelle
    t_min = scales[0]
    sizes_min = [len(n) for n in neighborhoods[t_min]]
    print(f"[CHECK] t={t_min:.3f} — min voisins: {min(sizes_min)}, points < 6 voisins: {sum(1 for s in sizes_min if s < 6)}")



    for t in scales:
        sizes = [len(n) for n in neighborhoods[t]]
        print(f"[INFO] avg neighbors for t={t}: {np.mean(sizes):.2f}")




    # =========================
    # STEP 3 — GLS FITTING
    # =========================
    normals_np = np.asarray(pcd.normals)   # (N, 3) normales estimées Open3D

    n_points = len(vertices)
    n_scales = len(scales)

    # Tableaux de descripteurs — NaN par défaut
    TAU   = np.full((n_points, n_scales), np.nan)
    ETA   = np.full((n_points, n_scales, 3), np.nan)
    KAPPA = np.full((n_points, n_scales), np.nan)
    DELTA = np.full((n_points, n_scales), np.nan)

    for j, t in enumerate(scales):
        print(f"[GLS] fitting scale t={t:.3f} ({j+1}/{n_scales}) ...")
        for i, p in enumerate(vertices):

            if not masks[t][i]:
                continue  # pas assez de voisins → NaN conservé

            idx  = neighborhoods[t][i]
            npts = vertices[idx]          # positions des voisins
            nrms = normals_np[idx]        # normales des voisins

            u = fit_algebraic_sphere(p, npts, t, normal_p=normals_np[i])

            if u is None:
                continue  # fitting échoué → NaN conservé

            TAU[i, j], ETA[i, j], KAPPA[i, j] = extract_descriptors(u, p)

            # fitness delta
            W = gaussian_weights(npts, p, t)
            DELTA[i, j] = compute_fitness(u, npts, p, nrms, W) 

    print("[GLS] Fitting terminé.")
    print(f"[INFO] KAPPA valides : {np.sum(~np.isnan(KAPPA[:, 0]))}/{n_points} à t_min")
    print(f"[INFO] KAPPA range   : [{np.nanmin(KAPPA):.4f}, {np.nanmax(KAPPA):.4f}]")
    print(f"[INFO] DELTA moyen   : {np.nanmean(DELTA):.4f}")

    # Dans main.py, après la boucle GLS, ajoute :
    print("\n[DIAG] DELTA par échelle :")
    for j, t in enumerate(scales):
        d = np.nanmean(DELTA[:, j])
        print(f"  t={t:.3f} → DELTA moyen = {d:.4f}")

    print("\n[DIAG] Distribution DELTA (toutes échelles) :")
    flat = DELTA[~np.isnan(DELTA)]
    print(f"  < 0.3  : {np.sum(flat < 0.3)} points")
    print(f"  0.3-0.6: {np.sum((flat >= 0.3) & (flat < 0.6))} points")
    print(f"  0.6-0.8: {np.sum((flat >= 0.6) & (flat < 0.8))} points")
    print(f"  > 0.8  : {np.sum(flat >= 0.8)} points")


    bad_mask  = DELTA[:, 0] < 0.3
    good_mask = DELTA[:, 0] > 0.8
    print(f"[DIAG] à t_min : {np.sum(bad_mask)} mauvais, {np.sum(good_mask)} bons")




    # Choisir un point avec bon DELTA (vert dans la visualisation précédente)
    good_indices = np.where(good_mask)[0]
    print(f"[VISU] Indices de bons points disponibles : {good_indices[:5]}")

    # Visualiser le même point à 3 échelles différentes
    for scale_idx in [0, 5, 11]:   # petite, moyenne, grande échelle
        visualize_sphere_at_point(
            vertices, neighborhoods, scales,
            TAU, KAPPA, ETA, normals_np,
            point_idx=good_indices[0],   # premier bon point
            scale_idx=scale_idx
        )


    # Trouve le point avec le meilleur DELTA à t_min
    best_point = np.nanargmax(DELTA[:, 0])
    print(f"[VISU] Meilleur point DELTA : #{best_point}, δ={DELTA[best_point, 0]:.4f}")

    for scale_idx in [0, 5, 11]:
        visualize_sphere_patch(
            vertices, neighborhoods, scales,
            TAU, KAPPA, ETA,
            point_idx=best_point,
            scale_idx=scale_idx
        )







    pcd_bad  = o3d.geometry.PointCloud()
    pcd_good = o3d.geometry.PointCloud()
    pcd_bad.points  = o3d.utility.Vector3dVector(vertices[bad_mask])
    pcd_good.points = o3d.utility.Vector3dVector(vertices[good_mask])
    pcd_bad.paint_uniform_color([1, 0, 0])
    pcd_good.paint_uniform_color([0, 0.8, 0])
    o3d.visualization.draw_geometries([pcd_bad, pcd_good])



    # Visualisation
    o3d.visualization.draw_geometries(
    [pcd],
    point_show_normal=True
    )  #avec normale

    visualize(pcd) #sans normale