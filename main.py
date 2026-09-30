# main.py
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
    save_pointcloud_screenshot,
)
from geometry import (
    clean_point_cloud,
    build_kdtree,
    knn_neighbors,
    estimate_mean_spacing,
    build_scales_from_spacing,
)
from gls      import gls_at_point
from notebooks import notebook_exists, save_results, load_results
from visualization import (
    save_tau_colormap_all_scales,
    save_eta_colormap_all_scales,   
    save_kappa_colormap_all_scales,
    save_nu_colormap_all_scales,
    show_tau_colormap_interactive,
    show_eta_colormap_interactive,   
    show_kappa_colormap_interactive,
    show_nu_colormap_interactive,
)


if __name__ == "__main__":

    # Step 1 — Séléction du fichier obj
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
    #print(f"\n[INFO] Fichier sélectionné : {path}")


    # Step 2 — Chargement + nettoyage
    vertices, faces, obj_normals = load_obj(path)
    vertices = clean_point_cloud(vertices)


    # Step 3 — Normales + PCD
    pcd = build_point_cloud_with_normals(vertices, faces, obj_normals)
    print_stats(vertices, faces, pcd)


    # Step 4 — Sauvegardes pcd (à chaque exécution)
    save_pointcloud_screenshot(pcd, obj_name, show_normals=False)
    save_pointcloud_screenshot(pcd, obj_name, show_normals=True)


    # Step 5 — Visualization open3d  
    visualize_points(pcd)
    visualize_normals(pcd)


    # Step 6 — GLS  (lecture si disponible, sinon calcul)
    if notebook_exists(obj_name):

        #print(f"[NOTEBOOK] Cache trouvé pour '{obj_name}' → lecture.")
        results   = load_results(obj_name)
        scales    = results["scales"]
        TAU       = results["TAU"]
        KAPPA     = results["KAPPA"]
        PHI       = results["PHI"]
        NU        = results["NU"]
        ETA_angle = results["ETA_angle"]

        #print(f"       scales : {np.round(scales, 4)}")
        #print(f"       TAU    : {TAU.shape}")
        #print(f"       KAPPA  : {KAPPA.shape}")
        #print(f"       PHI    : {PHI.shape}")
        #print(f"       NU     : {NU.shape}")
        #print(f"       ETA°   : {ETA_angle.shape}")

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
            n_scales   = 15, #12
            factor_min = 5,
            factor_max = 50, #15
            mode       = "log",
        )
        print(f"[SCALES] {np.round(scales, 4)}")

        # Fitting GLS
        normals_np = np.asarray(pcd.normals)
        N          = len(vertices)
        S          = len(scales)
        CHUNK      = 5000   

        TAU   = np.full((N, S),    np.nan)
        KAPPA = np.full((N, S),    np.nan)
        PHI   = np.full((N, S),    np.nan)
        NU    = np.full((N, S),    np.nan)
        ETA   = np.full((N, S, 3), np.nan)

        mean_nb_per_scale = np.zeros(S)
        n_valid_per_scale = np.zeros(S, dtype=int)

        for j, t in enumerate(scales):
            print(f"[GLS] échelle {j+1}/{S}  t={t:.4f}")

            n_valid  = 0
            sum_size = 0

            for start in range(0, N, CHUNK):
                end            = min(start + CHUNK, N)
                neighbor_lists = tree.query_ball_point(vertices[start:end],
                                                         r=t, workers=-1)

                for local_i, idx in enumerate(neighbor_lists):
                    sum_size += len(idx)
                    if len(idx) < 6:
                        continue
                    i      = start + local_i
                    idx    = np.asarray(idx, dtype=np.int64)
                    result = gls_at_point(vertices[i], vertices[idx], normals_np[idx],
                                          t, with_variation=True)
                    if result:
                        n_valid    += 1
                        TAU[i, j]   = result["tau"]
                        KAPPA[i, j] = result["kappa"]
                        PHI[i, j]   = result["phi"]
                        NU[i, j]    = result["nu"]
                        ETA[i, j]   = result["eta"]

                del neighbor_lists

            mean_nb_per_scale[j] = sum_size / N
            n_valid_per_scale[j] = n_valid
            print(f"       voisins moyens : {mean_nb_per_scale[j]:>8.1f}"
                  f"   valides : {n_valid}/{N}")

        print("[GLS] Calcul terminé.")


        ETA_angle = np.full((N, S), np.nan)
        valid_mask = ~np.isnan(ETA[:, :, 0])
        cos_theta  = np.einsum("ijk,ik->ij", ETA, normals_np)
        cos_theta  = np.clip(cos_theta, -1.0, 1.0)
        ETA_angle[valid_mask] = np.degrees(np.arccos(cos_theta[valid_mask]))


        # Sauvegarde dans notebooks/
        save_results(
            obj_name           = obj_name,
            vertices           = vertices,
            faces              = faces,
            pcd                = pcd,
            spacing            = spacing,
            scales             = scales,
            mean_nb_per_scale  = mean_nb_per_scale,
            n_valid_per_scale  = n_valid_per_scale,
            TAU                = TAU,
            ETA                = ETA,
            KAPPA              = KAPPA,
            PHI                = PHI,
            NU                 = NU,
            normals_np         = normals_np,
        )


    # Coloration multi-descripteurs (τ, η, κ, ν)
    save_tau_colormap_all_scales(pcd, TAU, scales, obj_name)
    save_eta_colormap_all_scales(pcd, ETA_angle, scales, obj_name)
    save_kappa_colormap_all_scales(pcd, KAPPA, scales, obj_name)
    save_nu_colormap_all_scales(pcd, NU, scales, obj_name)

    #for k in range(len(scales)):
    #    show_tau_colormap_interactive(pcd,   TAU,       scales, scale_index=k)



# Manual visualization of |τ|

while True:

    print("\n" + "=" * 50)
    print("VISUALISATION DE |τ|")
    print("=" * 50)

    # Choix de la normalisation
    while True:
        print("\nChoisissez la normalisation :")
        print("  1. Min / max global sur toutes les échelles")
        print("  2. Min / max de l'échelle choisie")

        try:
            norm_choice = int(input("\nVotre choix (1-2) : "))

            if norm_choice in (1, 2):
                break

            print("  Choix invalide.")

        except ValueError:
            print("  Veuillez entrer un nombre entier.")

    normalization = "global" if norm_choice == 1 else "scale"

    # Choix de l'échelle
    while True:
        print("\nÉchelles disponibles :")

        for i, t in enumerate(scales, start=1):
            print(f"  {i}. t = {t:.4f}")

        try:
            scale_choice = int(
                input(
                    f"\nChoisissez une échelle (1-{len(scales)}) : "
                )
            )

            if 1 <= scale_choice <= len(scales):
                break

            print(f"  Numéro invalide (1-{len(scales)}).")

        except ValueError:
            print("  Veuillez entrer un nombre entier.")

    scale_index = scale_choice - 1

    # Visualisation
    show_tau_colormap_interactive(
        pcd,
        TAU,
        scales,
        scale_index=scale_index,
        normalization=normalization
    )

    # Demander si l'utilisateur veut continuer
    while True:
        again = input(
            "\nVoulez-vous visualiser une autre échelle ? (o/n) : "
        ).strip().lower()

        if again in ("o", "oui"):
            break

        if again in ("n", "non"):
            print("\nFin de la visualisation.")
            raise SystemExit

        print("  Veuillez répondre par 'o' ou 'n'.")