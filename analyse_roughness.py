# analyse_roughness.py
# =========================================================
# Détection des points les plus RUGUEUX d'un nuage de points,
# d'après les descripteurs GLS (Mellado et al. 2012) :
#
#   tau (τ)   : écart algébrique au fit local — grand |τ| = point
#               qui s'écarte de la surface lisse ajustée
#   kappa (κ) : courbure signée — grand |κ| = forte variation de forme
#   phi (ϕ)   : fitness du fit ∈ [0,1] — ϕ bas = normales mal
#               expliquées par un ajustement lisse (indice de bruit)
#   nu (ν)    : variation géométrique (dérivée analytique en échelle)
#               — grand ν = descripteur instable à cette échelle,
#               donc structure locale non stable (bruit / rugosité)
#
# Contrairement à main.py, ce script ne balaye PAS toutes les
# échelles : tau/kappa/phi/nu sont tous calculés en UNE seule passe,
# à une échelle de référence t_ref (fine), car nu(p,t) est déjà une
# dérivée analytique en t (cf. src/gls.py, Section 4.2 Eq. 5) — un
# seul t suffit donc pour capturer la rugosité locale.
#
# Score composite (z-scores, cf. rapport texte pour le détail) :
#     score(p) = z(|tau|) + z(|kappa|) + z(1 - phi) + z(nu)
#
# Les N_TOP points de score maximal sont coloriés en ROUGE et
# marqués par de petites sphères pour rester visibles malgré leur
# faible nombre. Script autonome : aucun cache requis, séparé de
# main.py / analyse_tau_v.py.
# =========================================================

import os
import sys
sys.path.append("src")

import numpy as np
import open3d as o3d

from obj_reader import load_obj, build_point_cloud_with_normals
from geometry import clean_point_cloud, build_kdtree, estimate_mean_spacing
from gls import gls_at_point


N_TOP        = 20
DATA_FOLDER  = "data"
OUTPUT_DIR   = "results"
SCALE_FACTOR = 8.0     # t_ref = SCALE_FACTOR * espacement moyen (échelle fine)
CHUNK        = 5000    # taille de bloc pour limiter le pic mémoire


def select_obj_name(data_folder=DATA_FOLDER):
    """Liste les fichiers .obj de data/ et laisse l'utilisateur choisir."""
    obj_files = sorted(f for f in os.listdir(data_folder)
                       if f.lower().endswith(".obj"))

    if not obj_files:
        raise FileNotFoundError(
            f"Aucun fichier .obj trouve dans le dossier '{data_folder}'."
        )

    print("\nFichiers OBJ disponibles :")
    for i, fname in enumerate(obj_files, start=1):
        default_tag = "  (defaut)" if fname == "rough_smooth_test.obj" else ""
        print(f"  {i}. {fname}{default_tag}")

    raw = input(f"\nChoisissez un fichier (numero, defaut = "
                f"rough_smooth_test.obj si present) : ").strip()

    if raw == "":
        if "rough_smooth_test.obj" in obj_files:
            return "rough_smooth_test"
        raw = "1"

    choice = int(raw)
    if not (1 <= choice <= len(obj_files)):
        raise ValueError(f"Numero invalide (1 - {len(obj_files)}).")

    return os.path.splitext(obj_files[choice - 1])[0]


def compute_descriptors(vertices, normals_np, tree, t_ref):
    """
    Calcule tau, kappa, phi, nu pour chaque point, a l'echelle unique
    t_ref, par blocs (pour limiter la memoire sur les gros nuages).
    Les points au fit invalide (trop peu de voisins, sphere degeneree)
    recoivent NaN. N_NEIGH retient la taille du voisinage utilise,
    necessaire pour detecter les points de bord (cf. roughness_score).
    """
    N = len(vertices)
    TAU     = np.full(N, np.nan)
    KAPPA   = np.full(N, np.nan)
    PHI     = np.full(N, np.nan)
    NU      = np.full(N, np.nan)
    N_NEIGH = np.zeros(N, dtype=np.int64)

    n_valid = 0
    for start in range(0, N, CHUNK):
        end            = min(start + CHUNK, N)
        neighbor_lists = tree.query_ball_point(vertices[start:end], r=t_ref, workers=-1)

        for local_i, idx in enumerate(neighbor_lists):
            i             = start + local_i
            N_NEIGH[i]    = len(idx)
            if len(idx) < 6:
                continue
            idx    = np.asarray(idx, dtype=np.int64)
            result = gls_at_point(vertices[i], vertices[idx], normals_np[idx],
                                  t_ref, with_variation=True)
            if result:
                n_valid  += 1
                TAU[i]   = result["tau"]
                KAPPA[i] = result["kappa"]
                PHI[i]   = result["phi"]
                NU[i]    = result["nu"]

        print(f"  [GLS] {end}/{N} points traites ({n_valid} valides)")

    return TAU, KAPPA, PHI, NU, N_NEIGH


def zscore(x, valid_mask):
    """Normalisation z-score, calculee uniquement sur les points valides."""
    mu    = np.nanmean(x[valid_mask])
    sigma = np.nanstd(x[valid_mask])
    if sigma < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sigma


def roughness_score(TAU, KAPPA, PHI, NU, N_NEIGH):
    """
    Combine les 4 descripteurs GLS en un unique score de rugosite,
    par somme de z-scores (chaque terme contribue a poids egal) :

        score = z(|tau|) + z(|kappa|) + z(1 - phi) + z(nu)

    Un score eleve signifie : point qui s'ecarte du fit local (tau),
    forte courbure (kappa), fit mal aligne avec les normales (1-phi),
    et descripteur instable a cette echelle (nu) — la conjonction de
    ces 4 indices caracterise une rugosite locale plutot qu'une simple
    courbure geometrique lisse (ex: une arete franche aurait kappa
    eleve mais phi et nu resteraient bons si elle est bien definie).

    Exclusion des points de bord : un voisinage tronque par un bord
    ouvert du maillage (ou de la zone scannee) degrade artificiellement
    tau/phi/nu exactement comme le ferait du bruit, sans rapport avec
    une vraie rugosite de surface. On exclut donc les points dont le
    voisinage est nettement plus petit que la mediane (< 60%) — a la
    fois du calcul des z-scores (pour ne pas fausser mu/sigma) et de
    la selection finale du top N.
    """
    basic_mask = ~(np.isnan(TAU) | np.isnan(KAPPA) | np.isnan(PHI) | np.isnan(NU))

    median_neigh    = np.median(N_NEIGH[basic_mask])
    neighbor_thresh = max(6, 0.6 * median_neigh)
    valid_mask      = basic_mask & (N_NEIGH >= neighbor_thresh)

    score = np.full(TAU.shape, np.nan)
    score[valid_mask] = (
        zscore(np.abs(TAU),   valid_mask)[valid_mask] +
        zscore(np.abs(KAPPA), valid_mask)[valid_mask] +
        zscore(1.0 - PHI,     valid_mask)[valid_mask] +
        zscore(NU,            valid_mask)[valid_mask]
    )
    return score, valid_mask


def top_rough_points(score, valid_mask, n_top=N_TOP):
    idx_valid = np.where(valid_mask)[0]
    if len(idx_valid) < n_top:
        raise RuntimeError(
            f"Seulement {len(idx_valid)} points valides, "
            f"impossible d'en extraire {n_top}."
        )
    order = np.argsort(score[idx_valid])[::-1][:n_top]
    return idx_valid[order]


def build_colored_visualization(pcd, score, valid_mask, top_idx):
    """
    Construit le nuage colore (contexte en niveaux de gris d'apres le
    score continu, points hors-fit en gris neutre) + une petite sphere
    rouge par point du top N_TOP, pour rester visible malgre leur
    faible nombre au milieu de milliers de points.
    """
    N = len(np.asarray(pcd.points))
    colors = np.full((N, 3), 0.5)   # gris neutre par defaut (points invalides)

    s_min = float(np.nanmin(score[valid_mask]))
    s_max = float(np.nanmax(score[valid_mask]))
    delta = max(s_max - s_min, 1e-12)

    gray = 0.85 - 0.65 * ((score[valid_mask] - s_min) / delta)  # clair -> fonce
    colors[valid_mask] = np.stack([gray, gray, gray], axis=1)

    pcd_colored = o3d.geometry.PointCloud(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    points = np.asarray(pcd.points)
    extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    marker_radius = 0.01 * extent

    markers = []
    for i in top_idx:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(marker_radius, 1e-4))
        sphere.translate(points[i])
        sphere.paint_uniform_color([1.0, 0.0, 0.0])   # rouge
        sphere.compute_vertex_normals()
        markers.append(sphere)

    return pcd_colored, markers


def save_screenshot(pcd_colored, markers, obj_name, output_folder,
                    width=1280, height=960):
    path = os.path.join(output_folder, f"{obj_name}_top{N_TOP}_roughness.png")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd_colored)
    for m in markers:
        vis.add_geometry(m, reset_bounding_box=False)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size        = 3.0
    vis.reset_view_point(True)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path, do_render=True)
    vis.destroy_window()

    print(f"[ROUGHNESS] Capture sauvegardee : {path}")
    return path


def save_report(obj_name, output_folder, t_ref, top_idx, score, TAU, KAPPA, PHI, NU):
    path = os.path.join(output_folder, f"{obj_name}_top{N_TOP}_roughness_report.txt")

    header = (f"{'rang':>4} {'point':>8} {'score':>10} "
              f"{'tau':>12} {'kappa':>12} {'phi':>8} {'nu':>12}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Analyse de rugosite GLS -- {obj_name}\n")
        f.write(f"# echelle de reference t_ref = {t_ref:.6f}\n")
        f.write(f"# score = z(|tau|) + z(|kappa|) + z(1-phi) + z(nu)\n\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for rank, i in enumerate(top_idx, start=1):
            f.write(f"{rank:>4d} p{i:06d} {score[i]:>10.4f} "
                    f"{TAU[i]:>+12.6f} {KAPPA[i]:>+12.6f} "
                    f"{PHI[i]:>8.4f} {NU[i]:>12.6f}\n")

    print(f"[ROUGHNESS] Rapport texte sauvegarde : {path}")
    return path


def main():
    obj_name = select_obj_name()
    path     = os.path.join(DATA_FOLDER, obj_name + ".obj")
    print(f"\n[INFO] Objet selectionne : {path}")

    vertices, faces, obj_normals = load_obj(path)
    vertices = clean_point_cloud(vertices)

    pcd = build_point_cloud_with_normals(vertices, faces, obj_normals)
    normals_np = np.asarray(pcd.normals)

    tree    = build_kdtree(vertices)
    spacing = estimate_mean_spacing(vertices)
    t_ref   = SCALE_FACTOR * spacing
    print(f"[INFO] {len(vertices)} points, espacement moyen = {spacing:.6f}, "
          f"t_ref = {t_ref:.6f}")

    print("\n[GLS] Calcul de tau/kappa/phi/nu a l'echelle de reference...")
    TAU, KAPPA, PHI, NU, N_NEIGH = compute_descriptors(vertices, normals_np, tree, t_ref)

    score, valid_mask = roughness_score(TAU, KAPPA, PHI, NU, N_NEIGH)
    top_idx = top_rough_points(score, valid_mask, N_TOP)

    print(f"\n[ROUGHNESS] {N_TOP} points les plus rugueux (t_ref={t_ref:.4f}) :")
    for rank, i in enumerate(top_idx, start=1):
        print(f"  {rank:2d}. p{i:06d}  score={score[i]:+.4f}  "
              f"tau={TAU[i]:+.4f}  kappa={KAPPA[i]:+.4f}  "
              f"phi={PHI[i]:.4f}  nu={NU[i]:.4f}")

    output_folder = os.path.join(OUTPUT_DIR, obj_name, "roughness")
    os.makedirs(output_folder, exist_ok=True)

    pcd_colored, markers = build_colored_visualization(pcd, score, valid_mask, top_idx)
    save_screenshot(pcd_colored, markers, obj_name, output_folder)
    save_report(obj_name, output_folder, t_ref, top_idx, score, TAU, KAPPA, PHI, NU)

    ply_path = os.path.join(output_folder, f"{obj_name}_roughness_colored.ply")
    o3d.io.write_point_cloud(ply_path, pcd_colored, write_ascii=True)
    print(f"[ROUGHNESS] Nuage colore sauvegarde : {ply_path}")


if __name__ == "__main__":
    main()
