# src/notebooks.py
# =========================================================
# Gestion de la persistance des résultats GLS
# =========================================================
# Responsabilités :
#   - Vérifier si un obj a déjà été traité (cache)
#   - Créer le dossier notebooks/<obj_name>/
#   - Sauvegarder les 4 fichiers de résultats
#   - Relire les résultats depuis le cache
# =========================================================

import os
import numpy as np


# CHEMINS
NOTEBOOKS_ROOT = "notebooks"


def _notebook_dir(obj_name):
    """Retourne le chemin du dossier notebooks/<obj_name>/"""
    return os.path.join(NOTEBOOKS_ROOT, obj_name)


def _info_path(obj_name):
    return os.path.join(_notebook_dir(obj_name), f"{obj_name}_info.txt")


def _tau_path(obj_name):
    return os.path.join(_notebook_dir(obj_name), f"{obj_name}_tau.txt")


def _eta_path(obj_name):
    return os.path.join(_notebook_dir(obj_name), f"{obj_name}_eta.txt")


def _kappa_path(obj_name):
    return os.path.join(_notebook_dir(obj_name), f"{obj_name}_kappa.txt")



# VÉRIFICATION DU CACHE
def notebook_exists(obj_name):
    """
    Vérifie si le dossier notebooks/<obj_name>/ existe
    ET contient les 4 fichiers de résultats attendus.

    Retourne True uniquement si tout est présent et lisible.
    Un dossier vide ou incomplet est traité comme absent.
    """
    d = _notebook_dir(obj_name)
    if not os.path.isdir(d):
        return False

    required = [
        _info_path(obj_name),
        _tau_path(obj_name),
        _eta_path(obj_name),
        _kappa_path(obj_name),
    ]
    return all(os.path.isfile(p) for p in required)


# SAUVEGARDE — 4 fichiers
def save_info(obj_name, vertices, faces, pcd, spacing, scales, masks_dict, neighborhoods_dict):
    """
    Sauvegarde objnom_info.txt :
        - nom du fichier
        - nombre de points, normales, faces
        - espacement moyen
        - liste des échelles
        - statistiques de voisinage par échelle
    """
    path = _info_path(obj_name)
    n_points  = len(vertices)
    n_normals = len(pcd.normals)
    n_faces   = len(faces)

    with open(path, "w", encoding="utf-8") as f:

        f.write("=" * 52 + "\n")
        f.write(f"  OBJ  : {obj_name}\n")
        f.write("=" * 52 + "\n\n")

        f.write("[GÉOMÉTRIE]\n")
        f.write(f"  Points   : {n_points}\n")
        f.write(f"  Normales : {n_normals}\n")
        f.write(f"  Faces    : {n_faces}\n\n")

        f.write("[ANALYSE MULTI-ÉCHELLE]\n")
        f.write(f"  Espacement moyen    : {spacing:.6f}\n")
        f.write(f"  Nombre d'échelles   : {len(scales)}\n\n")

        f.write(f"  {'t':>10}  {'moy. voisins':>14}  {'valides':>12}\n")
        f.write("  " + "-" * 42 + "\n")

        for t in scales:
            neighbors  = neighborhoods_dict[t]                        # list of lists
            sizes_all  = np.array([len(n) for n in neighbors])        # nb voisins par point
            mean_nb    = float(np.mean(sizes_all))
            n_valid    = int(np.sum(masks_dict[t]))
            f.write(f"  t={t:8.4f}  {mean_nb:>12.1f}  {n_valid:>6}/{n_points}\n")

    print(f"[NOTEBOOK] Sauvegardé : {path}")


def save_tau(obj_name, scales, TAU):
    """
    Sauvegarde objnom_tau.txt :
        - nombre d'échelles
        - pour chaque échelle : valeur de t puis les N valeurs de τ (une par ligne)
    """
    path      = _tau_path(obj_name)
    n_scales  = len(scales)
    n_points  = TAU.shape[0]

    with open(path, "w", encoding="utf-8") as f:

        f.write(f"# TAU — offset algébrique (τ)\n")
        f.write(f"# Mellado et al. (2012) Eq. 4 : τ = s_û(p)\n")
        f.write(f"# τ > 0 : p à l'extérieur   τ = 0 : sur la surface   τ < 0 : à l'intérieur\n")
        f.write(f"#\n")
        f.write(f"n_scales  = {n_scales}\n")
        f.write(f"n_points  = {n_points}\n\n")

        for j, t in enumerate(scales):
            col        = TAU[:, j]
            valid_mask = ~np.isnan(col)
            n_valid    = int(np.sum(valid_mask))
            mean_val   = float(np.nanmean(col))
            std_val    = float(np.nanstd(col))

            f.write(f"[ECHELLE {j+1:02d}]\n")
            f.write(f"  t         = {t:.6f}\n")
            f.write(f"  n_valid   = {n_valid}/{n_points}\n")
            f.write(f"  mean(tau) = {mean_val:.6f}\n")
            f.write(f"  std(tau)  = {std_val:.6f}\n")
            f.write(f"  valeurs   :\n")

            for i, v in enumerate(col):
                if np.isnan(v):
                    f.write(f"    p{i:06d}  NaN\n")
                else:
                    f.write(f"    p{i:06d}  {v:+.6f}\n")
            f.write("\n")

    print(f"[NOTEBOOK] Sauvegardé : {path}")


def save_eta(obj_name, scales, ETA, normals_np):
    """
    Sauvegarde objnom_eta.txt :
        - nombre d'échelles
        - pour chaque échelle : valeur de t puis les N angles θ (en degrés)
          entre la normale GLS (η) et la normale de surface (n_i)

    η est de dimension 3 → on réduit à l'angle scalaire cos⁻¹(η · n_i)
    pour rendre le fichier lisible et interprétable directement.
    """
    path     = _eta_path(obj_name)
    n_scales = len(scales)
    n_points = ETA.shape[0]

    with open(path, "w", encoding="utf-8") as f:

        f.write(f"# ETA — angle entre η (normale GLS) et n_i (normale surface)\n")
        f.write(f"# Mellado et al. (2012) Eq. 4 : η = ∇s_û(p) / ||∇s_û(p)||\n")
        f.write(f"# Valeurs en DEGRÉS.  0° = aligné parfaitement  180° = opposé\n")
        f.write(f"#\n")
        f.write(f"n_scales  = {n_scales}\n")
        f.write(f"n_points  = {n_points}\n\n")

        for j, t in enumerate(scales):
            eta_j  = ETA[:, j, :]                     # (N, 3)
            valid  = ~np.isnan(eta_j[:, 0])
            n_valid = int(np.sum(valid))

            # Angle entre η et la normale de surface
            # cos θ = η · n_i  (les deux sont unitaires)
            angles = np.full(n_points, np.nan)
            if np.any(valid):
                cos_theta          = np.einsum("ij,ij->i",
                                               eta_j[valid],
                                               normals_np[valid])
                cos_theta          = np.clip(cos_theta, -1.0, 1.0)
                angles[valid]      = np.degrees(np.arccos(cos_theta))

            mean_angle = float(np.nanmean(angles))
            std_angle  = float(np.nanstd(angles))

            f.write(f"[ECHELLE {j+1:02d}]\n")
            f.write(f"  t             = {t:.6f}\n")
            f.write(f"  n_valid       = {n_valid}/{n_points}\n")
            f.write(f"  mean(angle°)  = {mean_angle:.4f}\n")
            f.write(f"  std(angle°)   = {std_angle:.4f}\n")
            f.write(f"  valeurs (degrés) :\n")

            for i, angle in enumerate(angles):
                if np.isnan(angle):
                    f.write(f"    p{i:06d}  NaN\n")
                else:
                    f.write(f"    p{i:06d}  {angle:8.4f}°\n")
            f.write("\n")

    print(f"[NOTEBOOK] Sauvegardé : {path}")


def save_kappa(obj_name, scales, KAPPA):
    """
    Sauvegarde objnom_kappa.txt :
        - nombre d'échelles
        - pour chaque échelle : valeur de t puis les N valeurs de κ

    κ > 0 → convexe   κ = 0 → plan   κ < 0 → concave
    """
    path     = _kappa_path(obj_name)
    n_scales = len(scales)
    n_points = KAPPA.shape[0]

    with open(path, "w", encoding="utf-8") as f:

        f.write(f"# KAPPA — courbure signée (κ)\n")
        f.write(f"# Mellado et al. (2012) Eq. 4 : κ = 2·ûq\n")
        f.write(f"# κ > 0 : convexe   κ = 0 : plan local   κ < 0 : concave\n")
        f.write(f"#\n")
        f.write(f"n_scales  = {n_scales}\n")
        f.write(f"n_points  = {n_points}\n\n")

        for j, t in enumerate(scales):
            col     = KAPPA[:, j]
            valid   = ~np.isnan(col)
            n_valid = int(np.sum(valid))

            mean_k  = float(np.nanmean(col))
            std_k   = float(np.nanstd(col))
            n_conv  = int(np.sum(col[valid] > 0))
            n_conc  = int(np.sum(col[valid] < 0))
            n_flat  = int(np.sum(np.abs(col[valid]) < 1e-4))

            f.write(f"[ECHELLE {j+1:02d}]\n")
            f.write(f"  t           = {t:.6f}\n")
            f.write(f"  n_valid     = {n_valid}/{n_points}\n")
            f.write(f"  mean(kappa) = {mean_k:+.6f}\n")
            f.write(f"  std(kappa)  = {std_k:.6f}\n")
            f.write(f"  convexe     : {n_conv}  ({100*n_conv/max(n_valid,1):.1f}%)\n")
            f.write(f"  concave     : {n_conc}  ({100*n_conc/max(n_valid,1):.1f}%)\n")
            f.write(f"  plan (~0)   : {n_flat}  ({100*n_flat/max(n_valid,1):.1f}%)\n")
            f.write(f"  valeurs :\n")

            for i, v in enumerate(col):
                if np.isnan(v):
                    f.write(f"    p{i:06d}  NaN\n")
                else:
                    f.write(f"    p{i:06d}  {v:+.6f}\n")
            f.write("\n")

    print(f"[NOTEBOOK] Sauvegardé : {path}")


# SAUVEGARDE COMPLÈTE — point d'entrée principal
def save_results(obj_name, vertices, faces, pcd,
                 spacing, scales, masks_dict, neighborhoods_dict, 
                 TAU, ETA, KAPPA, normals_np):
    """
    Crée le dossier notebooks/<obj_name>/ et sauvegarde les 4 fichiers.

    Appelé depuis main.py après la boucle GLS.

    Paramètres
    ----------
    obj_name    : str          — nom du fichier sans extension
    vertices    : np.ndarray   — (N, 3)
    faces       : list
    pcd         : o3d.PointCloud
    spacing     : float
    scales      : np.ndarray   — (S,)
    masks_dict  : dict
    TAU         : np.ndarray   — (N, S)
    ETA         : np.ndarray   — (N, S, 3)
    KAPPA       : np.ndarray   — (N, S)
    normals_np  : np.ndarray   — (N, 3)
    """
    d = _notebook_dir(obj_name)
    os.makedirs(d, exist_ok=True)
    print(f"\n[NOTEBOOK] Création du dossier : {d}")

    save_info(obj_name, vertices, faces, pcd, spacing, scales, masks_dict,neighborhoods_dict)
    save_tau(obj_name, scales, TAU)
    save_eta(obj_name, scales, ETA, normals_np)
    save_kappa(obj_name, scales, KAPPA)

    print(f"[NOTEBOOK] 4 fichiers sauvegardés dans {d}\n")



# LECTURE DU CACHE
def _parse_scale_block(lines, keyword):
    """
    Parse générique d'un bloc [ECHELLE XX] dans un fichier txt.
    Retourne (t, np.ndarray of values).
    """
    t      = None
    values = []

    for line in lines:
        line = line.strip()
        if line.startswith("t         =") or line.startswith("t             =") or line.startswith("t           ="):
            t = float(line.split("=")[1].strip())
        elif line.startswith("p") and len(line.split()) == 2:
            val_str = line.split()[1]
            if val_str == "NaN":
                values.append(np.nan)
            else:
                values.append(float(val_str.replace("°", "").replace("+", "")))

    return t, np.array(values)


def _read_descriptor_file(filepath):
    """
    Lit un fichier _tau / _kappa / _eta et retourne
    (scales, data) où data est shape (n_points, n_scales).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Récupérer n_points
    n_points = None
    for line in content.splitlines():
        if line.startswith("n_points"):
            n_points = int(line.split("=")[1].strip())
            break

    # Découper par blocs [ECHELLE XX]
    import re
    blocks = re.split(r"\[ECHELLE \d+\]", content)[1:]  # ignorer header

    scales_found = []
    cols         = []

    for block in blocks:
        lines = block.splitlines()
        t, values = _parse_scale_block(lines, "")
        if t is not None and len(values) == n_points:
            scales_found.append(t)
            cols.append(values)

    if not cols:
        return None, None

    data = np.column_stack(cols)   # (n_points, n_scales)
    return np.array(scales_found), data


def load_results(obj_name):
    """
    Lit les 4 fichiers de résultats depuis notebooks/<obj_name>/.

    Retourne
    --------
    dict avec clés :
        'scales'  : np.ndarray (S,)
        'TAU'     : np.ndarray (N, S)
        'KAPPA'   : np.ndarray (N, S)
        'ETA_angle': np.ndarray (N, S)  — angles en degrés η vs n_i
        'info'    : str                 — contenu brut de _info.txt
    """
    print(f"\n[NOTEBOOK] Lecture du cache : {_notebook_dir(obj_name)}")

    # Info
    with open(_info_path(obj_name), "r", encoding="utf-8") as f:
        info = f.read()
    print(info)

    # TAU
    scales_tau, TAU = _read_descriptor_file(_tau_path(obj_name))

    # KAPPA
    scales_kappa, KAPPA = _read_descriptor_file(_kappa_path(obj_name))

    # ETA (angles en degrés)
    scales_eta, ETA_angle = _read_descriptor_file(_eta_path(obj_name))

    print(f"[NOTEBOOK] TAU   chargé : {TAU.shape}")
    print(f"[NOTEBOOK] KAPPA chargé : {KAPPA.shape}")
    print(f"[NOTEBOOK] ETA   chargé : {ETA_angle.shape}")

    return {
        "scales"    : scales_tau,
        "TAU"       : TAU,
        "KAPPA"     : KAPPA,
        "ETA_angle" : ETA_angle,
        "info"      : info,
    }