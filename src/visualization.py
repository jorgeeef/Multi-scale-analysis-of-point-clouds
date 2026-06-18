# src/visualization.py
# =====================================================
# Visualtisation scalaire - coloration des points par τ
# =====================================================
# Pour chaque point p et chaque échelle t, mappe la valeur de τ
# vers une couleur RGB par interpolation linéaire :
#
#     β = (τ - τ_min) / (τ_max - τ_min)         ∈ [0, 1]
#     couleur = (β, 0, 1 - β)
#
#   τ_min  →  bleu  (0, 0, 1)
#   τ_max  →  rouge (1, 0, 0)
#
# τ_min et τ_max sont calculés globalement sur toutes les
# échelles pour garantir une comparaison visuelle cohérente
# entre les images.
import os
import copy
import numpy as np
import open3d as o3d


def _scalar_to_color(values, val_min, val_max):
    """
    Convertit un vecteur de scalaires en couleurs RGB linéaires.
    
        β = (val - val_min) / (val_max - val_min)
        couleur = (β, 0, 1 - β)
    
    val_min → bleu (0, 0, 1)
    val_max → rouge (1, 0, 0)
    NaN → gris (0.5, 0.5, 0.5)
    
    Retourne np.ndarray (N, 3) avec valeurs ∈ [0, 1].
    """
    n      = len(values)
    colors = np.zeros((n, 3))
    delta  = val_max - val_min

    if delta < 1e-12:
        colors[:] = [0.5, 0.5, 0.5]
        return colors

    betta            = (values - val_min) / delta
    colors[:, 0]     = betta            # R = β
    colors[:, 1]     = 0.0              # G = 0
    colors[:, 2]     = 1.0 - betta      # B = 1 - β

    nan_mask         = np.isnan(values)
    colors[nan_mask] = [0.5, 0.5, 0.5]

    return np.clip(colors, 0.0, 1.0)

def _render_colormap_loop(pcd, data_per_scale, scales, folder, file_prefix,
                          val_min, val_max, descriptor_label,
                          width=1024, height=768):
    """
    Helper interne — boucle de capture PNG commune à τ, η, κ.
    
    Paramètres
    ----------
    pcd              : o3d.geometry.PointCloud
    data_per_scale   : np.ndarray (N, S) — valeurs scalaires à chaque échelle
    scales           : np.ndarray (S,)
    folder           : str  — dossier de sortie déjà créé
    file_prefix      : str  — préfixe du nom de fichier (ex: "Format3_tau", "Format3_eta")
    val_min, val_max : bornes globales pour la colormap
    descriptor_label : str  — pour les logs (ex: "TAU", "ETA", "KAPPA")
    """
    original_colors = (np.asarray(pcd.colors).copy()
                       if len(pcd.colors) > 0 else None)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size       = 3.0
    vis.reset_view_point(True)

    n_scales = len(scales)

    for j, t in enumerate(scales):
        colors = _scalar_to_color(data_per_scale[:, j], val_min, val_max)

        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        png_path = os.path.join(folder, f"{file_prefix}_s{j+1:02d}_t{t:.4f}.png")
        vis.capture_screen_image(png_path, do_render=True)

        print(f"  s{j+1:02d}  t={t:.4f}  →  {os.path.basename(png_path)}")

    vis.destroy_window()

    if original_colors is not None and len(original_colors) > 0:
        pcd.colors = o3d.utility.Vector3dVector(original_colors)
    else:
        pcd.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))

    print(f"[{descriptor_label}-COLOR] {n_scales} images sauvegardées dans {folder}\n")

def save_tau_colormap_all_scales(pcd, TAU, scales, obj_name,
                                 output_dir="results",
                                 width=1024, height=768):
    """
    Génère une image PNG du nuage coloré par |τ| pour chaque échelle.
    Sauvegarde dans : results/<obj_name>/tau/
    """
    folder = os.path.join(output_dir, obj_name, "tau")
    os.makedirs(folder, exist_ok=True)

    # Bornes sur |τ| (signe ignoré)
    abs_TAU     = np.abs(TAU)
    abs_tau_min = float(np.nanmin(abs_TAU))
    abs_tau_max = float(np.nanmax(abs_TAU))

    tau_min_signed = float(np.nanmin(TAU))
    tau_max_signed = float(np.nanmax(TAU))

    print(f"\n[TAU-COLOR] Génération des images |τ| pour {len(scales)} échelles")
    print(f"[TAU-COLOR] τ signé     : min = {tau_min_signed:+.6f}   max = {tau_max_signed:+.6f}")
    print(f"[TAU-COLOR] |τ| utilisé : min = {abs_tau_min:.6f}        max = {abs_tau_max:.6f}")
    print(f"[TAU-COLOR] Gradient    : bleu (|τ| min)  →  rouge (|τ| max)")
    print(f"[TAU-COLOR] Dossier de sortie : {folder}")

    _render_colormap_loop(
        pcd              = pcd,
        data_per_scale   = abs_TAU,                   # ← on passe |τ|
        scales           = scales,
        folder           = folder,
        file_prefix      = f"{obj_name}_tau",
        val_min          = abs_tau_min,
        val_max          = abs_tau_max,
        descriptor_label = "TAU",
        width            = width,
        height           = height,
    )


def save_eta_colormap_all_scales(pcd, ETA_angle, scales, obj_name,
                                 output_dir="results",
                                 width=1024, height=768):
    """
    Génère une image PNG du nuage coloré par η (angle en degrés entre
    la normale GLS et la normale de surface) pour chaque échelle.
    
    Gradient :
        0°   → bleu  (normales parfaitement alignées)
        180° → rouge (normales opposées)
    
    Sauvegarde dans : results/<obj_name>/eta/
    """
    folder = os.path.join(output_dir, obj_name, "eta")
    os.makedirs(folder, exist_ok=True)

    # Bornes globales sur l'angle (toujours ∈ [0, 180°])
    eta_min = float(np.nanmin(ETA_angle))
    eta_max = float(np.nanmax(ETA_angle))

    print(f"\n[ETA-COLOR] Génération des images η pour {len(scales)} échelles")
    print(f"[ETA-COLOR] η min global = {eta_min:.4f}°  →  bleu  (alignées)")
    print(f"[ETA-COLOR] η max global = {eta_max:.4f}°  →  rouge (opposées)")
    print(f"[ETA-COLOR] Dossier de sortie : {folder}")

    _render_colormap_loop(
        pcd              = pcd,
        data_per_scale   = ETA_angle,
        scales           = scales,
        folder           = folder,
        file_prefix      = f"{obj_name}_eta",
        val_min          = eta_min,
        val_max          = eta_max,
        descriptor_label = "ETA",
        width            = width,
        height           = height,
    )



def save_kappa_colormap_all_scales(pcd, KAPPA, scales, obj_name,
                                   output_dir="results",
                                   width=1024, height=768):
    """
    Génère une image PNG du nuage coloré par |κ| (intensité de courbure
    signée) pour chaque échelle.
    
    Le signe de κ est ignoré : seule l'intensité de la courbure compte.
        |κ| faible → bleu  (plan local)
        |κ| élevé  → rouge (forte courbure, convexe OU concave)
    
    Sauvegarde dans : results/<obj_name>/kappa/
    """
    folder = os.path.join(output_dir, obj_name, "kappa")
    os.makedirs(folder, exist_ok=True)

    # Bornes sur |κ|
    abs_KAPPA     = np.abs(KAPPA)
    abs_kappa_min = float(np.nanmin(abs_KAPPA))
    abs_kappa_max = float(np.nanmax(abs_KAPPA))

    kappa_min_signed = float(np.nanmin(KAPPA))
    kappa_max_signed = float(np.nanmax(KAPPA))

    print(f"\n[KAPPA-COLOR] Génération des images |κ| pour {len(scales)} échelles")
    print(f"[KAPPA-COLOR] κ signé     : min = {kappa_min_signed:+.6f}   max = {kappa_max_signed:+.6f}")
    print(f"[KAPPA-COLOR] |κ| utilisé : min = {abs_kappa_min:.6f}        max = {abs_kappa_max:.6f}")
    print(f"[KAPPA-COLOR] Gradient    : bleu (plan, |κ|≈0)  →  rouge (forte courbure)")
    print(f"[KAPPA-COLOR] Dossier de sortie : {folder}")

    _render_colormap_loop(
        pcd              = pcd,
        data_per_scale   = abs_KAPPA,
        scales           = scales,
        folder           = folder,
        file_prefix      = f"{obj_name}_kappa",
        val_min          = abs_kappa_min,
        val_max          = abs_kappa_max,
        descriptor_label = "KAPPA",
        width            = width,
        height           = height,
    )

def show_tau_colormap_interactive(pcd, TAU, scales, scale_index=0):
    """
    Ouvre une fenêtre Open3D interactive avec coloration |τ| pour
    une échelle donnée.
    """
    abs_TAU     = np.abs(TAU)
    abs_tau_min = float(np.nanmin(abs_TAU))
    abs_tau_max = float(np.nanmax(abs_TAU))

    colors             = _scalar_to_color(np.abs(TAU[:, scale_index]),
                                          abs_tau_min, abs_tau_max)
    pcd_colored        = copy.deepcopy(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    title = (f"|τ| — échelle {scale_index+1}/{len(scales)} "
             f"(t={scales[scale_index]:.4f})")

    o3d.visualization.draw_geometries(
        [pcd_colored],
        window_name=title,
    )

def show_eta_colormap_interactive(pcd, ETA_angle, scales, scale_index=0):
    """
    Ouvre une fenêtre Open3D interactive avec coloration η pour
    une échelle donnée.

      0°   → bleu  (normale GLS alignée avec n_surface)
      180° → rouge (normale opposée)
    """
    eta_min = float(np.nanmin(ETA_angle))
    eta_max = float(np.nanmax(ETA_angle))

    colors             = _scalar_to_color(ETA_angle[:, scale_index],
                                          eta_min, eta_max)
    pcd_colored        = copy.deepcopy(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    title = (f"η — échelle {scale_index+1}/{len(scales)} "
             f"(t={scales[scale_index]:.4f})")

    o3d.visualization.draw_geometries(
        [pcd_colored],
        window_name=title,
    )


def show_kappa_colormap_interactive(pcd, KAPPA, scales, scale_index=0):
    """
    Ouvre une fenêtre Open3D interactive avec coloration |κ| pour
    une échelle donnée.

      |κ| faible → bleu  (zones plates)
      |κ| élevé  → rouge (forte courbure, convexe ou concave)
    """
    abs_KAPPA     = np.abs(KAPPA)
    abs_kappa_min = float(np.nanmin(abs_KAPPA))
    abs_kappa_max = float(np.nanmax(abs_KAPPA))

    colors             = _scalar_to_color(np.abs(KAPPA[:, scale_index]),
                                          abs_kappa_min, abs_kappa_max)
    pcd_colored        = copy.deepcopy(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    title = (f"|κ| — échelle {scale_index+1}/{len(scales)} "
             f"(t={scales[scale_index]:.4f})")

    o3d.visualization.draw_geometries(
        [pcd_colored],
        window_name=title,
    )