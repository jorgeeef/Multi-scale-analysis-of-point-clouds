# src/visualization.py
# =========================================================
# VISUALISATION SCALAIRE — coloration des points par τ
# =========================================================
# Pour chaque point p et chaque échelle t, mappe la valeur de τ
# vers une couleur RGB par interpolation linéaire :
#
#     β = (τ - τ_min) / (τ_max - τ_min)         ∈ [0, 1]
#     couleur = (β, 0, 1 - β)
#
#   τ_min  →  bleu  (0, 0, 1)
#   τ_max  →  rouge (1, 0, 0)
#
# τ_min et τ_max sont calculés GLOBALEMENT sur toutes les
# échelles pour garantir une comparaison visuelle cohérente
# entre les images.
#
# Points NaN (échec du fit) → gris (0.5, 0.5, 0.5)
# =========================================================

import os
import copy
import numpy as np
import open3d as o3d


def _tau_to_color(tau_values, tau_min, tau_max):
    """
    Convertit un vecteur de τ en couleurs RGB linéaires.
    NaN → gris.

    Retourne np.ndarray shape (N, 3) avec valeurs ∈ [0, 1].
    """
    n      = len(tau_values)
    colors = np.zeros((n, 3))
    delta  = tau_max - tau_min

    if delta < 1e-12:
        colors[:] = [0.5, 0.5, 0.5]
        return colors

    betta            = (tau_values - tau_min) / delta
    colors[:, 0]     = betta            # R = β
    colors[:, 1]     = 0.0              # G = 0
    colors[:, 2]     = 1.0 - betta      # B = 1 - β

    # NaN → gris neutre
    nan_mask         = np.isnan(tau_values)
    colors[nan_mask] = [0.5, 0.5, 0.5]

    return np.clip(colors, 0.0, 1.0)


def save_tau_colormap_all_scales(pcd, TAU, scales, obj_name,
                                 output_dir="notebooks",
                                 width=1024, height=768):
    """
    Génère une image PNG du nuage coloré par τ pour CHAQUE échelle.

    τ_min et τ_max sont calculés sur TOUTES les échelles → le gradient
    est identique entre les images.
    """
    folder = os.path.join(output_dir, obj_name)
    os.makedirs(folder, exist_ok=True)

    # Bornes globales
    tau_min = float(np.nanmin(TAU))
    tau_max = float(np.nanmax(TAU))

    print(f"\n[TAU-COLOR] Génération des images τ pour {len(scales)} échelles")
    print(f"[TAU-COLOR] τ_min global = {tau_min:+.6f}  →  bleu  (0, 0, 1)")
    print(f"[TAU-COLOR] τ_max global = {tau_max:+.6f}  →  rouge (1, 0, 0)")

    # Sauvegarder les couleurs originales pour restaurer plus tard
    original_colors = (np.asarray(pcd.colors).copy()
                       if len(pcd.colors) > 0 else None)

    n_scales = len(scales)

    # Créer UN SEUL visualiseur réutilisé pour toutes les échelles
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size       = 3.0

    # Cadrage initial sur le nuage complet
    vis.reset_view_point(True)

    for j, t in enumerate(scales):

        # 1. Calculer les couleurs pour cette échelle
        colors = _tau_to_color(TAU[:, j], tau_min, tau_max)

        # 2. Appliquer DIRECTEMENT sur le pcd original
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 3. Notifier Open3D que la géométrie a changé
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # 4. Capturer
        png_path = os.path.join(
            folder,
            f"{obj_name}_tau_s{j+1:02d}_t{t:.4f}.png"
        )
        vis.capture_screen_image(png_path, do_render=True)

        print(f"  s{j+1:02d}  t={t:.4f}  →  {os.path.basename(png_path)}")

    vis.destroy_window()

    # Restaurer les couleurs originales du PCD (par propreté)
    if original_colors is not None and len(original_colors) > 0:
        pcd.colors = o3d.utility.Vector3dVector(original_colors)
    else:
        pcd.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))

    print(f"[TAU-COLOR] {n_scales} images sauvegardées dans {folder}\n")


def show_tau_colormap_interactive(pcd, TAU, scales, scale_index=0):
    """
    Ouvre une fenêtre Open3D INTERACTIVE avec coloration τ pour
    une échelle donnée. L'utilisateur peut tourner, zoomer, etc.

    Touches :
      souris    : rotation
      molette   : zoom
      P         : capture d'écran
      Q / Échap : quitter
    """
    tau_min = float(np.nanmin(TAU))
    tau_max = float(np.nanmax(TAU))

    colors             = _tau_to_color(TAU[:, scale_index], tau_min, tau_max)
    pcd_colored        = copy.deepcopy(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    title = (f"τ — échelle {scale_index+1}/{len(scales)} "
             f"(t={scales[scale_index]:.4f})")

    o3d.visualization.draw_geometries(
        [pcd_colored],
        window_name=title,
    )