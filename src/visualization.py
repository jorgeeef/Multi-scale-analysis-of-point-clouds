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

def save_nu_colormap_all_scales(pcd, NU, scales, obj_name,
                                output_dir="results",
                                width=1024, height=768):
    """
    Génère une image PNG du nuage coloré par ν(p,t) (variation
    géométrique, Mellado et al. 2012 Eq. 5) pour chaque échelle.

    ν >= 0 toujours (pas de signe à ignorer, contrairement à τ/κ) :
        ν faible → bleu  (échelle pertinente, descripteur stable)
        ν élevé  → rouge (échelle instable / structure en train de changer)

    Sauvegarde dans : results/<obj_name>/nu/
    """
    folder = os.path.join(output_dir, obj_name, "nu")
    os.makedirs(folder, exist_ok=True)

    nu_min = float(np.nanmin(NU))
    nu_max = float(np.nanmax(NU))

    print(f"\n[NU-COLOR] Génération des images ν pour {len(scales)} échelles")
    print(f"[NU-COLOR] ν min global = {nu_min:.6f}  →  bleu  (échelle pertinente)")
    print(f"[NU-COLOR] ν max global = {nu_max:.6f}  →  rouge (échelle instable)")
    print(f"[NU-COLOR] Dossier de sortie : {folder}")

    _render_colormap_loop(
        pcd              = pcd,
        data_per_scale   = NU,
        scales           = scales,
        folder           = folder,
        file_prefix      = f"{obj_name}_nu",
        val_min          = nu_min,
        val_max          = nu_max,
        descriptor_label = "NU",
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


def show_nu_colormap_interactive(pcd, NU, scales, scale_index=0):
    """
    Ouvre une fenêtre Open3D interactive avec coloration ν(p,t) pour
    une échelle donnée.

      ν faible → bleu  (échelle pertinente, descripteur stable)
      ν élevé  → rouge (échelle instable / structure en train de changer)
    """
    nu_min = float(np.nanmin(NU))
    nu_max = float(np.nanmax(NU))

    colors             = _scalar_to_color(NU[:, scale_index], nu_min, nu_max)
    pcd_colored        = copy.deepcopy(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    title = (f"ν — échelle {scale_index+1}/{len(scales)} "
             f"(t={scales[scale_index]:.4f})")

    o3d.visualization.draw_geometries(
        [pcd_colored],
        window_name=title,
    )


def _build_sphere_fit_geometries(p, neighbor_points, center, radius,
                                 context_points=None):
    """
    Construit les géométries Open3D communes aux rendus d'ajustement de
    sphère : voisinage P_t(p) (bleu), point p (marqueur rouge), sphère
    algébrique ajustée (fil de fer noir), et un contexte optionnel
    (gris clair) — à la manière des Figures 1 / 3(a-b) / 5 de Mellado
    et al. (2012).

    Retourne
    --------
    geometries        : liste de géométries Open3D, contexte en premier
    n_context_offset  : int — nombre de géométries de contexte ajoutées
                         en tête de liste (0 ou 1), utile pour savoir
                         où commence la partie "locale" recadrable.
    """
    # Rayon local du voisinage : détermine la taille physique de la
    # calotte affichée autour de p.
    if len(neighbor_points) > 0:
        local_extent = float(np.max(np.linalg.norm(neighbor_points - p, axis=1)))
    else:
        local_extent = radius * 0.3
    crop_half = 1.3 * max(local_extent, 1e-6)

    context_geoms = []
    if context_points is not None and len(context_points) > 0:
        pcd_ctx = o3d.geometry.PointCloud()
        pcd_ctx.points = o3d.utility.Vector3dVector(context_points)
        pcd_ctx.paint_uniform_color([0.85, 0.85, 0.85])
        context_geoms.append(pcd_ctx)

    local_geometries = []

    pcd_nb = o3d.geometry.PointCloud()
    pcd_nb.points = o3d.utility.Vector3dVector(neighbor_points)
    pcd_nb.paint_uniform_color([0.20, 0.45, 0.95])   # bleu (cf. Fig. 3(a))
    local_geometries.append(pcd_nb)

    marker_radius = 0.05 * crop_half
    marker_p = o3d.geometry.TriangleMesh.create_sphere(radius=max(marker_radius, 1e-4))
    marker_p.translate(p)
    marker_p.paint_uniform_color([0.9, 0.1, 0.1])     # rouge : point p
    marker_p.compute_vertex_normals()
    local_geometries.append(marker_p)

    # Calotte locale de la sphère ajustée : on ne tessellise PAS la
    # sphère entière puis on découpe (échoue silencieusement quand le
    # rayon est très grand — quasi-plan — car la résolution globale
    # donne alors des triangles bien plus gros que la zone d'intérêt).
    # On construit à la place directement, dans le plan tangent en p,
    # une grille physiquement dimensionnée par crop_half, projetée
    # radialement sur la sphère depuis son centre : ceci reste précis
    # et bien résolu quel que soit le rayon, du fit quasi-plan (rayon
    # énorme) au fit très courbé (petit rayon).
    axis = p - center
    axis_norm = np.linalg.norm(axis)
    n = axis / axis_norm if axis_norm > 1e-9 else np.array([0., 0., 1.])
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u)

    grid_n = 30
    s_vals = np.linspace(-crop_half, crop_half, grid_n)
    S, T = np.meshgrid(s_vals, s_vals, indexing="ij")
    tangent_pts = p + S[..., None] * u + T[..., None] * v      # (grid_n, grid_n, 3)
    d = tangent_pts - center
    d_norm = np.linalg.norm(d, axis=-1, keepdims=True)
    grid_pts = center + radius * d / d_norm                    # projection radiale
    points = grid_pts.reshape(-1, 3)

    lines = []
    for i in range(grid_n):
        for j in range(grid_n - 1):
            lines.append([i * grid_n + j, i * grid_n + j + 1])
    for j in range(grid_n):
        for i in range(grid_n - 1):
            lines.append([i * grid_n + j, (i + 1) * grid_n + j])

    wireframe = o3d.geometry.LineSet()
    wireframe.points = o3d.utility.Vector3dVector(points)
    wireframe.lines   = o3d.utility.Vector2iVector(lines)
    wireframe.paint_uniform_color([0.1, 0.1, 0.1])    # noir : 0-isosurface ajustée
    local_geometries.append(wireframe)

    return context_geoms, local_geometries


def save_sphere_fit_visualization(p, neighbor_points, center, radius, obj_name,
                                  tag, output_dir="results",
                                  context_points=None,
                                  width=1024, height=768):
    """
    Rend et sauvegarde (headless) une image montrant, pour un point p
    et une échelle t donnés : le voisinage P_t(p), le point p, et la
    sphère algébrique ajustée. Voir _build_sphere_fit_geometries.

    Sauvegarde dans : results/<obj_name>/sphere_fit/{obj_name}_{tag}.png
    """
    folder = os.path.join(output_dir, obj_name, "sphere_fit")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{obj_name}_{tag}.png")

    context_geoms, local_geometries = _build_sphere_fit_geometries(
        p, neighbor_points, center, radius, context_points)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)

    for g in local_geometries:
        vis.add_geometry(g)          # reset_bounding_box=True par défaut
    vis.reset_view_point(True)       # caméra recadrée sur la zone locale

    for g in context_geoms:
        vis.add_geometry(g, reset_bounding_box=False)   # ne recadre pas

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size       = 5.0
    opt.line_width       = 2.0

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path, do_render=True)
    vis.destroy_window()

    print(f"[SPHERE-FIT] Sauvegardé : {path}  (r={radius:.4f}, K={len(neighbor_points)})")
    return path


def show_sphere_fit_interactive(p, neighbor_points, center, radius, tag,
                                context_points=None):
    """
    Ouvre une fenêtre Open3D INTERACTIVE (rotation/zoom à la souris)
    montrant, pour un point p et une échelle t donnés : le voisinage
    P_t(p) (bleu), le point p (marqueur rouge), et la sphère
    algébrique ajustée (fil de fer noir) — voir
    _build_sphere_fit_geometries pour la construction des géométries.

    Bloquant : la fenêtre doit être fermée pour passer au (point,
    échelle) suivant.
    """
    context_geoms, local_geometries = _build_sphere_fit_geometries(
        p, neighbor_points, center, radius, context_points)

    o3d.visualization.draw_geometries(
        context_geoms + local_geometries,
        window_name=tag,
    )