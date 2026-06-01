import open3d as o3d
import numpy as np



# =========================
# VISUALISATION D'UNE SPHÈRE GLS À UNE ÉCHELLE DONNÉE
# =========================
def visualize_sphere_at_point(vertices, neighborhoods, scales, TAU, KAPPA, ETA,
                               normals_np, point_idx=0, scale_idx=0):
    """
    Visualise la sphère GLS ajustée au point point_idx à l'échelle scales[scale_idx].
    """

    p = vertices[point_idx]
    t = scales[scale_idx]

    kappa = KAPPA[point_idx, scale_idx]
    tau   = TAU[point_idx, scale_idx]
    eta   = ETA[point_idx, scale_idx]

    if np.isnan(kappa):
        print(f"[VISU] Point {point_idx} invalide à l'échelle {t:.3f}")
        return

    # --- Rayon et centre de la sphère ---
    # κ = 2*uq,  centre = p - η/κ  (formule de Mellado)
    if abs(kappa) < 1e-6:
        print(f"[VISU] κ ≈ 0 → surface plane, pas de sphère à afficher")
        return

    #radius = abs(2.0 / kappa)          # rayon = |1/uq| = |2/κ|



    # APRÈS
    radius = abs(2.0 / kappa)
    tau_abs = abs(tau)
    radius_display = max(radius - tau_abs, radius * 0.1)  # rayon corrigé pour la visu






    center = p - (eta / kappa)         # centre de la sphère

    print(f"\n[VISU] Point #{point_idx} | échelle t={t:.3f}")
    print(f"  Centre sphère : {center}")
    print(f"  Rayon         : {radius:.4f}")
    print(f"  κ             : {kappa:.4f}")
    print(f"  τ             : {tau:.4f}")

    # --- Voisinage local ---
    idx      = neighborhoods[t][point_idx]
    npts     = vertices[idx]
    pcd_local = o3d.geometry.PointCloud()
    pcd_local.points = o3d.utility.Vector3dVector(npts)
    pcd_local.paint_uniform_color([0.5, 0.5, 0.5])   # gris

    # --- Point central (rouge) ---
    pcd_center = o3d.geometry.PointCloud()
    pcd_center.points = o3d.utility.Vector3dVector([p])
    pcd_center.paint_uniform_color([1, 0, 0])

    # --- Sphère ajustée (maillage) ---
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius_display)
    sphere_mesh.translate(center)
    sphere_mesh.paint_uniform_color([0.1, 0.5, 1.0])  # bleu
    sphere_mesh.compute_vertex_normals()

    # Rendre la sphère transparente (wireframe)
    sphere_wire = o3d.geometry.LineSet.create_from_triangle_mesh(sphere_mesh)
    sphere_wire.paint_uniform_color([0.1, 0.5, 1.0])

    o3d.visualization.draw_geometries(
        [pcd_local, pcd_center, sphere_wire],
        window_name=f"Sphère GLS — point #{point_idx}, t={t:.3f}, κ={kappa:.4f}"
    )


def visualize_sphere_patch(vertices, neighborhoods, scales, TAU, KAPPA, ETA,
                            point_idx=0, scale_idx=0):
    """
    Visualise le patch de sphère GLS — seulement la portion proche des points.
    Plus lisible que la sphère complète.
    """
    p     = vertices[point_idx]
    t     = scales[scale_idx]
    kappa = KAPPA[point_idx, scale_idx]
    tau   = TAU[point_idx, scale_idx]
    eta   = ETA[point_idx, scale_idx]

    if np.isnan(kappa) or abs(kappa) < 1e-6:
        print(f"[VISU] Point invalide ou plan à t={t:.3f}")
        return

    radius = abs(2.0 / kappa)
    center = p - (eta / kappa)

    print(f"\n[VISU] Point #{point_idx} | t={t:.3f} | r={radius:.4f} | κ={kappa:.4f}")

    # --- Voisinage local (points gris) ---
    idx       = neighborhoods[t][point_idx]
    npts      = vertices[idx]
    pcd_local = o3d.geometry.PointCloud()
    pcd_local.points = o3d.utility.Vector3dVector(npts)
    pcd_local.paint_uniform_color([0.6, 0.6, 0.6])

    # --- Point central (rouge, plus gros) ---
    pcd_center = o3d.geometry.PointCloud()
    pcd_center.points = o3d.utility.Vector3dVector([p])
    pcd_center.paint_uniform_color([1, 0, 0])

    # --- Normale au point p (verte) ---
    eta_line = o3d.geometry.LineSet()
    eta_line.points = o3d.utility.Vector3dVector([p, p + eta * t * 0.5])
    eta_line.lines  = o3d.utility.Vector2iVector([[0, 1]])
    eta_line.paint_uniform_color([0, 0.8, 0])

    # --- Patch de sphère : points échantillonnés sur la surface ---
    # On génère des points sur la sphère SEULEMENT dans la direction des voisins
    patch_pts = []
    for q in npts:
        # Projection de q sur la sphère (direction depuis le centre)
        direction = q - center
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            q_on_sphere = center + (direction / norm) * radius
            patch_pts.append(q_on_sphere)

    pcd_patch = o3d.geometry.PointCloud()
    pcd_patch.points = o3d.utility.Vector3dVector(patch_pts)
    pcd_patch.paint_uniform_color([0.1, 0.5, 1.0])   # bleu = patch sphère

    # --- Centre de la sphère (jaune) ---
    pcd_sphere_center = o3d.geometry.PointCloud()
    pcd_sphere_center.points = o3d.utility.Vector3dVector([center])
    pcd_sphere_center.paint_uniform_color([1, 0.8, 0])

    # --- Ligne centre sphère → point p (jaune) ---
    radius_line = o3d.geometry.LineSet()
    radius_line.points = o3d.utility.Vector3dVector([center, p])
    radius_line.lines  = o3d.utility.Vector2iVector([[0, 1]])
    radius_line.paint_uniform_color([1, 0.8, 0])



    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius, resolution=10   # ← résolution basse = moins de lignes
    )
    sphere_mesh.translate(center)
    sphere_wire = o3d.geometry.LineSet.create_from_triangle_mesh(sphere_mesh)
    sphere_wire.paint_uniform_color([0.1, 0.5, 1.0])

    o3d.visualization.draw_geometries(
        [pcd_local, pcd_center, sphere_wire, pcd_sphere_center, eta_line, radius_line],
        window_name=f"Patch GLS — point #{point_idx}, t={t:.3f}, κ={kappa:.4f}, r={radius:.3f}"
    )

"""
def visualiser_spheres(pcd, centres, rayons, filename="nuage_spheres.ply"):
    meshes = []

    # Créer des sphères pour chaque centre
    for c, r in zip(centres, rayons):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        sphere.translate(c)
        sphere.paint_uniform_color([1, 0, 0])  # rouge
        meshes.append(sphere)

    # Sauvegarder les sphères comme un mesh combiné
    combined_mesh = meshes[0] if meshes else o3d.geometry.TriangleMesh()
    for mesh in meshes[1:]:
        combined_mesh += mesh

    # Sauvegarder les sphères
    o3d.io.write_triangle_mesh("spheres_only.ply", combined_mesh)
    print("[INFO] Sphères sauvegardées dans 'spheres_only.ply'")

    # Sauvegarder le nuage de points séparément
    o3d.io.write_point_cloud("nuage_points.ply", pcd)
    print("[INFO] Nuage de points sauvegardé dans 'nuage_points.ply'")

    print("[INFO] Vous pouvez visualiser les fichiers PLY dans MeshLab ou Blender")
    """