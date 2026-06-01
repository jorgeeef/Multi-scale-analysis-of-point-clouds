# src/local_regression.py
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d

def algebraic_sphere_fit(points, normals=None, t=5.0):
    """
    Ajuste une sphère algébrique localement autour de chaque point.

    Args:
        points: array Nx3 des points 3D
        normals: array Nx3 des normales (optionnel)
        t: rayon de voisinage

    Returns:
        descriptors: liste de dictionnaires avec clés:
            'tau', 'eta', 'kappa', 'fitness'
    """
    tree = cKDTree(points)
    descriptors = []

    for i, p in enumerate(points):
        idx = tree.query_ball_point(p, r=t)
        Pt = points[idx]

        if len(Pt) < 4:
            # Pas assez de points pour ajuster une sphère
            descriptors.append({'tau':0, 'eta':np.array([0,0,1]), 'kappa':0, 'fitness':0})
            continue

        # Ajustement sphère: (x-c)^2 - r^2 = 0
        A = np.hstack((2*(Pt - p), np.ones((len(Pt),1))))
        b = np.sum(Pt**2, axis=1) - np.sum(p**2)
        x, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        c = x[:3] + p
        r2 = np.sum(x[:3]**2) + x[3]
        r = np.sqrt(abs(r2))

        # Paramètres géométriques
        tau = np.linalg.norm(p - c) - r
        eta = (p - c) / (np.linalg.norm(p - c) + 1e-8)
        kappa = 1.0 / (r + 1e-8)
        fitness = np.exp(-residuals.sum() / len(Pt)) if len(residuals) > 0 else 1.0

        descriptors.append({
            'tau': tau,
            'eta': eta,
            'kappa': kappa,
            'fitness': fitness
        })

    return descriptors

def visualize_descriptors(points, descriptors, scale=1.0):
    """
    Visualisation des normales et courbures en Open3D.
    
    Args:
        points: Nx3 array
        descriptors: liste descripteurs (tau, eta, kappa, fitness)
        scale: facteur d'échelle pour les vecteurs
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Créer des lignes pour les normales
    lines = []
    colors = []
    line_points = []
    for i, d in enumerate(descriptors):
        p0 = points[i]
        p1 = p0 + d['eta'] * scale
        line_points.append(p0)
        line_points.append(p1)
        lines.append([2*i, 2*i+1])
        # Couleur selon courbure
        color = [1, 0, 0] if d['kappa'] > 0.05 else [0,0,1]
        colors.append(color)
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.repeat(colors, 2, axis=0))
    
    o3d.visualization.draw_geometries([pcd, line_set])