import open3d as o3d
import numpy as np

def analyse_multiechelle_simple(pcd, rayons=[0.5, 1.0, 2.0]):
    points = np.asarray(pcd.points)
    resultats = []

    for r in rayons:
        voisins_par_point = []
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        for i, point in enumerate(points):
            [k, idx, _] = kdtree.search_radius_vector_3d(point, r)
            voisins_par_point.append(k)
        resultats.append(np.array(voisins_par_point))
    
    return resultats