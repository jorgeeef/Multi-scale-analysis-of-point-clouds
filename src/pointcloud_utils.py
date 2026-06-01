# src/pointcloud_utils.py
import open3d as o3d
import numpy as np

def creer_nuage_points(sommets):
    """
    Transforme les sommets en un nuage de points Open3D
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sommets)
    return pcd

def visualiser_nuage_points(pcd):
    """
    Affiche un nuage de points Open3D
    """
    o3d.visualization.draw_geometries([pcd])

def sauvegarder_nuage_points(pcd, filename):
    """
    Sauvegarde un nuage de points au format .ply
    """
    o3d.io.write_point_cloud(filename, pcd)