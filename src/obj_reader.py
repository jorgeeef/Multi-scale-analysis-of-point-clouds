import numpy as np
import open3d as o3d

# LOAD OBJ, extract vertices and faces
def load_obj(filepath):
    vertices = []
    faces = []

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])

            elif line.startswith("f "):
                parts = line.split()[1:]
                face = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(face)

    return np.array(vertices), faces 
# vertices → NumPy array (efficient math)
# faces → list of triangles


# CREATE POINT CLOUD: Convert raw vertices into Open3D structure
def create_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# PRINT STATS
def print_stats(vertices, faces, pcd=None):
    print("\n========== MESH STATISTICS ==========")
    print("Number of vertices :", len(vertices))
    print("Number of faces    :", len(faces))

    if pcd is not None:
        print("Number of points   :", len(pcd.points))
        print("Number of normals  :", len(pcd.normals))
    print("====================================\n")


def visualize(pcd):
    o3d.visualization.draw_geometries([pcd])