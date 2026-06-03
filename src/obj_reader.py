# src/obj_reader.py
import numpy as np
import open3d as o3d


def load_obj(filepath):
    """
    Lit un fichier .obj et extrait les vertices, faces et normales.
    Les indices de faces sont convertis de base-1 (OBJ) à base-0 (Python).
    
    Retourne
    --------
    vertices : np.ndarray, shape (N, 3)
    faces    : list of lists  (triangles, peut être vide)
    normals  : np.ndarray, shape (M, 3)  (peut être vide)
    """
    vertices = []
    faces    = []
    normals  = []

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])

            elif line.startswith("vn "):
                _, nx, ny, nz = line.split()
                normals.append([float(nx), float(ny), float(nz)])

            elif line.startswith("f "):
                parts = line.split()[1:]
                face  = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(face)

    return (
        np.array(vertices),
        faces,
        np.array(normals) if normals else np.empty((0, 3))
    )


def clean_faces(faces):
    """
    Nettoie et normalise la liste de faces.
    Filtre les faces None, les chaînes mal formées,
    et ne conserve que les triangles valides (>= 3 sommets).
    
    Retourne
    --------
    np.ndarray, shape (F, 3), dtype int32
    """
    clean = []
    for f in faces:
        if f is None:
            continue
        if isinstance(f, str):
            f = f.split('/')
        try:
            v = [int(str(x).split('/')[0]) for x in f]
            if len(v) >= 3:
                clean.append(v[:3])
        except (ValueError, IndexError):
            continue
    return np.asarray(clean, dtype=np.int32)


def build_point_cloud_with_normals(vertices, faces, obj_normals):
    """
    Construit un PointCloud Open3D avec les meilleures normales disponibles.

    Priorité :
        1. Normales calculées depuis le maillage (si faces disponibles)
        2. Normales lues depuis le fichier OBJ (si présentes et alignées)
        3. Normales estimées par ACP locale (fallback)

    Paramètres
    ----------
    vertices    : np.ndarray, shape (N, 3)
    faces       : list of lists ou np.ndarray
    obj_normals : np.ndarray, shape (N, 3) ou tableau vide

    Retourne
    --------
    pcd : o3d.geometry.PointCloud avec normales assignées
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # Priorité 1 : normales depuis le maillage
    if faces is not None and len(faces) > 0:
        print("[NORMALS] Source: mesh triangles")
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(clean_faces(faces))
        mesh.compute_vertex_normals()
        pcd.normals = mesh.vertex_normals
        return pcd

    # Priorité 2 : normales du fichier OBJ
    if obj_normals is not None and len(obj_normals) == len(vertices):
        print("[NORMALS] Source: OBJ file")
        pcd.normals = o3d.utility.Vector3dVector(obj_normals)
        return pcd

    # Priorité 3 : estimation ACP locale
    print("[NORMALS] Source: estimated (ACP k=30)")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )
    pcd.normalize_normals()

    # Orientation vers l'extérieur : on suppose que le centroïde
    # est à l'intérieur de l'objet (valide pour la grande majorité
    # des modèles scannés fermés)
    pcd.orient_normals_towards_camera_location(
        camera_location=np.mean(vertices, axis=0)
    )
    return pcd


def print_stats(vertices, faces, pcd=None):
    """Affiche les statistiques du modèle chargé."""
    print("\n========== MESH STATISTICS ==========")
    print("Vertices :", len(vertices))
    print("Faces    :", len(faces))
    if pcd is not None:
        print("Points   :", len(pcd.points))
        print("Normals  :", len(pcd.normals))
    print("=====================================\n")


def visualize_points(pcd):
    """Visualise le nuage de points sans normales."""
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Cloud — sans normales",
        point_show_normal=False
    )


def visualize_normals(pcd):
    """Visualise le nuage de points avec normales."""
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Cloud — avec normales",
        point_show_normal=True
    )


def compare_normals(vertices, faces, obj_normals):
    """
    Compare les trois sources de normales disponibles :
        1. Normales depuis le maillage (mesh triangles)
        2. Normales depuis le fichier OBJ
        3. Normales estimées par ACP locale (k=30)

    Pour chaque paire, calcule le cosinus de l'angle entre les normales
    (produit scalaire sur normales unitaires), puis en déduit :
        - la similarité moyenne  (cos θ proche de 1 → même direction)
        - la similarité médiane
        - le % de normales quasi-identiques  (cos θ > 0.99)
        - le % de normales similaires        (cos θ > 0.90)
        - le % de normales opposées          (cos θ < 0.0)

    Paramètres
    ----------
    vertices    : np.ndarray, shape (N, 3)
    faces       : list of lists ou np.ndarray
    obj_normals : np.ndarray, shape (N, 3) ou tableau vide

    Retourne
    --------
    None  (affichage console uniquement)
    """

    sources = {}

    # --- Source 1 : maillage ---
    if faces is not None and len(faces) > 0:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(clean_faces(faces))
        mesh.compute_vertex_normals()
        sources["mesh"] = np.asarray(mesh.vertex_normals)

    # --- Source 2 : fichier OBJ ---
    if obj_normals is not None and len(obj_normals) == len(vertices):
        sources["obj"] = np.asarray(obj_normals)

    # --- Source 3 : estimation ACP + orientation par référence mesh ---
    pcd_est = o3d.geometry.PointCloud()
    pcd_est.points = o3d.utility.Vector3dVector(vertices)
    pcd_est.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )
    pcd_est.normalize_normals()
    est_normals = np.asarray(pcd_est.normals).copy()

    # Orientation : si la normale estimée pointe dans le sens opposé
    # à la référence mesh, on la retourne.
    # Ceci simule ce que ferait un utilisateur sans topologie
    # mais avec un point de vue extérieur connu.
    if "mesh" in sources:
        ref = sources["mesh"]
        cos_signs = np.einsum("ij,ij->i", est_normals, ref)
        est_normals[cos_signs < 0] *= -1

    sources["estimated"] = est_normals


    if len(sources) < 2:
        print("[COMPARE] Pas assez de sources pour comparer.")
        return

    # --- Comparaison par paires ---
    names  = list(sources.keys())
    arrays = list(sources.values())

    print("\n========== COMPARAISON DES NORMALES ==========")
    print(f"  Points analysés : {len(vertices)}\n")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            na, nb   = names[i],  names[j]
            A,  B    = arrays[i], arrays[j]

            # Normalisation défensive (au cas où)
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
            B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)

            # Produit scalaire point à point  → cos θ ∈ [-1, 1]
            cos_theta = np.einsum("ij,ij->i", A, B)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            mean_cos   = np.mean(cos_theta)
            median_cos = np.median(cos_theta)
            pct_ident  = 100.0 * np.mean(cos_theta >  0.99)
            pct_sim    = 100.0 * np.mean(cos_theta >  0.90)
            pct_oppose = 100.0 * np.mean(cos_theta <  0.0)

            # Interprétation qualitative
            if mean_cos > 0.99:
                verdict = "QUASI-IDENTIQUES"
            elif mean_cos > 0.90:
                verdict = "TRES SIMILAIRES"
            elif mean_cos > 0.70:
                verdict = "SIMILAIRES"
            elif mean_cos > 0.0:
                verdict = "PARTIELLEMENT ALIGNEES"
            else:
                verdict = "OPPOSEES / INCOHERENTES"

            print(f"  [{na}]  vs  [{nb}]")
            print(f"    Verdict            : {verdict}")
            print(f"    cos θ moyen        : {mean_cos:.4f}")
            print(f"    cos θ médian       : {median_cos:.4f}")
            print(f"    quasi-identiques   : {pct_ident:.1f}%  (cos > 0.99)")
            print(f"    similaires         : {pct_sim:.1f}%  (cos > 0.90)")
            print(f"    opposées           : {pct_oppose:.1f}%  (cos < 0)")
            print()

    print("=" * 48 + "\n")