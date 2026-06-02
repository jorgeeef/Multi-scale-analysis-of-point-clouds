# src/geomety.py
# =========================================================
# Module de préparation géométrique:
# Ce module fournit toutes les fonctions nécessaires à la
# préparation du nuage de points avant l'analyse GLS 
# =========================================================
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# Nettoyage du nuage de points
def clean_point_cloud(points):
    """
    Nettoie le nuage de points en supprimant les valeurs invalides.
    Supprime les points contenant des NaN ou des valeurs infinies
    qui pourraient corrompre les calculs géométriques suivants.
    """
    points = np.asarray(points)
    points = points[~np.isnan(points).any(axis=1)]
    return points


# Estimation des normales
def estimate_normals(pcd, k=30):
    """
    Estime les normales du nuage de points par la méthode k-NN,
    puis oriente les normales de façon cohérente globalement.
 
    L'estimation utilise une ACP locale sur les k plus proches
    voisins de chaque point. L'orientation est ensuite rendue
    cohérente par propagation sur le graphe de voisinage tangent
    (orient_normals_consistent_tangent_plane).
 
    Note
    ----
    L'orientation automatique peut échouer dans les zones de forte
    concavité (orbites, mâchoire) ou aux bords ouverts du modèle.
    Ces zones seront identifiées par le paramètre DELTA lors du fitting GLS.
    """

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    pcd.orient_normals_consistent_tangent_plane(k=k)
    return pcd


# Construction du kd-tree
def build_kdtree(points):
    """
    Construit un KD-tree spatial à partir du nuage de points.
 
    Le KD-tree (scipy.cKDTree) permet des recherches de voisins
    en O(log N) au lieu de O(N), ce qui est essentiel pour les
    voisinages multi-échelle sur des nuages de grande taille.
 
    """
    return cKDTree(points)


# Voisinage K-NN
def knn_neighbors(tree, points, k=30):
    """
    Calcule les k plus proches voisins pour chaque point du nuage.
 
    Paramètres
    tree : scipy.spatial.cKDTree
        KD-tree construit sur le nuage de points.
    points : np.ndarray, shape (N, 3)
        Coordonnées des points pour lesquels on cherche les voisins.
    k : int
        Nombre de voisins à retourner (défaut : 30).
 
    Retourne
    idx : np.ndarray, shape (N, k)
        Indices des k plus proches voisins pour chaque point.
    """
    _, idx = tree.query(points, k=k)
    return idx


# Voisinage par rayon (cœur du multi-échelle)
def radius_neighbors(tree, points, radius):
    """
    Calcule le voisinage par rayon pour chaque point du nuage.
 
    Pour chaque point p, retourne tous les indices des points q
    tels que ‖q - p‖ ≤ radius. C'est le voisinage P_t(p) utilisé
    dans la formulation GLS de Mellado
 
    Paramètres
    tree : scipy.spatial.cKDTree
        KD-tree construit sur le nuage de points.
    points : np.ndarray, shape (N, 3)
        Coordonnées des points du nuage.
    radius : float
        Rayon de recherche (échelle t dans la notation GLS).
 
    Retourne
    neighbors : list of lists
        neighbors[i] contient les indices des voisins du point i
        dans la boule de rayon `radius`.
    """
    neighbors = []
    for p in points:
        idx = tree.query_ball_point(p, r=radius)
        neighbors.append(idx)
    return neighbors


# Voisinages multi-échelle
def multi_scale_neighbors(points, scales):
    """
    Calcule les voisinages par rayon pour toutes les échelles.
 
    Pour chaque échelle t dans `scales`, calcule le voisinage
    P_t(p) pour chaque point p. Le résultat est un dictionnaire
    indexé par l'échelle, prêt pour la boucle de fitting GLS.
 
    Paramètres
    points : np.ndarray, shape (N, 3)
        Coordonnées des points du nuage.
    scales : list of float
        Liste des rayons d'échelle [t_1, t_2, ..., t_S].
 
    Retourne
    all_neighbors : dict {float: list of lists}
        all_neighbors[t][i] = liste des indices voisins du point i
        à l'échelle t.
    """
    tree = build_kdtree(points)
    all_neighbors = {}
    for t in scales:
        all_neighbors[t] = radius_neighbors(tree, points, t)
    return all_neighbors


# Estimation de l'espacement moyens
def estimate_mean_spacing(points, k=2):
    """
    Estime la distance moyenne entre points voisins (espacement local).
 
    Cette valeur sert de référence pour construire les échelles
    d'analyse : les rayons t seront exprimés en multiples de
    cet espacement, garantissant une analyse adaptée à la densité
    réelle du nuage.
 
    Paramètres
    points : np.ndarray, shape (N, 3)
        Coordonnées des points du nuage.
    k : int
        k=2 car le plus proche voisin d'un point est lui-même
        (distance 0) ; on prend donc le 2ème plus proche (défaut : 2).
 
    Retourne
    spacing : float
        Distance moyenne au plus proche voisin réel.
    """
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k)
    nn_dist = dists[:, 1]  # 2ème colonne = vrai plus proche voisin
    return np.mean(nn_dist)


# Construction des échelles d'analyse
def build_scales_from_spacing(spacing, n_scales=15, factor_min=2, factor_max=20, mode="log"):
    """
    Génère la liste des échelles t en fonction de l'espacement moyen.
 
    Les échelles sont exprimées en multiples de l'espacement moyen :
        t_min = factor_min * spacing
        t_max = factor_max * spacing
 
    La distribution logarithmique est recommandée car elle donne
    plus d'échelles fines aux petits rayons (détails) et moins
    aux grands rayons (forme globale), ce qui est cohérent avec
    l'analyse multi-échelle de Mellado et al. (2012).
 
    Paramètres
    spacing : float
        Espacement moyen entre points voisins.
    n_scales : int
        Nombre d'échelles à générer (défaut : 12).
    factor_min : float
        Multiplicateur minimum (défaut : 5, soit t_min ≈ 5 * spacing).
    factor_max : float
        Multiplicateur maximum (défaut : 15).
    mode : str
        "log" pour distribution logarithmique (recommandé),
        "linear" pour distribution linéaire uniforme.
 
    Retourne
    scales : np.ndarray, shape (n_scales,)
        Tableau des rayons d'échelle triés par ordre croissant.
    """
    if mode == "linear":
        scales = np.linspace(factor_min * spacing, 
                             factor_max * spacing, 
                             n_scales)
    else: 
        scales = np.logspace(np.log10(factor_min * spacing),
                             np.log10(factor_max * spacing),
                             n_scales)
    return scales


# MASQUE DE VALIDITÉ POUR LE FITTING GLS
def compute_validity_mask(neighborhoods, min_neighbors=6):
    """
    Calcule un masque booléen de validité pour chaque point et échelle.
 
    Un point est considéré valide à l'échelle t si son voisinage
    contient au moins `min_neighbors` points. En dessous de ce seuil,
    le système linéaire du fitting GLS est sous-déterminé (le vecteur
    u a 5 composantes, il faut donc au moins 6 équations).
 
    Les points invalides recevront des descripteurs NaN dans la
    boucle de fitting GLS.
 
    Paramètres
    neighborhoods : dict {float: list of lists}
        Voisinages multi-échelle retournés par multi_scale_neighbors.
    min_neighbors : int
        Seuil minimum de voisins pour qu'un fitting soit fiable (défaut : 6).
 
    Retourne
    masks : dict {float: np.ndarray of bool}
        masks[t][i] = True si le point i est valide à l'échelle t.
    """

    masks = {}
    for t, neighbors in neighborhoods.items():
        masks[t] = np.array([len(n) >= min_neighbors for n in neighbors])
    return masks