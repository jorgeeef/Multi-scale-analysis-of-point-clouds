# src/gls.py
# =========================================================
# MODULE DE FITTING GLS — GROWING LEAST SQUARES
#
# Implémentation de la méthode de Mellado :
# "Growing Least Squares for the Analysis of Manifolds in Scale-Space"
# Computer Graphics Forum, vol. 31, pp. 1691-1701.
#
# Pour chaque point p et chaque échelle t, on ajuste une sphère
# algébrique au voisinage local P_t(p) par moindres carrés pondérés
# (WLS). Les descripteurs géométriques extraits (τ, η, κ, δ) forment
# la base de l'analyse multi-échelle.
#
# Pipeline pour chaque point p à l'échelle t :
#   1. Poids gaussiens w_i = exp(-‖q_i - p‖² / t²)
#   2. Matrice de design B  (coordonnées centrées autour de p)
#   3. Résolution WLS       (vecteur propre min de BᵀWB)
#   4. Normalisation Pratt  (contrainte ‖ul‖² - 4·uc·uq = 1)
#   5. Orientation de u     (alignement avec la normale estimée)
#   6. Extraction de τ, η, κ
#   7. Calcul de δ          (qualité du fitting)
# =========================================================

import numpy as np

# Nombre minimum de voisins pour qu'un fitting soit numériquement stable.
# Le vecteur u a 5 composantes → il faut au moins 6 équations.
MIN_NEIGHBORS = 6


# Poids gaussiens
def gaussian_weights(neighbors_pts, p, t):
    """
    Calcule les poids gaussiens pour les voisins de p à l'échelle t.
 
    Chaque voisin q_i reçoit un poids inversement proportionnel à
    sa distance à p : les voisins proches influencent davantage
    le fitting que les voisins lointains.
 
        w_i = exp(-‖q_i - p‖² / t²)
 
    Paramètres
    neighbors_pts : np.ndarray, shape (n, 3)
        Coordonnées des points voisins.
    p : np.ndarray, shape (3,)
        Point central (point d'analyse).
    t : float
        Échelle (rayon de voisinage). Contrôle la largeur de la gaussienne.
 
    Retourne
    weights : np.ndarray, shape (n,)
        Poids gaussiens pour chaque voisin.
    """
    diff = neighbors_pts - p                    # (n, 3) vecteurs q_i - p
    dist2 = np.sum(diff**2, axis=1)             # (n,)   distances au carré
    weights = np.exp(-dist2 / (t ** 2))         # (n,)   poids gaussiens
    return weights



# Matrice de design
def build_design_matrix(neighbors_pts, p):
    """
    Construit la matrice de design B pour le fitting de la sphère algébrique.
 
    La sphère algébrique est définie par :
        s_u(x) = uc + ul·x + uq·‖x‖²
 
    où x = q_i - p (coordonnées centrées autour de p pour la stabilité numérique).
 
    Chaque ligne de B correspond à un voisin q_i :
        B[i] = [1,  x_i,  y_i,  z_i,  ‖q_i - p‖²]
 
    Paramètres
    neighbors_pts : np.ndarray, shape (n, 3)
        Coordonnées des points voisins (en coordonnées absolues).
    p : np.ndarray, shape (3,)
        Point central — les voisins sont recentrés autour de p.
 
    Retourne
    B : np.ndarray, shape (n, 5)
        Matrice de design [1 | x | y | z | ‖q‖²] en coordonnées centrées.
    """
    q = neighbors_pts - p                           # centrage autour de p 
    n = len(q)
    B = np.ones((n, 5))
    B[:, 1:4] = q                                   # colonnes x, y, z
    B[:, 4]   = np.sum(q ** 2, axis=1)              # colonne ‖q‖²
    return B


# NORMALISATION DE PRATT
def pratt_normalize(u):
    """
    Applique la normalisation de Pratt au vecteur paramètre u.
 
    Sans contrainte, la sphère algébrique admet une infinité de
    représentations (u et λu définissent la même surface). La
    normalisation de Pratt impose une contrainte unique :
 
        ‖ul‖² - 4·uc·uq = 1
 
    Cette contrainte rend u stable numériquement et permet
    d'extraire des descripteurs géométriques interprétables (τ, κ).
 
    Paramètres
    u : np.ndarray, shape (5,)
        Vecteur paramètre brut [uc, ul_x, ul_y, ul_z, uq].
 
    Retourne
    u_norm : np.ndarray, shape (5,) ou None
        Vecteur normalisé selon Pratt, ou None si la contrainte
        est impossible (sphère dégénérée ou imaginaire).
    """
    uc = u[0]
    ul = u[1:4]
    uq = u[4]
    denom = np.dot(ul, ul) - 4.0 * uc * uq
    if denom <= 0:
        return None  # sphère dégénérée (plan ou sphère imaginaire)
    scale = 1.0 / np.sqrt(denom)
    return u * scale


# Résolution wls 
def solve_wls(B, W):
    """
    Résout le problème de moindres carrés pondérés (WLS).
 
    On cherche le vecteur u qui minimise l'énergie pondérée :
        E(u) = Σ_i w_i · s_u(q_i)²  =  uᵀ (BᵀWB) u
 
    La solution est le vecteur propre associé à la plus petite
    valeur propre de la matrice symétrique M = BᵀWB.
 
    Paramètres
    B : np.ndarray, shape (n, 5)
        Matrice de design.
    W : np.ndarray, shape (n,)
        Vecteur des poids gaussiens (diagonale de la matrice de poids).
 
    Retourne
    u : np.ndarray, shape (5,)
        Vecteur propre associé à la plus petite valeur propre de BᵀWB.
    """

    W_diag = np.diag(W)                    # matrice diagonale des poids
    M = B.T @ W_diag @ B                   # (5, 5) matrice symétrique
    # eigh est plus stable que eig pour les matrices symétriques réelles
    eigenvalues, eigenvectors = np.linalg.eigh(M)   
    u = eigenvectors[:, 0]                     # vecteur propre de valeur min
    return u


# Fitting principal - sphère algébrique
def fit_algebraic_sphere(p, neighbors_pts, t, normal_p=None):
    """
    Ajuste une sphère algébrique au voisinage local d'un point p
    à l'échelle t par la méthode GLS (Growing Least Squares).
 
    Étapes internes :
        1. Vérifie que le voisinage est suffisant (>= MIN_NEIGHBORS)
        2. Calcule les poids gaussiens
        3. Construit la matrice de design B (coordonnées centrées)
        4. Résout le WLS (vecteur propre minimal)
        5. Applique la normalisation de Pratt
        6. Oriente u pour que ∇s_u(p) soit aligné avec normal_p
 
    Paramètres
    p : np.ndarray, shape (3,)
        Point central d'analyse.
    neighbors_pts : np.ndarray, shape (n, 3)
        Coordonnées des voisins de p à l'échelle t.
    t : float
        Échelle courante (rayon de voisinage).
    normal_p : np.ndarray, shape (3,) ou None
        Normale estimée au point p (Open3D). Utilisée pour orienter
        le signe de u de façon cohérente.
 
    Retourne
    u : np.ndarray, shape (5,) ou None
        Vecteur paramètre normalisé [uc, ul_x, ul_y, ul_z, uq],
        ou None si le fitting est impossible (trop peu de voisins,
        matrice singulière, sphère dégénérée).
    """
    if len(neighbors_pts) < MIN_NEIGHBORS:
        return None

    try:
        W = gaussian_weights(neighbors_pts, p, t)
        B = build_design_matrix(neighbors_pts, p)
        u = solve_wls(B, W)
        u = pratt_normalize(u)

        if u is None:
            return None

        # Orientation de u : en coordonnées centrées, p_local = 0,
        # donc ∇s_u(0) = ul. On s'assure que ul pointe dans le même
        # sens que la normale estimée par Open3D.
        if normal_p is not None:
            ul = u[1:4]
            uq = u[4]
            grad_p = ul      
            if np.dot(grad_p, normal_p) < 0:
                u = -u                         # inversion globale du signe de u
 
        return u
    
    except np.linalg.LinAlgError:
        return None


# Extraction des descripteurs géométriques
def extract_descriptors(u, p):
    """
    Extrait les descripteurs géométriques locaux depuis u normalisé.
 
    En coordonnées centrées (p est l'origine locale), les formules
    se simplifient :
        τ  = s_u(0)   = uc               (offset algébrique)
        η  = ∇s_u(0) / ‖∇s_u(0)‖  = ul / ‖ul‖   (normale locale)
        κ  = 2·uq                        (courbure moyenne)
 
    Interprétation géométrique :
        - τ ≈ 0 : p est proche de la surface implicite
        - η     : direction normale à la surface en p
        - κ > 0 : surface convexe  (sphère pointant vers l'extérieur)
        - κ < 0 : surface concave  (sphère pointant vers l'intérieur)
        - κ ≈ 0 : surface plane localement
 
    Paramètres
    u : np.ndarray, shape (5,)
        Vecteur paramètre normalisé (Pratt).
    p : np.ndarray, shape (3,)
        Point central (non utilisé ici car coordonnées centrées,
        conservé pour cohérence de l'interface).
 
    Retourne
    tau   : float         — offset algébrique
    eta   : np.ndarray (3,) — normale locale normalisée
    kappa : float         — courbure (2 * uq)
    """

    uc = u[0]
    ul = u[1:4]
    uq = u[4]

    tau = uc                     # s_u(0) = uc en coordonnées centrées

    grad_norm = np.linalg.norm(ul)
    if grad_norm < 1e-10:
            eta = np.zeros(3)  # gradient nul → normale indéfinie
    else:
        eta = ul / grad_norm

    kappa = 2.0 * uq
    return tau, eta, kappa


# Qualité du fitting - paramètre delta (mellado)
def compute_fitness(u, neighbors_pts, p, normals_pts, weights):
    """
    Calcule le paramètre de fitness δ (qualité du fitting GLS).
 
    δ mesure l'alignement moyen entre la normale estimée par la sphère
    algébrique et les normales réelles des points voisins :
 
        δ = Σ_i w_i · (∇s_u(q_i) / ‖∇s_u(q_i)‖) · n_i
            ─────────────────────────────────────────────
                         Σ_i w_i
 
    Interprétation :
        δ = 1  → alignement parfait, fitting excellent
        δ ≈ 0  → mauvais alignement (zone concave, bord ouvert,
                  ou normales mal orientées)
 
    Ce paramètre permet de filtrer les zones géométriquement
    complexes avant l'analyse multi-échelle et multi-fractale.
 
    Paramètres
    u : np.ndarray, shape (5,)
        Vecteur paramètre normalisé (Pratt).
    neighbors_pts : np.ndarray, shape (n, 3)
        Coordonnées absolues des voisins.
    p : np.ndarray, shape (3,)
        Point central (pour centrer les coordonnées).
    normals_pts : np.ndarray, shape (n, 3)
        Normales estimées (Open3D) des points voisins.
    weights : np.ndarray, shape (n,)
        Poids gaussiens des voisins.
 
    Retourne
    delta : float
        Indice de qualité du fitting, dans [-1, 1].
    """

    ul = u[1:4]
    uq = u[4]

    q_local = neighbors_pts - p                             # centrage autour de p
    grads = ul + 2.0 * uq * q_local                        #  (n, 3) gradients aux voisins
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)           # évite la division par zéro
    grads_normalized = grads / norms                      # (n, 3) gradients normalisés

    dot_products = np.sum(grads_normalized * normals_pts, axis=1)  # (n,)
    delta = np.sum(weights * dot_products) / np.sum(weights)
    return delta