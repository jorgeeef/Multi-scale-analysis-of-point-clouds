# src/gls.py
# =========================================================
# Growing Least Squares — Mellado et al. (2012)
# =========================================================
# Implémente :
#   - Fonction de poids de Wendland C2        (Eq. 2)
#   - Fitting de sphère algébrique par WLS    (Appendix Eq. 7)
#   - Normalisation de Pratt                  (Eq. 3)
#   - Extraction des descripteurs τ, η, κ     (Eq. 4)
#   - Calcul du fitness ϕ                     (Section 4.1)
# =========================================================

import numpy as np

# 1. FONCTION DE POIDS DE WENDLAND C2  (Eq. 2)

def wendland_weights(neighbors, p, t):
    """
    Calcule les poids de Wendland C2 pour les voisins de p à l'échelle t.

    Formule (Eq. 2) :
        w_i(t) = ( ||qi - p||² / t² - 1 )²

    Support compact : w_i = 0 exactement pour ||qi - p|| = t.
    Classe C² → garantit la dérivabilité du fitting par rapport à t,
    propriété essentielle pour l'analyse continue en scale-space.

    Paramètres
    ----------
    neighbors : np.ndarray, shape (K, 3)
        Coordonnées des K points voisins qi.
    p : Point central d'évaluation.
    t : Rayon d'échelle.

    Retourne
    --------
    w : poids w_i >= 0
    """
    diff  = neighbors - p
    d2    = np.einsum("ij,ij->i", diff, diff)    # ||qi - p||²
    ratio = d2 / (t * t)
    w     = (ratio - 1.0) ** 2
    return w

# 2. FITTING DE LA SPHÈRE ALGÉBRIQUE  (Appendix Eq. 7)

def fit_algebraic_sphere(p, neighbors, normals, t):
    """
    Ajuste une sphère algébrique sur le voisinage P_t(p) par
    Weighted Least Squares avec intégration des normales.

    Modèle implicite (Eq. 1) :
        s_u(x) = [1, x^T, ||x||²] · u
        u = [uc, ul_x, ul_y, ul_z, uq]^T

    Le gradient du modèle est :
        ∇s_u(x) = ul + 2·uq·x

    Deux minimisations successives (GGG08) :

    Étape A — intégration des normales :
        min_u Σ w_i ||∇s_u(qi) - ni||²
        → donne ul et uq en closed-form

    Étape B — distance algébrique :
        min_uc Σ w_i s_u(qi)²
        → donne uc en closed-form

    Formules (Appendix Eq. 7), avec w̃_i = w_i / Σw_j :

        num_uq   = Σw̃i·(qi·ni) - (Σw̃i·qi)·(Σwi·ni)
        denom_uq = Σw̃i·||qi||² - (Σw̃i·qi)·(Σwi·qi)
        uq = num_uq / denom_uq

        ul = Σw̃i·ni - 2·uq·(Σw̃i·qi)

        uc = -(ul · Σw̃i·qi) - uq·(Σw̃i·||qi||²)

    Retourne
    --------
    u : [uc, ul_x, ul_y, ul_z, uq]
        None si le fitting échoue.
    """
    K = len(neighbors)
    if K < 6:
        return None

    # Poids de Wendland
    w = wendland_weights(neighbors, p, t)        # (K,)
    W_sum = np.sum(w)
    if W_sum < 1e-10:
        return None
    w_tilde = w / W_sum                          # w̃i

    # Termes agrégés
    Wt_q  = np.einsum("i,ij->j",  w_tilde, neighbors)   # Σw̃i·qi       (3,)
    Wt_n  = np.einsum("i,ij->j",  w_tilde, normals)     # Σw̃i·ni       (3,)
    W_n   = np.einsum("i,ij->j",  w,       normals)     # Σwi·ni        (3,)
    W_q   = np.einsum("i,ij->j",  w,       neighbors)   # Σwi·qi        (3,)
    Wt_q2 = np.einsum("i,ij,ij->", w_tilde, neighbors, neighbors)  # Σw̃i·||qi||²
    Wt_qn = np.einsum("i,ij,ij->", w_tilde, neighbors, normals)    # Σw̃i·(qi·ni)

    # Étape A : uq
    num_uq   = Wt_qn  - np.dot(Wt_q, W_n)
    denom_uq = Wt_q2  - np.dot(Wt_q, W_q)
    if abs(denom_uq) < 1e-12:
        return None
    uq = num_uq / denom_uq

    # Étape A : ul
    ul = Wt_n - 2.0 * uq * Wt_q                 # (3,)

    # Étape B : uc
    uc = -np.dot(ul, Wt_q) - uq * Wt_q2

    u = np.concatenate(([uc], ul, [uq]))         # (5,)

    if not np.all(np.isfinite(u)):
        return None

    return u


# ---------------------------------------------------------
# 3. NORMALISATION DE PRATT  (Eq. 3)
# ---------------------------------------------------------

def pratt_normalize(u):
    """
    Applique la normalisation de Pratt au vecteur de paramètres u.

    Formule (Eq. 3) :
        û = u / sqrt( ||ul||² - 4·uc·uq )

    Effet : contraint le gradient de s_û à être unitaire sur la
    0-isosurface → distances algébriques quasi-euclidiennes.
    Résout l'ambiguïté de signe (u et λu définissent la même sphère).

    Paramètres
    ----------
    u : np.ndarray, shape (5,)

    Retourne
    --------
    u_hat : np.ndarray, shape (5,)
        None si dégénéré (sphère imaginaire ou point).
    """
    uc = u[0]
    ul = u[1:4]
    uq = u[4]

    pratt_sq = np.dot(ul, ul) - 4.0 * uc * uq
    if pratt_sq <= 1e-10:
        return None

    u_hat = u / np.sqrt(pratt_sq)
    return u_hat

# 4. EXTRACTION DES DESCRIPTEURS  (Eq. 4)

def extract_descriptors(u_hat, p):
    """
    Extrait les descripteurs géométriques (τ, η, κ) depuis û.

    Formules (Eq. 4) :
        τ = s_û(p) = ûc + ûl^T·p + ûq·||p||²
        η = ∇s_û(p) / ||∇s_û(p)||    avec ∇s_û(p) = ûl + 2·ûq·p
        κ = 2·ûq

    Interprétation :
        τ > 0  → p est à l'extérieur de la sphère ajustée
        τ = 0  → p est sur la surface
        τ < 0  → p est à l'intérieur
        η      → direction vers la surface (normale locale)
        κ > 0  → convexe   (comme vu depuis l'extérieur)
        κ = 0  → plan local (rayon infini)
        κ < 0  → concave

    Paramètres
    ----------
    u_hat : np.ndarray, shape (5,)
    p     : np.ndarray, shape (3,)

    Retourne
    --------
    tau   : float
    eta   : np.ndarray, shape (3,)
    kappa : float
    """
    uc_hat = u_hat[0]
    ul_hat = u_hat[1:4]
    uq_hat = u_hat[4]

    # τ = s_û(p)
    tau = uc_hat + np.dot(ul_hat, p) + uq_hat * np.dot(p, p)

    # η = ∇s_û(p) / ||∇s_û(p)||
    grad      = ul_hat + 2.0 * uq_hat * p
    grad_norm = np.linalg.norm(grad)
    eta       = grad / grad_norm if grad_norm > 1e-10 else np.zeros(3)

    # κ = 2·ûq
    kappa = 2.0 * uq_hat

    return tau, eta, kappa


# ---------------------------------------------------------
# 5. FITNESS  (Section 4.1)
# ---------------------------------------------------------

def compute_fitness(u, neighbors, normals, p, t):
    """
    Calcule le fitness ϕ : qualité d'alignement du champ scalaire
    ajusté avec les normales d'entrée.

    Formule (Section 4.1) :
        ϕ = Σ w_i(t) · ∇s_u(qi) · ni  /  Σ w_i(t)

    où ∇s_u(x) = ul + 2·uq·x  est le gradient du champ ajusté.

    Propriétés :
        ϕ = 1.0  → alignement parfait (surface lisse, normales cohérentes)
        ϕ < 0.9  → fit dégradé (bruit, zone concave, bord ouvert)
        ϕ ≈ 0    → fit incohérent

    Paramètres
    ----------
    u         : np.ndarray, shape (5,)   — paramètres bruts
    neighbors : np.ndarray, shape (K, 3)
    normals   : np.ndarray, shape (K, 3)
    p         : np.ndarray, shape (3,)
    t         : float

    Retourne
    --------
    phi : float ∈ [0, 1]
    """
    w = wendland_weights(neighbors, p, t)
    W_sum = np.sum(w)
    if W_sum < 1e-10:
        return 0.0

    ul = u[1:4]
    uq = u[4]

    # ∇s_u(qi) = ul + 2·uq·qi
    grad_su      = ul[None, :] + 2.0 * uq * neighbors   # (K, 3)
    dot_products = np.einsum("ij,ij->i", grad_su, normals)  # (K,)

    phi = np.sum(w * dot_products) / W_sum
    return float(np.clip(phi, 0.0, 1.0))

# 6. PIPELINE GLS COMPLET POUR UN POINT

def gls_at_point(p, neighbors, normals, t):
    """
    Pipeline GLS complet pour un point p à l'échelle t.

    Enchaîne :
        1. fit_algebraic_sphere  — WLS avec normales (Eq. 7)
        2. pratt_normalize       — unicité (Eq. 3)
        3. extract_descriptors   — τ, η, κ (Eq. 4)
        4. compute_fitness       — ϕ (Section 4.1)

    Paramètres
    ----------
    p         : np.ndarray, shape (3,)
    neighbors : np.ndarray, shape (K, 3)
    normals   : np.ndarray, shape (K, 3)
    t         : float

    Retourne
    --------
    dict : { 'tau', 'eta', 'kappa', 'phi', 'u', 'u_hat' }
    None : si le fitting échoue à n'importe quelle étape.
    """
    u = fit_algebraic_sphere(p, neighbors, normals, t)
    if u is None:
        return None

    u_hat = pratt_normalize(u)
    if u_hat is None:
        return None

    tau, eta, kappa = extract_descriptors(u_hat, p)
    phi             = compute_fitness(u, neighbors, normals, p, t)

    return {
        "tau"   : tau,
        "eta"   : eta,
        "kappa" : kappa,
        "phi"   : phi,
        "u"     : u,
        "u_hat" : u_hat,
    }