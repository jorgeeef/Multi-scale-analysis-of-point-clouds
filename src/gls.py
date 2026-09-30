# src/gls.py
# =========================================================
# Growing Least Squares — Mellado 

import numpy as np

# 1. FONCTION DE POIDS

def wendland_weights(neighbors, p, t):
    """
    Calcule les poids de Wendland C2 pour les voisins de p à l'échelle t.

    Formule (Eq. 2) :
        w_i(t) = ( ||qi - p||² / t² - 1 )²

    """
    diff  = neighbors - p
    d2    = np.einsum("ij,ij->i", diff, diff)    # ||qi - p||²
    ratio = d2 / (t * t)
    w     = (ratio - 1.0) ** 2
    return w


def wendland_weight_derivative(neighbors, p, t):
    """
    Dérivée du poids w_i(t) = (d_i²/t² - 1)² par rapport à l'échelle t,
    avec d_i = ||qi - p|| (Appendix Eq. 8).

        dw_i/dt = 4·d_i²/t³ · (1 - d_i²/t²)

    """
    diff = neighbors - p
    d2   = np.einsum("ij,ij->i", diff, diff)
    dw   = 4.0 * d2 / (t ** 3) * (1.0 - d2 / (t * t))
    return dw

# 2. FITTING DE LA SPHÈRE ALGÉBRIQUE  (Appendix Eq. 7)

def fit_algebraic_sphere(p, neighbors, normals, t):
    """
    Ajuste une sphère algébrique sur le voisinage P_t(p) par
    Weighted Least Squares avec intégration des normales.
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
    Wt_q2 = np.einsum("i,ij,ij->", w_tilde, neighbors, neighbors)  # Σw̃i·||qi||²  (pour uc)
    W_q2  = np.einsum("i,ij,ij->", w,       neighbors, neighbors)  # Σwi·||qi||²   (pour uq)
    W_qn  = np.einsum("i,ij,ij->", w,       neighbors, normals)    # Σwi·(qi·ni)   (pour uq)

    # Étape A : uq
    num_uq   = W_qn   - np.dot(Wt_q, W_n)
    denom_uq = W_q2   - np.dot(Wt_q, W_q)
    if abs(denom_uq) < 1e-12:
        return None
    uq = 0.5 * num_uq / denom_uq

    # Étape A : ul
    ul = Wt_n - 2.0 * uq * Wt_q                 # (3,)

    # Étape B : uc
    uc = -np.dot(ul, Wt_q) - uq * Wt_q2

    u = np.concatenate(([uc], ul, [uq]))         # (5,)

    if not np.all(np.isfinite(u)):
        return None

    return u


def fit_algebraic_sphere_with_derivative(p, neighbors, normals, t):
    """
    Comme fit_algebraic_sphere, mais calcule en plus du/dt : la dérivée
    du vecteur de paramètres bruts u par rapport à l'échelle t, en
    différentiant analytiquement toute la chaîne de l'Eq. 7 (Section 4.2).
    """
    K = len(neighbors)
    if K < 6:
        return None, None

    w  = wendland_weights(neighbors, p, t)
    dw = wendland_weight_derivative(neighbors, p, t)

    S0  = np.sum(w)
    S0p = np.sum(dw)
    if S0 < 1e-10:
        return None, None

    q, n = neighbors, normals
    qq = np.einsum("ij,ij->i", q, q)   # ||qi||²  (K,)
    qn = np.einsum("ij,ij->i", q, n)   # qi·ni    (K,)

    def A(f):  return np.einsum("i,i...->...", w,  f)
    def Ap(f): return np.einsum("i,i...->...", dw, f)

    W_q,  W_n,  W_q2,  W_qn  = A(q),  A(n),  A(qq),  A(qn)
    W_qp, W_np, W_q2p, W_qnp = Ap(q), Ap(n), Ap(qq), Ap(qn)

    # sommes normalisées T[f]=A[f]/S0 (les w̃i·fi) et leurs dérivées T'[f]
    Wt_q,  Wt_n,  Wt_q2  = W_q / S0,  W_n / S0,  W_q2 / S0
    Wt_qp  = (W_qp  - Wt_q  * S0p) / S0
    Wt_np  = (W_np  - Wt_n  * S0p) / S0
    Wt_q2p = (W_q2p - Wt_q2 * S0p) / S0

    # --- uq et sa dérivée ---
    num_uq   = W_qn  - np.dot(Wt_q, W_n)
    denom_uq = W_q2  - np.dot(Wt_q, W_q)
    if abs(denom_uq) < 1e-12:
        return None, None

    num_uq_p   = W_qnp - (np.dot(Wt_qp, W_n) + np.dot(Wt_q, W_np))
    denom_uq_p = W_q2p - (np.dot(Wt_qp, W_q) + np.dot(Wt_q, W_qp))

    uq   = 0.5 * num_uq / denom_uq
    uq_p = 0.5 * (num_uq_p * denom_uq - num_uq * denom_uq_p) / (denom_uq ** 2)

    # --- ul et sa dérivée ---
    ul   = Wt_n  - 2.0 * uq   * Wt_q
    ul_p = Wt_np - 2.0 * (uq_p * Wt_q + uq * Wt_qp)

    # --- uc et sa dérivée ---
    uc   = -np.dot(ul, Wt_q)          - uq   * Wt_q2
    uc_p = -(np.dot(ul_p, Wt_q) + np.dot(ul, Wt_qp)) - (uq_p * Wt_q2 + uq * Wt_q2p)

    u     = np.concatenate(([uc],   ul,   [uq]))
    du_dt = np.concatenate(([uc_p], ul_p, [uq_p]))

    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(du_dt))):
        return None, None

    return u, du_dt


# 3. NORMALISATION DE PRATT  
def pratt_normalize(u):
    """
    Applique la normalisation de Pratt au vecteur de paramètres u.

    Formule :
        û = u / sqrt( ||ul||² - 4·uc·uq )

    Effet : contraint le gradient de s_û à être unitaire sur la
    0-isosurface → distances algébriques quasi-euclidiennes.
    Résout l'ambiguïté de signe (u et λu définissent la même sphère).

    """
    uc = u[0]
    ul = u[1:4]
    uq = u[4]

    pratt_sq = np.dot(ul, ul) - 4.0 * uc * uq
    if pratt_sq <= 1e-10:
        return None

    u_hat = u / np.sqrt(pratt_sq)
    return u_hat


def pratt_normalize_with_derivative(u, du_dt):
    """
    Comme pratt_normalize, mais calcule aussi dû/dt en différentiant
    û = u·P^{-1/2} par la règle du produit, avec P = ||ul||²-4ucuq :

        dP/dt = 2·ul·dul/dt - 4·(duc/dt·uq + uc·duq/dt)
        dû/dt = du/dt/√P  -  û · dP/dt / (2P)

    """
    uc,  ul,  uq  = u[0],     u[1:4],     u[4]
    duc, dul, duq = du_dt[0], du_dt[1:4], du_dt[4]

    P = np.dot(ul, ul) - 4.0 * uc * uq
    if P <= 1e-10:
        return None, None
    dP = 2.0 * np.dot(ul, dul) - 4.0 * (duc * uq + uc * duq)

    sqrtP = np.sqrt(P)
    u_hat     = u / sqrtP
    du_hat_dt = du_dt / sqrtP - u_hat * dP / (2.0 * P)

    return u_hat, du_hat_dt

# 4. EXTRACTION DES DESCRIPTEURS  (Eq. 4)
def extract_descriptors(u_hat, p):
    """
    Extrait les descripteurs géométriques depuis û.

    Formules  :
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


def extract_descriptors_with_derivative(u_hat, du_hat_dt, p):
    """
    Comme extract_descriptors, mais calcule aussi les dérivées
    d'échelle dτ/dt, dη/dt, dκ/dt (Section 4.2), en différentiant
    l'Eq. 4 par rapport à t, à p FIXE (p ne dépend pas de l'échelle) :

        τ = ûc + ûl·p + ûq·||p||²
        dτ/dt = dûc/dt + dûl/dt·p + dûq/dt·||p||²

        grad = ûl + 2ûq·p                    (∇s_û(p), non normalisé)
        dgrad/dt = dûl/dt + 2·dûq/dt·p

        η = grad / ||grad||
        dη/dt = (dgrad/dt - η·(η·dgrad/dt)) / ||grad||
              (dérivée standard d'un champ vectoriel normalisé)

        κ = 2ûq  ⟹  dκ/dt = 2·dûq/dt

    """
    uc_hat, ul_hat, uq_hat = u_hat[0], u_hat[1:4], u_hat[4]
    duc,    dul,    duq    = du_hat_dt[0], du_hat_dt[1:4], du_hat_dt[4]

    pp = np.dot(p, p)
    tau     = uc_hat + np.dot(ul_hat, p) + uq_hat * pp
    dtau_dt = duc     + np.dot(dul, p)   + duq     * pp

    grad     = ul_hat + 2.0 * uq_hat * p
    dgrad_dt = dul     + 2.0 * duq     * p
    grad_norm = np.linalg.norm(grad)

    if grad_norm > 1e-10:
        eta     = grad / grad_norm
        deta_dt = (dgrad_dt - eta * np.dot(eta, dgrad_dt)) / grad_norm
    else:
        eta     = np.zeros(3)
        deta_dt = np.zeros(3)

    kappa     = 2.0 * uq_hat
    dkappa_dt = 2.0 * duq

    return tau, eta, kappa, dtau_dt, deta_dt, dkappa_dt


def sphere_center_radius(u_hat, kappa_eps=1e-8):
    """
    Retrouve le centre c et le rayon r de la sphère algébrique associée
    à û, en complétant le carré de sa forme implicite :

        s_û(x) = ûq·||x||² + ûl·x + ûc = 0
        ⟺ ||x - c||² = r²,   c = -ûl/(2ûq),
                              r² = (||ûl||² - 4ûcûq) / (4ûq²)

    Grâce à la normalisation de Pratt, ||ûl||² - 4ûcûq = 1 exactement
    (c'est la quantité par laquelle on a divisé, Eq. 3), donc :

        r = 1 / (2|ûq|) = 1/|κ|     (κ = 2ûq, cohérent avec l'Eq. 4 :
                                      κ est bien l'inverse du rayon)

    """
    uq = u_hat[4]
    if abs(uq) < kappa_eps:
        return None

    ul = u_hat[1:4]
    center = -ul / (2.0 * uq)
    radius = 1.0 / (2.0 * abs(uq))
    return center, radius


# 5. FITNESS  
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


# 6. VARIATION GÉOMÉTRIQUE ν(p,t)  
def geometric_variation(dtau_dt, deta_dt, dkappa_dt, t):
    """
    Fonction de variation géométrique ν(p,t)  (Eq. 5) :

        ν(p,t) = (dτ/dt)² + (t·dη/dt)² + (t²·dκ/dt)²
    """
    deta_norm2 = float(np.dot(deta_dt, deta_dt))
    nu = dtau_dt ** 2 + (t ** 2) * deta_norm2 + (t ** 4) * (dkappa_dt ** 2)
    return float(nu)


# Appliquer le pipeline GLS complet pour un point p.

def gls_at_point(p, neighbors, normals, t, with_variation=False):
    if with_variation:
        u, du_dt = fit_algebraic_sphere_with_derivative(p, neighbors, normals, t)
        # Ajuster la sphère et calculer sa dérivée.
        if u is None:
            return None 

        u_hat, du_hat_dt = pratt_normalize_with_derivative(u, du_dt)
        # Normaliser la solution et sa dérivée.
        if u_hat is None:
            return None

        tau, eta, kappa, dtau_dt, deta_dt, dkappa_dt = \
            extract_descriptors_with_derivative(u_hat, du_hat_dt, p)
         # Extraire les descripteurs et leurs dérivées.
        nu  = geometric_variation(dtau_dt, deta_dt, dkappa_dt, t) # Calculer la variation géométrique.
        phi = compute_fitness(u, neighbors, normals, p, t) # Évaluer la qualité de l'ajustement.

        return {
            "tau"       : tau,
            "eta"       : eta,
            "kappa"     : kappa,
            "phi"       : phi,
            "dtau_dt"   : dtau_dt,
            "deta_dt"   : deta_dt,
            "dkappa_dt" : dkappa_dt,
            "nu"        : nu,
            "u"         : u,
            "u_hat"     : u_hat,
        }
    

    # Ajuster une sphère algébrique.
    u = fit_algebraic_sphere(p, neighbors, normals, t)
    if u is None:
        return None

    u_hat = pratt_normalize(u) # Normaliser la solution.
    if u_hat is None:
        return None

    tau, eta, kappa = extract_descriptors(u_hat, p) # Extraire les descripteurs géométriques.
    phi             = compute_fitness(u, neighbors, normals, p, t) # Calculer la qualité de l'ajustement.

    return {
        "tau"   : tau,
        "eta"   : eta,
        "kappa" : kappa,
        "phi"   : phi,
        "u"     : u,
        "u_hat" : u_hat,
    }