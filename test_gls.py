# test_gls.py
# =========================================================
# Tests de validation du module GLS
# Chaque test vérifie une propriété analytique connue
# =========================================================

import sys
sys.path.append("src")

import numpy as np
from gls import gls_at_point, wendland_weights

# Constantes de tolérance
TOL_EXACT  = 1e-5   # pour les cas analytiquement parfaits
TOL_APPROX = 0.15   # pour les cas avec échantillonnage fini

PASS = 0
FAIL = 0

def check(name, condition, got, expected, tol=None):
    global PASS, FAIL
    tol_str = f"  (tol={tol})" if tol else ""
    if condition:
        print(f"  ✓  {name}")
        print(f"       got={got}  expected={expected}{tol_str}")
        PASS += 1
    else:
        print(f"  ✗  {name}  ← ECHEC")
        print(f"       got={got}  expected={expected}{tol_str}")
        FAIL += 1


# SETUP : générateur de sphère analytique
def make_sphere(R, n=1000, seed=42):
    np.random.seed(seed)
    theta = np.random.uniform(0, np.pi,   n)
    phi   = np.random.uniform(0, 2*np.pi, n)
    pts   = np.column_stack([
        R * np.sin(theta) * np.cos(phi),
        R * np.sin(theta) * np.sin(phi),
        R * np.cos(theta)
    ])
    nrms  = pts / R   # normales vers l'extérieur
    return pts, nrms

def make_plane(n=300, seed=0):
    np.random.seed(seed)
    xy   = np.random.uniform(-2, 2, (n, 2))
    pts  = np.column_stack([xy, np.zeros(n)])
    nrms = np.tile([0., 0., 1.], (n, 1))
    return pts, nrms

def get_neighbors(pts, p, t):
    d    = np.linalg.norm(pts - p, axis=1)
    mask = d <= t
    return pts[mask], mask

# TEST 1 : Poids
print("\n" + "="*55)
print("TEST 1 — Equation du Poids (Eq. 2)")
print("="*55)

p = np.zeros(3)
t = 1.0

# Point au bord → w doit être exactement 0
q_bord = np.array([[1., 0., 0.]])
w_bord = wendland_weights(q_bord, p, t)
check("w = 0 au bord (||q-p|| = t)", w_bord[0] == 0.0,
      got=w_bord[0], expected=0.0)

# Point au centre → w doit être maximal (= 1)
q_centre = np.array([[0., 0., 0.]])
w_centre = wendland_weights(q_centre, p, t)
check("w = 1 au centre (||q-p|| = 0)", abs(w_centre[0] - 1.0) < TOL_EXACT,
      got=round(w_centre[0], 6), expected=1.0, tol=TOL_EXACT)

# Poids toujours positifs
pts_rand = np.random.uniform(-0.9, 0.9, (50, 3))
w_rand   = wendland_weights(pts_rand, p, t)
check("Tous les poids >= 0", np.all(w_rand >= 0),
      got=f"min={w_rand.min():.4f}", expected=">= 0")

# Poids décroissants avec la distance
d1 = np.array([[0.3, 0., 0.]]); d2 = np.array([[0.7, 0., 0.]])
w1 = wendland_weights(d1, p, t)[0]
w2 = wendland_weights(d2, p, t)[0]
check("Poids décroissant avec la distance", w1 > w2,
      got=f"w(0.3)={w1:.4f} > w(0.7)={w2:.4f}", expected="w(0.3) > w(0.7)")


# TEST 2 : PLAN PARFAIT  (κ = 0)
print("\n" + "="*55)
print("TEST 2 — Plan parfait z=0")
print("  Attendu : κ=0, η=[0,0,1], τ=0, ϕ=1")
print("="*55)

pts_pl, nrms_pl = make_plane(n=400)
p_pl = np.zeros(3)
t_pl = 2.5

r = gls_at_point(p_pl, pts_pl, nrms_pl, t_pl)

if r is None:
    print("  ✗  gls_at_point a retourné None ← ECHEC")
    FAIL += 1
else:
    check("κ = 0 (plan, courbure nulle)",
          abs(r["kappa"]) < TOL_EXACT,
          got=f"{r['kappa']:.8f}", expected=0.0, tol=TOL_EXACT)

    check("η = [0,0,1] (normale au plan)",
          abs(r["eta"][2] - 1.0) < TOL_EXACT,
          got=np.round(r["eta"], 5), expected=[0,0,1], tol=TOL_EXACT)

    check("τ = 0 (p sur le plan)",
          abs(r["tau"]) < TOL_EXACT,
          got=f"{r['tau']:.8f}", expected=0.0, tol=TOL_EXACT)

    check("ϕ = 1 (fit parfait)",
          abs(r["phi"] - 1.0) < TOL_EXACT,
          got=f"{r['phi']:.6f}", expected=1.0, tol=TOL_EXACT)


# TEST 3 : SPHÈRE — SIGNE DE κ
print("\n" + "="*55)
print("TEST 3 — Sphère : signe de κ")
print("  Convexe (normales ext.) → κ > 0")
print("  Concave (normales int.) → κ < 0")
print("="*55)

R    = 2.0
pts_sph, nrms_sph = make_sphere(R, n=2000)
p_sph = np.array([R, 0., 0.])
t_sph = 1.2

nb_pts, mask = get_neighbors(pts_sph, p_sph, t_sph)
nb_nrms      = nrms_sph[mask]

r_conv = gls_at_point(p_sph, nb_pts, nb_nrms,   t_sph)
r_conc = gls_at_point(p_sph, nb_pts, -nb_nrms,  t_sph)

if r_conv and r_conc:
    check("κ > 0 pour surface convexe",
          r_conv["kappa"] > 0,
          got=f"{r_conv['kappa']:.4f}", expected="> 0")

    check("κ < 0 pour surface concave (normales inversées)",
          r_conc["kappa"] < 0,
          got=f"{r_conc['kappa']:.4f}", expected="< 0")

    check("κ symétrique : κ_conv = -κ_conc",
          abs(r_conv["kappa"] + r_conc["kappa"]) < TOL_EXACT,
          got=f"{r_conv['kappa']:.4f} vs {-r_conc['kappa']:.4f}", expected="symétriques")


# TEST 4 : SPHÈRE — SIGNE DE τ
print("\n" + "="*55)
print("TEST 4 — Sphère : signe de τ (position relative)")
print("  p sur la surface → τ ≈ 0")
print("  p à l'extérieur  → τ > 0")
print("  p à l'intérieur  → τ < 0")
print("="*55)

# p SUR la surface (R, 0, 0)
nb_pts, mask = get_neighbors(pts_sph, p_sph, t_sph)
r_on  = gls_at_point(p_sph, nb_pts, nrms_sph[mask], t_sph)

# p À L'EXTÉRIEUR
p_out = np.array([R + 0.4, 0., 0.])
nb_out, mask_out = get_neighbors(pts_sph, p_out, t_sph)
r_out = gls_at_point(p_out, nb_out, nrms_sph[mask_out], t_sph)

# p À L'INTÉRIEUR
p_in  = np.array([R - 0.4, 0., 0.])
nb_in, mask_in = get_neighbors(pts_sph, p_in, t_sph)
r_in  = gls_at_point(p_in, nb_in, nrms_sph[mask_in], t_sph)

if r_on:
    check("τ ≈ 0 sur la surface",
          abs(r_on["tau"]) < TOL_APPROX,
          got=f"{r_on['tau']:.4f}", expected="≈ 0", tol=TOL_APPROX)

if r_out:
    check("τ > 0 à l'extérieur",
          r_out["tau"] > 0,
          got=f"{r_out['tau']:.4f}", expected="> 0")

if r_in:
    check("τ < 0 à l'intérieur",
          r_in["tau"] < 0,
          got=f"{r_in['tau']:.4f}", expected="< 0")

# TEST 5 : SPHÈRE — DIRECTION DE η
print("\n" + "="*55)
print("TEST 5 — Sphère : direction de η")
print("  Pour p=(R,0,0) → η doit pointer vers [1,0,0]")
print("="*55)

nb_pts, mask = get_neighbors(pts_sph, p_sph, t_sph)
r_eta = gls_at_point(p_sph, nb_pts, nrms_sph[mask], t_sph)

if r_eta:
    check("η[0] ≈ 1 (composante x dominante)",
          r_eta["eta"][0] > 0.98,
          got=np.round(r_eta["eta"], 4), expected="≈ [1,0,0]", tol=0.02)

    check("η est unitaire (||η|| = 1)",
          abs(np.linalg.norm(r_eta["eta"]) - 1.0) < TOL_EXACT,
          got=f"{np.linalg.norm(r_eta['eta']):.8f}", expected=1.0)


# TEST 6 : FITNESS ϕ — propre vs bruité
print("\n" + "="*55)
print("TEST 6 — Fitness ϕ")
print("  Surface propre  → ϕ ≈ 1")
print("  Normales bruitées → ϕ < ϕ_propre")
print("="*55)

nb_pts, mask = get_neighbors(pts_sph, p_sph, t_sph)
nrms_clean   = nrms_sph[mask]

# Normales bruitées
np.random.seed(99)
nrms_noisy = nrms_clean + 0.3 * np.random.randn(*nrms_clean.shape)
nrms_noisy /= np.linalg.norm(nrms_noisy, axis=1, keepdims=True)

r_clean = gls_at_point(p_sph, nb_pts, nrms_clean, t_sph)
r_noisy = gls_at_point(p_sph, nb_pts, nrms_noisy, t_sph)

if r_clean and r_noisy:
    check("ϕ = 1 sur surface propre",
          abs(r_clean["phi"] - 1.0) < TOL_EXACT,
          got=f"{r_clean['phi']:.6f}", expected=1.0)

    check("ϕ < 1 sur normales bruitées",
          r_noisy["phi"] < 1.0,
          got=f"{r_noisy['phi']:.4f}", expected="< 1.0")

    check("ϕ_propre > ϕ_bruité",
          r_clean["phi"] > r_noisy["phi"],
          got=f"{r_clean['phi']:.4f} > {r_noisy['phi']:.4f}", expected="propre > bruité")


# TEST 7 : ROBUSTESSE — minimum de voisins
print("\n" + "="*55)
print("TEST 7 — Robustesse avec trop peu de voisins")
print("  K < 6 → retourner None (pas de crash)")
print("="*55)

pts_few  = np.random.randn(4, 3)
nrms_few = np.random.randn(4, 3)
nrms_few /= np.linalg.norm(nrms_few, axis=1, keepdims=True)

r_few = gls_at_point(np.zeros(3), pts_few, nrms_few, 5.0)
check("K=4 → None (pas de crash)",
      r_few is None,
      got="None" if r_few is None else "dict", expected="None")


# TEST 8 : COHÉRENCE MULTI-ÉCHELLE
print("\n" + "="*55)
print("TEST 8 — Cohérence multi-échelle sur plan")
print("  κ doit rester = 0 à toutes les échelles")
print("  ϕ doit rester = 1 à toutes les échelles")
print("="*55)

pts_big, nrms_big = make_plane(n=2000)
p_center = np.zeros(3)

all_ok_kappa = True
all_ok_phi   = True

for t_val in [0.5, 1.0, 1.5, 2.0]:
    nb, m = get_neighbors(pts_big, p_center, t_val)
    if len(nb) < 6:
        continue
    res = gls_at_point(p_center, nb, nrms_big[m], t_val)
    if res:
        if abs(res["kappa"]) > TOL_EXACT:
            all_ok_kappa = False
            print(f"    t={t_val}  κ={res['kappa']:.8f}  ← ECHEC")
        if abs(res["phi"] - 1.0) > TOL_EXACT:
            all_ok_phi = False
            print(f"    t={t_val}  ϕ={res['phi']:.8f}  ← ECHEC")
        else:
            print(f"    t={t_val}  K={len(nb):4d}  κ={res['kappa']:.8f}  ϕ={res['phi']:.6f}")

check("κ = 0 à toutes les échelles (plan)", all_ok_kappa,
      got="OK" if all_ok_kappa else "ECHEC", expected="κ=0 partout")
check("ϕ = 1 à toutes les échelles (plan)", all_ok_phi,
      got="OK" if all_ok_phi else "ECHEC", expected="ϕ=1 partout")

# BILAN
print("\n" + "="*55)
total = PASS + FAIL
print(f"BILAN : {PASS}/{total} tests passés", end="")
if FAIL == 0:
    print("  ← TOUS CORRECTS ✓")
else:
    print(f"  ← {FAIL} ECHEC(S) ✗")
print("="*55 + "\n")