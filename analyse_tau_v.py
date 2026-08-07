# analyse_tau_v.py
# =========================================================
# Analyse des points à τ extrême sur la plus grande échelle :
#   1. recherche des 20 points de τ maximal à la plus grande échelle
#   2. tracé, sur un même graphe, des courbes (échelle, ν) de ces points
#      (v = ν(p,t), variation géométrique — Mellado et al. 2012, Eq. 5)
#   3. pour les points dont ν décroît de la plus grande à la plus
#      petite échelle, régression linéaire et affichage de la pente
# =========================================================
# Script autonome, séparé de main.py : il ne recalcule PAS le GLS,
# il lit uniquement le cache déjà produit dans notebooks/<obj>/.
# Lancez d'abord main.py pour l'objet choisi si le cache n'existe pas.
# =========================================================

import os
import sys
sys.path.append("src")

import numpy as np
import matplotlib.pyplot as plt

from notebooks import notebook_exists, load_results


N_TOP       = 20
DATA_FOLDER = "data"
OUTPUT_DIR  = "results"


def select_obj_name(data_folder=DATA_FOLDER):
    """Liste les fichiers .obj de data/ et laisse l'utilisateur choisir."""
    obj_files = sorted(f for f in os.listdir(data_folder)
                       if f.lower().endswith(".obj"))

    if not obj_files:
        raise FileNotFoundError(
            f"Aucun fichier .obj trouvé dans le dossier '{data_folder}'."
        )

    print("\nFichiers OBJ disponibles :")
    for i, fname in enumerate(obj_files, start=1):
        print(f"  {i}. {fname}")

    while True:
        try:
            choice = int(input("\nChoisissez un fichier (numéro) : "))
            if 1 <= choice <= len(obj_files):
                break
            print(f"  Numéro invalide (1 – {len(obj_files)}).")
        except ValueError:
            print("  Veuillez entrer un nombre entier.")

    selected = obj_files[choice - 1]
    return os.path.splitext(selected)[0]


def top_tau_points_at_largest_scale(TAU, n_top=N_TOP):
    """
    Retourne les indices des n_top points ayant les plus grandes
    valeurs de τ à la plus grande échelle (dernière colonne de TAU).
    """
    tau_last  = TAU[:, -1]
    idx_valid = np.where(~np.isnan(tau_last))[0]

    if len(idx_valid) < n_top:
        raise RuntimeError(
            f"Seulement {len(idx_valid)} points valides à la plus grande "
            f"échelle, impossible d'en extraire {n_top}."
        )

    order = np.argsort(tau_last[idx_valid])[::-1][:n_top]
    return idx_valid[order]


def is_decreasing_large_to_small(v, tol=0.0):
    """
    Teste si la courbe v(t) (v fourni à échelles t croissantes) est en
    décroissance quand on la relit de la plus grande à la plus petite
    échelle — ce qui équivaut à v croissant (au sens large) le long des
    échelles croissantes.

    Les NaN sont ignorés (au moins 3 valeurs valides requises).

    Retourne
    --------
    (decreasing, valid_mask) : bool, np.ndarray(bool)
    """
    valid_mask = ~np.isnan(v)
    if np.sum(valid_mask) < 3:
        return False, valid_mask

    diffs      = np.diff(v[valid_mask])
    decreasing = bool(np.all(diffs >= -tol))
    return decreasing, valid_mask


def plot_descriptor_curves(scales, data, top_idx, TAU, t_max, obj_name,
                           output_folder, ylabel, descriptor_name, file_tag):
    """
    Trace, sur un même graphe, les courbes (échelle, valeur) d'un
    descripteur (ν, τ ou κ) pour les points de top_idx, et sauvegarde
    la figure en PNG dans output_folder.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab20")

    for rank, i in enumerate(top_idx):
        ax.semilogx(scales, data[i, :], marker="o", markersize=3,
                    color=cmap(rank % 20),
                    label=f"p{i:06d} (τ={TAU[i, -1]:+.3f})")

    ax.set_xlabel("échelle t")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{obj_name} — {descriptor_name} pour les {N_TOP} points "
                 f"de τ maximal à t={t_max:.4f}")
    ax.legend(fontsize=6, ncol=2, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    fig_path = os.path.join(output_folder, f"{obj_name}_top{N_TOP}_{file_tag}_curves.png")
    fig.savefig(fig_path, dpi=150)
    print(f"\n[TAU-V] Graphe sauvegardé : {fig_path}")


def main():
    obj_name = select_obj_name()
    print(f"\n[INFO] Objet sélectionné : {obj_name}")

    if not notebook_exists(obj_name):
        raise RuntimeError(
            f"Aucun résultat en cache pour '{obj_name}'. "
            f"Lancez d'abord main.py pour cet objet (calcul GLS)."
        )

    results = load_results(obj_name)
    scales  = results["scales"]
    TAU     = results["TAU"]
    NU      = results["NU"]        # v = ν(p,t), variation géométrique
    KAPPA   = results["KAPPA"]

    t_max = scales[-1]
    print(f"\n[TAU-V] Plus grande échelle : t = {t_max:.4f} "
          f"(échelle {len(scales)}/{len(scales)})")

    top_idx = top_tau_points_at_largest_scale(TAU, N_TOP)

    print(f"\n[TAU-V] {N_TOP} points à τ maximal (t={t_max:.4f}) :")
    for rank, i in enumerate(top_idx, start=1):
        print(f"  {rank:2d}. p{i:06d}   τ = {TAU[i, -1]:+.6f}")

    # ---------------------------------------------------
    # Graphes (échelle, ν) / (échelle, τ) / (échelle, κ) pour les N_TOP points
    # ---------------------------------------------------
    output_folder = os.path.join(OUTPUT_DIR, obj_name, "tau_v_analysis")
    os.makedirs(output_folder, exist_ok=True)

    plot_descriptor_curves(scales, NU, top_idx, TAU, t_max, obj_name,
                           output_folder, ylabel="ν(p,t)  (v)",
                           descriptor_name="ν(p,t)", file_tag="nu")

    plot_descriptor_curves(scales, TAU, top_idx, TAU, t_max, obj_name,
                           output_folder, ylabel="τ(p,t)",
                           descriptor_name="τ(p,t)", file_tag="tau")

    plot_descriptor_curves(scales, KAPPA, top_idx, TAU, t_max, obj_name,
                           output_folder, ylabel="κ(p,t)",
                           descriptor_name="κ(p,t)", file_tag="kappa")

    # ---------------------------------------------------
    # Décroissance grande → petite échelle + régression linéaire
    # ---------------------------------------------------
    print(f"\n[TAU-V] Analyse de décroissance (grande → petite échelle) :")

    header = (f"{'rang':>4} {'point':>8} {'tau(t_max)':>12} "
              f"{'decroissant':>12} {'pente (a)':>14}")
    report_lines = [
        f"# Analyse tau/nu — {obj_name}\n",
        f"# {N_TOP} points de tau maximal a la plus grande echelle (t={t_max:.4f})\n",
        f"# v = nu(p,t), variation geometrique (Mellado et al. 2012, Eq. 5)\n",
        f"# decroissance testee de la plus grande a la plus petite echelle\n\n",
        header + "\n",
        "-" * len(header) + "\n",
    ]

    n_decreasing = 0
    for rank, i in enumerate(top_idx, start=1):
        v = NU[i, :]
        decreasing, valid_mask = is_decreasing_large_to_small(v)

        slope_str = "—"
        if decreasing:
            n_decreasing += 1
            slope, _ = np.polyfit(scales[valid_mask], v[valid_mask], 1)
            slope_str = f"{slope:+.6e}"
            print(f"  p{i:06d}  τ={TAU[i, -1]:+.6f}  décroissant  a={slope:+.6e}")
        else:
            print(f"  p{i:06d}  τ={TAU[i, -1]:+.6f}  non-décroissant")

        report_lines.append(
            f"{rank:>4d} p{i:06d} {TAU[i, -1]:>+12.6f} "
            f"{'oui' if decreasing else 'non':>12} {slope_str:>14}\n"
        )

    report_lines.append(
        f"\n{n_decreasing}/{N_TOP} points montrent une decroissance de nu "
        f"de la plus grande a la plus petite echelle.\n"
    )

    report_path = os.path.join(output_folder, f"{obj_name}_top{N_TOP}_tau_v_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"\n[TAU-V] Rapport texte sauvegardé : {report_path}")

    # ---------------------------------------------------
    # Interprétation automatique (indicative)
    # ---------------------------------------------------
    print("\n[INTERPRÉTATION]")
    print(f"  {n_decreasing}/{N_TOP} des points de τ maximal ont une "
          f"variation géométrique ν qui décroît de la plus grande à la "
          f"plus petite échelle.")
    if n_decreasing > N_TOP / 2:
        print("  → Majoritaire : pour ces points extrêmes en τ, ν diminue "
              "quand l'échelle diminue — la structure locale devient plus "
              "stable/prévisible à petite échelle, cohérent avec des "
              "points situés sur des structures saillantes (bords, "
              "pointes) bien définies localement.")
    else:
        print("  → Minoritaire : la plupart des points de τ maximal ont un "
              "comportement de ν non monotone (bruit local, ou structure "
              "qui change de nature selon l'échelle) — le τ élevé mesuré "
              "à la plus grande échelle seule ne suffit pas à garantir "
              "une structure stable à toutes les échelles.")

    plt.show()

if __name__ == "__main__":
    main()