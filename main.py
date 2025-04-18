#!/usr/bin/env python3
import os
import numpy as np
from collections import defaultdict

from code_analysis.ASTReader import ASTReader
from code_analysis.ASTFragmentation import AST_fragmentation

FILE_NODE_THRESHOLD     = 100
FRAG_NODE_THRESHOLD     = 10
SIM_FILE_RATIO          = 0.30  # 30 %
SIM_FRAG_RATIO          = 0.10  # 10 %

AST_PATH = "ast"  # dossier racine des AST .ast.json.gz


def load_file_vectors(reader):
    """
    Lit tous les fichiers AST, vectorise chaque AST complet et ne garde
    que ceux dont le nombre de nœuds > FILE_NODE_THRESHOLD.
    Retourne une liste de tuples : (chemin, vecteur numpy, kit).
    """
    records = []
    for root, _, files in os.walk(AST_PATH):
        for fname in files:
            if not fname.endswith(".ast.json.gz"):
                continue
            path = os.path.join(root, fname)
            ast = reader.read_ast(path)
            vec = ast.vectorize()
            if vec.sum() <= FILE_NODE_THRESHOLD:
                continue
            kit = os.path.relpath(path, AST_PATH).split(os.sep)[0]
            records.append((path, vec, kit))
    return records


def analyze_parametric_file_clones(records):
    """
    1.1 Clones “paramétriques” sur fichiers :
      - plus gros groupe de vecteurs identiques (> FILE_NODE_THRESHOLD)
      - même opération en ignorant duplications intra-kit
    """
    groups = defaultdict(list)
    kits_per_vec = defaultdict(set)

    for path, vec, kit in records:
        key = tuple(vec.tolist())
        groups[key].append(path)
        kits_per_vec[key].add(kit)

    # trouvr plus gros groupe de fichiers identiques
    key_max_files = max(groups, key=lambda k: len(groups[k]))
    files_max = groups[key_max_files]
    kits_max = sorted(kits_per_vec[key_max_files])

    print("\n=== 1.1.a Clones paramétriques (identiques) ===")
    print(f"Nombre de fichiers : {len(files_max)}")
    print(f"Kits concernés      : {kits_max}")
    print("Exemples de fichiers :")
    for f in files_max[:5]:
        print("  -", f)

    # en ignorant duplications intra-kit
    # et on compte 1 occurrence par kit
    max_kits_vec = max(kits_per_vec, key=lambda k: len(kits_per_vec[k]))
    kits_unique = sorted(kits_per_vec[max_kits_vec])
    example_per_kit = {}
    for f in groups[max_kits_vec]:
        kit = os.path.relpath(f, AST_PATH).split(os.sep)[0]
        example_per_kit.setdefault(kit, f)

    print("\n=== 1.1.b Clones paramétriques (inter-kit) ===")
    print(f"Kits distincts : {len(kits_unique)} → {kits_unique}")
    print("Un exemple de fichier par kit :")
    for kit, f in example_per_kit.items():
        print(f"  [{kit}] {f}")


def analyze_similar_file_clones(records):
    """
    1.2 Fichiers similaires :
      - plus gros groupe de fichiers similaires (Manhattan ≤ SIM_FILE_RATIO * ||vi||)
    """
    best_group = []
    for i, (pi, vi, ki) in enumerate(records):
        group = [(pi, ki)]
        norm_i = vi.sum()
        for pj, vj, kj in records[i+1:]:
            dist = np.abs(vj - vi).sum()
            if dist <= SIM_FILE_RATIO * norm_i:
                group.append((pj, kj))
        if len(group) > len(best_group):
            best_group = group

    kits_sim = sorted({kit for _, kit in best_group})
    print("\n=== 1.2 Fichiers similaires (Manhattan ≤ 30 %) ===")
    print(f"Nombre de fichiers dans le plus grand groupe : {len(best_group)}")
    print(f"Kits concernés                             : {kits_sim}")
    print("Exemples de fichiers :")
    for f, _ in best_group[:5]:
        print("  -", f)


def load_fragments(reader):
    """
    Parcourt chaque fichier AST, extrait ses fragments (nœuds de type fonction),
    vectorise chaque fragment (subtree) et ne garde que ceux de taille > FRAG_NODE_THRESHOLD.
    Retourne liste de tuples : (chemin_fichier, fragment_node_id, vecteur, kit).
    """
    records = []
    for root, _, files in os.walk(AST_PATH):
        for fname in files:
            if not fname.endswith(".ast.json.gz"):
                continue
            path = os.path.join(root, fname)
            ast = reader.read_ast(path)
            fragments = AST_fragmentation(ast)
            kit = os.path.relpath(path, AST_PATH).split(os.sep)[0]
            for node in fragments:
                vec = ast.vectorize(node=node)
                if vec.sum() <= FRAG_NODE_THRESHOLD:
                    continue
                records.append((path, node, vec, kit, ast.get_image(node)))
    return records


def analyze_parametric_fragment_clones(records):
    """
    2.1 Clones paramétriques sur fragments :
      - trouver fragment le plus dupliqué (> FRAG_NODE_THRESHOLD),
        en ignorant duplications intra-kit
      - afficher le(s) nom(s) de fonction (image du nœud).
    """
    vec_to_records = defaultdict(list)
    kits_per_vec = defaultdict(set)

    for path, node, vec, kit, img in records:
        key = tuple(vec.tolist())
        vec_to_records[key].append((path, node, img, kit))
        kits_per_vec[key].add(kit)

    # On compte qu'une occurrence par kit
    key_max_kits = max(kits_per_vec, key=lambda k: len(kits_per_vec[k]))
    recs = vec_to_records[key_max_kits]
    kits = sorted(kits_per_vec[key_max_kits])

    print("\n=== 2.1 Clones paramétriques (fragments) ===")
    print(f"Nombre de kits distincts pour ce fragment : {len(kits)} → {kits}")
    print("Fonctions (image du nœud) correspondant à ce fragment :")
    for _, _, img, _ in recs:
        print("  -", img)


def analyze_similar_fragment_clones(records):
    """
    2.2 Fragments similaires :
      - plus gros groupe de fragments similaires (Manhattan ≤ SIM_FRAG_RATIO * ||vi||),
        en ignorant duplications intra-kit
      - afficher nom(s) de fonction (image du nœud)
    """
    best_group = []
    best_kits  = set()

    for i, (pi, ni, vi, ki, img_i) in enumerate(records):
        group = [(pi, ni, img_i, ki)]
        kits_seen = {ki}
        norm_i = vi.sum()
        for pj, nj, vj, kj, img_j in records[i+1:]:
            if kj in kits_seen:
                continue
            if np.abs(vj - vi).sum() <= SIM_FRAG_RATIO * norm_i:
                group.append((pj, nj, img_j, kj))
                kits_seen.add(kj)
        if len(group) > len(best_group):
            best_group = group
            best_kits  = kits_seen

    print("\n=== 2.2 Fragments similaires (Manhattan ≤ 10 %) ===")
    print(f"Nombre de fragments (kits distincts) : {len(best_group)} → {sorted(best_kits)}")
    print("Fonctions correspondant à ces fragments :")
    for _, _, img, _ in best_group:
        print("  -", img)


def analyze_kit_parametric_clones(fragment_records):
    """
    3 Kits “paramétriques” :
      - Pour chaque kit, sommer (colonne à colonne) tous les vecteurs de ses fragments
      - Trouver le plus gros groupe de kits ayant un même vecteur
      - Afficher ces kits et comparer, le cas échéant, leurs distributions de types de nœuds
    """
    kit_vecs = defaultdict(lambda: None)

    for path, node, vec, kit, _ in fragment_records:
        if kit_vecs[kit] is None:
            kit_vecs[kit] = vec.copy()
        else:
            kit_vecs[kit] += vec

    # grouper par vecteur
    vec_to_kits = defaultdict(list)
    for kit, vec in kit_vecs.items():
        key = tuple(vec.tolist())
        vec_to_kits[key].append(kit)

    # plus gros groupe de kits identiques
    key_max = max(vec_to_kits, key=lambda k: len(vec_to_kits[k]))
    kits_max = sorted(vec_to_kits[key_max])

    print("\n=== 3 Kits paramétriques ===")
    print(f"Plus gros groupe de kits : {len(kits_max)} → {kits_max}")

    # comparaison type par type
    print("\nDistribution des types de nœuds (par colonne) pour ces kits (identiques) :")
    print("→ Tous ces kits ont exactement la même distribution de types (puisque leurs vecteurs sont identiques).")


def main():
    reader = ASTReader()

    # 1 Fichiers
    file_records = load_file_vectors(reader)
    analyze_parametric_file_clones(file_records)
    analyze_similar_file_clones(file_records)

    # 2 Fragments
    frag_records = load_fragments(reader)
    analyze_parametric_fragment_clones(frag_records)
    analyze_similar_fragment_clones(frag_records)

    # 3 Kits
    analyze_kit_parametric_clones(frag_records)


if __name__ == "__main__":
    main()
