#!/usr/bin/env python3
import os
import numpy as np
from code_analysis.ASTReader    import ASTReader
from code_analysis.ASTFragmentation import AST_fragmentation

def main():
    reader   = ASTReader()
    ast_root = 'ast'  # dossier racine des .ast.json.gz

    # --- Chargement et filtrage des AST entiers (>100 nœuds) ---
    vec_records = []  # (chemin, vecteur, kit)
    for root, _, files in os.walk(ast_root):
        for fname in files:
            if not fname.endswith('.ast.json.gz'):
                continue
            path = os.path.join(root, fname)
            ast  = reader.read_ast(path)
            vec  = ast.vectorize()
            if vec.sum() <= 100:
                continue
            kit = os.path.relpath(path, ast_root).split(os.sep)[0]
            vec_records.append((path, vec, kit))

    # --- 1. Clones paramétriques sur AST complets ---
    groups     = {}  # vecteur → [chemins]
    kit_groups = {}  # vecteur → {kits}
    for path, vec, kit in vec_records:
        key = tuple(vec.tolist())
        groups.setdefault(key, []).append(path)
        kit_groups.setdefault(key, set()).add(kit)

    # 1.1 Plus gros groupe de fichiers identiques
    largest_key = max(groups, key=lambda k: len(groups[k]))
    files_largest = groups[largest_key]
    kits_largest  = sorted({p.split(os.sep)[1] for p in files_largest})

    print("\n=== 1.1 Clones paramétriques (AST identiques) ===")
    print(f"  • {len(files_largest)} fichiers")
    print(f"  • Kits : {kits_largest}")
    for f in files_largest[:5]:
        print("    -", f)

    # 1.2 En ignorant duplications intra-kit
    largest_key_kits = max(kit_groups, key=lambda k: len(kit_groups[k]))
    kits_unique      = sorted(kit_groups[largest_key_kits])
    example_per_kit  = {}
    for p in groups[largest_key_kits]:
        kit = p.split(os.sep)[1]
        example_per_kit.setdefault(kit, p)

    print("\n=== 1.2 Clones paramétriques (inter-kit) ===")
    print(f"  • {len(kits_unique)} kits → {kits_unique}")
    for kit, p in example_per_kit.items():
        print(f"    [{kit}] {p}")

    # --- 2.1 Clones paramétriques sur fragments de fonctions (>10 nœuds) ---
    frag_groups     = {}  # vecteur_fragment → [(chemin, node_id)]
    frag_kit_groups = {}  # vecteur_fragment → {kits}

    for path, _, kit in vec_records:
        ast = reader.read_ast(path)
        fragments = AST_fragmentation(ast)
        for node in fragments:
            vec_f = ast.vectorize(node)
            if vec_f.sum() <= 10:
                continue
            key = tuple(vec_f.tolist())
            frag_groups.setdefault(key, []).append((path, node))
            frag_kit_groups.setdefault(key, set()).add(kit)

    # Ignorer multiples occurences within le même kit
    best_frag_key = max(frag_kit_groups, key=lambda k: len(frag_kit_groups[k]))
    kits_best     = sorted(frag_kit_groups[best_frag_key])
    occ_count     = len(kits_best)

    # Récupérer le(s) nom(s) de fonction pour ce fragment
    func_names = set()
    for path, node in frag_groups[best_frag_key]:
        ast = reader.read_ast(path)
        func_names.add(ast.get_image(node))

    print("\n=== 2.1 Clones paramétriques (fragments) ===")
    print(f"  • Fragments >10 nœuds dupliqués dans {occ_count} kits : {kits_best}")
    print(f"  • Fonction(s) concernée(s) : {sorted(func_names)}")
    print("  • Exemple de chemins + nœud :")
    for path, node in frag_groups[best_frag_key][:5]:
        print(f"    - {path} (node {node})")

    # --- 3. Clones similaires (AST entiers, Manhattan ≤30%) ---
    threshold = 0.3
    best_sim_group = []
    for i, (pi, vi, _) in enumerate(vec_records):
        group = [pi]
        size = vi.sum()
        for pj, vj, _ in vec_records[i+1:]:
            if np.abs(vj - vi).sum() <= threshold * size:
                group.append(pj)
        if len(group) > len(best_sim_group):
            best_sim_group = group

    kits_sim = sorted({p.split(os.sep)[1] for p in best_sim_group})

    print("\n=== 3. Clones similaires (AST): Manhattan ≤ 30% ===")
    print(f"  • {len(best_sim_group)} fichiers dans {len(kits_sim)} kits : {kits_sim}")
    for f in best_sim_group[:5]:
        print("    -", f)

if __name__ == "__main__":
    main()
