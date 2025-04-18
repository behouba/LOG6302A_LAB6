# -*- coding: utf-8 -*-

import os
import sys
from collections import defaultdict

# Assurez-vous que le dossier parent est dans le PYTHONPATH
# pour que "import code_analysis" fonctionne :
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from code_analysis import ASTReader

def find_ast_files(ast_root):
    """Recursively collect tous les fichiers .ast.json(.gz) sous ast_root."""
    ast_files = []
    for dirpath, _, filenames in os.walk(ast_root):
        for fn in filenames:
            if fn.endswith('.ast.json') or fn.endswith('.ast.json.gz'):
                ast_files.append(os.path.join(dirpath, fn))
    return ast_files

def group_by_vector(ast_files, min_nodes=100):
    """
    Lit chaque AST, vectorise, et groupe les chemins de fichiers par
    vecteur (tuple). Ne garde que les AST dont le nombre de nœuds > min_nodes.
    """
    reader = ASTReader()
    groups = defaultdict(list)
    node_counts = {}

    for path in ast_files:
        try:
            ast = reader.read_ast(path)
            vec = ast.vectorize()
            total = int(vec.sum())
            if total <= min_nodes:
                continue
            key = tuple(vec.tolist())
            groups[key].append(path)
            node_counts[key] = total
        except Exception as e:
            print(f"⚠️ Erreur lecture AST {path}: {e}", file=sys.stderr)

    return groups, node_counts

def largest_within_group(groups, node_counts):
    """Retourne le vecteur qui a le plus de fichiers (clones intra‑kit)."""
    best = max(groups.keys(), key=lambda k: len(groups[k]))
    return best, groups[best], node_counts[best]

def largest_across_kits(groups):
    """
    Calcule, pour chaque vecteur, le nombre de kits distincts qui partagent ce vecteur.
    Retourne le vecteur maximisant cette valeur (clones inter‑kit).
    """
    vec_to_kits = {}
    for vec, paths in groups.items():
        kits = set(os.path.relpath(p, 'ast').split(os.sep)[0] for p in paths)
        vec_to_kits[vec] = kits

    best = max(vec_to_kits.keys(), key=lambda v: len(vec_to_kits[v]))
    return best, vec_to_kits[best]

def main():
    ast_root = 'ast'  # ou chemin absolu si besoin
    print(f"🔍 Recherche des fichiers AST sous « {ast_root} »…")
    ast_files = find_ast_files(ast_root)
    print(f"Total AST trouvés : {len(ast_files)}")

    print("🔢 Groupement par vecteur de métriques (nœuds > 100)…")
    groups, node_counts = group_by_vector(ast_files, min_nodes=100)
    print(f"Vecteurs distincts conservés : {len(groups)}")

    # Intra‑kit
    vec1, files1, count1 = largest_within_group(groups, node_counts)
    print("\n🟢 Plus grand groupe INTRA‑KIT")
    print(f" • Taille du groupe : {len(files1)} fichiers")
    print(f" • Nombre de nœuds : {count1}")
    print(" • Exemples de chemins :")
    for p in files1[:5]:
        print("    -", p)

    # Inter‑kit
    vec2, kits2 = largest_across_kits(groups)
    print("\n🔵 Plus grand groupe INTER‑KIT")
    print(f" • Nombre de kits : {len(kits2)}")
    print(" • Kits concernés :", ", ".join(list(kits2)[:10]), ("…" if len(kits2)>10 else ""))
    print(f" • Nombre de nœuds : {node_counts[vec2]}")
    print(" • Exemple de chemin :")
    print("    -", groups[vec2][0])

if __name__ == '__main__':
    main()
