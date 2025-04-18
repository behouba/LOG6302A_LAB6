#!/usr/bin/env python3
"""
Script d'analyse de similarité des vecteurs de métriques AST.
- Parcourt tous les fichiers AST (.ast.json et .ast.json.gz) sous le dossier "ast".
- Extrait les vecteurs via ASTReader, ne conserve que ceux de plus de 100 nœuds.
- Calcule pour chaque vecteur le groupe de fichiers dont la distance de Manhattan
  au vecteur de référence est ≤ 30% de la taille de ce vecteur.
- Affiche les détails du plus grand groupe.

Usage:
    python3 similarity_analysis.py

Assurez-vous que le module code_analysis est importable (chemin depuis la racine du projet).
"""
import os
import sys
import numpy as np

def find_ast_files(ast_root):
    """Retourne la liste des chemins vers tous les fichiers .ast.json(.gz) sous ast_root."""
    ast_files = []
    for root, _, files in os.walk(ast_root):
        for fname in files:
            if fname.endswith('.ast.json') or fname.endswith('.ast.json.gz'):
                ast_files.append(os.path.join(root, fname))
    return ast_files


def load_vectors(ast_files, min_nodes=100):
    """Lit chaque AST, vectorise et ne garde que les vecteurs de taille ≥ min_nodes."""
    from code_analysis.ASTReader import ASTReader
    reader = ASTReader()
    valid_files = []
    vectors = []
    sizes = []
    for path in ast_files:
        try:
            ast = reader.read_ast(path)
            vec = ast.vectorize()
            total = int(vec.sum())
            if total > min_nodes:
                valid_files.append(path)
                vectors.append(vec)
                sizes.append(total)
        except Exception as e:
            print(f"[Warning] Échec lecture AST {path}: {e}")
    return valid_files, vectors, sizes


def find_largest_similar_group(files, vectors, sizes, threshold_ratio=0.3):
    """Retourne l'indice et la liste des indices du plus grand groupe de similarité."""
    n = len(files)
    best_size = 0
    best_center = -1
    best_group = []
    for i in range(n):
        center = vectors[i]
        limit = threshold_ratio * sizes[i]
        group = []
        for j in range(n):
            dist = np.abs(vectors[j] - center).sum()
            if dist <= limit:
                group.append(j)
        if len(group) > best_size:
            best_size = len(group)
            best_center = i
            best_group = group
    return best_center, best_group


def main():
    # Détermine les chemins
    project_root = os.path.dirname(os.path.abspath(__file__))
    ast_root = os.path.join(project_root, 'ast')

    if not os.path.isdir(ast_root):
        print(f"Erreur: dossier AST non trouvé sous {ast_root}")
        sys.exit(1)

    print("Collecte des fichiers AST...")
    ast_files = find_ast_files(ast_root)
    print(f"  {len(ast_files)} fichiers AST trouvés.")

    print("Chargement et vectorisation (AST > 100 nœuds)...")
    files, vectors, sizes = load_vectors(ast_files, min_nodes=100)
    print(f"  {len(files)} vecteurs conservés (>100 nœuds).")

    print("Recherche du plus grand groupe de similarité...")
    center_idx, group_idx = find_largest_similar_group(files, vectors, sizes, threshold_ratio=0.3)
    print(f"Résultat: groupe de taille {len(group_idx)}")
    print(f"Vecteur de référence: fichier #{center_idx} -> {files[center_idx]}")
    print("Exemples de fichiers dans le groupe:")
    for idx in group_idx[:10]:
        print(f"  - {files[idx]}")

if __name__ == '__main__':
    main()
