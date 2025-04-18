#!/usr/bin/env python3
"""
Analyse des kits « paramétriques » avec logs de progression :
1. Calcule pour chaque kit un vecteur représentant le kit,
   en sommant tous les fragments (>10 nœuds) qui le composent.
2. Cherche le plus grand groupe de vecteurs identiques entre kits.
3. Affiche les kits concernés et compare leurs types/images AST.

Usage:
    python3 kits_parametriques.py
"""
import os
import sys
import numpy as np

# Se fixe le seuil minimum de nœuds par fragment
MIN_NODES = 10

def find_ast_files(ast_root):
    """Retourne la liste des chemins vers tous les fichiers AST (.ast.json/.ast.json.gz)."""
    ast_files = []
    for root, _, files in os.walk(ast_root):
        for fname in files:
            if fname.endswith('.ast.json') or fname.endswith('.ast.json.gz'):
                ast_files.append(os.path.join(root, fname))
    return ast_files


def extract_fragments(ast_files, min_nodes=MIN_NODES):
    """
    Lit chaque AST, isole les fragments de fonctions/méthodes,
    vectorise chaque fragment, et collecte son kit et son vecteur.
    Affiche un log de progression.
    """
    from code_analysis.ASTReader import ASTReader
    reader = ASTReader()
    fragments = []
    total_files = len(ast_files)
    for idx, path in enumerate(ast_files, start=1):
        print(f"Extraction fragments : fichier {idx}/{total_files}", end="\r")
        kit_id = os.path.relpath(path, ast_root).split(os.sep)[0]
        try:
            ast = reader.read_ast(path)
            for node in ast.get_node_ids():
                ntype = ast.get_type(node)
                if ntype.endswith('FunctionStatement') or ntype.endswith('MethodStatement'):
                    vec = ast.vectorize(node)
                    size = int(vec.sum())
                    if size <= min_nodes:
                        continue
                    # Récupère le nom de la fonction/méthode
                    name = '<anonymous>'
                    for child in ast.get_children(node) or []:
                        if ast.get_type(child) == 'EId':
                            img = ast.get_image(child)
                            if img:
                                name = img
                                break
                    fragments.append({'kit': kit_id, 'name': name, 'vector': vec, 'size': size})
        except Exception as e:
            print(f"[Warning] Lecture échouée {path}: {e}")
    print()  # Fin du log
    return fragments


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    global ast_root
    ast_root = os.path.join(project_root, 'ast')
    if not os.path.isdir(ast_root):
        print(f"Erreur: dossier AST non trouvé sous {ast_root}")
        sys.exit(1)

    # Étape 1 : collecte des fichiers AST
    print("Collecte des fichiers AST...")
    ast_files = find_ast_files(ast_root)
    print(f"  {len(ast_files)} fichiers AST trouvés.")

    # Étape 2 : extraction des fragments
    print("Extraction des fragments (>10 nœuds)...")
    fragments = extract_fragments(ast_files)
    print(f"Total fragments isolés : {len(fragments)}")

    # Étape 3 : calcul des vecteurs représentatifs par kit
    print("Calcul des vecteurs par kit...")
    kit_fragments = {}
    total_frags = len(fragments)
    for i, frag in enumerate(fragments, start=1):
        print(f"Aggregation kit : fragment {i}/{total_frags}", end="\r")
        kit_fragments.setdefault(frag['kit'], []).append(frag['vector'])
    print()
    kit_repr = {kit: np.sum(vecs, axis=0) for kit, vecs in kit_fragments.items()}

    # Étape 4 : recherche des kits paramétriques
    print("Recherche des kits à vecteurs identiques...")
    groups = {}
    all_kits = list(kit_repr.keys())
    total_kits = len(all_kits)
    for idx, kit in enumerate(all_kits, start=1):
        print(f"Comparaison vecteurs : kit {idx}/{total_kits}", end="\r")
        key = tuple(kit_repr[kit].tolist())
        groups.setdefault(key, []).append(kit)
    print()

    largest_key = max(groups.keys(), key=lambda k: len(groups[k]))
    largest_kits = groups[largest_key]

    print("\n=== Kits paramétriques (vecteurs identiques) ===")
    print(f"Nombre de kits dans le groupe : {len(largest_kits)}")
    print("Kits concernés :", ", ".join(largest_kits))

    # Étape 5 : vérification AST types et images
    print("\nVérification des types et images AST par kit...")
    from code_analysis.ASTReader import ASTReader
    reader = ASTReader()
    ast_paths_per_kit = {k: [] for k in largest_kits}
    for idx, path in enumerate(ast_files, start=1):
        kit = os.path.relpath(path, ast_root).split(os.sep)[0]
        if kit in ast_paths_per_kit:
            print(f"Collecte AST kit {kit}: fichier {idx}/{len(ast_files)}", end="\r")
            ast_paths_per_kit[kit].append(path)
    print()

    kit_types = {}
    kit_images = {}
    for kit in largest_kits:
        types_set = set()
        images_set = set()
        paths = ast_paths_per_kit[kit]
        total_paths = len(paths)
        for j, path in enumerate(paths, start=1):
            print(f"Analyse AST kit {kit}: fichier {j}/{total_paths}", end="\r")
            ast = reader.read_ast(path)
            for node in ast.get_node_ids():
                types_set.add(ast.get_type(node))
                img = ast.get_image(node)
                if img:
                    images_set.add(img)
        kit_types[kit] = types_set
        kit_images[kit] = images_set
        print()

    base = largest_kits[0]
    same_types = all(kit_types[k] == kit_types[base] for k in largest_kits)
    print("→ Les kits partagent les mêmes types de nœuds AST." if same_types else "→ Différences détectées dans les types AST.")

    print("\nAnalyse des valeurs d'image (literals) :")
    for kit in largest_kits:
        diff = kit_images[kit] - kit_images[base]
        if diff:
            print(f"Kit {kit} possède {len(diff)} images uniques comparé à {base}.")
    print("Analyse terminée.")

if __name__ == '__main__':
    main()
