import os
import numpy as np
import collections
import sys
import time
from code_analysis import ASTReader, AST_fragmentation

AST_PATH = "ast"  # Chemin vers les kits d'AST
MIN_NODES_FILE = 100
MIN_NODES_FRAGMENT = 10
SIMILARITY_THRESHOLD_FILE = 0.30
SIMILARITY_THRESHOLD_FRAGMENT = 0.10


# Trouve tous les fichiers .ast.json dans les sous-dossiers de base_path.
def get_ast_files(base_path):
    kit_files_map = collections.defaultdict(list)
    kits = []
    if not os.path.isdir(base_path):
        print(f"Erreur: Le dossier AST '{base_path}' n'existe pas.")
        return {}, []
        
    for dir_kit in sorted(os.listdir(base_path)):
        kit_path = os.path.join(base_path, dir_kit)
        if os.path.isdir(kit_path):
            kits.append(dir_kit)
            for root, _, files in os.walk(kit_path):
                for file in files:
                    if file.endswith(".ast.json") or file.endswith(".ast.json.gz"):
                        full_path = os.path.join(root, file)
                        kit_files_map[dir_kit].append(full_path)
    if not kits:
        print(f"Aucun sous-dossier (kit) trouvé dans '{base_path}'.")
    return kit_files_map, kits

# Calcule la distance de Manhattan entre deux vecteurs numpy.
def calculate_manhattan_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))

def get_function_name(ast, fragment_node_id):
    # Tente d'extraire le nom de la fonction/méthode d'un nœud fragment.
    # chercher un enfant de type 'Id' qui pourrait être le nom
    try:
        children = ast.get_children(fragment_node_id)
        for child_id in children:
            # Souvent le nom est un Id ou similaire, directement enfant ou petit-enfant
            if ast.get_type(child_id) == 'Id' and ast.get_image(child_id):
                return ast.get_image(child_id)
            # Parfois encapsulé dans un autre noeud (ex: ReturnValueFunction)
            grandchildren = ast.get_children(child_id)
            for grandchild_id in grandchildren:
                 if ast.get_type(grandchild_id) == 'Id' and ast.get_image(grandchild_id):
                     return ast.get_image(grandchild_id)
    except Exception:
        pass
    # Si non trouvé, retourner une indication
    node_type = ast.get_type(fragment_node_id)
    return f"[{node_type}_SansNomTrouve_{fragment_node_id}]"


# Chargement et Prétraitement des Données

print("Chargement des données AST...")
start_time = time.time()
reader = ASTReader()
kit_files_map, kits = get_ast_files(AST_PATH)

all_files_data = [] # (filepath, kit_id, vector, node_count)
all_fragments_data = [] # (filepath, kit_id, fragment_node_id, vector, node_count, function_name)

asts_cache = {} # Cache pour éviter de relire les ASTs: {filepath: ast}
def get_ast(path):
    if path not in asts_cache:
        asts_cache[path] = reader.read_ast(path)
    return asts_cache[path]

total_files_processed = 0
total_fragments_processed = 0


# Initialisation d'un AST pour obtenir la taille du vecteur
try:
    # Essayer de lire un fichier pour obtenir la structure AST et la taille du vecteur
    first_kit = next(iter(kit_files_map)) if kit_files_map else None
    first_file = kit_files_map[first_kit][0] if first_kit and kit_files_map[first_kit] else None
    if first_file:
        temp_ast = get_ast(first_file)
        VECTOR_SIZE = len(temp_ast.types)
        print(f"Taille de vecteur détectée : {VECTOR_SIZE}")
        del temp_ast # vider la mémoire
    else:
        print("Aucun fichier AST trouvé pour déterminer la taille du vecteur. Arrêt.")
        sys.exit(1)
except Exception as e:
    print(f"Erreur lors de la lecture initiale pour déterminer la taille du vecteur : {e}")
    sys.exit(1)


for kit_id in kits:
    print(f"  Traitement du kit : {kit_id} ({len(kit_files_map[kit_id])} fichiers)")
    for filepath in kit_files_map[kit_id]:
        try:
            ast = get_ast(filepath)
            asts_cache[filepath] = ast
            total_files_processed += 1

            file_vector = ast.vectorize()
            file_node_count = int(file_vector.sum()) 
            if file_node_count > 0:
                 all_files_data.append((filepath, kit_id, file_vector, file_node_count))

            fragment_nodes = AST_fragmentation(ast)
            
            ast_for_fragments = get_ast(filepath)

            for fragment_node_id in fragment_nodes:
                 fragment_vector = ast_for_fragments.vectorize(node=fragment_node_id)
                 fragment_node_count = int(fragment_vector.sum())
                 if fragment_node_count > 0:
                     function_name = get_function_name(ast_for_fragments, fragment_node_id)
                     all_fragments_data.append((filepath, kit_id, fragment_node_id, fragment_vector, fragment_node_count, function_name))
                     total_fragments_processed += 1

        except FileNotFoundError:
            print(f"    Attention: Fichier non trouvé: {filepath}")
        except Exception as e:
            print(f"    Erreur lors du traitement de {filepath}: {e}")

loading_time = time.time() - start_time
print(f"Chargement terminé en {loading_time:.2f} secondes.")
print(f"Nombre total de fichiers AST traités : {total_files_processed}")
print(f"Nombre total de fragments extraits : {total_fragments_processed}")
print("-" * 30)

print("\nQuestion 1: Analyse au niveau des fichiers")

print("\n1.1 Clones 'paramétriques' (Fichiers exacts, >100 nœuds)")
start_time = time.time()

files_filtered = [(fp, k_id, vec, count) for fp, k_id, vec, count in all_files_data if count >= MIN_NODES_FILE]
print(f"Nombre de fichiers avec >= {MIN_NODES_FILE} nœuds : {len(files_filtered)}")

# Groupement par vecteur (ignorant les doublons intra-kit pour l'instant)
vectors_files_exact = collections.defaultdict(list)
for filepath, kit_id, vector, _ in files_filtered:
    vector_tuple = tuple(vector)
    vectors_files_exact[vector_tuple].append((filepath, kit_id))

largest_group_files_exact = []
if vectors_files_exact:
    largest_group_files_exact = max(vectors_files_exact.values(), key=len)

print(f"Taille du plus grand groupe de fichiers identiques : {len(largest_group_files_exact)}")
print(f"Kits concernés : {', '.join(set(k_id for fp, k_id in largest_group_files_exact))}")
if largest_group_files_exact:
    print("Exemples de fichiers dans ce groupe :")
    for i, (fp, k_id) in enumerate(largest_group_files_exact[:5]):
        print(f"  - {fp} (Kit: {k_id})")
    if len(largest_group_files_exact) > 5:
        print("  ...")

    # Question : Que remarquez-vous concernant ces fichiers ?
    print("\nRemarques sur ces fichiers identiques :")
    kits_in_group = set(k_id for _, k_id in largest_group_files_exact)
    print("Kits concernés :", ", ".join(sorted(kits_in_group)))
    if len(kits_in_group) == 1:
        print(f"- Tous les fichiers de ce groupe appartiennent au même kit: {list(kits_in_group)[0]}.")
        print("- Il s'agit probablement de copies exactes du même fichier au sein de ce kit.")
    elif len(kits_in_group) < len(largest_group_files_exact):
         print(f"- Le groupe contient des fichiers provenant de {len(kits_in_group)} kits différents.")
         print("- Certains kits contiennent plusieurs copies de ce fichier.")
         print("- D'autres kits pourraient être des copies ou des évolutions très proches les uns des autres.")
    else:
         print(f"- Le groupe contient des fichiers provenant de {len(kits_in_group)} kits différents (un fichier par kit).")
         print("- Cela suggère une réutilisation directe d'un fichier spécifique à travers plusieurs kits distincts.")

# Refaire l'opération en ignorant les vecteurs dupliqués au sein d'un même kit
print("\nAnalyse des clones paramétriques (fichiers) en ignorant les duplicats intra-kit:")
vectors_files_exact_unique_kit = collections.defaultdict(set)
for filepath, kit_id, vector, _ in files_filtered:
    vector_tuple = tuple(vector)
    vectors_files_exact_unique_kit[vector_tuple].add(kit_id)

largest_group_size_unique_kit = 0
if vectors_files_exact_unique_kit:
   largest_group_size_unique_kit = max(len(kits) for kits in vectors_files_exact_unique_kit.values())

print(f"Taille du plus grand groupe de kits partageant un fichier identique (après filtrage intra-kit) : {largest_group_size_unique_kit}")

kits_in_largest_groups = set()
if largest_group_size_unique_kit > 0:
    for vector_tuple, kit_id_set in vectors_files_exact_unique_kit.items():
        if len(kit_id_set) == largest_group_size_unique_kit:
            kits_in_largest_groups.update(kit_id_set)

print(f"Kits concernés : {', '.join(sorted(list(kits_in_largest_groups)))}")

if largest_group_size_unique_kit > 0:
     print("Cela indique le nombre maximum de kits *différents* qui contiennent au moins une copie d'un fichier spécifique (ayant >100 nœuds).")

# print(f"Analyse 1.1 terminée en {time.time() - start_time:.2f} secondes.")

print("\n1.2 Fichiers similaires (>100 nœuds, distance <= 30%)")
start_time = time.time()

n_files = len(files_filtered)
similarity_groups = collections.defaultdict(list)

if n_files > 1:
    print(f"Calcul des similarités par paires pour {n_files} fichiers...")
    for i in range(n_files):
        filepath_i, kit_id_i, vector_i, node_count_i = files_filtered[i]
        similarity_groups[i].append(i)
        threshold = SIMILARITY_THRESHOLD_FILE * node_count_i

        for j in range(i + 1, n_files):
            filepath_j, kit_j, vector_j, node_count_j = files_filtered[j]

            distance = calculate_manhattan_distance(vector_i, vector_j)

            # Vérifier si similaire par rapport à i OU j
            threshold_j = SIMILARITY_THRESHOLD_FILE * node_count_j
            if distance <= threshold or distance <= threshold_j:
                 # Ils sont considérés comme similaires
                 similarity_groups[i].append(j)
                 similarity_groups[j].append(i)
else:
    print("Pas assez de fichiers (>1) pour comparer la similarité.")

largest_group_files_similar_indices = []
if similarity_groups:
    center_index = max(similarity_groups, key=lambda k: len(similarity_groups[k]))
    largest_group_files_similar_indices = similarity_groups[center_index]

print(f"Taille du plus grand groupe de fichiers similaires trouvé : {len(largest_group_files_similar_indices)}")
if largest_group_files_similar_indices:
    print("Exemples de fichiers dans ce groupe similaire :")
    representative_files = [files_filtered[idx] for idx in largest_group_files_similar_indices[:5]]
    for i, (fp, k_id, vec, count) in enumerate(representative_files):
        print(f"  - {fp} (Kit: {k_id}, Nœuds: {count})")
    if len(largest_group_files_similar_indices) > 5:
        print("  ...")

    kits_in_similar_group = set(files_filtered[idx][1] for idx in largest_group_files_similar_indices)
    print(f"\nCe groupe de fichiers similaires provient de {len(kits_in_similar_group)} kits différents.")
    print(f"Kits concernés : {', '.join(sorted(list(kits_in_similar_group)))}")
    print("Cela suggère une base de code commune ou des modifications mineures (paramétrisation, renommage)")
    print("entre des fichiers servant potentiellement le même objectif dans différents kits.")

print(f"Analyse 1.2 terminée en {time.time() - start_time:.2f} secondes.")
print("-" * 30)


print("\nQuestion 2: Analyse au niveau des fragments")

print("\n2.1 Clones 'paramétriques' (Fragments exacts, >10 nœuds, hors duplicats intra-kit)")
start_time = time.time()

fragments_filtered = [(fp, k_id, f_id, vec, count, name) for fp, k_id, f_id, vec, count, name in all_fragments_data if count >= MIN_NODES_FRAGMENT]
print(f"Nombre de fragments avec >= {MIN_NODES_FRAGMENT} nœuds : {len(fragments_filtered)}")

# Groupement par vecteur, en ne comptant chaque kit qu'une fois par vecteur
vectors_fragments_exact_unique_kit = collections.defaultdict(lambda: {'kits': set(), 'names': set(), 'examples': []})
processed_frag_kit_vector = set()

for filepath, kit_id, frag_id, vector, count, name in fragments_filtered:
    vector_tuple = tuple(vector)
    frag_kit_vector_key = (kit_id, vector_tuple)

    if frag_kit_vector_key not in processed_frag_kit_vector:
        vectors_fragments_exact_unique_kit[vector_tuple]['kits'].add(kit_id)
        if name:
             vectors_fragments_exact_unique_kit[vector_tuple]['names'].add(name)
        if len(vectors_fragments_exact_unique_kit[vector_tuple]['examples']) < 1:
             vectors_fragments_exact_unique_kit[vector_tuple]['examples'].append((filepath, frag_id))
        processed_frag_kit_vector.add(frag_kit_vector_key)

most_duplicated_fragment_info = None
max_kits_count = 0
if vectors_fragments_exact_unique_kit:
    most_duplicated_vector_tuple = max(vectors_fragments_exact_unique_kit,
                                       key=lambda k: len(vectors_fragments_exact_unique_kit[k]['kits']))
    most_duplicated_fragment_info = vectors_fragments_exact_unique_kit[most_duplicated_vector_tuple]
    max_kits_count = len(most_duplicated_fragment_info['kits'])


print(f"\nLe fragment le plus dupliqué (structure exacte) est présent dans {max_kits_count} kits différents.")
print(f"- Kits concernés : {', '.join(sorted(list(most_duplicated_fragment_info['kits'])))}")

if most_duplicated_fragment_info:
    # Question: Quelle est le nom(s) de la fonction correspondant à ce fragment ?
    fragment_names = most_duplicated_fragment_info['names']
    if fragment_names:
        print(f"Nom(s) de fonction/méthode associés à ce fragment : {', '.join(sorted(list(fragment_names)))}")
    else:
        print("Aucun nom de fonction/méthode n'a pu être extrait pour ce fragment.")
    example_fp, example_fid = most_duplicated_fragment_info['examples'][0]
    print(f"Exemple d'occurrence : Fichier '{example_fp}', Nœud fragment ID: {example_fid}")
    print("Ce fragment représente probablement une fonction utilitaire commune ou une partie standard")
    print("d'un framework réutilisé dans de nombreux kits de phishing.")

print(f"Analyse 2.1 terminée en {time.time() - start_time:.2f} secondes.")


print("\n2.2 Fragments similaires (>10 nœuds, distance <= 10%, hors duplicats intra-kit)")
start_time = time.time()

# Créer une liste de fragments où chaque (vecteur, kit_id) est unique.
unique_fragments_per_kit = {}
for filepath, kit_id, frag_id, vector, count, name in fragments_filtered:
    vector_tuple = tuple(vector)
    key = (vector_tuple, kit_id)
    if key not in unique_fragments_per_kit:
         unique_fragments_per_kit[key] = (vector, kit_id, name, filepath, count)

fragments_to_compare = list(unique_fragments_per_kit.values())
n_fragments = len(fragments_to_compare)
print(f"Nombre de fragments uniques par kit (avec >= {MIN_NODES_FRAGMENT} nœuds) à comparer : {n_fragments}")

# Calculer les similarités par paires sur cette liste unique
similarity_groups_frags = collections.defaultdict(list)

if n_fragments > 1:
    print(f"Calcul des similarités par paires pour {n_fragments} fragments uniques par kit...")
    for i in range(n_fragments):
        vector_i, kit_id_i, name_i, fp_i, node_count_i = fragments_to_compare[i]
        similarity_groups_frags[i].append(i)
        threshold = SIMILARITY_THRESHOLD_FRAGMENT * node_count_i

        for j in range(i + 1, n_fragments):
            vector_j, kit_id_j, name_j, fp_j, node_count_j = fragments_to_compare[j]

            # Assurer que les kits sont différents pour la similarité inter-kit
            if kit_id_i == kit_id_j:
                 continue # On ignore la similarité intra-kit ici comme demandé implicitement

            distance = calculate_manhattan_distance(vector_i, vector_j)
            threshold_j = SIMILARITY_THRESHOLD_FRAGMENT * node_count_j

            if distance <= threshold or distance <= threshold_j:
                 similarity_groups_frags[i].append(j)
                 similarity_groups_frags[j].append(i)
else:
     print("Pas assez de fragments uniques par kit (>1) pour comparer la similarité.")


largest_group_frags_similar_indices = []
if similarity_groups_frags:
    center_index_frag = max(similarity_groups_frags, key=lambda k: len(similarity_groups_frags[k]))
    largest_group_frags_similar_indices = similarity_groups_frags[center_index_frag]

print(f"\nTaille du plus grand groupe de fragments similaires trouvé (inter-kit) : {len(largest_group_frags_similar_indices)}")

if largest_group_frags_similar_indices:
    # Question: Quelle est le nom(s) de la fonction correspondant à ces fragments?
    names_in_similar_group = set()
    representative_frags = []
    for idx in largest_group_frags_similar_indices:
         frag_data = fragments_to_compare[idx]
         representative_frags.append(frag_data)
         if frag_data[2]:
              names_in_similar_group.add(frag_data[2])

    print("Exemples de fragments dans ce groupe similaire :")
    for i, (vec, k_id, name, fp, count) in enumerate(representative_frags):
        print(f"  - Nom: '{name}', Kit: {k_id}, Fichier: {fp}, Nœuds: {count}")
    if len(largest_group_frags_similar_indices) > 5:
        print("  ...")

    if names_in_similar_group:
        print(f"\nNom(s) de fonction/méthode associés à ces fragments similaires : {', '.join(sorted(list(names_in_similar_group)))}")
    else:
        print("\nAucun nom de fonction/méthode n'a pu être extrait pour les fragments de ce groupe.")
    print("Ces fragments représentent probablement des variations mineures de fonctions communes,")
    print("peut-être dues à l'obfuscation, au renommage de variables ou à de petites modifications fonctionnelles.")

print(f"Analyse 2.2 terminée en {time.time() - start_time:.2f} secondes.")
print("-" * 30)

print("\nSection 3: Analyse au niveau des kits 'paramétriques'")
start_time = time.time()

kit_vectors_sum = collections.defaultdict(lambda: np.zeros(VECTOR_SIZE, dtype=int))
kit_fragment_count = collections.defaultdict(int)

print("Calcul des vecteurs représentatifs des kits (somme des fragments)...")
# Utiliser tous les fragments (pas seulement > 10 noeuds) pour le vecteur kit ?
# L'énoncé dit "sommant (par colonne) tout les fragments qui composent ce kit."
# Utilisons all_fragments_data
for filepath, kit_id, frag_id, vector, count, name in all_fragments_data:
     kit_vectors_sum[kit_id] += vector
     kit_fragment_count[kit_id] += 1

print(f"{len(kit_vectors_sum)} kits ont au moins un fragment et ont un vecteur représentatif.")

kit_vector_groups = collections.defaultdict(list)
for kit_id, sum_vector in kit_vectors_sum.items():
    if sum_vector.sum() > 0:
        vector_tuple = tuple(sum_vector)
        kit_vector_groups[vector_tuple].append(kit_id)

largest_group_kits_exact = []
if kit_vector_groups:
     largest_group_kits_exact = max(kit_vector_groups.values(), key=len)

print(f"\nTaille du plus grand groupe de kits ayant des vecteurs représentatifs identiques : {len(largest_group_kits_exact)}")

if largest_group_kits_exact:
    # Question: Quels sont ces kits?
    print(f"Kits dans ce groupe : {', '.join(sorted(largest_group_kits_exact))}")

    # Question: Y a-t-il des différences entre ces kits au niveaux des types des nœuds de l'AST (type, image)?
    print("\nDifférences au niveau des types de nœuds AST pour ces kits identiques (basé sur le vecteur somme) :")
    # Par définition, si les vecteurs sommes sont identiques, la *distribution globale* des types de nœuds
    # (telle que capturée par la somme des vecteurs fragments) est identique entre ces kits.
    print("- Non, par définition de ce groupe, leurs vecteurs représentatifs (somme des vecteurs fragments) sont identiques.")
    print("- Cela signifie que la somme totale des occurrences de chaque type de nœud AST, sur l'ensemble des fragments de chaque kit, est la même.")
    print("- Cette mesure ne capture cependant pas la structure fine, l'organisation des fragments, ou le contenu textuel ('image') des nœuds.")
    print("- Il est possible que les kits soient structurellement différents mais aboutissent à la même somme vectorielle (moins probable si les kits sont complexes),")
    print("  ou plus probablement, qu'ils soient des copies très proches ou exactes au niveau du code source analysé.")

    # On pourrait vérifier si le nombre de fragments est aussi identique (indice supplémentaire)
    num_fragments_in_group = [kit_fragment_count[k] for k in largest_group_kits_exact]
    if len(set(num_fragments_in_group)) == 1:
        print(f"- De plus, tous les kits de ce groupe ont le même nombre de fragments détectés : {num_fragments_in_group[0]}.")
    else:
        print(f"- Fait intéressant : bien que les vecteurs sommes soient égaux, le nombre de fragments détectés varie dans ce groupe: {num_fragments_in_group}.")
        print("  Cela pourrait indiquer des différences structurelles (ex: fonctions vides vs non vides, seuils de détection).")
