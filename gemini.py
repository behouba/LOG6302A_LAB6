import os
import numpy as np
from collections import defaultdict
import gzip
from tqdm import tqdm  # Optional: for progress bars

# Assuming the code_analysis package is in the same directory or accessible
from code_analysis import ASTReader, AST, AST_fragmentation, GraphException

# --- Configuration ---
PATH_AST = "ast"
MIN_FILE_NODES = 100
MIN_FRAGMENT_NODES = 10
FILE_SIMILARITY_THRESHOLD = 0.30
FRAGMENT_SIMILARITY_THRESHOLD = 0.10

# --- Data Loading ---

reader = ASTReader()
all_files_data = []
all_fragments_data = []

print("Starting data loading and initial processing...")

kits = sorted([d for d in os.listdir(PATH_AST) if os.path.isdir(os.path.join(PATH_AST, d))])

for kit_id in tqdm(kits, desc="Processing Kits"):
    kit_path = os.path.join(PATH_AST, kit_id)
    for root, _, files in os.walk(kit_path):
        for filename in files:
            if filename.endswith(".ast.json") or filename.endswith(".ast.json.gz"):
                filepath = os.path.join(root, filename)
                try:
                    # Load AST - Create a copy for fragmentation later if needed
                    # However, vectorizing first and then fragmenting should be okay
                    # as fragmentation only detaches nodes, not altering types/counts within subtrees.
                    ast = reader.read_ast(filepath)
                    if ast is None or not ast.get_node_ids():
                        # print(f"Warning: Skipping empty or invalid AST: {filepath}")
                        continue

                    # 1. Process Full File AST
                    file_vector = ast.vectorize()
                    file_size = int(file_vector.sum())

                    if file_size >= MIN_FILE_NODES:
                        all_files_data.append({
                            'kit_id': kit_id,
                            'filepath': filepath,
                            'vector': file_vector,
                            'size': file_size
                        })

                    # 2. Process Fragments
                    # Apply fragmentation (modifies the AST by removing parent links of fragments)
                    try:
                       fragment_roots = AST_fragmentation(ast)
                    except Exception as frag_e:
                       # print(f"Warning: Fragmentation error in {filepath}: {frag_e}")
                       fragment_roots = [] # Continue without fragments for this file

                    for frag_root_id in fragment_roots:
                        try:
                            # Vectorize the subtree rooted at the fragment node
                            fragment_vector = ast.vectorize(frag_root_id)
                            fragment_size = int(fragment_vector.sum())

                            if fragment_size >= MIN_FRAGMENT_NODES:
                                # Try to get function name
                                func_name = ast.get_image(frag_root_id) # Root might have it
                                if not func_name: # Check children for 'Id' node
                                    children_ids = ast.get_children(frag_root_id)
                                    if children_ids:
                                        # Often the first child or an 'Id' node holds the name
                                        # Let's look specifically for 'Id' type
                                        name_node_id = None
                                        for child_id in children_ids:
                                            if ast.get_type(child_id) == 'Id':
                                                 name_node_id = child_id
                                                 break
                                        # If no 'Id', maybe take image of first child if it exists
                                        if name_node_id:
                                             func_name = ast.get_image(name_node_id)
                                        elif children_ids:
                                             # As a fallback, check image of the first child? Risky.
                                             pass # Keep func_name as None/Empty


                                all_fragments_data.append({
                                    'kit_id': kit_id,
                                    'filepath': filepath,
                                    'frag_root_id': frag_root_id,
                                    'vector': fragment_vector,
                                    'size': fragment_size,
                                    'func_name': func_name if func_name else "Unnamed"
                                })
                        except GraphException as ge:
                            # print(f"Warning: Graph error vectorizing fragment {frag_root_id} in {filepath}: {ge}")
                            pass # Skip this fragment
                        except Exception as e:
                            # print(f"Warning: Error processing fragment {frag_root_id} in {filepath}: {e}")
                            pass # Skip this fragment


                except GraphException as ge:
                    # print(f"Warning: Graph error reading {filepath}: {ge}")
                    pass # Skip this file
                except Exception as e:
                    # print(f"Warning: Generic error reading {filepath}: {e}")
                    pass # Skip this file

print(f"Finished loading. Found {len(all_files_data)} files >= {MIN_FILE_NODES} nodes.")
print(f"Found {len(all_fragments_data)} fragments >= {MIN_FRAGMENT_NODES} nodes.")

# --- Analysis Functions ---

def find_largest_identical_group(data, key_field='vector', info_fields=['kit_id', 'filepath'], ignore_intra_kit=False):
    """Finds the largest group of items with identical vectors."""
    vector_groups = defaultdict(list)
    # Track kits per vector if ignoring intra-kit duplicates
    processed_kits_per_vector = defaultdict(set)

    for item in data:
        vec_tuple = tuple(item[key_field])
        kit_id = item['kit_id']
        info = {field: item[field] for field in info_fields if field in item}

        if ignore_intra_kit:
            if kit_id not in processed_kits_per_vector[vec_tuple]:
                vector_groups[vec_tuple].append(info)
                processed_kits_per_vector[vec_tuple].add(kit_id)
        else:
            vector_groups[vec_tuple].append(info)

    largest_group_size = 0
    largest_group_items = []
    largest_vector = None

    if not vector_groups: # Handle case where no data meets criteria
        return 0, [], None

    for vec, items in vector_groups.items():
        if len(items) > largest_group_size:
            largest_group_size = len(items)
            largest_group_items = items
            largest_vector = vec # Keep the vector itself

    return largest_group_size, largest_group_items, largest_vector

# --- Disjoint Set Union (DSU) for Similarity Grouping ---
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.num_sets = n

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def unite(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            self.num_sets -= 1
            return True
        return False

def find_largest_similar_group(data, threshold_ratio, ignore_intra_kit=False):
    """Finds the largest group of similar items using DSU."""
    if not data: return 0, []

    # Filter data for unique inter-kit instances if needed FIRST
    items_to_compare = []
    if ignore_intra_kit:
        seen_kit_vector = set()
        for i, item in enumerate(data):
             # Need a unique identifier per kit for a given vector shape
             # Let's use (kit_id, tuple(vector))
             vec_tuple = tuple(item['vector'])
             kit_id = item['kit_id']
             # Store index mapping if needed? Or just store the item.
             # Let's store the item itself if it's the first time we see this vector from this kit
             # Note: This assumes identical vectors from the same kit should only count once for similarity grouping.
             # This interpretation might be slightly different than just ignoring pairs during comparison.
             # Let's stick to pairwise comparison with kit check for now, it's simpler.
             items_to_compare = data # Revert to using all data and check kits during compare
    else:
        items_to_compare = data

    n = len(items_to_compare)
    if n < 2: return n, items_to_compare # Handle small N

    dsu = DSU(n)

    print(f"Calculating similarities for {n} items...")
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            item_i = items_to_compare[i]
            item_j = items_to_compare[j]

            # Skip if ignoring intra-kit and items are from the same kit
            if ignore_intra_kit and item_i['kit_id'] == item_j['kit_id']:
                continue

            # Manhattan distance
            manhattan_dist = np.abs(item_i['vector'] - item_j['vector']).sum()
            # Use vector_i's size for threshold as per example
            threshold = threshold_ratio * item_i['size']

            if manhattan_dist <= threshold:
                dsu.unite(i, j)
            # Also check the other way around? The prompt example only uses vector_i.sum()
            # Let's check symmetric condition just in case it matters.
            threshold_j = threshold_ratio * item_j['size']
            if manhattan_dist <= threshold_j:
                 dsu.unite(i, j) # DSU unite handles duplicates

    # Group results
    groups = defaultdict(list)
    for i in range(n):
        root = dsu.find(i)
        groups[root].append(items_to_compare[i]) # Store original item data

    largest_group_size = 0
    largest_group_items = []
    if not groups: return 0, []

    for root_id, items in groups.items():
        if len(items) > largest_group_size:
            largest_group_size = len(items)
            largest_group_items = items # List of item dictionaries

    return largest_group_size, largest_group_items


# --- Section 1: Files ---

print("\n--- Section 1: Files ---")

# 1.1 Clones "paramétriques" (Files)
print("\n--- 1.1 Parametric Clones (Files) ---")
size_1_1a, items_1_1a, _ = find_largest_identical_group(all_files_data, ignore_intra_kit=False)
print(f"Largest group size (including intra-kit duplicates): {size_1_1a}")
print("Examples from largest group (first 5):")
for item in items_1_1a[:5]: print(f"  - Kit: {item['kit_id']}, Path: {item['filepath']}")
# print("\nObservation:")
# print("Files with identical vectors often represent configuration files, library includes,")
# print("or very simple scripts that are commonly reused without modification.")
# print("Seeing a large group including intra-kit duplicates is expected if a kit reuses a file internally.")

size_1_1b, items_1_1b, _ = find_largest_identical_group(all_files_data, ignore_intra_kit=True)
print(f"\nLargest group size (ignoring intra-kit duplicates): {size_1_1b}")
print("Examples from largest inter-kit group (first 5):")
for item in items_1_1b[:5]: print(f"  - Kit: {item['kit_id']}, Path: {item['filepath']}")
# print("\nObservation:")
# print("A large group when ignoring intra-kit duplicates strongly suggests that")
# print("different phishing kits are reusing the exact same file, indicating")
# print("code sharing, kit copying, or derivation from a common source.")


# 1.2 Fichiers similaires
print("\n--- 1.2 Similar Files ---")
size_1_2, items_1_2 = find_largest_similar_group(all_files_data, FILE_SIMILARITY_THRESHOLD, ignore_intra_kit=False) # Prompt doesn't explicitly say ignore intra-kit here, let's include them.
print(f"Largest group size (similar files, Manhattan dist <= {FILE_SIMILARITY_THRESHOLD*100}% of size): {size_1_2}")
print("Examples from largest similar group (first 5):")
for item in items_1_2[:5]: print(f"  - Kit: {item['kit_id']}, Path: {item['filepath']}")
# print("\nObservation:")
# print("Similar files suggest variations of a base template or script. Differences")
# print("might be in variable names, minor logic changes, or added/removed features,")
# print("but the overall structure (reflected in node counts) remains close.")


# --- Section 2: Fragments ---

print("\n\n--- Section 2: Fragments ---")
frag_info_fields = ['kit_id', 'filepath', 'frag_root_id', 'func_name']

# 2.1 Clones "paramétriques" (Fragments)
print("\n--- 2.1 Parametric Clones (Fragments) ---")
# Ignore intra-kit duplicates for fragments as requested
size_2_1, items_2_1, _ = find_largest_identical_group(all_fragments_data, info_fields=frag_info_fields, ignore_intra_kit=True)
print(f"Most duplicated fragment count (inter-kit): {size_2_1}")
if items_2_1:
    func_names = sorted(list(set(f['func_name'] for f in items_2_1)))
    print(f"Corresponding function name(s): {', '.join(func_names)}")
    print("Example instances (first 5):")
    for item in items_2_1[:5]: print(f"  - Kit: {item['kit_id']}, Path: {item['filepath']}, RootID: {item['frag_root_id']}, Func: {item['func_name']}")
else:
    print("No duplicated fragments found meeting criteria.")
# print("\nObservation:")
# print("Identical functions across different kits point to shared libraries or core functionalities")
# print("that are directly copied between kits. Function names can provide clues to their purpose.")


# 2.2 Fragments similaires
print("\n--- 2.2 Similar Fragments ---")
# Ignore intra-kit duplicates for similar fragments as requested
size_2_2, items_2_2 = find_largest_similar_group(all_fragments_data, FRAGMENT_SIMILARITY_THRESHOLD, ignore_intra_kit=True)
print(f"Largest group size (similar fragments, inter-kit, Manhattan dist <= {FRAGMENT_SIMILARITY_THRESHOLD*100}% of size): {size_2_2}")
if items_2_2:
    func_names_2_2 = sorted(list(set(f['func_name'] for f in items_2_2)))
    print(f"Corresponding function name(s) in largest group: {', '.join(func_names_2_2)}")
    print("Example instances (first 5):")
    for item in items_2_2[:5]: print(f"  - Kit: {item['kit_id']}, Path: {item['filepath']}, RootID: {item['frag_root_id']}, Func: {item['func_name']}")
else:
    print("No similar fragments found meeting criteria.")
# print("\nObservation:")
# print("Similar functions across kits suggest evolution or customization of common code snippets.")
# print("The core logic might be the same, but with minor tweaks (e.g., different variable names, slight logic changes).")


# --- Section 3: Kits "paramétriques" ---

print("\n\n--- Section 3: Kits 'Paramétriques' ---")

kit_vectors = defaultdict(lambda: np.zeros_like(all_fragments_data[0]['vector']) if all_fragments_data else np.array([])) # Use first fragment's vector shape if possible
kit_has_fragments = set()

# Sum fragment vectors per kit
if all_fragments_data:
    for data in all_fragments_data:
        kit_id = data['kit_id']
        kit_vectors[kit_id] += data['vector']
        kit_has_fragments.add(kit_id)
else:
    print("Warning: No fragment data available to calculate kit vectors.")

# Prepare data for grouping (only kits that had fragments)
kit_vector_data = []
for kit_id, vector in kit_vectors.items():
    if kit_id in kit_has_fragments and vector.sum() > 0: # Ensure kit had fragments and vector is not zero
        kit_vector_data.append({'kit_id': kit_id, 'vector': vector})

# Find largest group of identical kit vectors
size_3, items_3, vec_3 = find_largest_identical_group(kit_vector_data, info_fields=['kit_id'], ignore_intra_kit=False) # ignore_intra_kit=False is default, here we compare distinct kits

print(f"\nLargest group of kits with identical summed fragment vectors: {size_3}")
if items_3:
    kit_ids_3 = sorted([item['kit_id'] for item in items_3])
    print(f"Kit IDs in this group: {', '.join(kit_ids_3)}")
    print(f"Representative summed vector (first 30 elements): {tuple(vec_3)[:30]}...")

    # print("\nAnalysis of differences:")
    # print(f"  - The kit vectors for these {size_3} kits (sum of their fragment vectors) are identical.")
    # print(f"  - This implies the total count of each AST node *type* across all extracted functions/methods (>= {MIN_FRAGMENT_NODES} nodes) is the same for these kits.")
    # print(f"  - Differences could still exist in:")
    # print(f"    * Node 'image' contents (e.g., variable names $a vs $b, string literals 'X' vs 'Y').")
    # print(f"    * Code *outside* of functions/methods (global scope code, includes, HTML).")
    # print(f"    * The specific arrangement or number of files containing these functions.")
    # print(f"    * Very small functions/methods (< {MIN_FRAGMENT_NODES} nodes) that were not included in the sum.")
    # print(f"    * Non-PHP files (images, CSS, JS) which are not represented in the ASTs.")
    # print("  - Essentially, these kits likely share the vast majority of their functional PHP code structure.")
else:
    print("No group of kits with identical summed fragment vectors found (or only groups of size 1).")


# --- Section 4: Rapport ---
# print("\n\n--- Section 4: Rapport ---")
# print("Please collect the results printed above and structure them in your PDF report.")
# print("Ensure you include:")
# print("  - The sizes of the largest groups found for each section (1.1a, 1.1b, 1.2, 2.1, 2.2, 3).")
# print("  - Example file paths/kit IDs for the largest groups.")
# print("  - The function names associated with the fragment groups (2.1, 2.2).")
# print("  - The Kit IDs for the largest group in Section 3.")
# print("  - Your observations and interpretations for each result, as prompted in the assignment and added in the print statements.")
# print("  - Discussion of any problems encountered during the analysis.")