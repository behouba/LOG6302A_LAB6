### 1. Fichiers

#### 1.1 Clones “paramétriques” (vecteurs identiques, AST > 100 nœuds)

- **Plus gros groupe identifié**  
  - **47 fichiers** partagent exactement le même vecteur.  
  - Ils proviennent exclusivement de **2 kits** : `2099` et `4417`.  
- **Que remarque‑t‑on ?**  
  - Au sein du kit `2099`, de très nombreux chemins (« cc/index.php », « info/index.php », « start/index.php », …) sont **dupliqués** dans plusieurs sous‑répertoires.  
  - Le même phénomène se produit dans `4417`. Ces kits contiennent donc une même base de code copiée à l’identique, simplement répliquée sous plusieurs hashes de répertoire.  
- **En ignorant les duplications intra‑kit**  
  - On trouve un vecteur commun à **7 kits distincts** :  
    ```
    ['0415', '0485', '1110', '1442', '3135', '3662', '3716']
    ```  
  - Dans chaque kit, c’est le même fichier « anti2.php » (ou « anti10.php ») qui est repris à l’identique.  
  - **Conclusion** : ce composant “anti‑bot” est cloné tel quel dans de nombreux kits.

#### 1.2 Fichiers similaires (Manhattan ≤ 30 %, AST > 100 nœuds)

- **Plus gros groupe similaire**  
  - **81 fichiers** forment le plus grand ensemble.  
  - Ils couvrent **3 kits** : `2099`, `2110` et `4417`.  
- **Interprétation**  
  - Les trois kits partagent une structure de code très proche (mêmes branches « cc », « info », « start », « vbv »…), avec quelques variations mineures (≤ 30 % de différence en taille de vecteur).  
  - Cela révèle une base commune fortement réutilisée entre ces archives de phishing.

---

### 2. Fragments

#### 2.1 Clones “paramétriques” (fragments, > 10 nœuds)

- **Fragment le plus dupliqué (inter‑kit)**  
  - Présent dans **12 kits** :  
    ```
    ['0009','0229','1110','1651','2070','2099','2110','3180','3548','3676','4218','4417']
    ```  
  - **Nom de la fonction** (image du nœud) :  
    - Majoritairement `recurse_copy`  
    - Quelques occurrences sous le nom `dublicate`  
- **Conclusion**  
  - Cette routine de copie récursive est un composant commun à un très grand nombre de kits, repris sans modification ou sous une variante de nom.

#### 2.2 Fragments similaires (Manhattan ≤ 10 %, > 10 nœuds)

- **Plus gros groupe similaire**  
  - **16 fragments** issus de 16 kits :  
    ```
    ['0109','0229','0406','0415','0431','0481','1033',
     '1110','1442','1651','1926','2866','3148','3716','3858','4273']
    ```  
  - **Noms de fonction concernés** :  
    ```
    getOS, get_user_os, getOs, XB_OS, X_OS
    ```  
- **Interprétation**  
  - Il s’agit d’une **détection du système d’exploitation** implémentée sous différentes variantes de nom (casse, underscore, préfixe XB_ ou X_).  
  - La similarité montre qu’il s’agit du même algorithme, adapté nominalement.

---

### 3. Kits “paramétriques”

- **Calcul kit‑niveau** : on somme les vecteurs de tous les fragments d’un kit.  
- **Plus gros groupe de kits identiques**  
  - **3 kits** partagent exactement la même signature :  
    ```
    ['0398','0556','2784']
    ```  
- **Y a‑t‑il des différences de type de nœuds ?**  
  - Non : ces kits ont **la même distribution** de types de nœuds (leurs vecteurs agrégés sont identiques).  
  - Ils correspondent à des kits quasiment clones dans leur ensemble de fonctionnalités AST.# LOG6302A_LAB6
