# ModelFinal.py
# Classification hiérarchique du rapport molaire (RM) en 4 classes :
#   0 : [1.1, 1.16)   (dans l'intervalle, bas)
#   1 : [1.16, 1.2]   (dans l'intervalle, haut)
#   2 : rm < 1.1      (hors intervalle, bas)
#   3 : rm > 1.2      (hors intervalle, haut)
# Modèle : XGBoost (détection dans [1.1,1.2]) + LightGBM (affinage dans chaque région)

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import xgboost as xgb

# =============================================================================
# CONFIGURATION
# =============================================================================
FICHIER_DONNEES = 'db_net_7.csv'   # Base préparée (imputée, sans outliers)
SEUIL_BAS = 1.1
SEUIL_HAUT = 1.2
SEUIL_MILIEU = 1.16                  # séparation à l'intérieur de [1.1,1.2]
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Hyperparamètres pour XGBoost (étape 1)
XGB_PARAMS = {
    'colsample_bytree': 0.6,
    'gamma': 0,
    'learning_rate': 0.05,
    'max_depth': 3,
    'min_child_weight': 1,
    'n_estimators': 200,
    'subsample': 0.8,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# Hyperparamètres pour LightGBM (étape 2)
LGB_PARAMS = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.05,
    'max_depth': 3,
    'min_child_samples': 20,
    'n_estimators': 200,
    'subsample': 0.6,
    'random_state': 42,
    'verbose': -1
}

def preparer_donnees(df):
    """
    Nettoie le DataFrame : suppression de colonnes non utiles,
    création de la cible hiérarchique (4 classes).
    Retourne X (features), y (code 0-3) et la liste des features.
    """
    df = df.copy()
    # Supprimer les colonnes indésirables si présentes
    cols_a_supprimer = ['datetime', 'date', 'RM1', 'classe_rm', 'cible']
    for col in cols_a_supprimer:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            print(f"Colonne supprimée : {col}")
    
    # Vérifier la présence de 'rm'
    if 'rm' not in df.columns:
        raise ValueError("La colonne cible 'rm' est absente du fichier.")
    
    # Définition de la classe réelle (0..3)
    def real_class(rm):
        inside = SEUIL_BAS <= rm <= SEUIL_HAUT
        low = rm < SEUIL_MILIEU
        if inside and low:
            return 0   # [1.1, 1.16)
        elif inside and not low:
            return 1   # [1.16, 1.2]
        elif not inside and low:
            return 2   # rm < 1.1
        else:
            return 3   # rm > 1.2
    
    df['classe_reelle'] = df['rm'].apply(real_class)
    
    # Séparer features (X) et cible (y)
    X = df.drop(columns=['rm', 'classe_reelle'], errors='ignore')
    # Ne garder que les colonnes numériques
    X = X.select_dtypes(include=[np.number])
    y = df['classe_reelle']
    
    # Supprimer les lignes avec des NaN (normalement plus, mais sécurité)
    avant = len(X)
    X = X.dropna()
    y = y.loc[X.index]
    apres = len(X)
    if apres < avant:
        print(f"Attention : {avant - apres} lignes supprimées car valeurs manquantes.")
    
    return X, y, list(X.columns)

# =============================================================================
# ENTRAÎNEMENT DES 3 MODÈLES HIÉRARCHIQUES
# =============================================================================
def entrainer_modele_hiérarchique(X, y):
    """
    Entraîne :
        - model_A : XGBoost binaire (dans [1.1,1.2] ?)
        - model_B_in : LightGBM binaire (selon <1.16 ?) pour les lignes dans l'intervalle
        - model_B_out : LightGBM binaire (selon <1.1 ?) pour les lignes hors intervalle
    Retourne les trois modèles, les indices de split, et les métriques.
    """
    # Train / test split (stratifié sur les 4 classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # --- Variable A : est-ce que rm est dans [1.1, 1.2] ? (classes 0 ou 1) ---
    y_train_A = y_train.isin([0,1])   # True si classe 0 ou 1 (dans intervalle)
    y_test_A  = y_test.isin([0,1])
    
    # --- Sous-classes pour ceux DANS l'intervalle (A=true) : classe 0 vs 1 ---
    train_in_A = X_train.index[y_train_A]
    test_in_A  = X_test.index[y_test_A]
    # y_train_B_inA = 1 si classe 0 ([1.1,1.16)), 0 si classe 1 ([1.16,1.2])
    y_train_B_inA = (y_train.loc[train_in_A] == 0).astype(int)
    y_test_B_inA  = (y_test.loc[test_in_A] == 0).astype(int)
    
    # --- Sous-classes pour ceux HORS intervalle (A=false) : classe 2 vs 3 ---
    train_out_A = X_train.index[~y_train_A]
    test_out_A  = X_test.index[~y_test_A]
    # y_train_B_outA = 1 si classe 2 (rm<1.1), 0 si classe 3 (rm>1.2)
    y_train_B_outA = (y_train.loc[train_out_A] == 2).astype(int)
    y_test_B_outA  = (y_test.loc[test_out_A] == 2).astype(int)
    
    print(f"Train : dans intervalle = {len(train_in_A)}, hors intervalle = {len(train_out_A)}")
    print(f"Test  : dans intervalle = {len(test_in_A)}, hors intervalle = {len(test_out_A)}")
    
    # --- Modèle XGBoost pour prédire A ---
    model_A = xgb.XGBClassifier(**XGB_PARAMS)
    model_A.fit(X_train, y_train_A)
    y_pred_A = model_A.predict(X_test)
    acc_A = accuracy_score(y_test_A, y_pred_A)
    print(f"XGBoost (dans intervalle) - Accuracy : {acc_A:.4f}")
    
    # --- Modèle LightGBM pour les échantillons DANS intervalle ---
    model_B_in = None
    if len(train_in_A) > 1:
        model_B_in = lgb.LGBMClassifier(**LGB_PARAMS)
        model_B_in.fit(X_train.loc[train_in_A], y_train_B_inA)
        # Évaluation sur test_in_A
        if len(test_in_A) > 0:
            y_pred_B_in = model_B_in.predict(X_test.loc[test_in_A])
            acc_B_in = accuracy_score(y_test_B_inA, y_pred_B_in)
            print(f"LightGBM (dans intervalle) - Accuracy : {acc_B_in:.4f}")
    else:
        print("Pas assez d'échantillons DANS l'intervalle pour LightGBM.")
    
    # --- Modèle LightGBM pour les échantillons HORS intervalle ---
    model_B_out = None
    if len(train_out_A) > 1:
        model_B_out = lgb.LGBMClassifier(**LGB_PARAMS)
        model_B_out.fit(X_train.loc[train_out_A], y_train_B_outA)
        if len(test_out_A) > 0:
            y_pred_B_out = model_B_out.predict(X_test.loc[test_out_A])
            acc_B_out = accuracy_score(y_test_B_outA, y_pred_B_out)
            print(f"LightGBM (hors intervalle) - Accuracy : {acc_B_out:.4f}")
    else:
        print("Pas assez d'échantillons HORS intervalle pour LightGBM.")
    
    # --- Prédiction finale sur l'ensemble test (pour évaluation globale) ---
    def predict_final(X_row, pred_A, model_in, model_out):
        if pred_A:   # prédit dans l'intervalle
            if model_in is not None:
                sub = model_in.predict(X_row)[0]
                return 0 if sub == 1 else 1  # sub=1 -> classe0, sub=0 -> classe1
            else:
                return 0  # fallback
        else:        # prédit hors intervalle
            if model_out is not None:
                sub = model_out.predict(X_row)[0]
                return 2 if sub == 1 else 3  # sub=1 -> classe2, sub=0 -> classe3
            else:
                return 2
    
    y_pred_final = []
    for i, idx in enumerate(X_test.index):
        X_row = X_test.loc[[idx]]
        pred_A = y_pred_A[i]
        yp = predict_final(X_row, pred_A, model_B_in, model_B_out)
        y_pred_final.append(yp)
    y_pred_final = np.array(y_pred_final)
    
    acc_final = accuracy_score(y_test, y_pred_final)
    print(f"\n--- Classification hiérarchique finale (4 classes) ---")
    print(f"Accuracy globale : {acc_final:.4f}")
    print("\nRapport de classification :")
    target_names = ['[1.1,1.16)', '[1.16,1.2]', 'rm<1.1', 'rm>1.2']
    print(classification_report(y_test, y_pred_final, target_names=target_names))
    
    return model_A, model_B_in, model_B_out

def sauvegarder_modele(model_A, model_B_in, model_B_out, features, chemin='modele_hiérarchique.pkl'):
    """
    Sauvegarde les trois modèles, les features et les seuils.
    """
    data = {
        'model_A': model_A,
        'model_B_in': model_B_in,
        'model_B_out': model_B_out,
        'features': features,
        'seuils': {'bas': SEUIL_BAS, 'haut': SEUIL_HAUT, 'milieu': SEUIL_MILIEU}
    }
    joblib.dump(data, chemin)
    print(f"Modèle sauvegardé dans {chemin}")

# =============================================================================
# FONCTION DE PRÉDICTION (pour le monitoring)
# =============================================================================
def predire_classe_rm(variables):
    """
    Prédit la classe du rapport molaire à partir d'un dictionnaire ou d'un DataFrame.
    
    Paramètres
    ----------
    variables : dict ou pandas.DataFrame
        Doit contenir toutes les features utilisées lors de l'entraînement.
    
    Retourne
    --------
    classe : int (0,1,2,3)
    label  : str (description de la classe)
    proba  : dict des probabilités pour les 4 classes finales (optionnel)
    """
    # Chargement du modèle (une seule fois, en cache)
    if not hasattr(predire_classe_rm, 'modele'):
        try:
            data = joblib.load('modele_hiérarchique.pkl')
            predire_classe_rm.model_A = data['model_A']
            predire_classe_rm.model_B_in = data['model_B_in']
            predire_classe_rm.model_B_out = data['model_B_out']
            predire_classe_rm.features = data['features']
            predire_classe_rm.seuils = data['seuils']
        except FileNotFoundError:
            raise FileNotFoundError("Modèle non trouvé. Exécutez d'abord l'entraînement.")
    
    # Convertir en DataFrame si nécessaire
    if isinstance(variables, dict):
        X = pd.DataFrame([variables])
    else:
        X = variables.copy()
    
    # Vérifier que toutes les features sont présentes
    manquantes = set(predire_classe_rm.features) - set(X.columns)
    if manquantes:
        raise ValueError(f"Variables manquantes : {manquantes}")
    X = X[predire_classe_rm.features]
    
    # Prédiction étape A (XGBoost)
    pred_A = predire_classe_rm.model_A.predict(X)[0]  # True/False (bool)
    # Prédiction sous-classe
    if pred_A:
        if predire_classe_rm.model_B_in is not None:
            sub = predire_classe_rm.model_B_in.predict(X)[0]
            classe = 0 if sub == 1 else 1
        else:
            classe = 0
    else:
        if predire_classe_rm.model_B_out is not None:
            sub = predire_classe_rm.model_B_out.predict(X)[0]
            classe = 2 if sub == 1 else 3
        else:
            classe = 2
    
    label = {
        0: f"[{predire_classe_rm.seuils['bas']}, {predire_classe_rm.seuils['milieu']})",
        1: f"[{predire_classe_rm.seuils['milieu']}, {predire_classe_rm.seuils['haut']}]",
        2: f"rm < {predire_classe_rm.seuils['bas']}",
        3: f"rm > {predire_classe_rm.seuils['haut']}"
    }[classe]
    
    # Calcul approximatif des probabilités finales
    proba_A = predire_classe_rm.model_A.predict_proba(X)[0]  # [p(hors), p(dans)] si modèle binaire
    p_in = proba_A[1]   # probabilité d'être dans intervalle
    p_out = proba_A[0]
    
    proba_final = np.zeros(4)
    if predire_classe_rm.model_B_in is not None and predire_classe_rm.model_B_out is not None:
        proba_in = predire_classe_rm.model_B_in.predict_proba(X)[0]   # [p(classe1), p(classe0)] ? Attention ordre
        proba_out = predire_classe_rm.model_B_out.predict_proba(X)[0]
        # Pour LightGBM, le premier élément correspond à la classe 0 (négative) et le second à la classe 1 (positive)
        # Ici, pour 'in' : classe 1 = bas (rm<1.16), classe 0 = haut (rm>=1.16)
        # Pour 'out': classe 1 = bas (rm<1.1), classe 0 = haut (rm>1.2)
        p_low_inside = proba_in[1] if len(proba_in) > 1 else 0.5
        p_low_outside = proba_out[1] if len(proba_out) > 1 else 0.5
        
        proba_final[0] = p_in * p_low_inside      # [1.1,1.16)
        proba_final[1] = p_in * (1 - p_low_inside) # [1.16,1.2]
        proba_final[2] = p_out * p_low_outside     # rm<1.1
        proba_final[3] = p_out * (1 - p_low_outside) # rm>1.2
        # Normalisation (éviter les erreurs d'arrondi)
        proba_final = proba_final / proba_final.sum()
        proba_detail = {
            '[1.1,1.16)': proba_final[0],
            '[1.16,1.2]': proba_final[1],
            'rm<1.1': proba_final[2],
            'rm>1.2': proba_final[3]
        }
    else:
        proba_detail = None
    
    return classe, label, proba_detail

# =============================================================================
# EXÉCUTION PRINCIPALE (entraînement)
# =============================================================================
if __name__ == "__main__":
    print("Chargement de db_net_7.csv...")
    df = pd.read_csv(FICHIER_DONNEES)
    print(f"Dimensions : {df.shape}")
    
    X, y, features = preparer_donnees(df)
    print(f"Nombre d'observations : {len(X)}")
    print(f"Nombre de features : {len(features)}")
    
    print("\nEntraînement du modèle hiérarchique (XGBoost → LightGBM)...")
    model_A, model_B_in, model_B_out = entrainer_modele_hiérarchique(X, y)
    
    sauvegarder_modele(model_A, model_B_in, model_B_out, features)
    
    # Petit test sur la première ligne
    print("\nTest de prédiction sur la première observation :")
    premiere_ligne = X.iloc[0].to_dict()
    classe, label, probas = predire_classe_rm(premiere_ligne)
    print(f"Classe prédite : {classe} → {label}")
    if probas:
        print("Probabilités associées :")
        for k, v in probas.items():
            print(f"  {k}: {v:.3f}")