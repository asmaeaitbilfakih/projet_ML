# ARBRE_Décision_C.py - Version complète et corrigée
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

print("=== DÉBUT DE L'ENTRAÎNEMENT ARBRE DE DÉCISION ===")

# Chargement des données
df = pd.read_csv('../European_Airbnb_Price_Classification.csv')
print("✅ Données chargées avec succès!")
print(f"Shape: {df.shape}")

# 1. Nettoyage initial
print("Shape initial:", df.shape)
print("\nValeurs manquantes:")
print(df.isnull().sum())

# Supprimer les lignes avec valeurs manquantes
df = df.dropna()
print("Shape après suppression des NA:", df.shape)

# 2. Renommer les colonnes si nécessaire
df = df.rename(columns={"dist": "center_dist"})

# 3. Traitement des outliers
def traiter_outliers(df, colonne):
    Q1 = df[colonne].quantile(0.25)
    Q3 = df[colonne].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[colonne] >= lower) & (df[colonne] <= upper)]

# Appliquer seulement sur les colonnes numériques
colonnes_numeriques = ["lat", "lng", "attr_index", "center_dist", "metro_dist", "guest_satisfaction_overall", "person_capacity", "bedrooms"]
for col in colonnes_numeriques:
    if col in df.columns:
        df = traiter_outliers(df, col)

print("Shape après traitement des outliers:", df.shape)

# 4. Analyse de la target
print(f"\n📊 Distribution de la target 'recommend':")
print(df['recommend'].value_counts())
print(f"Pourcentage de 'Recommandé': {df['recommend'].mean():.2%}")

# 5. Préparation des features et target
y = df["recommend"]

# Identifier les colonnes catégorielles et numériques
colonnes_categorielles = df.select_dtypes(include=['object']).columns.tolist()
colonnes_numeriques = df.select_dtypes(include=[np.number]).columns.tolist()

# Supprimer la target des features
if "recommend" in colonnes_numeriques:
    colonnes_numeriques.remove("recommend")

print(f"\n🔍 Colonnes catégorielles: {colonnes_categorielles}")
print(f"🔍 Colonnes numériques: {colonnes_numeriques}")

# Encoder les variables catégorielles
label_encoders = {}
for col in colonnes_categorielles:
    if col != "recommend":  # ne pas encoder la target
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Encodage de {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Features finales
X = df.drop(columns=["recommend"])

# 6. Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Split des données
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\n📐 Dimensions finales:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# 8. Entraînement du modèle
clf = DecisionTreeClassifier(
    criterion='gini', 
    max_depth=5,  # Augmenté pour plus de précision
    random_state=42
)

clf.fit(X_train, y_train)
print("✅ Modèle entraîné avec succès!")

# 9. Prédictions et évaluation
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Résultats:")
print(f"Accuracy: {accuracy:.4f}")

print("\nRapport de classification:")
print(classification_report(y_test, y_pred))

# 10. Information sur les features
feature_names = X.columns.tolist()
print(f"\n🔧 Features utilisées ({len(feature_names)}):")
for i, feature in enumerate(feature_names, 1):
    print(f"  {i}. {feature}")

# 11. Importance des features
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 Importance des features:")
print(feature_importance)

# 12. Sauvegarder le modèle pour l'utilisation web
models_dir = '../models_ai'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"✅ Dossier '{models_dir}' créé!")

# Sauvegarder le modèle, scaler et feature names
joblib.dump(clf, os.path.join(models_dir, 'decision_tree_model.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.pkl'))

# Sauvegarder les label encoders
joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoders.pkl'))

print("\n💾 SAUVEGARDE TERMINÉE:")
print(f"Modèle: {os.path.join(models_dir, 'decision_tree_model.pkl')}")
print(f"Scaler: {os.path.join(models_dir, 'scaler.pkl')}")
print(f"Features: {os.path.join(models_dir, 'feature_names.pkl')}")
print(f"Label encoders: {os.path.join(models_dir, 'label_encoders.pkl')}")

# Vérification finale
if (os.path.exists(os.path.join(models_dir, 'decision_tree_model.pkl')) and
    os.path.exists(os.path.join(models_dir, 'scaler.pkl')) and
    os.path.exists(os.path.join(models_dir, 'feature_names.pkl'))):
    print("\n🎉 SUCCÈS: Tous les fichiers ont été créés!")
    print("Vous pouvez maintenant utiliser le modèle dans Django.")
else:
    print("\n❌ ERREUR: Certains fichiers n'ont pas été créés.")

print("\n=== FIN DE L'ENTRAÎNEMENT ===")