import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_logistic_regression_model():
    # 1. Charger les données
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '../European_Airbnb_Price_Classification.csv')  # Adaptez le nom du fichier
    
    print(f"Chargement du fichier: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print("Fichier CSV chargé avec succès!")
    print(f"Dimensions: {df.shape}")
    print("\nAperçu des données:")
    print(df.head())
    print(f"\nDistribution de la variable cible 'recommend':")
    print(df['recommend'].value_counts())

    # Nettoyer les données
    df = df.dropna()
    print(f"\nDimensions après suppression des NaN: {df.shape}")

    # Gérer les outliers avec la méthode IQR
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
           
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    # Préparer les features et la target
    X = df.drop(columns=["recommend"])
    y = df["recommend"]
    
    print(f"\nFeatures utilisées: {list(X.columns)}")
    print(f"Target: recommend")
    
    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nSplit des données:")
    print(f"- Train: {X_train.shape[0]} échantillons")
    print(f"- Test: {X_test.shape[0]} échantillons")
    
    # Entraîner le modèle de régression logistique
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Évaluation du modèle
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== PERFORMANCES DU MODÈLE ===")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"\nMatrice de confusion:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nRapport de classification:")
    print(classification_report(y_test, y_pred))
    
    # Affichage des coefficients
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\nImportance des features (coefficients):")
    print(feature_importance)
    
    # Sauvegarder le modèle et le scaler
    models_dir = os.path.join(current_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'logistic_regression_model.pkl')
    scaler_path = os.path.join(models_dir, 'logistic_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModèle sauvegardé dans: {model_path}")
    print(f"Scaler sauvegardé dans: {scaler_path}")
    
    return model, scaler, accuracy, feature_importance

# Fonction pour faire des prédictions
def predict_recommendation(new_data, model, scaler):
    """
    Fonction pour prédire si un logement sera recommandé
    """
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    probability = model.predict_proba(new_data_scaled)
    
    return prediction, probability

# Exécuter l'entraînement
if __name__ == "__main__":
    model, scaler, accuracy, feature_importance = train_logistic_regression_model()
    
    # Exemple de prédiction
    print(f"\n=== EXEMPLE DE PRÉDICTION ===")
    example_data = pd.DataFrame({
        'lat': [52.38],
        'person_capacity': [2.0],
        'dist': [1.5],
        'bedrooms': [1],
        'metro_dist': [0.5],
        'guest_satisfaction_overall': [95.0],
        'lng': [4.90],
        'attr_index': [300.0]
    })
    
    pred, proba = predict_recommendation(example_data, model, scaler)
    print(f"Prédiction: {pred[0]} (0 = Non recommandé, 1 = Recommandé)")
    print(f"Probabilités: [Non recommandé: {proba[0][0]:.3f}, Recommandé: {proba[0][1]:.3f}]")