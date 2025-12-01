import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_linear_gold_model():
    # 1. Charger les données
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    # Un seul nom de fichier
    csv_path = os.path.join(project_dir, 'gold_selected_features.csv')
    
    # Créer le dossier models
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Charger les données
    df = pd.read_csv(csv_path)
    
    # Vérifier les valeurs manquantes
    print("Valeurs manquantes avant nettoyage:")
    print(df.isnull().sum())

    # Remplir les valeurs manquantes pour TOUTES les colonnes
    df["nasdaq open"] = df["nasdaq open"].fillna(df["nasdaq open"].mean())
    df["nasdaq_high_low"] = df["nasdaq_high_low"].fillna(df["nasdaq_high_low"].mean())
    df["nasdaq close"] = df["nasdaq close"].fillna(df["nasdaq close"].mean())
    df["oil open"] = df["oil open"].fillna(df["oil open"].mean())
    df["oil_high_low"] = df["oil_high_low"].fillna(df["oil_high_low"].mean())
    df["oil close"] = df["oil close"].fillna(df["oil close"].mean())
    df["gold open"] = df["gold open"].fillna(df["gold open"].mean())
    df["gold_high_low"] = df["gold_high_low"].fillna(df["gold_high_low"].mean())
    df["gold close"] = df["gold close"].fillna(df["gold close"].mean())  # AJOUTÉ

    # Vérifier qu'il n'y a plus de NaN
    print("Valeurs manquantes après nettoyage:")
    print(df.isnull().sum())
    
    # Gérer les outliers
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    # Préparer les données
    X = df.drop(columns=["gold close"])
    y = df["gold close"]
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Vérifier qu'il n'y a pas de NaN dans y
    print(f"NaN dans y: {y.isnull().sum()}")
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Entraînement
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prédictions
    predictions = model.predict(X_test)
    
    # Évaluation
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n=== RÉSULTATS DU MODÈLE ===")
    print(f"- Mean Squared Error: {mse:.2f}")
    print(f"- R² Score: {r2:.4f}")
    print(f"- RMSE: {np.sqrt(mse):.2f}")
    
    # Sauvegarder
    model_path = os.path.join(models_dir, 'linear_gold_model.pkl')
    scaler_path = os.path.join(models_dir, 'gold_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModèle sauvegardé dans: {model_path}")
    
    return model, scaler, mse, r2

if __name__ == "__main__":
    train_linear_gold_model()