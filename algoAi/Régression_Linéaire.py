import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_linear_gold_model():
    # 1. Charger les données
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(current_dir), 'gold_selected_features3 (2).csv')
    
    print(f"Chargement du fichier: {csv_path}")
    
    if not os.path.exists(csv_path):
        print("ERREUR: Fichier CSV non trouvé!")
        return
    
    df = pd.read_csv(csv_path)
    print("Fichier CSV chargé avec succès!")
    print(f"Dimensions: {df.shape}")
    
    # 2. Nettoyer les données
    df = df.dropna()
    print(f"Après nettoyage: {df.shape}")
    
    # 3. Préparer les features (X) et la target (y)
    # Features : 8 variables
    X = df[[
        'nasdaq open', 'nasdaq_high_low', 'nasdaq close',
        'oil open', 'oil_high_low', 'oil close', 
        'gold open', 'gold_high_low'
    ]]
    
    # Target : gold_close (valeur exacte)
    y = df['gold close']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}")
    
    # 4. Diviser en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 5. Entraîner le modèle de régression linéaire
    print("Entraînement du modèle Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 6. Évaluer le modèle
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Performance du modèle:")
    print(f"- Mean Squared Error: {mse:.2f}")
    print(f"- R² Score: {r2:.2f}")
    print(f"- RMSE: {np.sqrt(mse):.2f}")
    
    # 7. Afficher les coefficients
    print("\nCoefficients du modèle:")
    features = X.columns
    for i, coef in enumerate(model.coef_):
        print(f"  {features[i]}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    
    # 8. Sauvegarder le modèle
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models_ai')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'linear_gold_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nModèle sauvegardé: {model_path}")
    
    return model, mse, r2

# Exécuter l'entraînement
if __name__ == "__main__":
    train_linear_gold_model()