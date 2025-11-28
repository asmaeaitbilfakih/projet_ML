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
    csv_path = os.path.join(os.path.dirname(current_dir), 'gold_selected_features.csv')
    
    print(f"Chargement du fichier: {csv_path}")
    
    
    
    df = pd.read_csv(csv_path)
    print("Fichier CSV chargé avec succès!")
    print(f"Dimensions: {df.shape}")

    df = df.dropna() # Nettoyer les données
    for column in df.columns: # Gérer les outliers avec la méthode IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
           
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    X = df.drop(columns=["gold close"])
    y = df["gold close"]
    
    scaler = StandardScaler() #Normalisation des données
    X_scaled = scaler.fit_transform(X)
    
    #Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Entraîner le modèle 
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)


    print(f"- Mean Squared Error: {mse:.2f}")
    print(f"- R² Score: {r2:.4f}")
    print(f"- RMSE: {np.sqrt(mse):.2f}")
    
    # Sauvegarder le modèle et le scaler 
    model_path = os.path.join(models_dir, 'linear_gold_model.pkl')
    scaler_path = os.path.join(models_dir, 'gold_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    return model, scaler, mse, r2

# Exécuter l'entraînement
if __name__ == "__main__":
    train_linear_gold_model()