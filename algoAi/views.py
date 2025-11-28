from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

def regLog_details(request):
    return render(request, 'regLog_details.html')

def regLog_atelier(request):
    return render(request, 'regLog_atelier.html')

def reglog_tester(request):
    return render(request,'reglog_tester.html')

def vehicles_form(request):
    return render(request, 'vehicles_form.html')

def reglog_results(request):
    return render(request,'reglog_results.html')



import os
import joblib

# 1. D'ABORD la fonction load_models
def load_models(name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models_ai')
    model_path = os.path.join(models_dir, name)
    ml_model = joblib.load(model_path)
    return ml_model

def linear_prediction(request):
    """Prédiction avec régression linéaire pour gold_close"""
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            nasdaq_open = float(request.POST.get('nasdaq_open', 0))
            nasdaq_high_low = float(request.POST.get('nasdaq_high_low', 0))
            nasdaq_close = float(request.POST.get('nasdaq_close', 0))
            oil_open = float(request.POST.get('oil_open', 0))
            oil_high_low = float(request.POST.get('oil_high_low', 0))
            oil_close = float(request.POST.get('oil_close', 0))
            gold_open = float(request.POST.get('gold_open', 0))
            gold_high_low = float(request.POST.get('gold_high_low', 0))
            
            # Charger le modèle de régression linéaire
            model = load_models('linear_gold_model.pkl')
            
            # Faire la prédiction (valeur exacte)
            predicted_gold_close = model.predict([[
                nasdaq_open, nasdaq_high_low, nasdaq_close,
                oil_open, oil_high_low, oil_close,
                gold_open, gold_high_low
            ]])
            
            predicted_value = predicted_gold_close[0]
            
            # Préparer les données pour l'affichage
            input_data = {
                'nasdaq_open': nasdaq_open,
                'nasdaq_high_low': nasdaq_high_low,
                'nasdaq_close': nasdaq_close,
                'oil_open': oil_open,
                'oil_high_low': oil_high_low,
                'oil_close': oil_close,
                'gold_open': gold_open,
                'gold_high_low': gold_high_low
            }
            
            context = {
                'predicted_gold_close': f"{predicted_value:.2f}",
                'initial_data': input_data,
                'model_type': 'Régression Linéaire'
            }
            
            return render(request, 'linear_results.html', context)
            
        except Exception as e:
            return render(request, 'prediction_or.html', {
                'error': f"Erreur lors de la prédiction: {str(e)}"
            })
    
    return render(request, 'prediction_or.html')

def linreg_details(request):
    """Page de détails sur la régression linéaire"""
    return render(request, 'linreg_details.html')

def linreg_atelier(request):
    """Page d'atelier pratique pour la régression linéaire"""
    return render(request, 'linreg_atelier.html')

def prediction_or(request):
    return render(request, 'prediction_or.html')

def linear_results(request):
    return render(request, 'linear_results.html')

def load_housing_model(model_name):
    """Charge les modèles de classification de logements"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models', model_name)
        return joblib.load(model_path)
    except Exception as e:
        print(f"Erreur chargement modèle {model_name}: {e}")
        return None

def reglog_tester(request):
    """Page de test du modèle de régression logistique"""
    return render(request, 'vehicles_form.html')

def housing_prediction(request):
    """Prédiction avec régression logistique pour les recommandations de logements"""
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            lat = float(request.POST.get('lat', 0))
            person_capacity = float(request.POST.get('person_capacity', 0))
            dist = float(request.POST.get('dist', 0))  # Note: 'dist' pas 'center_dist'
            bedrooms = float(request.POST.get('bedrooms', 0))
            metro_dist = float(request.POST.get('metro_dist', 0))
            guest_satisfaction = float(request.POST.get('guest_satisfaction_overall', 0))
            lng = float(request.POST.get('lng', 0))
            attr_index = float(request.POST.get('attr_index', 0))
            
            # Charger le modèle et le scaler
            model = load_housing_model('logistic_regression_model.pkl')
            scaler = load_housing_model('logistic_scaler.pkl')
            
            if model is None or scaler is None:
                return render(request, 'vehicles_form.html', {
                    'error': "Modèle non disponible. Veuillez contacter l'administrateur."
                })
            
            # Préparer les données et normaliser
            input_data = [[lat, person_capacity, dist, bedrooms, metro_dist, 
                         guest_satisfaction, lng, attr_index]]
            input_scaled = scaler.transform(input_data)
            
            # Faire la prédiction
            prediction = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)
            
            predicted_class = prediction[0]
            probability_recommended = probabilities[0][1] * 100
            
            # Préparer les données pour l'affichage
            input_data_display = {
                'lat': lat,
                'person_capacity': person_capacity,
                'dist': dist,
                'bedrooms': bedrooms,
                'metro_dist': metro_dist,
                'guest_satisfaction_overall': guest_satisfaction,
                'lng': lng,
                'attr_index': attr_index
            }
            
            context = {
                'prediction': predicted_class,
                'prediction_label': 'Recommandé' if predicted_class == 1 else 'Non recommandé',
                'probability': f"{probability_recommended:.2f}%",
                'initial_data': input_data_display,
                'model_type': 'Régression Logistique'
            }
            
            return render(request, 'reglog_results.html', context)
            
        except Exception as e:
            return render(request, 'vehicles_form.html', {
                'error': f"Erreur lors de la prédiction: {str(e)}"
            })
    
    return render(request, 'vehicles_form.html')


def decision_tree_atelier(request):
    """Page de prédiction avec l'arbre de décision"""
    return render(request, 'decision_tree_atelier.html') 
def decision_tree_details(request):
    """Page de prédiction avec l'arbre de décision"""
    return render(request, 'decision_tree_details.html') 
def decision_tree_prediction(request):
    """Page de prédiction avec l'arbre de décision"""
    return render(request, 'decision_tree_prediction.html') 
def decision_tree_results(request):
    """Page de prédiction avec l'arbre de décision"""
    return render(request, 'decision_tree_results.html') 
def prediction_tree_c(request):
    """Page de prédiction avec l'arbre de décision"""
    return render(request, 'prediction_tree_c.html') 



import numpy as np

# Fonction pour charger le modèle
def load_decision_tree_model():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models_ai')
        
        model_path = os.path.join(models_dir, 'decision_tree_model.pkl')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        feature_names_path = os.path.join(models_dir, 'feature_names.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(feature_names_path)
        
        print(f"✅ Modèle chargé avec {len(feature_names)} features")
        print(f"Features: {feature_names}")
        
        return model, scaler, feature_names
        
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        raise

# Vue pour la page d'accueil de l'arbre de décision
def decision_tree_atelier(request):
    return render(request, 'decision_tree_atelier.html')

# Vue pour les détails de l'algorithme
def decision_tree_details(request):
    return render(request, 'decision_tree_details.html')

# Vue pour le formulaire de prédiction
def decision_tree_prediction(request):
    if request.method == 'POST':
        try:
            # Fonctions helper pour récupérer les valeurs
            def get_float_value(post_data, key, default=0.0):
                value = post_data.get(key)
                if value is None or value == '':
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            def get_string_value(post_data, key, default=""):
                value = post_data.get(key)
                if value is None or value == '':
                    return default
                return value
            
            # Récupérer toutes les valeurs du formulaire
            lat = get_float_value(request.POST, 'lat', 52.3676)
            lng = get_float_value(request.POST, 'lng', 4.9041)
            attr_index = get_float_value(request.POST, 'attr_index', 100.0)
            center_dist = get_float_value(request.POST, 'center_dist', 2.0)
            metro_dist = get_float_value(request.POST, 'metro_dist', 0.5)
            guest_satisfaction = get_float_value(request.POST, 'guest_satisfaction_overall', 85.0)
            person_capacity = get_float_value(request.POST, 'person_capacity', 2.0)
            bedrooms = get_float_value(request.POST, 'bedrooms', 1.0)
            room_type = get_string_value(request.POST, 'room_type', 'Entire home/apt')
            city = get_string_value(request.POST, 'city', 'Amsterdam')
            
            # DEBUG: Afficher les valeurs reçues
            print("=== VALEURS RECUES DU FORMULAIRE ===")
            print(f"lat: {lat}")
            print(f"lng: {lng}")
            print(f"attr_index: {attr_index}")
            print(f"center_dist: {center_dist}")
            print(f"metro_dist: {metro_dist}")
            print(f"guest_satisfaction: {guest_satisfaction}")
            print(f"person_capacity: {person_capacity}")
            print(f"bedrooms: {bedrooms}")
            print(f"room_type: {room_type}")
            print(f"city: {city}")
            
            # Charger le modèle, scaler et feature names
            model, scaler, feature_names = load_decision_tree_model()
            
            print("=== FEATURES ATTENDUES PAR LE MODELE ===")
            print(f"Features: {feature_names}")
            print(f"Nombre de features: {len(feature_names)}")
            
            # Mapping des valeurs catégorielles
            room_type_mapping = {
                'Entire home/apt': 0, 
                'Private room': 1, 
                'Shared room': 2,
                'Hotel room': 3
            }
            
            city_mapping = {
                'Amsterdam': 0, 
                'Barcelona': 1, 
                'Berlin': 2, 
                'Paris': 3, 
                'Rome': 4,
                'London': 5,
                'Vienna': 6
            }
            
            room_type_encoded = room_type_mapping.get(room_type, 0)
            city_encoded = city_mapping.get(city, 0)
            
            # Préparer toutes les valeurs dans un dictionnaire
            all_values = {
                'lat': lat,
                'lng': lng,
                'attr_index': attr_index,
                'center_dist': center_dist,
                'metro_dist': metro_dist,
                'guest_satisfaction_overall': guest_satisfaction,
                'person_capacity': person_capacity,
                'bedrooms': bedrooms,
                'room_type': room_type_encoded,
                'city': city_encoded
            }
            
            # Créer le tableau dans l'ordre exact des features du modèle
            input_data = []
            for feature in feature_names:
                if feature in all_values:
                    input_data.append(all_values[feature])
                else:
                    # Si une feature manque, utiliser 0 et logger un warning
                    print(f"⚠️ Feature manquante: {feature}")
                    input_data.append(0)
            
            input_array = np.array([input_data])
            
            print("=== DONNEES ENVOYEES AU MODELE ===")
            print(f"Données: {input_array}")
            print(f"Shape: {input_array.shape}")
            
            # Appliquer la même normalisation
            input_scaled = scaler.transform(input_array)
            
            # Faire la prédiction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            print("=== RESULTATS DE PREDICTION ===")
            print(f"Prédiction: {prediction}")
            print(f"Probabilités: {prediction_proba}")
            
            # Interpréter le résultat
            recommendation = "Recommandé" if prediction[0] == 1 else "Non Recommandé"
            confidence = prediction_proba[0][prediction[0]] * 100
            
            # Préparer le contexte pour le template
            context = {
                'recommendation': recommendation,
                'confidence': f"{confidence:.1f}%",
                'probability_recommended': f"{prediction_proba[0][1] * 100:.1f}%",
                'probability_not_recommended': f"{prediction_proba[0][0] * 100:.1f}%",
                'input_data': {
                    'lat': lat,
                    'lng': lng,
                    'attr_index': attr_index,
                    'center_dist': center_dist,
                    'metro_dist': metro_dist,
                    'guest_satisfaction': guest_satisfaction,
                    'person_capacity': person_capacity,
                    'bedrooms': bedrooms,
                    'room_type': room_type,
                    'city': city,
                }
            }
            
            return render(request, 'decision_tree_results.html', context)
            
        except Exception as e:
            # En cas d'erreur
            import traceback
            error_details = traceback.format_exc()
            print(f"ERREUR DETAILLEE: {error_details}")
            
            error_context = {
                'error_message': f"Une erreur s'est produite lors de la prédiction: {str(e)}"
            }
            return render(request, 'decision_tree_results.html', error_context)
    
    # Si ce n'est pas une requête POST, afficher le formulaire
    return render(request, 'prediction_tree_c.html')

# Vue pour tester le modèle (optionnel)
def decision_tree_tester(request):
    return render(request, 'prediction_tree_c.html')




