import pandas as pd  
import numpy as np 
import matplotlib
# Set the backend to 'Agg' which doesn't require a GUI
matplotlib.use('Agg')  # This must be done before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
import os
import base64
from io import BytesIO
import json
import webbrowser
import time
import threading
from flask import Flask, render_template, request, jsonify

# Disattiviamo i messaggi di avviso
warnings.filterwarnings('ignore')

# Inizializziamo Flask
app = Flask(__name__)

# Variabili globali per il modello e lo scaler
optimized_model = None
scaler = None
X = None
df = None

# Configurazione dei percorsi
script_dir = os.path.dirname(os.path.abspath(__file__))
grafici_dir = os.path.join(script_dir, 'static', 'grafici')
if not os.path.exists(grafici_dir):
    os.makedirs(grafici_dir)
    print(f"Directory creata: {grafici_dir}")

def open_browser():
    time.sleep(1.5)  # Aspetta che il server sia completamente avviato
    webbrowser.open('http://127.0.0.1:5000/')

def encode_image(fig):
    """Converte un figura matplotlib in una stringa base64 per HTML"""
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)  # Chiude la figura per liberare memoria
    return f"data:image/png;base64,{encoded}"

@app.route('/')
def home():
    """Renderizza la pagina principale"""
    return render_template('index.html')

@app.route('/api/charts')
def get_charts():
    """Restituisce i grafici generati durante l'analisi"""
    global df, X, optimized_model
    charts_data = {}
    
    # Distribuzione target
    fig1 = plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Distribuzione Variabile Target')
    plt.xlabel('Presenza di Malattia Cardiaca (0=No, 1=Sì)')
    plt.ylabel('Conteggio')
    charts_data['target_distribution'] = encode_image(fig1)
    
    # Matrice di correlazione
    fig2 = plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matrice di Correlazione')
    charts_data['correlation_matrix'] = encode_image(fig2)
    
    # Se il modello ha feature importances, aggiungi anche quel grafico
    if hasattr(optimized_model, 'feature_importances_'):
        importances = optimized_model.feature_importances_
        feature_names = X.columns
        indices = np.argsort(importances)[::-1]
        
        fig3 = plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.title('Importanza delle Caratteristiche')
        plt.tight_layout()
        charts_data['feature_importance'] = encode_image(fig3)
    elif hasattr(optimized_model, 'coef_'):
        coefficients = optimized_model.coef_[0]
        feature_names = X.columns
        indices = np.argsort(np.abs(coefficients))[::-1]
        
        fig3 = plt.figure(figsize=(10, 6))
        plt.bar(range(len(coefficients)), coefficients[indices])
        plt.xticks(range(len(coefficients)), [feature_names[i] for i in indices], rotation=90)
        plt.title('Coefficienti del Modello')
        plt.tight_layout()
        charts_data['model_coefficients'] = encode_image(fig3)
    
    return jsonify(charts_data)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predice la probabilità di malattia cardiaca dato l'input dell'utente"""
    global optimized_model, scaler, X
    
    # Otteniamo i dati dal form
    data = request.json
    
    # Validazione dell'input
    for key, value in data.items():
        try:
            data[key] = float(value)
        except ValueError:
            return jsonify({'error': f'Valore non valido per {key}'}), 400
    
    # Create DataFrame from input data
    patient_data = [
        data['age'], data['sex'], data['cp'], data['trestbps'], 
        data['chol'], data['fbs'], data['restecg'], data['thalach'], 
        data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
    ]
    
    patient_df = pd.DataFrame([patient_data], columns=X.columns)
    
    # Apply scaling
    patient_scaled = scaler.transform(patient_df)
    
    # Make prediction
    prediction = int(optimized_model.predict(patient_scaled)[0])
    probability = float(optimized_model.predict_proba(patient_scaled)[0, 1])
    
    # Determine risk level
    risk = "BASSO"
    if probability > 0.7:
        risk = "ALTO"
    elif probability > 0.3:
        risk = "MEDIO"
    
    return jsonify({
        'prediction': prediction,
        'probability': probability,
        'risk': risk
    })

def train_model():
    """Addestra il modello di Machine Learning"""
    global optimized_model, scaler, X, df
    
    # CARICAMENTO DATASET
    print("1. CARICAMENTO DEL DATASET")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=column_names)
    
    print(f"Dimensioni del dataset: {df.shape}")
    
    # ESPLORAZIONE E PULIZIA DEI DATI
    print("\n2. ESPLORAZIONE E PULIZIA DEI DATI")
    
    df = df.replace('?', np.nan)
    
    for col in ['ca', 'thal']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in ['ca', 'thal']:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
    
    # Conversione della variabile target in binaria
    df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)
    
    # PREPARAZIONE DEI DATI
    print("\n3. PREPARAZIONE DEI DATI")
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    # ADDESTRAMENTO E VALUTAZIONE MODELLI
    print("\n4. ADDESTRAMENTO E VALUTAZIONE DEI MODELLI")
    
    def evaluate_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return model, acc
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_accuracy = 0
    
    for name, model in models.items():
        print(f"\nValutazione del modello: {name}")
        trained_model, accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = (trained_model, accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
    
    print(f"\nIl miglior modello è {best_model_name} con accuracy {best_accuracy:.4f}")
    
    # OTTIMIZZAZIONE MIGLIOR MODELLO
    print("\n5. OTTIMIZZAZIONE DEL MIGLIOR MODELLO")
    best_model = results[best_model_name][0]
    
    if best_model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        }
    elif best_model_name == 'K-Nearest Neighbors':
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance']
        }
    elif best_model_name == 'Support Vector Machine':
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    elif best_model_name == 'Decision Tree':
        param_grid = {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
    elif best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    
    grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(f"Migliori parametri: {grid_search.best_params_}")
    print(f"Miglior punteggio di cross-validation: {grid_search.best_score_:.4f}")
    
    optimized_model = grid_search.best_estimator_
    
    # CARATTERISTICHE PIÙ IMPORTANTI
    print("\n7. ANALISI DELLE CARATTERISTICHE PIÙ IMPORTANTI")
    if hasattr(optimized_model, 'feature_importances_'):
        importances = optimized_model.feature_importances_
        feature_names = X.columns
        indices = np.argsort(importances)[::-1]
        
        print("\nImportanza delle caratteristiche:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    elif hasattr(optimized_model, 'coef_'):
        coefficients = optimized_model.coef_[0]
        feature_names = X.columns
        indices = np.argsort(np.abs(coefficients))[::-1]
        
        print("\nCoefficienti del modello:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {feature_names[idx]}: {coefficients[idx]:.4f}")
    
    print("\n8. CONCLUSIONI")
    y_pred = optimized_model.predict(X_test)
    print(f"Il miglior modello è {best_model_name}, ottimizzato tramite GridSearchCV")
    print(f"L'accuracy sul test set è {accuracy_score(y_test, y_pred):.4f}.")

# Descrizioni e limiti per i campi del form
@app.route('/api/field_metadata')
def get_field_metadata():
    feature_descriptions = {
        'age': "Età del paziente in anni",
        'sex': "Sesso (0 = F, 1 = M)",
        'cp': "Tipo di dolore toracico (0: Angina tipica, 1: Angina atipica, 2: Dolore non anginoso, 3: Asintomatico)",
        'trestbps': "Pressione sanguigna a riposo in mm Hg",
        'chol': "Colesterolo sierico in mg/dl",
        'fbs': "Glicemia a digiuno > 120 mg/dl (0 = No, 1 = Sì)",
        'restecg': "Risultati elettrocardiografici a riposo (0: Normale, 1: Anomalia dell'onda ST-T, 2: Ipertrofia ventricolare sinistra)",
        'thalach': "Frequenza cardiaca massima raggiunta",
        'exang': "Angina indotta dall'esercizio (0 = No, 1 = Sì)",
        'oldpeak': "Depressione del segmento ST indotta dall'esercizio rispetto al riposo",
        'slope': "Pendenza del segmento ST durante l'esercizio (0: Ascendente, 1: Piatto, 2: Discendente)",
        'ca': "Numero di vasi principali colorati dalla fluoroscopia (0-4)",
        'thal': "Talassemia (1: Difetto fisso, 2: Normale, 3: Difetto reversibile)"
    }
    
    field_limits = {
        'age': [20, 100],
        'sex': [0, 1],
        'cp': [0, 3],
        'trestbps': [80, 250],
        'chol': [100, 600],
        'fbs': [0, 1],
        'restecg': [0, 2],
        'thalach': [60, 220],
        'exang': [0, 1],
        'oldpeak': [0.0, 10.0],
        'slope': [0, 2],
        'ca': [0, 4],
        'thal': [1, 3]
    }
    
    default_values = {
        'age': 55,
        'sex': 1,
        'cp': 1,
        'trestbps': 130,
        'chol': 150,
        'fbs': 0,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 0.0,
        'slope': 1,
        'ca': 0,
        'thal': 2
    }
    
    return jsonify({
        'descriptions': feature_descriptions,
        'limits': field_limits,
        'defaults': default_values
    })

if __name__ == "__main__":
    # Addestriamo il modello prima di avviare l'applicazione
    train_model()
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.start()

    # Avviamo l'applicazione Flask
    app.run(debug=False)  # Set debug to False to avoid reloading which can cause issues
