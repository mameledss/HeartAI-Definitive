import pandas as pd #per manipolazione dati in formato di tabella
import numpy as np #libreria per calcoli numerici
import matplotlib #libreria per grafici
matplotlib.use('Agg') #per generare grafici senza GUI
import matplotlib.pyplot as plt #per creazione grafici
import seaborn as sns #per grafici statistici avanzati
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV #funzioni per divisione dataset e ottimizzazione modelli
from sklearn.preprocessing import StandardScaler #per normalizzare i dati
from sklearn.tree import DecisionTreeClassifier #decision tree 
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.svm import SVC #SVC
from sklearn.linear_model import LogisticRegression #regressione logistica
from sklearn.ensemble import RandomForestClassifier #random forest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc #metriche per valutazione modelli
import warnings #per gestire messaggi di avviso
import os #per interazione con sistema operativo
import base64 #per codifica/decodifica base64
from io import BytesIO #per gestire dati binari in memoria
import json #per lavorare con formato JSON
import webbrowser #per aprire il browser
import time #per gestire ritardi
import threading #per gestire thread paralleli
from flask import Flask, render_template, request, jsonify #importa componenti Flask per web server

warnings.filterwarnings('ignore') #disattiva i warning per evitare messaggi non necessari

app = Flask(__name__) #crea istanza dell'applicazione Flask

optimized_model = None #variabile per memorizzare il modello ottimizzato
scaler = None #lo scaler dei dati
X = None #le features del dataset
df = None #e il dataframe completo

script_dir = os.path.dirname(os.path.abspath(__file__)) #ottiene la directory del file corrente
grafici_dir = os.path.join(script_dir, 'static', 'grafici') #crea percorso per directory grafici
if not os.path.exists(grafici_dir): #se la directory non esiste
    os.makedirs(grafici_dir) #la crea
    print(f"Directory creata: {grafici_dir}")

def open_browser():
    time.sleep(1.5) #aspetta che il server sia completamente avviato
    webbrowser.open('http://127.0.0.1:5000/') #apre il browser all'indirizzo locale

def encode_image(fig):
    """Converte un figura matplotlib in una stringa base64 per HTML"""
    img = BytesIO() #crea un buffer di memoria
    fig.savefig(img, format='png', bbox_inches='tight') #salva la figura nel buffer come PNG
    img.seek(0) #riporta il puntatore all'inizio del buffer
    encoded = base64.b64encode(img.getvalue()).decode('utf-8') #codifica l'immagine in base64
    plt.close(fig) #chiude la figura
    return f"data:image/png;base64,{encoded}" #restituisce l'immagine come stringa data URL

@app.route('/') #renderizza la pagina principale
def home():
    return render_template('index.html') #restituisce il template HTML principale

@app.route('/api/charts') #restituisce grafici generati dopo l'analisi
def get_charts():
    global df, X, optimized_model # Usa le variabili globali
    charts_data = {} # Dizionario per memorizzare i dati dei grafici
    
    fig1 = plt.figure(figsize=(8, 6)) #crea figura matplotlib di dimensione 8x6
    sns.countplot(x='target', data=df) #crea grafico a barre della distribuzione variabile target
    plt.title('Distribuzione Variabile Target') #titolo del grafico
    plt.xlabel('Presenza di Malattia Cardiaca (0=No, 1=Sì)') #etichetta asse x
    plt.ylabel('Conteggio') #etichetta asse y
    charts_data['target_distribution'] = encode_image(fig1) #codifica e salva grafico
    
    fig2 = plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr() #calcola la matrice di correlazione
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f') #crea heatmap con annotazioni
    plt.title('Matrice di Correlazione')
    charts_data['correlation_matrix'] = encode_image(fig2) #codifica e salva grafico
    
    if hasattr(optimized_model, 'feature_importances_'): #se il modello ha "feature_importances_"
        importances = optimized_model.feature_importances_ #ottiene i valori di importanza
        feature_names = X.columns #ottiene i nomi delle features
        indices = np.argsort(importances)[::-1] #ordina gli indici in base all'importanza (decrescente)
        
        fig3 = plt.figure(figsize=(10, 6)) #crea una nuova figura matplotlib
        plt.bar(range(len(importances)), importances[indices])  # Crea grafico a barre dell'importanza
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)  # Imposta etichette asse x ruotate
        plt.title('Importanza delle Caratteristiche')  # Aggiunge titolo al grafico
        plt.tight_layout()  # Aggiusta layout per evitare sovrapposizioni
        charts_data['feature_importance'] = encode_image(fig3)  # Codifica e salva grafico
    elif hasattr(optimized_model, 'coef_'):  # Se il modello ha coefficienti (es. regressione logistica)
        coefficients = optimized_model.coef_[0]  # Ottiene i coefficienti
        feature_names = X.columns  # Ottiene i nomi delle features
        indices = np.argsort(np.abs(coefficients))[::-1]  # Ordina gli indici in base al valore assoluto dei coefficienti
        
        fig3 = plt.figure(figsize=(10, 6))  # Crea una nuova figura matplotlib
        plt.bar(range(len(coefficients)), coefficients[indices])  # Crea grafico a barre dei coefficienti
        plt.xticks(range(len(coefficients)), [feature_names[i] for i in indices], rotation=90)  # Imposta etichette asse x ruotate
        plt.title('Coefficienti del Modello')  # Aggiunge titolo al grafico
        plt.tight_layout()  # Aggiusta layout per evitare sovrapposizioni
        charts_data['model_coefficients'] = encode_image(fig3)  # Codifica e salva grafico
    
    return jsonify(charts_data)  # Restituisce dati dei grafici in formato JSON

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predice la probabilità di malattia cardiaca dato l'input dell'utente"""
    global optimized_model, scaler, X  # Usa le variabili globali
    
    # Otteniamo i dati dal form
    data = request.json  # Estrae dati JSON dalla richiesta
    
    # Validazione dell'input
    for key, value in data.items():
        try:
            data[key] = float(value)  # Converte ogni valore in float
        except ValueError:
            return jsonify({'error': f'Valore non valido per {key}'}), 400  # Restituisce errore se la conversione fallisce
    
    # Create DataFrame from input data
    patient_data = [
        data['age'], data['sex'], data['cp'], data['trestbps'], 
        data['chol'], data['fbs'], data['restecg'], data['thalach'], 
        data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
    ]  # Crea lista con i dati del paziente
    
    patient_df = pd.DataFrame([patient_data], columns=X.columns)  # Crea DataFrame con stesse colonne di X
    
    # Apply scaling
    patient_scaled = scaler.transform(patient_df)  # Applica la stessa normalizzazione usata per il training
    
    # Make prediction
    prediction = int(optimized_model.predict(patient_scaled)[0])  # Predice la classe (0 o 1)
    probability = float(optimized_model.predict_proba(patient_scaled)[0, 1])  # Ottiene la probabilità della classe positiva
    
    # Determine risk level
    risk = "BASSO"  # Imposta rischio default
    if probability > 0.7:  # Se probabilità maggiore di 0.7
        risk = "ALTO"  # Rischio alto
    elif probability > 0.3:  # Se probabilità maggiore di 0.3
        risk = "MEDIO"  # Rischio medio
    
    return jsonify({
        'prediction': prediction,
        'probability': probability,
        'risk': risk
    })  # Restituisce risultati in formato JSON

def train_model():
    """Addestra il modello di Machine Learning"""
    global optimized_model, scaler, X, df  # Usa le variabili globali
    
    # CARICAMENTO DATASET
    print("1. CARICAMENTO DEL DATASET")  # Stampa fase di elaborazione
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"  # URL del dataset
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']  # Nomi delle colonne
    df = pd.read_csv(url, names=column_names)  # Carica il dataset dal URL
    
    print(f"Dimensioni del dataset: {df.shape}")  # Stampa dimensioni del dataset
    
    # ESPLORAZIONE E PULIZIA DEI DATI
    print("\n2. ESPLORAZIONE E PULIZIA DEI DATI")  # Stampa fase di elaborazione
    
    df = df.replace('?', np.nan)  # Sostituisce '?' con NaN
    
    for col in ['ca', 'thal']:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Converte le colonne in numeriche, con errori gestiti
    
    for col in ['ca', 'thal']:
        median_val = df[col].median()  # Calcola la mediana della colonna
        df[col].fillna(median_val, inplace=True)  # Riempie i valori NaN con la mediana
    
    # Conversione della variabile target in binaria
    df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)  # Converte target in binario (0 o 1)
    
    # PREPARAZIONE DEI DATI
    print("\n3. PREPARAZIONE DEI DATI")  # Stampa fase di elaborazione
    
    X = df.drop('target', axis=1)  # Features (rimuove colonna target)
    y = df['target']  # Target
    
    scaler = StandardScaler()  # Crea un oggetto StandardScaler
    X_scaled = scaler.fit_transform(X)  # Normalizza le features
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)  # Divide il dataset in training e test
    
    # ADDESTRAMENTO E VALUTAZIONE MODELLI
    print("\n4. ADDESTRAMENTO E VALUTAZIONE DEI MODELLI")  # Stampa fase di elaborazione
    
    def evaluate_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)  # Addestra il modello sui dati di training
        y_pred = model.predict(X_test)  # Predice i valori sui dati di test
        acc = accuracy_score(y_test, y_pred)  # Calcola l'accuratezza
        return model, acc  # Restituisce il modello addestrato e l'accuratezza
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }  # Dizionario con modelli da valutare
    
    results = {}  # Dizionario per salvare i risultati
    best_model_name = None  # Nome del miglior modello
    best_accuracy = 0  # Migliore accuratezza
    
    for name, model in models.items():
        print(f"\nValutazione del modello: {name}")  # Stampa modello in valutazione
        trained_model, accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)  # Valuta il modello
        results[name] = (trained_model, accuracy)  # Salva risultati
        
        if accuracy > best_accuracy:  # Se accuratezza migliore della precedente
            best_accuracy = accuracy  # Aggiorna migliore accuratezza
            best_model_name = name  # Aggiorna nome miglior modello
    
    print(f"\nIl miglior modello è {best_model_name} con accuracy {best_accuracy:.4f}")  # Stampa miglior modello
    
    # OTTIMIZZAZIONE MIGLIOR MODELLO
    print("\n5. OTTIMIZZAZIONE DEL MIGLIOR MODELLO")  # Stampa fase di elaborazione
    best_model = results[best_model_name][0]  # Ottiene il miglior modello
    
    if best_model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        }  # Parametri da ottimizzare per regressione logistica
    elif best_model_name == 'K-Nearest Neighbors':
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance']
        }  # Parametri da ottimizzare per KNN
    elif best_model_name == 'Support Vector Machine':
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }  # Parametri da ottimizzare per SVM
    elif best_model_name == 'Decision Tree':
        param_grid = {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }  # Parametri da ottimizzare per Decision Tree
    elif best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }  # Parametri da ottimizzare per Random Forest
    
    grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy')  # Crea GridSearchCV con validazione incrociata
    grid_search.fit(X_train, y_train)  # Esegue ricerca degli iperparametri ottimali
    
    print(f"Migliori parametri: {grid_search.best_params_}")  # Stampa migliori parametri
    print(f"Miglior punteggio di cross-validation: {grid_search.best_score_:.4f}")  # Stampa miglior punteggio
    
    optimized_model = grid_search.best_estimator_  # Salva il modello ottimizzato
    
    # CARATTERISTICHE PIÙ IMPORTANTI
    print("\n7. ANALISI DELLE CARATTERISTICHE PIÙ IMPORTANTI")  # Stampa fase di elaborazione
    if hasattr(optimized_model, 'feature_importances_'):  # Se il modello ha feature importances
        importances = optimized_model.feature_importances_  # Ottiene importanza delle features
        feature_names = X.columns  # Ottiene nomi delle features
        indices = np.argsort(importances)[::-1]  # Ordina per importanza decrescente
        
        print("\nImportanza delle caratteristiche:")  # Stampa intestazione
        for i, idx in enumerate(indices):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")  # Stampa feature e importanza
    
    elif hasattr(optimized_model, 'coef_'):  # Se il modello ha coefficienti
        coefficients = optimized_model.coef_[0]  # Ottiene coefficienti
        feature_names = X.columns  # Ottiene nomi delle features
        indices = np.argsort(np.abs(coefficients))[::-1]  # Ordina per valore assoluto dei coefficienti
        
        print("\nCoefficienti del modello:")  # Stampa intestazione
        for i, idx in enumerate(indices):
            print(f"{i+1}. {feature_names[idx]}: {coefficients[idx]:.4f}")  # Stampa feature e coefficiente
    
    print("\n8. CONCLUSIONI")  # Stampa fase di elaborazione
    y_pred = optimized_model.predict(X_test)  # Predice sul test set
    print(f"Il miglior modello è {best_model_name}, ottimizzato tramite GridSearchCV")  # Stampa miglior modello
    print(f"L'accuracy sul test set è {accuracy_score(y_test, y_pred):.4f}.")  # Stampa accuratezza finale

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
    }  # Dizionario con descrizioni delle features
    
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
    }  # Dizionario con limiti accettabili per ogni feature
    
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
    }  # Dizionario con valori predefiniti per il form
    
    return jsonify({
        'descriptions': feature_descriptions,
        'limits': field_limits,
        'defaults': default_values
    })  # Restituisce metadati in formato JSON

if __name__ == "__main__":
    # Addestriamo il modello prima di avviare l'applicazione
    train_model()  # Avvia l'addestramento del modello
    
    browser_thread = threading.Thread(target=open_browser)  # Crea thread per apertura browser
    browser_thread.start()  # Avvia il thread

    # Avviamo l'applicazione Flask
    app.run(debug=False)  # Avvia il server Flask (senza debug per evitare riavvii automatici)
