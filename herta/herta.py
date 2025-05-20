import pandas as pd #per manipolazione dati in formato di tabella
import numpy as np #libreria per calcoli numerici
import matplotlib #libreria per grafici
matplotlib.use('Agg') #per generare grafici senza GUI
import matplotlib.pyplot as plt #per creazione grafici
import seaborn as sns #per grafici statistici avanzati
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV #funzioni per divisione dati e ottimizzazione modelli
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

modelloOttimizzato = None #variabile per memorizzare il modello ottimizzato
scaler = None #lo scaler dei dati
X = None #le features del dataset
df = None #e il dataframe completo

def apriBrowser():
    time.sleep(1.5) #aspetta che il server sia completamente avviato
    webbrowser.open('http://127.0.0.1:5000/') #apre il browser all'indirizzo locale

def codificaImmagine(fig): #converte un figura matplotlib in una stringa base64 per HTML evitando di dover salvare il file
    img = BytesIO() #crea un buffer di memoria
    fig.savefig(img, format='png', bbox_inches='tight') #salva la figura nel buffer come PNG
    img.seek(0) #riporta il puntatore all'inizio del buffer
    codificata = base64.b64encode(img.getvalue()).decode('utf-8') #codifica l'immagine in base64
    plt.close(fig) #chiude la figura
    return f"data:image/png;base64,{codificata}" #restituisce l'immagine come stringa data URL

@app.route('/') #renderizza la pagina principale
def home():
    return render_template('index.html') #restituisce il template HTML principale

@app.route('/api/charts') #restituisce grafici generati dopo l'analisi
def ottieniGrafici():
    global df, X, modelloOttimizzato #usa variabili globali
    datiGrafici = {} #dizionario per memorizzare i dati dei grafici
    
    fig1 = plt.figure(figsize=(8, 6)) #crea figura matplotlib di dimensione 8x6
    sns.countplot(x='target', data=df) #crea grafico a barre della distribuzione variabile target
    plt.title('Distribuzione Variabile Target') #titolo del grafico
    plt.xlabel('Presenza di Malattia Cardiaca (0=No, 1=Sì)') #etichetta asse x
    plt.ylabel('Conteggio') #etichetta asse y
    datiGrafici['target_distribution'] = codificaImmagine(fig1) #codifica e salva grafico
    
    fig2 = plt.figure(figsize=(12, 10))
    matriceCorrelaz = df.corr() #calcola la matrice di correlazione
    sns.heatmap(matriceCorrelaz, annot=True, cmap='coolwarm', fmt='.2f') #crea heatmap con annotazioni
    plt.title('Matrice di Correlazione')
    datiGrafici['correlation_matrix'] = codificaImmagine(fig2) #codifica e salva grafico
    
    if hasattr(modelloOttimizzato, 'feature_importances_'): #se il modello ha "feature_importances_"
        importanze = modelloOttimizzato.feature_importances_ #ottiene i valori di importanza
        nomiFeature = X.columns #ottiene i nomi delle features
        indici = np.argsort(importanze)[::-1] #ordina gli indici in base all'importanza (decrescente)
        
        fig3 = plt.figure(figsize=(10, 6)) #crea una nuova figura matplotlib
        plt.bar(range(len(importanze)), importanze[indici]) #crea grafico a barre dell'importanza
        plt.xticks(range(len(importanze)), [nomiFeature[i] for i in indici], rotation=90) #etichette asse x ruotate
        plt.title('Importanza delle Caratteristiche')
        plt.tight_layout() #aggiusta layout per evitare sovrapposizioni
        datiGrafici['feature_importance'] = codificaImmagine(fig3)
    elif hasattr(modelloOttimizzato, 'coef_'): #se il modello ha coefficienti (es. regressione logistica)
        coefficienti = modelloOttimizzato.coef_[0] #li ottiene
        nomiFeature = X.columns #ottiene i nomi delle features
        indici = np.argsort(np.abs(coefficienti))[::-1] #ordina gli indici in base al valore assoluto dei coefficienti
        
        fig3 = plt.figure(figsize=(10, 6)) #crea una nuova figura matplotlib
        plt.bar(range(len(coefficienti)), coefficienti[indici]) #crea grafico a barre dei coefficienti
        plt.xticks(range(len(coefficienti)), [nomiFeature[i] for i in indici], rotation=90)
        plt.title('Coefficienti del Modello')
        plt.tight_layout()  
        datiGrafici['model_coefficients'] = codificaImmagine(fig3) 
    
    return jsonify(datiGrafici) #ritorna dati dei grafici in formato JSON

@app.route('/api/predict', methods=['POST']) #quando qualcuno fa una richiesta HTTP all'indirizzo /api/predict, esegue la funzione subito sotto
def predict():
    global modelloOttimizzato, scaler, X #usa variabili globali
    
    data = request.json #estrae dati del paziente in formato JSON dalla richiesta
    
    for key, value in data.items(): #per ogni valore
        try:
            data[key] = float(value)  #lo converte in float
        except ValueError:
            return jsonify({'error': f'Valore non valido per {key}'}), 400 #genera un'eccezione se la conversione fallisce (error 400 bad request)
    
    datiPaziente = [ #crea lista con i dati del paziente
        data['age'], data['sex'], data['cp'], data['trestbps'], 
        data['chol'], data['fbs'], data['restecg'], data['thalach'], 
        data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
    ]  
    
    dataFramePaziente = pd.DataFrame([datiPaziente], columns=X.columns) #crea un dataframe con stesse colonne di X
    
    pazienteNorm = scaler.transform(dataFramePaziente) #applica la stessa normalizzazione usata per il training
    
    prediz = int(modelloOttimizzato.predict(pazienteNorm)[0]) #predice la classe (0 o 1)
    probab = float(modelloOttimizzato.predict_proba(pazienteNorm)[0, 1]) #ottiene la probabilità della classe positiva
    
    risk = "BASSO" #rischio di default
    if probab > 0.7: #se probabilità maggiore di 0.7
        risk = "ALTO" #rischio alto
    elif probab > 0.3: #se maggiore di 0.3
        risk = "MEDIO" #rischio medio
    
    return jsonify({
        'prediction': prediz,
        'probability': probab,
        'risk': risk
    }) #restituisce risultati in formato JSON

def allenaModello(): #addestra modello
    global modelloOttimizzato, scaler, X, df
    
    print("[Caricamento del dataset]:")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data" #URL dataset
    nomiColonne = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'] #nomi delle colonne
    df = pd.read_csv(url, names=nomiColonne) #caricamento dataset da URL
    print(f"Dimensioni del dataset: {df.shape}")  
    
    print("\n[Pulizia dati]:")
    
    df = df.replace('?', np.nan) #sostituisce "?" con NaN
    
    for col in ['ca', 'thal']: #per ogni colonna in "ca" e "thal"
        df[col] = pd.to_numeric(df[col], errors='coerce') #converte le colonne in numeriche, con errori gestiti
    
    for col in ['ca', 'thal']:
        valMediana = df[col].median() #calcola la mediana della colonna
        df[col].fillna(valMediana, inplace=True) #riempie i valori NaN con la mediana
    
    df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1) #converte target in binario (0 no malattia, 1 malattia)
    
    print("\n[Preparazione dati]:")
    
    X = df.drop('target', axis=1) #features (rimuove colonna target)
    y = df['target'] #target
    
    scaler = StandardScaler() #crea oggetto StandardScaler
    X_scaled = scaler.fit_transform(X) #normalizza le features (converte valori in una scala comune)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y) #divide il dataset in training e test (70% e 30%)
    
    print("\n[Addestramento/Valutazione modelli]:")
    
    def valutaModello(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train) #addestra il modello sui dati di training
        y_pred = model.predict(X_test) #predice i valori sui dati di test
        acc = accuracy_score(y_test, y_pred) #calcola l'accuratezza
        return model, acc #ritorna il modello addestrato e l'accuratezza
    
    modelli = { #dizionario con modelli da valutare
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }  
    
    risultati = {} #dizionario per salvare i risultati
    nomeMigliorModello = None #nome miglior modello
    migliorPrecis = 0 #migliore accuratezza
    
    for name, model in modelli.items():
        print(f"\nValutazione del modello: {name}")
        modelloTrainato, accuracy = valutaModello(model, X_train, X_test, y_train, y_test) #valuta il modello
        risultati[name] = (modelloTrainato, accuracy) #salva risultati
        
        if accuracy > migliorPrecis: #se l'accuratezza è migliore della precedente
            migliorPrecis = accuracy #aggiorna migliore accuratezza
            nomeMigliorModello = name #aggiorna nome miglior modello
    
    print(f"\nIl miglior modello è {nomeMigliorModello} con accuracy {migliorPrecis:.4f}")
    
    print("\n[Ottimizzazione miglior modello]:")
    migliorMod = risultati[nomeMigliorModello][0] #ottiene il miglior modello
    
    if nomeMigliorModello == 'Logistic Regression': #parametri da ottimizzare per regressione logistica
        gridParametri = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        } 
    elif nomeMigliorModello == 'K-Nearest Neighbors': #per KNN
        gridParametri = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance']
        }  
    elif nomeMigliorModello == 'Support Vector Machine': #per SVM
        gridParametri = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }  
    elif nomeMigliorModello == 'Decision Tree': #per Decision Tree
        gridParametri = {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        } 
    elif nomeMigliorModello == 'Random Forest': #e per Random Forest
        gridParametri = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        } 
    
    gridSearch = GridSearchCV(migliorMod, gridParametri, cv=5, scoring='accuracy') #crea GridSearchCV per trovare parametri migliori per i modelli
    gridSearch.fit(X_train, y_train) #esegue ricerca degli iperparametri ottimali
    
    print(f"Migliori parametri: {gridSearch.best_params_}")
    print(f"Miglior punteggio di cross-validation: {gridSearch.best_score_:.4f}")
    
    modelloOttimizzato = gridSearch.best_estimator_ #salva il modello ottimizzato
    
    print("\n[Analisi caratteristiche più importanti]:") 
    if hasattr(modelloOttimizzato, 'feature_importances_'): #se il modello ha feature importances (decision tree, random forest, )
        importanze = modelloOttimizzato.feature_importances_ #ottiene quanto ogni feature ha influito sulle decisioni del modello
        nomiFeature = X.columns #e i nomi delle features
        indici = np.argsort(importanze)[::-1] #ordina gli indici delle feature dalla più importante alla meno importante
        
        print("\nImportanza delle caratteristiche:")
        for i, idx in enumerate(indici): #stampa l'elenco ordinato
            print(f"{i+1}. {nomiFeature[idx]}: {importanze[idx]:.4f}")
    
    elif hasattr(modelloOttimizzato, 'coef_'): #se il modello ha coefficienti (SVC, Linear Regression)
        coefficienti = modelloOttimizzato.coef_[0] #ottiene coefficienti (pesature delle features)
        nomiFeature = X.columns #ottiene nomi delle features
        indici = np.argsort(np.abs(coefficienti))[::-1] #ordina per valore assoluto dei coefficienti
        
        print("\nCoefficienti del modello:")
        for i, idx in enumerate(indici):
            print(f"{i+1}. {nomiFeature[idx]}: {coefficienti[idx]:.4f}")
    
    print("\n[Conclusione]:")  
    y_pred = modelloOttimizzato.predict(X_test) #predice in base al test set
    print(f"Il miglior modello è {nomeMigliorModello} (dopo ottimizzazione)")  
    print(f"L'accuracy  è {accuracy_score(y_test, y_pred):.4f}.")

@app.route('/api/field_metadata')
def ottieniCampiDati():
    descrizioniFeatures = { #dizionario con descrizioni delle features
        'age': "Età del paziente",
        'sex': "Sesso",
        'cp': "Tipo di dolore toracico (0: Angina tipica, 1: Angina atipica, 2: Dolore non anginoso, 3: Asintomatico)",
        'trestbps': "Pressione sanguigna a riposo in mm Hg",
        'chol': "Colesterolo sierico in mg/dl",
        'fbs': "Glicemia a digiuno > 120 mg/dl",
        'restecg': "Risultati elettrocardiografici a riposo (0: Normale, 1: Anomalia dell'onda ST-T, 2: Ipertrofia ventricolare sinistra)",
        'thalach': "Frequenza cardiaca massima raggiunta",
        'exang': "Angina indotta dall'esercizio",
        'oldpeak': "Depressione del segmento ST indotta dall'esercizio rispetto al riposo",
        'slope': "Pendenza del segmento ST durante l'esercizio (0: Ascendente, 1: Piatto, 2: Discendente)",
        'ca': "Numero di vasi principali colorati dalla fluoroscopia (0-4)",
        'thal': "Talassemia (1: Difetto fisso, 2: Normale, 3: Difetto reversibile)"
    }  
    
    limitiCampi = { #dizionario con limiti accettabili per ogni feature
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
    
    valoriDefault = { #dizionario con valori predefiniti per il form
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
        'descriptions': descrizioniFeatures,
        'limits': limitiCampi,
        'defaults': valoriDefault
    }) #restituisce in formato JSON

if __name__ == "__main__":
    allenaModello() #avvia l'addestramento del modello
    
    threadBrowser = threading.Thread(target=apriBrowser) #crea thread per apertura browser
    threadBrowser.start() #avvia il thread

    app.run(debug=False) #avvia il server flask (senza debug)
