document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form'); //ottiene il form di predizione
    const placeHolderrisult = document.querySelector('.results-placeholder'); //placeholder per i risultati (mostrato quando non ci sono ancora predizioni)
    const dettagliPrediz = document.querySelector('.prediction-details'); //contenitore per i dettagli della predizione
    const valorePrediz = document.getElementById('prediction-value'); //mostrerà il valore della predizione (positivo/negativo)
    const barraProbab = document.getElementById('probability-bar'); //barra che mostra la probabilità visivamente
    const valoreProbab = document.getElementById('probability-value'); //mostrerà il valore numerico della probabilità
    const rischioValore = document.getElementById('risk-value'); //mostrerà il livello di rischio (BASSO/MEDIO/ALTO)
    const graficiSchede = document.querySelectorAll('.chart-tab'); //schede per selezionare i diversi grafici
    const graficoCorrente = document.getElementById('current-chart'); //elemento img che mostra il grafico attualmente selezionato
    
    let datiGrafici = {}; //memorizza dati grafici ricevuti dall'API
    
    fetch('/api/field_metadata') //richiesta GET per ottenere i dati dei campi del form (descrizioni, limiti, valori predefiniti)
        .then(risposta => risposta.json()) //converte la risposta in JSON
        .then(data => {
            //imposta i valori predefiniti per tutti i campi del form
            for (const [campo, value] of Object.entries(data.defaults)) {
                const elemento = document.getElementById(campo);  // Trova l'elemento corrispondente al campo
                if (elemento) {
                    if (elemento.type === 'number' || elemento.tagName === 'SELECT') { //se l'elemento è un input numerico o select
                        elemento.value = value; //imposta direttamente il valore
                    } else if (elemento.type === 'radio') { //per i radio button
                        const radio = document.querySelector(`input[name="${campo}"][value="${value}"]`); //trova e seleziona quello con il valore corretto
                        if (radio) radio.checked = true;
                    }
                }
            }
        })
        .catch(error => console.error('Errore ottenimento dati', error));
    
    fetch('/api/charts') //richiesta HTTP per ottenere i grafici generati da Python
        .then(risposta => risposta.json()) //converte la risposta in JSON
        .then(data => {
            datiGrafici = data; //memorizza i dati dei grafici (stringhe base64)
            if (data['target_distribution']) {
                graficoCorrente.src = data['target_distribution']; //imposta la sorgente dell'immagine al grafico di distribuzione target
            }
        })
        .catch(error => console.error('Errore ottenimento grafici', error));
    
    graficiSchede.forEach(tab => { //gestisce il cambio di scheda per i grafici
        tab.addEventListener('click', function() {
            graficiSchede.forEach(t => t.classList.remove('active')); //rimuove la classe active da tutte le schede
            this.classList.add('active'); //aggiunge la classe active alla scheda cliccata
        
            const chartType = this.getAttribute('data-chart'); //ottiene il tipo di grafico dall'attributo data-chart
            if (datiGrafici[chartType]) {
                graficoCorrente.src = datiGrafici[chartType]; //imposta la sorgente dell'immagine al grafico selezionato
            }
        });
    });
    
    form.addEventListener('submit', function(e) { //gestisce l'invio del form di predizione
        e.preventDefault(); //previene l'invio tradizionale del form
        
        const datiForm = {}; //prepara un oggetto per raccogliere i dati del form
        
        //ottiene i valori di tutti gli input numerici e select
        const inputs = form.querySelectorAll('input[type="number"], select');
        inputs.forEach(input => { //per ogni input
            datiForm[input.name] = input.value; //salva ogni valore nell'oggetto formdata
        });
        
        //ottiene i valori dei radio button (sesso, glicemia a digiuno, angina indotta dall'esercizio)
        const gruppoRadio = ['sex', 'fbs', 'exang'];
        gruppoRadio.forEach(group => {
            const radioPremuto = form.querySelector(`input[name="${group}"]:checked`); //trova il radio button selezionato
            if (radioPremuto) {
                datiForm[group] = radioPremuto.value; //salva il valore nell'oggetto datiForm
            }
        });
        
        
        const campiRichiesti = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', //lista di tutti i campi richiesti per la predizione
                               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'];
        
        let tuttiCampiPresenti = true; 
        campiRichiesti.forEach(campo => { //se un campo di quelli richiesti non è presente
            if (datiForm[campo] === undefined) {
                console.error(`Campo mancante: ${campo}`); //scrive in console il campo mancante
                tuttiCampiPresenti = false;
            }
        });
        
        if (!tuttiCampiPresenti) { //se mancano campi
            alert("Compilare tutti i campi richiesti"); //mostra messaggio di errore
            return;
        }
        
        for (const key in datiForm) { //per ogni valore nei dati
            datiForm[key] = Number(datiForm[key]); //lo converte a numero
        }
        
        valorePrediz.textContent = 'Elaborazione...'; //aggiorna il testo della predizione
        dettagliPrediz.style.display = 'flex'; //rende visibile il contenitore dei dettagli
        placeHolderrisult.style.display = 'none'; //nasconde il segnaposto
        
        fetch('/api/predict', {
            method: 'POST', //metodo HTTP per l'invio dei dati
            headers: {
                'Content-Type': 'application/json', //formato dei dati inviati
            },
            body: JSON.stringify(datiForm)  //converte l'oggetto datiForm in una stringa JSON
        })
        .then(risposta => {
            if (!risposta.ok) { //se ci sono errori nella risposta
                throw new Error(`Errore HTTP: ${risposta.status}`); 
            }
            return risposta.json();  //converte la risposta in JSON
        })
        .then(result => {
            
            valorePrediz.textContent = result.prediction === 1 ? 'POSITIVO' : 'NEGATIVO'; //aggiorna il valore della predizione (POSITIVO/NEGATIVO)
            valorePrediz.className = 'prediction-value ' + (result.prediction === 1 ? 'positive' : 'negative'); //aggiunge classe CSS in base al risultato
            
            //aggiorna la barra di probabilità
            const percentProbab = Math.round(result.probability * 100); //converte la probabilità in percentuale
            barraProbab.style.width = `${percentProbab}%`; //imposta la larghezza della barra
            valoreProbab.textContent = `${percentProbab}%`; //mostra la percentuale
            
            rischioValore.textContent = result.risk; //imposta il testo (BASSO/MEDIO/ALTO)
            rischioValore.className = 'risk-value ' + result.risk.toLowerCase(); //aggiunge classe CSS in base al rischio
        })
        .catch(error => {
            console.error("Errore nella predizione:", error);
            valorePrediz.textContent = 'Errore'; //mostra "Errore" nel campo predizione
            valorePrediz.className = 'prediction-value error'; //classe CSS di errore
            alert("Si è verificato un errore durante la predizione.");
        });
    });
    
    form.addEventListener('reset', function() { //gestione reset del form
        dettagliPrediz.style.display = 'none'; //nasconde i dettagli della predizione
        placeHolderrisult.style.display = 'block'; //mostra nuovamente il segnaposto
        
        setTimeout(() => { //ripristina i valori predefiniti dopo un breve tempo
            fetch('/api/field_metadata') //nuova richiesta per ottenere i valori predefiniti
                .then(risposta => risposta.json())
                .then(data => { //imposta nuovamente i valori predefiniti per tutti i campi
                    for (const [campo, value] of Object.entries(data.defaults)) { //per ogni coppia di valori campo-valore
                        const elemento = document.getElementById(campo);
                        if (elemento) {
                            if (elemento.type === 'number' || elemento.tagName === 'SELECT') { //se l'elemento è un numero o un select
                                elemento.value = value; //ripristina il valore predefinito
                            } else if (elemento.type === 'radio') { //se è un radio
                                const radio = document.querySelector(`input[name="${campo}"][value="${value}"]`); //lo imposta al valore predefinito
                                if (radio) radio.checked = true;
                            }
                        }
                    }
                });
        }, 100); //attende 100ms prima di ripristinare i valori predefiniti
    });
});