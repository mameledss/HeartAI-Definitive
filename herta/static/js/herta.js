// script.js corretto
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const form = document.getElementById('prediction-form');
    const resultsPlaceholder = document.querySelector('.results-placeholder');
    const predictionDetails = document.querySelector('.prediction-details');
    const predictionValue = document.getElementById('prediction-value');
    const probabilityBar = document.getElementById('probability-bar');
    const probabilityValue = document.getElementById('probability-value');
    const riskValue = document.getElementById('risk-value');
    const chartTabs = document.querySelectorAll('.chart-tab');
    const currentChart = document.getElementById('current-chart');
    
    // Store charts data
    let chartsData = {};
    
    // Fetch field metadata and set default values
    fetch('/api/field_metadata')
        .then(response => response.json())
        .then(data => {
            // Set default values for all fields
            for (const [field, value] of Object.entries(data.defaults)) {
                const element = document.getElementById(field);
                if (element) {
                    if (element.type === 'number' || element.tagName === 'SELECT') {
                        element.value = value;
                    } else if (element.type === 'radio') {
                        // For radio buttons, select the appropriate one
                        const radio = document.querySelector(`input[name="${field}"][value="${value}"]`);
                        if (radio) radio.checked = true;
                    }
                }
            }
        })
        .catch(error => console.error('Error fetching field metadata:', error));
    
    // Fetch charts data
    fetch('/api/charts')
        .then(response => response.json())
        .then(data => {
            chartsData = data;
            
            // Set the default chart (target distribution)
            if (data['target_distribution']) {
                currentChart.src = data['target_distribution'];
            }
        })
        .catch(error => console.error('Error fetching charts:', error));
    
    // Handle chart tab switching
    chartTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active class from all tabs
            chartTabs.forEach(t => t.classList.remove('active'));
            
            // Add active class to clicked tab
            this.classList.add('active');
            
            // Update current chart
            const chartType = this.getAttribute('data-chart');
            if (chartsData[chartType]) {
                currentChart.src = chartsData[chartType];
            }
        });
    });
    
    // Handle form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Gather form data - assicurarsi che tutti i campi siano inclusi
        const formData = {};
        
        // Raccogliamo i valori dei campi input e select
        const inputs = form.querySelectorAll('input[type="number"], select');
        inputs.forEach(input => {
            formData[input.name] = input.value;
        });
        
        // Raccogliamo i valori dei radio button
        const radioGroups = ['sex', 'fbs', 'exang'];
        radioGroups.forEach(group => {
            const checkedRadio = form.querySelector(`input[name="${group}"]:checked`);
            if (checkedRadio) {
                formData[group] = checkedRadio.value;
            }
        });
        
        // Verifica che tutti i campi richiesti siano presenti
        const requiredFields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'];
        
        let allFieldsPresent = true;
        requiredFields.forEach(field => {
            if (formData[field] === undefined) {
                console.error(`Campo mancante: ${field}`);
                allFieldsPresent = false;
            }
        });
        
        if (!allFieldsPresent) {
            alert('Compilare tutti i campi richiesti');
            return;
        }
        
        // Converti tutti i valori in numeri
        for (const key in formData) {
            formData[key] = Number(formData[key]);
        }
        
        // Show loading state
        predictionValue.textContent = 'Elaborazione...';
        predictionDetails.style.display = 'flex';
        resultsPlaceholder.style.display = 'none';
        
        // Log dei dati inviati per debugging
        console.log('Dati inviati al backend:', formData);
        
        // Make prediction request
        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Errore HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(result => {
            console.log('Risultato ricevuto:', result);
            
            // Update prediction value
            predictionValue.textContent = result.prediction === 1 ? 'POSITIVO' : 'NEGATIVO';
            predictionValue.className = 'prediction-value ' + (result.prediction === 1 ? 'positive' : 'negative');
            
            // Update probability bar
            const probabilityPercent = Math.round(result.probability * 100);
            probabilityBar.style.width = `${probabilityPercent}%`;
            probabilityValue.textContent = `${probabilityPercent}%`;
            
            // Update risk level
            riskValue.textContent = result.risk;
            riskValue.className = 'risk-value ' + result.risk.toLowerCase();
        })
        .catch(error => {
            console.error('Error making prediction:', error);
            predictionValue.textContent = 'Errore';
            predictionValue.className = 'prediction-value error';
            alert('Si è verificato un errore durante la predizione. Controllare la console per i dettagli.');
        });
    });
    
    // Handle form reset
    form.addEventListener('reset', function() {
        // Hide prediction results
        predictionDetails.style.display = 'none';
        resultsPlaceholder.style.display = 'block';
        
        // Reset to default values after a short delay
        setTimeout(() => {
            fetch('/api/field_metadata')
                .then(response => response.json())
                .then(data => {
                    // Set default values for all fields
                    for (const [field, value] of Object.entries(data.defaults)) {
                        const element = document.getElementById(field);
                        if (element) {
                            if (element.type === 'number' || element.tagName === 'SELECT') {
                                element.value = value;
                            } else if (element.type === 'radio') {
                                // For radio buttons, select the appropriate one
                                const radio = document.querySelector(`input[name="${field}"][value="${value}"]`);
                                if (radio) radio.checked = true;
                            }
                        }
                    }
                });
        }, 100);
    });
});