document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const predictionOutput = document.getElementById('prediction-output');

    // API URL - Assuming backend runs on default port 8000
    const API_URL = 'http://localhost:8000/predict';

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // 1. Collect Input Values
        const modelYear = parseInt(document.getElementById('model-year').value);
        const engineSize = parseFloat(document.getElementById('engine-size').value);
        const cylinders = parseInt(document.getElementById('cylinders').value);
        const fuelType = document.getElementById('fuel-type').value;
        const transmission = document.getElementById('transmission').value;
        const vehicleClass = document.getElementById('vehicle-class').value;

        // 2. Prepare Payload
        const payload = {
            model_year: modelYear,
            engine_size: engineSize,
            cylinders: cylinders,
            fuel_type: fuelType,
            transmission: transmission,
            vehicle_class: vehicleClass
        };

        try {
            // 3. Send Request
            // NOTE: In a real production app, this key should not be hardcoded. 
            // It should be injected during deployment or the frontend should be served by the backend.
            const API_KEY = 'change_me_to_a_secure_key';

            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': API_KEY
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }

            const data = await response.json();

            // 4. Update UI
            predictionOutput.textContent = data.prediction.toFixed(1);
            resultContainer.classList.remove('hidden');

            // Scroll to result on mobile
            if (window.innerWidth < 850) {
                resultContainer.scrollIntoView({ behavior: 'smooth' });
            }

        } catch (error) {
            console.error('Error:', error);
            // Fallback for demo purposes if backend is offline
            alert(`Error: ${error.message}\nMake sure backend is running.`);
        }
    });

    // Fetch model metrics on load
    fetchMetrics();
});

async function fetchMetrics() {
    try {
        const response = await fetch('http://localhost:8000/metrics');
        if (response.ok) {
            const data = await response.json();
            document.getElementById('mae-value').textContent = data.mae;
            document.getElementById('r2-value').textContent = data.r2;
        }
    } catch (e) {
        console.log('Metrics not available yet');
    }
}
