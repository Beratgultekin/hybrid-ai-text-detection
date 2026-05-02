function clearText() {
    document.getElementById('textInput').value = '';
    
    // Reset UI to default state
    document.getElementById('errorMessage').classList.add('hidden');
    document.getElementById('loading').classList.add('hidden');
    
    const progressFill = document.getElementById('progressFill');
    const percentageText = document.getElementById('percentageText');
    const decisionCard = document.getElementById('decisionCard');
    const statusDot = document.getElementById('statusDot');
    const decisionTitle = document.getElementById('decisionTitle');
    const decisionDesc = document.getElementById('decisionDesc');
    
    progressFill.style.strokeDashoffset = 314; // Default empty
    progressFill.style.stroke = '#7aa2f7';
    
    percentageText.textContent = '%0.0';
    percentageText.style.color = '#7aa2f7';
    
    decisionCard.style.borderColor = '#3d4054';
    statusDot.style.backgroundColor = '#565f89';
    statusDot.style.boxShadow = 'none';
    
    decisionTitle.textContent = 'WAITING';
    decisionDesc.textContent = 'Please enter a text on the left and click the analyze button.';
    decisionDesc.style.color = '#a9b1d6';
}

async function analyzeText() {
    const textInput = document.getElementById('textInput').value;
    const errorMsg = document.getElementById('errorMessage');
    const loadingMsg = document.getElementById('loading');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');

    if (!textInput.trim()) {
        errorMsg.innerHTML = '<strong>Error:</strong> Please enter a text to analyze.';
        errorMsg.classList.remove('hidden');
        return;
    }

    errorMsg.classList.add('hidden');
    loadingMsg.classList.remove('hidden');
    analyzeBtn.disabled = true;
    clearBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: textInput })
        });

        const data = await response.json();

        if (response.ok) {
            updateUI(data.ai_probability, data.is_ai);
        } else {
            throw new Error(data.error || 'An unknown error occurred.');
        }
    } catch (error) {
        errorMsg.innerHTML = `<strong>Warning:</strong> ${error.message}`;
        errorMsg.classList.remove('hidden');
        
        // Reset progress bar and decision card on error
        const progressFill = document.getElementById('progressFill');
        progressFill.style.strokeDashoffset = 314;
        progressFill.style.stroke = '#7aa2f7';
        
        const percentageText = document.getElementById('percentageText');
        percentageText.textContent = '%0.0';
        percentageText.style.color = '#7aa2f7';
        
        const decisionCard = document.getElementById('decisionCard');
        const statusDot = document.getElementById('statusDot');
        const decisionTitle = document.getElementById('decisionTitle');
        const decisionDesc = document.getElementById('decisionDesc');
        
        decisionCard.style.borderColor = '#3d4054';
        statusDot.style.backgroundColor = '#565f89';
        statusDot.style.boxShadow = 'none';
        
        decisionTitle.textContent = 'COULD NOT CALCULATE';
        decisionTitle.style.color = '#c0caf5';
        decisionDesc.innerHTML = 'Result could not be calculated. Please check the warning on the left.';
        decisionDesc.style.color = '#a9b1d6';
    } finally {
        loadingMsg.classList.add('hidden');
        analyzeBtn.disabled = false;
        clearBtn.disabled = false;
    }
}

function updateUI(aiProb, isAI) {
    const probPercentage = (aiProb * 100).toFixed(1);
    
    // UI Elements
    const progressFill = document.getElementById('progressFill');
    const percentageText = document.getElementById('percentageText');
    const decisionCard = document.getElementById('decisionCard');
    const statusDot = document.getElementById('statusDot');
    const decisionTitle = document.getElementById('decisionTitle');
    const decisionDesc = document.getElementById('decisionDesc');

    // Premium Colors
    const colorAI = '#f7768e'; // Bright pastel red
    const colorHuman = '#9ece6a'; // Bright pastel green

    const activeColor = isAI ? colorAI : colorHuman;

    // Update Probability Circle
    const circumference = 314; // 2 * pi * 50
    const offset = circumference - (aiProb * circumference);
    progressFill.style.strokeDashoffset = offset;
    progressFill.style.stroke = activeColor;
    
    // Update Percentage Text
    percentageText.textContent = `%${probPercentage}`;
    percentageText.style.color = activeColor;

    // Update Decision Card
    decisionCard.style.borderColor = activeColor;
    statusDot.style.backgroundColor = activeColor;
    statusDot.style.boxShadow = `0 0 15px ${activeColor}80`; // Glowing effect
    
    if (isAI) {
        decisionTitle.textContent = 'ARTIFICIAL INTELLIGENCE (AI)';
        decisionDesc.innerHTML = `This text is <strong>%${probPercentage}</strong> likely written by an AI model.`;
        decisionDesc.style.color = colorAI;
    } else {
        decisionTitle.textContent = 'HUMAN';
        decisionDesc.innerHTML = `This text is <strong>%${(100 - probPercentage).toFixed(1)}</strong> likely written by a human.`;
        decisionDesc.style.color = colorHuman;
    }
}
