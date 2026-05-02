from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from autogluon.tabular import TabularPredictor
import pandas as pd
import os
import pathlib
import platform

# PosixPath patch for cross-platform compatibility (Linux to Windows)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

# --- MODEL DEFINITIONS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MultiTaskRoBERTa(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super(MultiTaskRoBERTa, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.head_label = nn.Linear(768, 1)
        self.head_topic = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        out = outputs.pooler_output
        return torch.sigmoid(self.head_label(out)), self.head_topic(out)

# Global variables for models
tokenizer = None
nlp_model = None
label_predictor = None

def load_models():
    global tokenizer, nlp_model, label_predictor
    print("Loading models...")
    
    # Load RoBERTa
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    nlp_model = MultiTaskRoBERTa().to(device)
    
    # Check if weights exist
    weights_path = 'roberta_multitask_state_dict.pt'
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('base.', 'roberta.')
            new_state_dict[new_key] = value
        try:
            nlp_model.load_state_dict(new_state_dict)
            print("RoBERTa weights loaded successfully.")
        except Exception as e:
            print(f"Error loading RoBERTa weights: {e}")
    else:
        print(f"Warning: {weights_path} not found.")
        
    nlp_model.eval()
    
    # Load AutoGluon
    ag_path = 'Label_Model_Final'
    if os.path.exists(ag_path):
        label_predictor = TabularPredictor.load(ag_path, require_py_version_match=False)
        print("AutoGluon model loaded successfully.")
    else:
        print(f"Warning: {ag_path} not found.")

# Load models when the app starts
load_models()

def predict_text(text):
    # 1. Get RoBERTa probabilities
    with torch.no_grad():
        inputs = tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        l_out, t_out = nlp_model(inputs['input_ids'], inputs['attention_mask'])
        
        roberta_prob = (1 - l_out).cpu().numpy().flatten()[0]
        t_probs = torch.softmax(t_out, dim=1).cpu().numpy().flatten()
        prob_sayisal, prob_sosyal, prob_teknoloji = t_probs
        
    # 2. Prepare dataframe for AutoGluon
    df = pd.DataFrame([{
        'text': text,
        'roberta_prob': roberta_prob,
        'prob_sayisal': prob_sayisal,
        'prob_sosyal': prob_sosyal,
        'prob_teknoloji': prob_teknoloji
    }])
    
    # 3. Predict with AutoGluon
    if label_predictor is not None:
        # Get class probabilities
        pred_proba = label_predictor.predict_proba(df).iloc[0]
        
        classes = list(pred_proba.index)
        
        # Extract Human probability (assuming binary classification: Human vs AI)
        # Convert class names to lowercase for robust matching
        lower_classes = [str(c).strip().lower() for c in classes]
        
        if 'human' in lower_classes:
            human_idx = lower_classes.index('human')
            human_prob = float(pred_proba.iloc[human_idx])
            ai_probability = 1.0 - human_prob
        elif 'ai' in lower_classes:
            ai_idx = lower_classes.index('ai')
            ai_probability = float(pred_proba.iloc[ai_idx])
        else:
            # Fallback if classes are numerical (Human=1, AI=0)
            if 1 in classes:
                human_prob = float(pred_proba[1])
                ai_probability = 1.0 - human_prob
            else:
                ai_probability = float(pred_proba.iloc[0]) # Fallback
            
        is_ai = ai_probability >= 0.5
    else:
        # Fallback to pure roberta if autogluon is missing
        # Calculate fallback probability if AutoGluon is unavailable
        ai_probability = float(1.0 - roberta_prob)
        is_ai = ai_probability >= 0.5
        
    return {
        'ai_probability': ai_probability,
        'is_ai': bool(is_ai)
    }

@app.route('/')
def index():
    return render_template('index.html')

from langdetect import detect_langs, LangDetectException

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Please enter a text to analyze.'}), 400
        
    # --- LENGTH CHECK ---
    words = text.split()
    if len(words) < 20:
        return jsonify({
            'error': 'The text is too short. Please enter a text with at least 20 words for a reliable AI detection.'
        }), 400
        
    # --- LANGUAGE DETECTION ---
    try:
        langs = detect_langs(text)
        is_english = False
        for lang in langs:
            if lang.lang == 'en' and lang.prob >= 0.5:
                is_english = True
                break
                
        if not is_english:
            return jsonify({
                'error': 'The English content of this text is insufficient. The model can only analyze English texts accurately. Please provide an English text.'
            }), 400
    except LangDetectException:
        return jsonify({'error': 'Language could not be detected. Please enter a longer and more meaningful text.'}), 400

    try:
        result = predict_text(text)
        return jsonify(result)
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'An error occurred. The models might not be loaded.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
