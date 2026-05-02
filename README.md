# AI-Generated Text Detection via Hybrid Meta-Ensemble Framework

## Overview
This repository contains the source code, pre-trained weights, and the interactive web interface for our academic research on AI-generated text detection. As Large Language Models (LLMs) become increasingly sophisticated, distinguishing between human-written and AI-generated content is critical for maintaining academic integrity, fighting misinformation, and ensuring data quality. 

To address the limitations of standalone text classifiers, we propose a novel **Hybrid Meta-Ensemble Architecture** that synergizes the deep semantic and contextual understanding of **RoBERTa** with the robust, multi-algorithm predictive power of **AutoGluon**. 

This repository includes a fully functional, real-time web application dashboard to interact with and test the hybrid model.

---

## 🧠 Model Architecture

Our hybrid framework operates in a sophisticated two-stage pipeline, maximizing both deep learning feature extraction and machine learning decision boundaries:

### Stage 1: Deep Feature Extraction (Multi-Task RoBERTa)
*   **Base Architecture:** `roberta-base` fine-tuned using a Multi-Task Learning (MTL) objective.
*   **Mechanism:** Raw input text is tokenized and processed through the RoBERTa transformer blocks. Rather than relying solely on a standard classification head, we extract the dense probability vectors (logits) that capture complex syntactic patterns, semantic coherence, and stylistic anomalies typical of generative AI.

### Stage 2: Meta-Ensemble Classification (AutoGluon)
*   **Architecture:** `AutoGluon TabularPredictor` (Weighted Ensemble L2).
*   **Mechanism:** The high-dimensional features extracted by RoBERTa are fed as tabular inputs into an advanced AutoGluon ensemble. This ensemble dynamically weights predictions from multiple state-of-the-art algorithms:
    *   **CatBoost:** Excellent at handling categorical data structures.
    *   **LightGBM:** Provides high-speed, memory-efficient tree boosting.
    *   **XGBoost:** Ensures robust regularization to prevent overfitting.
    *   **FastAI:** Deep neural networks for tabular data to capture non-linear relationships.
*   **Synergy:** By utilizing RoBERTa as an advanced semantic feature extractor and AutoGluon as the ultimate meta-decision-maker, the model effectively neutralizes the weaknesses of individual algorithms. This hybrid approach yields superior classification accuracy, improved convergence speed, and a highly resilient decision boundary.

---

## ⚙️ Key Features

- **Real-Time Inference:** Instant probability calculation via a Flask-based RESTful API.
- **Automated Language Filtering:** Integrated `langdetect` module that automatically blocks non-English inputs. Since the model was exclusively trained on English datasets, this filter maintains the statistical validity and integrity of the predictions.
- **Statistical Significance Threshold:** Prevents anomalous predictions by enforcing a strict 20-word minimum limit, ensuring the model has sufficient contextual data to analyze.
- **Premium Dashboard UI:** A highly responsive, dark-mode Graphical User Interface (GUI) built with vanilla CSS and JavaScript, featuring dynamic confidence level rings and instant visual feedback.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- PyTorch (CPU or CUDA)
- AutoGluon 1.1+

### Quick Start
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/hybrid-ai-detection.git
   cd hybrid-ai-detection
   ```

2. **Install dependencies:**
   Ensure you have the required ML libraries installed.
   ```bash
   pip install torch transformers pandas autogluon.tabular flask langdetect catboost lightgbm xgboost fastai
   ```

3. **Download Model Weights (Important):**
   Due to GitHub's file size limits, the pre-trained model weights are hosted externally. Download them from the link below and place them in the root directory:
   - [Download Model Weights (Google Drive)](#) *(Link will be added soon)*
   - Extract the contents so that `roberta_multitask_state_dict.pt` and `Label_Model_Final/` are in the main folder.

4. **Run the Application:**
   Start the local Flask server.
   ```bash
   python app.py
   ```

5. **Access the Dashboard:**
   Open your preferred web browser and navigate to:
   `http://127.0.0.1:5000`

---

## 📁 Repository Structure
*   `app.py`: Core Flask application, API routing, and hybrid model inference logic.
*   `templates/index.html`: Frontend dashboard layout structure.
*   `static/css/style.css`: UI styling, dark-mode themes, and animations.
*   `static/js/main.js`: Asynchronous API requests, error handling, and dynamic UI updates.
*   `Label_Model_Final/`: AutoGluon TabularPredictor ensemble metadata and weights.
*   `roberta_multitask_state_dict.pt`: Fine-tuned state dictionary for the RoBERTa feature extractor.

---

## 📝 Academic Citation
If you use this codebase, methodology, or model in your academic research, please consider citing our forthcoming paper:

> *(Citation details to be added upon publication)*

## 📄 License
This project is licensed under the MIT License.
