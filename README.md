# 📈 LSTM Stock Market Future Price Predictor

A Streamlit web application for time-series forecasting of the Nifty 50 index (`^NSEI`') using a Long Short-Term Memory (LSTM) neural network.

The model is trained on **15 years of historical price data** to predict the closing prices for the next 30 trading days.

---

## ✨ Features

* **Data Source:** Fetches historical data for the Nifty 50 index (`^NSEI`) from Yahoo Finance.
* **Model:** Uses a deep learning **LSTM** architecture (built with TensorFlow/Keras) for highly accurate sequence prediction.
* **Time Horizon:** Trained on 15 years of data for robust long-term context.
* **Visualization:** Interactive Plotly chart displaying historical price data and the 30-day forecast seamlessly.
* **Web Interface:** Hosted via **Streamlit** for an easy-to-use, interactive, and shareable experience.

---

## 🛠️ Installation & Setup

To run this application locally, follow these steps.

### 1. Clone the Repository

```bash
git clone [https://github.com/SwethaNarayanasamy1920/Stock-Hunter.git](https://github.com/SwethaNarayanasamy1920/Stock-Hunter.git)
cd Stock-Hunter
2. Create and Activate Virtual Environment (Recommended)
This isolates the project dependencies from your system Python environment.

Bash

# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
3. Install Dependencies
All necessary libraries, including Streamlit, yfinance, TensorFlow, and Plotly, are listed in requirements.txt.

Bash

pip install -r requirements.txt
4. Run the Application
Execute the Python script using Streamlit:

Bash

streamlit run Stockpp.py
The application will automatically open in your default web browser (usually at http://localhost:8501).

Model Architecture
The LSTM model employs a simple, effective sequential structure:

Layer	Type	Units / Output Shape	Purpose
Input	LSTM	50	Processes the 60-day price sequence.
Dropout	20%	Prevents overfitting.
LSTM	50	Extracts higher-level temporal features.
Dropout	20%	Prevents overfitting.
Output	Dense	25	Intermediate layer for feature mapping.
Dense	1	Final output (the predicted next-day closing price).

Export to Sheets
Sequence Length: 60 days

Optimizer: Adam

Loss Function: Mean Squared Error (MSE)

Disclaimer
This project is intended for educational and demonstration purposes only. It should not be used for making real-world trading decisions. Stock market prediction is highly complex and volatile, and the results from this model should be treated as an academic forecast.

🤝 Contribution
Feel free to open issues or submit pull requests for any improvements!
