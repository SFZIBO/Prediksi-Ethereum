# app.py
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import json
import os
import re

app = Flask(__name__)

# --- Memuat Model dan Informasi ---
model_path = 'ethereum_price_lstm_model.h5'
info_path = 'model_training_info.json'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' tidak ditemukan.")
if not os.path.exists(info_path):
    raise FileNotFoundError(f"Info file '{info_path}' tidak ditemukan.")

model = load_model(model_path)

with open(info_path, 'r') as f:
    training_info = json.load(f)

# --- Membaca dan Menyiapkan Scaler dari Data Asli ---
# Kita tetap membutuhkan scaler untuk transformasi input user
data_path = 'eth-usd-max.csv'
df = pd.read_csv(data_path)
df['snapped_at'] = pd.to_datetime(df['snapped_at'])
df.set_index('snapped_at', inplace=True)
df.sort_index(inplace=True)

# Ambil kolom harga ('price') untuk fit scaler
price_data = df[['price']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(price_data) # Fit scaler terhadap data harga asli

# --- Fungsi untuk Membuat Prediksi dari Input User ---
def make_prediction_from_input(model, scaler, input_prices, timesteps=60):
    try:
        # Ubah string input menjadi list angka
        prices_list = [float(x.strip()) for x in re.split(r'[,\s\n]+', input_prices) if x.strip()]
        
        if len(prices_list) != timesteps:
            return None, f"Input harus berisi tepat {timesteps} harga. Anda memasukkan {len(prices_list)} harga."

        input_array = np.array(prices_list).reshape(-1, 1)
        
        # Transformasi menggunakan scaler yang telah disesuaikan
        scaled_input = scaler.transform(input_array)
        X_pred = scaled_input.reshape(1, timesteps, 1)

        # Lakukan prediksi
        prediction_scaled = model.predict(X_pred, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)
        
        return prediction[0, 0], None # Kembalikan prediksi dan pesan error (jika ada)
    except ValueError:
        return None, "Input mengandung nilai yang bukan angka."
    except Exception as e:
        return None, f"Terjadi kesalahan saat memproses input: {str(e)}"

# --- Route untuk Halaman Utama ---
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    error_message = None
    input_prices_str = "" # Untuk mengisi kembali form jika terjadi error

    if request.method == 'POST':
        input_prices_str = request.form.get('prices', '')
        pred, err = make_prediction_from_input(model, scaler, input_prices_str)
        
        if err:
            error_message = err
        else:
            predicted_price = pred

    # Ambil informasi model untuk ditampilkan
    model_name = training_info.get("model_name", "N/A")
    rmse = training_info.get("rmse", "N/A")
    mse = training_info.get("mse", "N/A")
    timesteps = training_info.get("timesteps", "N/A")

    return render_template('index.html',
                           predicted_price=predicted_price,
                           error_message=error_message,
                           input_prices_str=input_prices_str, # Kirim input kembali ke template
                           model_name=model_name,
                           rmse=rmse,
                           mse=mse,
                           timesteps=timesteps)

# --- Route untuk Health Check ---
@app.route('/health')
def health():
    return {"status": "healthy"}, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))