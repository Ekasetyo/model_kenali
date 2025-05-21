from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pymongo import MongoClient
from flask_cors import CORS
from datetime import datetime  # Import datetime

app = Flask(__name__)
CORS(app)  # Aktifkan CORS untuk semua route

# Load model
model = joblib.load("stroke_model.pkl")

# Koneksi ke MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["kenali"]  # Ganti dengan nama database Anda

# Kolom yang diharapkan
expected_columns = [
    'sex', 'age', 'hypertension', 'heart_disease', 
    'ever_married', 'work_type', 'Residence_type', 
    'avg_glucose_level', 'bmi', 'smoking_status'
]

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Ambil data JSON dari request
        data = request.get_json()
        
        # Validasi data
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Ambil user_id
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        # Konversi ke tipe data yang tepat
        input_data = [
            int(data.get('sex', 0)),
            float(data.get('age', 0)),
            int(data.get('hypertension', 0)),
            int(data.get('heart_disease', 0)),
            int(data.get('ever_married', 0)),
            int(data.get('work_type', 0)),
            int(data.get('Residence_type', 0)),
            float(data.get('avg_glucose_level', 0)),
            float(data.get('bmi', 0)),
            int(data.get('smoking_status', 0))
        ]
        
        # Buat DataFrame dengan nama kolom yang benar
        sample_df = pd.DataFrame([input_data], columns=expected_columns)
        
        # Prediksi
        pred = model.predict(sample_df)
        
        # Hasil prediksi
        result = "anda beresiko terkena stroke" if pred[0] == 1 else "anda tidak beresiko"
        
        # Simpan ke MongoDB
        data_to_save = data.copy()
        data_to_save['user_id'] = user_id  # Tambahkan user_id
        data_to_save['prediction'] = result
        data_to_save['created_at'] = datetime.now().isoformat()  # Tambahkan timestamp
        inserted = db['hasil_deteksi'].insert_one(data_to_save)

        # Siapkan response tanpa ObjectId
        data['prediction'] = result
        data['id'] = str(inserted.inserted_id)

        return jsonify({
            'status': 'success',
            'prediction': result,
            'data': data
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)