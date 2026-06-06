from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --- TỰ ĐỘNG ĐỌC FILE CSV VÀ HUẤN LUYỆN AI ---
def load_data_and_train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
    if not csv_files:
        return None, []
    
    file_path = os.path.join(current_dir, csv_files[0])
    df = pd.read_csv(file_path)
    
    X = df.drop(columns=['Financial_Status'])
    y = df['Financial_Status']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_names = load_data_and_train()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Không tìm thấy file dữ liệu CSV!'})
        
    try:
        # Lấy dữ liệu từ giao diện web gửi lên
        data = request.json
        val_assets = float(data.get('assets', 0))
        val_liabilities = float(data.get('liabilities', 0))
        val_revenue = float(data.get('revenue', 0))
        
        # Bù các cột dữ liệu cho mô hình AI
        user_inputs = {}
        for feat in feature_names:
            if "asset" in feat.lower(): user_inputs[feat] = val_assets
            elif "liabilit" in feat.lower(): user_inputs[feat] = val_liabilities
            elif "revenue" in feat.lower() or "sales" in feat.lower(): user_inputs[feat] = val_revenue
            else: user_inputs[feat] = 0.0
            
        input_df = pd.DataFrame([user_inputs])[feature_names]
        
        # Dự đoán
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        confidence = probabilities[1] * 100 if prediction == "Normal" else probabilities[0] * 100
        
        return jsonify({
            'status': prediction,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)