from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Thay thế RandomForestClassifier bằng LinearRegression

app = Flask(__name__)

# --- TỰ ĐỘNG ĐỌC FILE CSV VÀ HUẤN LUYỆN AI ---
def load_data_and_train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
    
    if not csv_files:
        return None, []
        
    file_path = os.path.join(current_dir, csv_files[0])
    df = pd.read_csv(file_path)
    
    # XỬ LÝ ĐẶC BIỆT CHO HỒI QUY TUYẾN TÍNH:
    # Nếu cột Financial_Status đang là dạng chữ (Normal, Risky...), ta cần chuyển về dạng số để mô hình tính toán đường thẳng.
    if df['Financial_Status'].dtype == 'object':
        # Chuyển đổi: 'Normal' sẽ thành số 1, các trạng thái khác (Risky, Bankruptcy...) thành số 0
        df['Financial_Status_Numeric'] = df['Financial_Status'].apply(lambda x: 1 if str(x).lower() == 'normal' else 0)
        X = df.drop(columns=['Financial_Status', 'Financial_Status_Numeric'])
        y = df['Financial_Status_Numeric']
    else:
        X = df.drop(columns=['Financial_Status'])
        y = df['Financial_Status']
        
    model = LinearRegression()  # Thay thế sang mô hình Hồi quy tuyến tính
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
        # Lấy dữ liệu từ giao diện web gửi lên (Giữ nguyên)
        data = request.json
        val_assets = float(data.get('assets', 0))
        val_liabilities = float(data.get('liabilities', 0))
        val_revenue = float(data.get('revenue', 0))
        
        # Bù các cột dữ liệu cho mô hình AI (Giữ nguyên tên biến và logic so khớp từ khóa)
        user_inputs = {}
        for feat in feature_names:
            if "asset" in feat.lower():
                user_inputs[feat] = val_assets
            elif "liabilit" in feat.lower():
                user_inputs[feat] = val_liabilities
            elif "revenue" in feat.lower() or "sales" in feat.lower():
                user_inputs[feat] = val_revenue
            else:
                user_inputs[feat] = 0.0
                
        input_df = pd.DataFrame([user_inputs])[feature_names]
        
        # Dự đoán bằng mô hình Hồi quy tuyến tính -> Trả về 1 con số thực (Ví dụ: 0.85 hoặc 0.23)
        prediction_val = model.predict(input_df)[0]
        
        # ĐỔI LOGIC XỬ LÝ KẾT QUẢ CHO PHÙ HỢP VỚI LÀM VIỆC VỚI SỐ THỰC:
        # Vì Hồi quy tuyến tính không có hàm tính xác suất .predict_proba(), ta dùng ngưỡng 0.5 để phân loại
        if prediction_val >= 0.5:
            prediction_status = "Normal"
            # Tính độ tự tin mô phỏng dựa trên khoảng cách số thực tiến gần về 1
            confidence = min(100.0, max(50.0, prediction_val * 100))
        else:
            prediction_status = "Risky"
            # Tính độ tự tin mô phỏng dựa trên khoảng cách số thực lùi về hướng 0
            confidence = min(100.0, max(50.0, (1 - prediction_val) * 100))
            
        return jsonify({
            'status': prediction_status,
            'confidence': f"{confidence:.2f}%",
            'raw_score': f"{prediction_val:.4f}"  # Trả thêm điểm số thô để bạn tiện debug nếu cần
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)