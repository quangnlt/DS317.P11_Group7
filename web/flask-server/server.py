from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np # Import numpy
import warnings
import os
# from models import Students, db
from flask_sqlalchemy import SQLAlchemy
warnings.filterwarnings("ignore", category=FutureWarning)
# Bỏ qua UserWarning chỉ từ pytorch_tabnet
warnings.filterwarnings("ignore", message="Device used : cpu")

app = Flask(__name__)
CORS(app, support_credentials=True)  # Tự động cấu hình CORS cho toàn bộ ứng dụng

# Cấu hình database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///my_database.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db.init_app(app)
db = SQLAlchemy(app)

class Students(db.Model):
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)  # Khóa chính
    mssv = db.Column(db.String(20), nullable=False, unique=True)  # Mã số sinh viên
    namsinh = db.Column(db.Integer, nullable=False)  # Năm sinh
    gioitinh = db.Column(db.String(10), nullable=False)  # Giới tính
    drl = db.Column(db.Float, nullable=True)  # Điểm rèn luyện
    diem_tt = db.Column(db.Float, nullable=True)  # Điểm thực tập
    dtb_toankhoa = db.Column(db.Float, nullable=True)  # Điểm trung bình toàn khóa
    dtb_tichluy = db.Column(db.Float, nullable=True)  # Điểm trung bình tích lũy
    sotc_tichluy = db.Column(db.Integer, nullable=True)  # Số tín chỉ tích lũy
    diemtbhk_1 = db.Column(db.Float, nullable=True)  # Điểm TB học kỳ 1
    diemtbhk_2 = db.Column(db.Float, nullable=True)  # Điểm TB học kỳ 2
    diemtbhk_3 = db.Column(db.Float, nullable=True)  # Điểm TB học kỳ 3
    diemtbhk_4 = db.Column(db.Float, nullable=True)  # Điểm TB học kỳ 4
    diemtbhk_5 = db.Column(db.Float, nullable=True)  # Điểm TB học kỳ 5
    diemtbhk_6 = db.Column(db.Float, nullable=True)  # Điểm TB học kỳ 6
    diemtbhk_7 = db.Column(db.Float, nullable=True)  # Điểm TB học kỳ 7
    diemtbhk_8 = db.Column(db.Float, nullable=True)  # Điểm TB học kỳ 8
    mssv_login = db.Column(db.Integer, nullable=False, unique=True)  # Mã login đánh số từ 1 đến n

    def __repr__(self):
        return f'<Student {self.mssv}>'

model = TabNetRegressor()

@app.route('/predict', methods=['POST'])
def predict():

    coefficients = {
        "namsinh": 0,
        "gioitinh": 0,
        "dtb_toankhoa": 0,
        "dtb_tichluy": 0,
        "sotc_tichluy": 0,
        "diemtbhk_1": 0,
        "diemtbhk_2": 0,
        "diemtbhk_3": 0,
        "diemtbhk_4": 0,
        "diemtbhk_5": 0,
        "diemtbhk_6": 0,
        "diemtbhk_7": 0
    }
    form = request.json
    if not form:
        return jsonify({'error': 'No formData provided'}), 400
    
    namsinh = int(form['namsinh'])
    if namsinh > 1985:
        namsinh -= 1985

    predictors = [
        namsinh,  # Không cần trừ nữa vì đã xử lý trong điều kiện
        float(form['gioitinh']),
        float(form['dtb_toankhoa']),
        float(form['dtb_tichluy']),
        float(form['sotc_tichluy'])
    ]
    i = 1
    while 'diemtbhk_' + str(i) in form:
        if form['diemtbhk_' + str(i)] != 0:
            predictors.append(float(form['diemtbhk_' + str(i)]))
            i += 1
        else:
            break
    
    
    # print(predictors)
    model_path = f'models/tabnet_diemtbhk_{len(predictors)-4}.pt.zip'
    model.load_model(model_path)
    X = np.array(predictors).reshape(1, -1)
    prediction = model.predict(X)
    result = float(prediction[0][0])

    # Define coefficients for each scenario
    coefficients_map = {
        12: {
            "namsinh": 2.75,
            "gioitinh": 1.97,
            "dtb_toankhoa": 57.49,
            "dtb_tichluy": 2.22,
            "sotc_tichluy": 13.29,
            "diemtbhk_1": 4.65,
            "diemtbhk_2": 2.70,
            "diemtbhk_3": 1.60,
            "diemtbhk_4": 3.39,
            "diemtbhk_5": 4.91,
            "diemtbhk_6": 1.59,
            "diemtbhk_7": 3.43
        },
        11: {
            "namsinh": 2.75,
            "gioitinh": 1.97,
            "dtb_toankhoa": 57.49,
            "dtb_tichluy": 2.22,
            "sotc_tichluy": 13.29,
            "diemtbhk_1": 5.22,
            "diemtbhk_2": 3.27,
            "diemtbhk_3": 2.17,
            "diemtbhk_4": 3.96,
            "diemtbhk_5": 5.48,
            "diemtbhk_6": 2.16
        },
        10: {
            "namsinh": 2.75,
            "gioitinh": 1.97,
            "dtb_toankhoa": 57.49,
            "dtb_tichluy": 2.22,
            "sotc_tichluy": 13.29,
            "diemtbhk_1": 6.27,
            "diemtbhk_2": 4.32,
            "diemtbhk_3": 3.22,
            "diemtbhk_4": 4.96,
            "diemtbhk_5": 6.48
        },
        9: {
            "namsinh": 2.75,
            "gioitinh": 1.97,
            "dtb_toankhoa": 57.49,
            "dtb_tichluy": 2.22,
            "sotc_tichluy": 13.29,
            "diemtbhk_1": 7.13,
            "diemtbhk_2": 5.18,
            "diemtbhk_3": 4.08,
            "diemtbhk_4": 5.82
        },
        8: {
            "namsinh": 2.75,
            "gioitinh": 1.97,
            "dtb_toankhoa": 57.49,
            "dtb_tichluy": 2.22,
            "sotc_tichluy": 13.29,
            "diemtbhk_1": 9.10,
            "diemtbhk_2": 7.15,
            "diemtbhk_3": 6.05
        },
        7: {
            "namsinh": 2.75,
            "gioitinh": 1.97,
            "dtb_toankhoa": 57.49,
            "dtb_tichluy": 2.22,
            "sotc_tichluy": 13.29,
            "diemtbhk_1": 12.21,
            "diemtbhk_2": 10.26
        },
        6: {
            "namsinh": 2.75,
            "gioitinh": 1.97,
            "dtb_toankhoa": 57.49,
            "dtb_tichluy": 2.22,
            "sotc_tichluy": 13.29,
            "diemtbhk_1": 22.27
        }
    }

    # Update coefficients based on the number of predictors
    predictor_count = len(predictors)
    print(len(predictors))
    print(predictors)

    if predictor_count in coefficients_map:
        coefficients = coefficients_map[predictor_count]
        print("Updated coefficients:", coefficients)
        print(coefficients)
        print(len(predictors))
    else:
        print(f"No coefficients available for {predictor_count} predictors.")
        
    
    return jsonify({'result': result, 'message': 'Đã nhận kết quả dự đoán', 'coefficients': coefficients})


@app.route("/login", methods=["POST"])
def login():
    # Lấy dữ liệu mssv_login từ yêu cầu JSON
    data = request.get_json()
    mssv_login = data.get("mssv_login")

    # Tìm kiếm sinh viên trong database theo mssv_login
    student = Students.query.filter_by(mssv_login=mssv_login).first()
    if student:
        # Trả về thông tin của tất cả các cột
        return jsonify({
            "mssv_login": student.mssv_login,
            "mssv": student.mssv,
            "namsinh": student.namsinh,
            "gioitinh": student.gioitinh,
            "drl": student.drl,
            "diem_tt": student.diem_tt,
            "dtb_toankhoa": student.dtb_toankhoa,
            "dtb_tichluy": student.dtb_tichluy,
            "sotc_tichluy": student.sotc_tichluy,
            "diemtbhk_1": student.diemtbhk_1,
            "diemtbhk_2": student.diemtbhk_2,
            "diemtbhk_3": student.diemtbhk_3,
            "diemtbhk_4": student.diemtbhk_4,
            "diemtbhk_5": student.diemtbhk_5,
            "diemtbhk_6": student.diemtbhk_6,
            "diemtbhk_7": student.diemtbhk_7,
            "diemtbhk_8": student.diemtbhk_8,
        }), 200
    
    else:
        return jsonify({"message": "Mã số sinh viên không tồn tại"}), 400

if __name__ == '__main__':
    with app.app_context():  # This ensures that db.create_all() is executed in the correct context
        db.create_all()  # Create the database tables if they don't exist already
    app.run(debug=True)
