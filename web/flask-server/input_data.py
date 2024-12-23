
'''
file này chỉ chạy duy nhất 1 lần để đọc dữ liệu từ csv vào db
cận thận khi sử dụng vì có thể làm tăng dung lượng của db
'''

import pandas as pd
from server import db, app, Students # Import app và db từ server.py
# from models import Students  # Import model Students từ models.py

# Đọc dữ liệu từ file CSV
df = pd.read_csv('flask-server/datasets/students_dataset_v2.csv')  # Cập nhật đường dẫn chính xác

# Kiểm tra xem cột 'mssv_login' đã có trong file CSV chưa
if 'mssv_login' not in df.columns:
    # Tạo giá trị cho mssv_login nếu chưa có
    df['mssv_login'] = range(1, len(df) + 1)

# Xóa tất cả dữ liệu cũ trong bảng Students
with app.app_context():
    db.session.query(Students).delete()  # Xóa dữ liệu cũ
    db.session.commit()  # Commit để thay đổi có hiệu lực
    print("Dữ liệu cũ đã được xóa!")

    # Nhập dữ liệu mới từ CSV vào bảng Students
    for index, row in df.iterrows():
        student = Students(
            mssv=row['mssv'],
            namsinh=row['namsinh'],
            gioitinh=row['gioitinh'],
            drl=row['drl'],
            diem_tt=row['diem_tt'],
            dtb_toankhoa=row['dtb_toankhoa'],
            dtb_tichluy=row['dtb_tichluy'],
            sotc_tichluy=row['sotc_tichluy'],
            diemtbhk_1=row['diemtbhk_1'],
            diemtbhk_2=row['diemtbhk_2'],
            diemtbhk_3=row['diemtbhk_3'],
            diemtbhk_4=row['diemtbhk_4'],
            diemtbhk_5=row['diemtbhk_5'],
            diemtbhk_6=row['diemtbhk_6'],
            diemtbhk_7=row['diemtbhk_7'],
            diemtbhk_8=row['diemtbhk_8'],
            mssv_login=row['mssv_login']  # Đảm bảo mssv_login có trong dữ liệu
        )
        db.session.add(student)
    
    db.session.commit()  # Lưu tất cả dữ liệu vào cơ sở dữ liệu
    print("Dữ liệu đã được nhập thành công vào bảng Students!")
