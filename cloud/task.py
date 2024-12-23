# Import các thư viện cần thiết
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from google.cloud import bigquery, bigquery_storage
from google import auth
import numpy as np
import argparse
import os
import pickle

# Thêm parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('--project-id', dest='project_id', type=str, help='Project ID.')
parser.add_argument('--training-dir', dest='training_dir', default=os.getenv("AIP_MODEL_DIR"),
                    type=str, help='Dir to save the data and the trained model.')
parser.add_argument('--bq-source', dest='bq_source', type=str, help='BigQuery data source for training data.')
args = parser.parse_args()

# Query để lấy data từ BigQuery
BQ_QUERY = """
SELECT 
    namsinh,
    gioitinh,
    drl,
    diem_tt,
    dtb_toankhoa,
    dtb_tichluy,
    sotc_tichluy,
    diemtbhk_1,
    diemtbhk_2,
    diemtbhk_3,
    diemtbhk_4,
    diemtbhk_5,
    diemtbhk_6,
    diemtbhk_7,
    diemtbhk_8
FROM `{}`
""".format("main-train-445501.mlops.student_information")

# Lấy credentials và khởi tạo clients
credentials, project = auth.default()
bqclient = bigquery.Client(credentials=credentials, project="main-train-445501")
bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)

# Query data
df = (
    bqclient.query(BQ_QUERY)
    .result()
    .to_dataframe(bqstorage_client=bqstorageclient)
)

# Split features và target
X = df.drop(['diemtbhk_8'], axis=1)  # Giả sử dtb_tichluy là target
y = df['diemtbhk_8']

# Split data thành train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Khởi tạo mô hình TabNet
model = TabNetRegressor(
    n_d=8,  # Width of the decision prediction layer
    n_a=8,  # Width of the attention embedding
    n_steps=3  # Number of steps in the architectur
)

# Train mô hình
model.fit(
    X_train.values, y_train.values.reshape(-1, 1),
    max_epochs=1,
    patience=10
)

# Lưu mô hình
save_path = os.path.join(args.training_dir, "model.pkl")
saved_filepath = model.save_model(save_path)


# In kết quả
#print("Train performance:", model.explain(X_train.values)["loss"])
#print("Test performance:", model.explain(X_test.values)["loss"])

# Lưu test data để dùng cho batch prediction
X_test.to_csv(os.path.join(args.training_dir, "test.csv"), index=False)