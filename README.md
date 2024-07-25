## Data Description
Import và đọc dữ liệu với thư việt Pandas,
Sử dụng các method head(), describe(), info() để kiểm tra thông tin sơ lược về dữ liêu.
Data ban đầu có 83 trường dữ liệu, trong đó target là “label”
![image](https://github.com/user-attachments/assets/1d979bee-0bd2-459f-896e-caf894a24cc3)

Sử dụng các hàm insna() để kiểm tra các dữ liệu bị thiếu, để xử lý các dữ liệu này ta thay thế chúng với mode của các cột.
![image](https://github.com/user-attachments/assets/9cc31df9-9f80-4c0b-8f82-d0bae12d7b40)

Đồng thời kiểm tra các dữ liêụ bị trùng lặp và xóa toàn bộ những dữ liệu này, chỉ giữ lại một bản duy nhất của các dữ liệu trùng lặp
![image](https://github.com/user-attachments/assets/38fbb255-974e-4387-9b38-8af98827b1ed)

## Feature Engineering
Sử dụng label encoding cho các dữ liệu category
![image](https://github.com/user-attachments/assets/1e13c393-2156-48c8-a683-90f83483109d)

## Data Filtering
## Exploratory Data Analysis
Sử dụng heatmap để thấy được correlation giữa các cột khác nhau trong bảng dữ liệu,
Để ý các cột correlate lẫn nhau dọc theo đường chéo của heatmap.
![image](https://github.com/user-attachments/assets/79b13d0b-a392-4e56-8a09-84cfd74c2905)
Nhận thấy có 7 nhóm cột có mối quan hệ dạng này và để tránh gây ra hiện tượng đa cộng tuyến đồng thời giảm chiều dữ liệu để tiết kiệm hiệu năng tính toán, chúng ta sử dụng PCA để biến chúng thành một chiều duy nhất
![image](https://github.com/user-attachments/assets/51675d22-5ab7-457a-a4b9-a2bc08f25a6c)
Dữ liệu mới có được từ PCA giảm xuống 43 cột thay vì hơn 80 như cũ
Bên cạnh đó có thể thấy, dữ liệu bị imbalenced:
![image](https://github.com/user-attachments/assets/4e13143f-651e-4e01-af96-17d3fa8d5af0)
Vì vậy, chúng ta cân nhắc cả 2 phương pháp là
- Under Sampling bằng cách loại bỏ một cách random các dữ liệu có label là "0" để nó trở nên cân bằng về mặt kích thướng với các dữ liệu có label là 1
- Over Sampling  thì ngược lại, gia tăng các điểm dữ liệu của nhóm label "0" để nó trở nên cân bằng về mặt kích thướng với các dữ liệu có label là 1, ở đây chúng ta sử dụng SMOTE (Synthetic Minority Oversampling Technique) method: 
![image](https://github.com/user-attachments/assets/2c696de7-903e-488d-b5e7-2eaee20e1c19)
![image](https://github.com/user-attachments/assets/cc80ebf5-aa9e-48f5-97fe-383c1042c73c)


## Data Preparation
Loại bỏ f25 vì không có ý nghĩa thống kê, toàn bộ dữ liệu đều là 1
## Feature Selection
Sử dụng "label" làm target và các biến còn lại là input của prediction
## Machine Learning Modeling
Sử dụng dữ liệu undersampling trên một loạt các model phân loại:
![image](https://github.com/user-attachments/assets/e6588369-c797-4d82-97b1-80478f364de0)
Nhận thấy Xgboost classifier và Random Forest có hiệu suất cao hơn hẳn các model còn lại
Tiếp tục sử dụng dữ liệu từ over sampling cho 2 model này, nhận thấy kết quả không có khác biệt giữa 2 model:
![image](https://github.com/user-attachments/assets/ff334484-4608-488f-a959-884ef98038bc)
 Quyết định sử dụng cả 2 model vào bước cuối, đó là sử dụng hyperparameter tuning để tối ưu hóa model, kết quả là:
![image](https://github.com/user-attachments/assets/6b3e7800-a76e-4b84-b820-1e2f4d0d11f1)
![image](https://github.com/user-attachments/assets/64d40738-202e-4931-a7f7-18bf417acdfc)
Chúng có hiệu suất gần như là tương đương. Đặc biệt nếu sử dụng ROC để theo dỡi thì:
![image](https://github.com/user-attachments/assets/ea69d699-5b0a-4eb6-848f-c4af83c3ff26)
## Conclusions
Cả hai mô hình dự đoán Random Forest và XGBoost đều cho khả năng dự đoán chính xác cao, ngay cả khi sử dụng các metric như Recall, Precision cho trường hợp dữ liệu imbalanced đều xấu xỉ 97%

## Model Deploy(Docker)
Mô hình dự đoán được triển khai dưới dạng API Flask và triển khai với docker
dữ liệu đầu vào cho API: dưới dạng json
ví dụ:
{"label":0.0,"f0":3,"f1":0.0,"f2":-15.0,"f3":0.0,"f4":0.0,"f12":0.0,"f13":0.0,"f14":0.0,"f15":0.0,"f16":0.0,"f21":85.0,"f23":0.0,"f24":29.0,"f25":0.0,"f26":0.0,"f34":0.0,"f35":48.0,"f36":0.0,"f37":0.0,"f45":0.0,"f46":0.0,"f47":0.0,"f48":0.0,"f56":500.0,"f57":0.0,"f58":86.0,"f59":0.0,"f60":0.0,"f68":2,"f69":10,"f70":0.0,"f71":61.0,"f72":0.0,"f73":0.0,"f81":245,"PCA_f5":-1284370.1356532155,"PCA_f17":-0.0071385538,"PCA_f27":-34.5720788949,"PCA_f38":-28.1619356953,"PCA_f49":-18.3254612004,"PCA_f61":-349.9820946494,"PCA_f74":-1276390.4333535051}
## command deploy model:
#### > docker build -t fraud_pred_model .
#### > docker run -p 5000:5000 fraud_pred_model

