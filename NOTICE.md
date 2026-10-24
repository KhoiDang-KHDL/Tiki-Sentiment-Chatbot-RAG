# Provenance

Repository này được chuẩn bị từ bản mã tham khảo:
https://github.com/phngahn/sentiment-chatbot-mlops

`src/`, các DAG, training/export/evaluation scripts và model evaluation JSON giữ nguồn gốc
bản tham khảo. Thông tin môn học, bảng thành viên và phần triển khai EC2 được loại khỏi
README; automatic deployment workflow của môi trường cũ không được đưa sang.

`runtime/`, các runtime regression tests, metrics/provisioning, incremental retry và CI
được chuyển và điều chỉnh từ:
https://github.com/KhoiDang-KHDL/Tiki-Sentiment-Chatbot

Namespace, Docker build inputs, model path resolver và Compose được điều chỉnh cho repo mới.
Benchmark/evaluation của upstream không được coi là phép đo lại trên runtime mới.
Thông tin bản quyền hoặc license có sẵn trong mã nguồn vẫn được giữ nguyên.
