# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603643**
- **Họ và tên: Trần Quốc ĐÔng**
- **SEED** (in ở ô đầu notebook): 6930

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.902      0.949        0.947   
2. BoW + MLP (deep learning)         1.000   0.957      0.978        0.974   
3a. Encoder (MiniLM) + kNN           0.880   0.957      0.917        1.000   
3b. Encoder (MiniLM) + LogReg        0.944   0.913      0.928        0.923   
3c. Encoder (MiniLM) + MLP head      0.935   0.935      0.935        0.950   
4a. LLM nhỏ zero-shot                1.000   1.000      1.000        1.000   
4b. LLM lớn zero-shot                1.000   1.000      1.000        1.000   
4c. LLM nhỏ few-shot                 0.950   0.950      0.950        0.950   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            5.593      19521.962       0.054   
2. BoW + MLP (deep learning)         8.176      42308.034       0.226   
3a. Encoder (MiniLM) + kNN          80.955         23.616      55.890   
3b. Encoder (MiniLM) + LogReg       81.433         31.508      29.023   
3c. Encoder (MiniLM) + MLP head     81.951         31.399      28.268   
4a. LLM nhỏ zero-shot                0.000          0.732    1365.854   
4b. LLM lớn zero-shot                0.000          4.239     235.906   
4c. LLM nhỏ few-shot                 0.000          0.790     870.108   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.083           0.0  
2. BoW + MLP (deep learning)          0.352           0.0  
3a. Encoder (MiniLM) + kNN           80.600           0.0  
3b. Encoder (MiniLM) + LogReg        39.005           0.0  
3c. Encoder (MiniLM) + MLP head      38.622           0.0  
4a. LLM nhỏ zero-shot              1365.854           0.0  
4b. LLM lớn zero-shot               235.906           0.0  
4c. LLM nhỏ few-shot               2836.736           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.


## 3. Dự đoán và thực tế

1. LLM lớn vs LLM nhỏ (4.A)
    Dự đoán: Model lớn có F1 cao hơn model nhỏ.
    Thực tế: Cả hai đều đạt F1 = 1.000 trên 5 tin, nên dự đoán không khớp.
    Nguyên nhân: Tập thử chỉ có 5 tin nên chưa đủ để phân biệt khả năng phân loại. Tốc độ API còn phụ thuộc vào máy chủ, tải hệ thống và thời điểm gọi.
2. Few-shot (4.B)
    Dự đoán: Thêm 4 ví dụ sẽ làm F1 tăng.
    Thực tế: Few-shot đạt F1 = 0.950, trong khi zero-shot từng đạt 1.000 trên 1 tin.
    Nguyên nhân: Few-shot không phải lúc nào cũng cải thiện kết quả. Các ví dụ có thể chưa đại diện tốt; hơn nữa, hai lần thử dùng số lượng tin khác nhau nên chưa thể kết luận F1 thực sự giảm.
3. Đường Pareto
    Dự đoán: Logistic Regression và Embedding + Logistic Regression có khả năng nằm trên đường Pareto.
    Thực tế: BoW + Logistic Regression, BoW + MLP và Encoder + kNN nằm trên đường Pareto; Encoder + Logistic Regression bị lấn át.
    Nguyên nhân: Encoder + Logistic Regression có độ trễ cao nhưng F1 không đủ tốt để bù lại. Encoder + kNN đạt F1 cao hơn nên có mặt trên đường Pareto.
## 4. Phân tích 50 case khó

Model tụt nhiều nhất là BoW + Logistic Regression, F1 giảm từ 0.947 xuống 0.333 (giảm 0.614), độ chính xác trên 50 case khó chỉ đạt 52%.
Hai nhóm case tiêu biểu:
1. Thao túng/lừa đảo: BoW + Logistic Regression chỉ đoán đúng 12%. Nguyên nhân là mô hình dựa vào tần suất từ khóa, không hiểu ngữ cảnh và ý đồ lừa đảo. Những tin nhắn không chứa từ khóa spam quen thuộc dễ bị phân loại nhầm thành ham.
2. Spam giống tin nhắn tự nhiên: Mô hình chỉ đoán đúng 10%. Nguyên nhân là tin spam sử dụng ngôn ngữ giống hội thoại bình thường, ít từ khóa như “free”, “win”, “prize”, khiến BoW không nhận diện được dấu hiệu spam.
## 5. Thử thách cá nhân

🎯 Đề #1: ML · Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1. Báo precision/recall/F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?
# X_te: dữ liệu test đã vector hóa
# w, b: trọng số và bias đã huấn luyện ở Phần 1
# y_te: nhãn thật (spam = 1, ham = 0)

probs = predict_proba(X_te, w, b)

for threshold in [0.2, 0.5, 0.8]:
    preds = (probs >= threshold).astype(int)
    m = metrics(y_te, preds)

    print(
        f"Threshold = {threshold:.1f} | "
        f"Precision = {m['precision']:.3f} | "
        f"Recall = {m['recall']:.3f} | "
        f"F1 = {m['f1']:.3f}"
    )
Threshold = 0.2 | Precision = 0.882 | Recall = 0.978 | F1 = 0.928
Threshold = 0.5 | Precision = 1.000 | Recall = 0.902 | F1 = 0.949
Threshold = 0.8 | Precision = 1.000 | Recall = 0.663 | F1 = 0.797

Khi giảm ngưỡng xuống 0.2, mô hình dễ đánh dấu tin nhắn là spam hơn, nên có thể chặn nhầm nhiều tin ham. Khi tăng ngưỡng lên 0.8, mô hình chỉ đánh dấu spam khi xác suất dự đoán đủ cao, giúp giảm nguy cơ chặn nhầm.

## 6. Phương án cho sản phẩm thật

chọn BoW + MLP để xử lý 10 triệu SMS/ngày với yêu cầu p95 dưới 50 ms/tin.

Số liệu: Theo kết quả thí nghiệm, BoW + MLP có F1 ≈ 0.975, tương ứng \(1-F1\approx2.5\%\), và nằm trong nhóm đáp ứng SLA 50 ms. Đây là mô hình có lỗi thấp nhất trong các phương án đạt SLA trên biểu đồ Pareto.

Đánh đổi: Chấp nhận F1 thấp hơn một số mô hình như Encoder + kNN để đổi lấy tốc độ xử lý nhanh, chi phí vận hành thấp và không phụ thuộc API LLM. Ngoài ra, BoW + MLP có thể bỏ sót spam tinh vi do hạn chế trong việc hiểu ngữ cảnh.

Kết luận: BoW + MLP là lựa chọn phù hợp vì cân bằng giữa độ chính xác, tốc độ và chi phí. Tuy nhiên, cần kiểm tra p95 thực tế dưới tải cao và khả năng xử lý trung bình khoảng 116 SMS/giây để bảo đảm đáp ứng 10 triệu SMS/ngày.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
