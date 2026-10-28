# Dữ liệu sức khỏe: quy trình trước khi train

Phạm vi đợt mở rộng: khoảng 1.000 bài tổng cộng, gồm dữ liệu CafeF đã có và hai nguồn Sức khỏe & Đời sống, Tuổi Trẻ. Chỉ tải văn bản/HTML công khai; không tải model, ảnh, video hay chạy huấn luyện.

## Các phiên bản

- `data/cafef_health`: dữ liệu crawl cũ, giữ nguyên.
- `data/cafef_health_prepared_v1`: bản xử lý cũ và manifest chia tập đã cố định, giữ nguyên.
- `data/health_multisource_raw_v1`: bài mới, nguồn gốc và nhật ký chấp nhận/loại bỏ.
- Bản xử lý nhiều nguồn ghi vào thư mục phiên bản mới. Không ghi đè bản cũ.

## Kiểm tra tự động

1. Chỉ lấy URL trong chuyên mục sức khỏe đã xác định, tuân thủ robots theo User-Agent công khai, nghỉ giữa các yêu cầu, có ngân sách tải và tín hiệu dừng của runner.
2. Chỉ lấy thân bài; loại khối quảng cáo, chú thích ảnh, tin liên quan. Xử lý cả HTML lược bỏ thẻ đóng đoạn văn. Lưu URL chuẩn, ngày đăng, tác giả khi có, liên kết tham khảo ngoài báo, hash văn bản và hash HTML.
3. Kiểm tra Unicode, ngày đăng có múi giờ, độ dài 250–2.500 từ cách nhau bởi khoảng trắng, ít nhất bốn đoạn và bằng chứng nội dung sức khỏe. Loại dấu hiệu quảng bá thương mại rõ ràng. Đây là bộ lọc thực dụng, có thể bỏ sót hoặc loại nhầm.
4. Loại trùng chính xác và cách ly bản gần như sao chép (Jaccard 5-gram ≥ 0,85). Gom nhóm bài tương tự còn lại xuyên nguồn bằng 5-gram và tiêu đề; nhóm không được đi qua các tập. Số bài không đồng nghĩa số mẫu độc lập.
5. Giữ phân chia cũ cho các ID đã có. Nếu nhóm mới nối hai tập cũ, cách ly nhóm đó để tránh rò rỉ; không sửa nhãn tập để làm đẹp kết quả.
6. Bản tóm tắt nháp chỉ dùng câu nguyên văn, kèm vị trí ký tự để truy ngược từng câu. Không dùng sapo làm nhãn chuẩn. Mục tiêu độ dài 20–25% theo quy ước hiện có; bản không đạt được gắn cờ, không tự nới tiêu chí.
7. Bảo toàn nguồn đầu vào bằng hash, kiểm tra vị trí câu trích, trạng thái nhãn và bất biến của manifest. Chưa kiểm tra độ dài theo tokenizer vì chưa tải model/tokenizer.

## Phần cần duyệt trước khi coi là nhãn chuẩn

Mọi nhãn tự tạo đều là `extractive_pseudo`, `pending_human_review`, `approved=false`. Không đổi thành `human_reviewed` bằng việc chạy kiểm tra tự động.

Người duyệt cần kiểm tra: đúng chủ đề; đủ ý chính; giữ chủ thể, thời gian, đối tượng áp dụng; không mất phủ định hoặc điều kiện; không ghép các con số/liều lượng sai ngữ cảnh; phân biệt lời bác sĩ, lời bệnh nhân và kết quả nghiên cứu; không biến tương quan thành quan hệ nhân quả; không dùng tiêu đề giật gân làm kết luận. Với nội dung có thể ảnh hưởng điều trị, cần người có chuyên môn kiểm tra dẫn chứng.

Thiếu tác giả hoặc liên kết tham khảo là cờ ưu tiên duyệt, không tự chứng minh bài sai. Có tác giả hoặc dẫn nguồn cũng không chứng minh bài đúng. Từ khóa sức khỏe chỉ chứng minh mức liên quan sơ bộ; bài báo không phải hướng dẫn lâm sàng đã được xác thực.

## Đủ để train chưa?

Khoảng 1.000 bài sạch là quy mô thử nghiệm ban đầu cho mô hình đã được tiền huấn luyện, không phải căn cứ để train mô hình ngôn ngữ từ đầu. Chưa có learning curve và nhãn đánh giá chuẩn nên chưa thể kết luận đủ. Ưu tiên tạo tập validation/test được duyệt độc lập và kiểm tra tokenizer trước; sau đó mới đo hiệu quả khi tăng lượng train. Không dùng điểm trên nhãn trích xuất tự tạo để tuyên bố chất lượng tóm tắt thực tế hoặc độ an toàn y khoa.

Các nguồn có thể thiên lệch theo thời gian, chủ đề và phong cách viết; cần đọc báo cáo phân bố theo nguồn và số nhóm độc lập trước khi chọn thêm dữ liệu. Quyền truy cập robots không đồng nghĩa giấy phép tái phân phối hoặc sử dụng lại toàn bộ bài báo; không tự xuất bản corpus.
