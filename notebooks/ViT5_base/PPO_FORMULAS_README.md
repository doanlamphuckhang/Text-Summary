# PPO Stage 3 - Công Thức Và Cách Hiểu Nhanh

Tài liệu này tổng hợp các công thức chính trong `stage3_vit5_RLUsingPPO.ipynb`: reward, KL penalty, advantage và PPO loss. 
## 1. Ký Hiệu

- `src`: văn bản nguồn.
- `gold`: bản tóm tắt tham chiếu.
- `pred`: bản tóm tắt do Actor sinh ra.
- `Actor`: mô hình Stage 2 được tiếp tục cập nhật bằng PPO.
- `Reference`: bản sao Stage 2 được đóng băng, dùng để tính KL.
- `π_actor`: phân phối sinh token của Actor.
- `π_ref`: phân phối sinh token của Reference.
- `V(s)`: reward kỳ vọng tại trạng thái `s`, được ước lượng bởi value head.

Trong sinh văn bản:

```text
s_t = source + các token summary đã sinh trước đó
a_t = token tiếp theo
```

## 2. Reward Function

Trong code, reward được tính bởi:

```python
compute_reward_breakdown(src, pred, gold)
```

Reward gồm 4 thành phần chính:

```text
raw_reward =
    0.45 * number_recall
  + 0.20 * length_score
  + 0.15 * diversity_score
  - 0.20 * number_hallucination
```

Nếu `number_hallucination > 0.5`, code trừ thêm hard penalty:

```text
raw_reward = raw_reward - 1.0
```

Reward cuối cùng được clip:

```text
reward = clip(raw_reward, -1.0, 1.0)
```

Lưu ý: notebook hiện tại không thấy bước batch-normalize reward thủ công trước `ppo_trainer.step(...)`. Reward sau khi tính được chuyển thành tensor và đưa trực tiếp vào PPO.

## 3. Number Recall

Trong Stage 3, `number_recall` ưu tiên các số trong `gold`, không phải toàn bộ số trong `src`:

```text
target_numbers = numbers(gold)
number_recall = |target_numbers ∩ numbers(pred)| / |target_numbers|
```

Nếu `gold` không có số, code trả về:

```text
NUMBER_NO_TARGET_REWARD = 0.0
```

Ý nghĩa: mô hình chỉ được khuyến khích giữ các số quan trọng xuất hiện trong bản tóm tắt tham chiếu. Nếu dùng toàn bộ số trong source, mô hình có thể bị ép copy quá nhiều số phụ, làm summary dài và kém cô đọng.

Ví dụ:

```text
gold: "Bệnh nhân dùng 5mg thuốc trong 7 ngày."
pred: "Bệnh nhân dùng 5mg thuốc."

target_numbers = {5, 7}
pred_numbers = {5}
number_recall = 1 / 2 = 0.5
```

## 4. Number Hallucination

Trong code:

```text
number_hallucination = |numbers(pred) - numbers(src)| / |numbers(pred)|
```

Nếu `pred` không sinh số nào:

```text
number_hallucination = 0.0
```

Ý nghĩa: đo tỷ lệ số trong output không tồn tại trong source.

Ví dụ:

```text
src:  "Bệnh nhân sốt 38 độ."
pred: "Bệnh nhân sốt 39 độ."

numbers(src) = {38}
numbers(pred) = {39}
hallucination = 1 / 1 = 1.0
```

## 5. Length Score

Trước hết tính tỷ lệ độ dài:

```text
length_ratio = words(pred) / words(src)
```

Nếu ratio nằm trong khoảng:

```text
MIN_RATIO = 0.18
MAX_RATIO = 0.27
```

thì:

```text
length_score = 1.0
```

Nếu lệch khỏi khoảng này:

```text
length_score = max(0, 1 - |ratio - TARGET_RATIO| / TARGET_RATIO)
```

với:

```text
TARGET_RATIO = 0.225
```

Ý nghĩa: khuyến khích bản tóm tắt có độ dài gần 18%-27% so với source.

## 6. Diversity Score

```text
diversity_score = số từ khác nhau trong pred / tổng số từ trong pred
```

Ví dụ:

```text
pred = "bệnh nhân bệnh nhân cần nghỉ ngơi"
unique_words = {bệnh, nhân, cần, nghỉ, ngơi}
total_words = 6
diversity = 5 / 6 = 0.8333
```

Ý nghĩa: giảm rủi ro mô hình lặp từ hoặc lặp cụm từ để ăn reward.

## 7. KL Penalty

`KL_COEF` trong notebook:

```python
"KL_COEF": 0.3
```

Không phải là KL. Nó là hệ số phạt KL:

```text
KL penalty = KL_COEF * KL(actor || reference)
```

Trong `PPOConfig`:

```python
init_kl_coef=CONFIG["KL_COEF"]
kl_penalty=CONFIG["KL_PENALTY_MODE"]  # "abs"
```

Với `kl_penalty = "abs"`, TRL dùng dạng độ lệch tuyệt đối log-prob:

```text
KL ≈ |log_p_actor - log_p_ref|
```

Ý nghĩa: nếu Actor sinh theo phân phối quá khác Reference model, reward hiệu dụng bị phạt. Reference model là bản sao SFT đóng băng, nên KL giúp Actor không lệch quá xa khỏi khả năng tóm tắt gốc.

Ví dụ:

```text
reward = 0.8
KL = 0.5
KL_COEF = 0.3

reward_effective = 0.8 - 0.3 * 0.5
                 = 0.65
```

## 8. Advantage

Slide viết:

```text
A_t = reward - KL_COEF * KL - V(s)
```

Ý nghĩa:

```text
Advantage = reward hiệu dụng - reward kỳ vọng
```

Trong đó:

- `reward - KL_COEF * KL`: reward sau khi trừ phạt KL.
- `V(s)`: giá trị kỳ vọng do value head dự đoán.

Nếu:

```text
A_t > 0
```

output tốt hơn kỳ vọng, PPO tăng xác suất sinh output/token tương tự.

Nếu:

```text
A_t < 0
```

output kém hơn kỳ vọng, PPO giảm xác suất sinh output/token tương tự.

Ví dụ:

```text
reward = 0.8
KL penalty = 0.1
V(s) = 0.5

A = 0.8 - 0.1 - 0.5 = 0.2
```

Output tốt hơn kỳ vọng `0.2`.

## 9. Importance Ratio

PPO không so sánh token hiện tại với token trước đó. Nó so sánh xác suất của cùng một token, trong cùng context, giữa policy mới và policy cũ:

```text
r_t = π_actor_new(a_t | s_t) / π_actor_old(a_t | s_t)
```

Thường tính bằng log-prob để ổn định:

```text
r_t = exp(logπ_new(a_t|s_t) - logπ_old(a_t|s_t))
```

Ví dụ:

```text
π_old("5mg" | context) = 0.10
π_new("5mg" | context) = 0.13

r_t = 0.13 / 0.10 = 1.3
```

Nghĩa là policy mới đang tăng xác suất sinh token `"5mg"` lên 30% so với policy cũ.

Trong notebook, ta không tự lưu log-prob. `PPOTrainer.step(...)` xử lý nội bộ.

## 10. PPO Clip

Clip trong PPO không clip weight và không clip reward. Nó clip importance ratio:

```text
clip(r_t, 1 - ε, 1 + ε)
```

Trong notebook:

```text
ε = CLIP_RANGE = 0.2
```

nên:

```text
clip(r_t, 0.8, 1.2)
```

Ý nghĩa: PPO giới hạn mức thay đổi xác suất sinh token trong một bước update.

Ví dụ với advantage dương:

```text
A_t = +2
r_t = 1.8

Không clip: 1.8 * 2 = 3.6
Có clip:    1.2 * 2 = 2.4
```

PPO chỉ công nhận mức tăng vừa phải.

Ví dụ với advantage âm:

```text
A_t = -2
r_t = 0.3

Không clip: 0.3 * (-2) = -0.6
Có clip:    0.8 * (-2) = -1.6
```

Phần `min(...)` trong objective chọn nhánh bảo thủ hơn, tránh policy thay đổi quá mạnh.

## 11. PPO Objective / Loss

Slide viết:

```text
L_PPO =
E[ min( r_t * A_t,
        clip(r_t, 1 - ε, 1 + ε) * A_t ) ]
- β * KL(π_actor || π_ref)
```

Trong đó:

- `r_t`: importance ratio.
- `A_t`: advantage.
- `ε`: clip range, trong notebook là `0.2`.
- `β`: KL coefficient, trong notebook là `0.3`.
- `KL(π_actor || π_ref)`: độ lệch giữa Actor và Reference.

Cách hiểu:

1. Nếu output tốt hơn kỳ vọng, PPO muốn tăng xác suất sinh output đó.
2. Nếu output kém hơn kỳ vọng, PPO muốn giảm xác suất sinh output đó.
3. Clip giới hạn mức thay đổi policy mỗi bước.
4. KL penalty giữ Actor không lệch quá xa Reference model.

Trong notebook, công thức này không được tự code bằng tay. Nó được tính bên trong TRL tại:

```python
stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

Sau đó lấy total PPO loss bằng:

```python
ppo_loss = tensor_to_float(stats["ppo/loss/total"])
```

## 12. Sampling Khi Train PPO

Trong notebook:

```python
do_sample=True
top_k=0
top_p=1.0
temperature=1.0
```

Ý nghĩa:

- `do_sample=True`: sinh bằng sampling.
- `top_k=0`: không cắt theo top-k.
- `top_p=1.0`: không cắt theo nucleus sampling.
- `temperature=1.0`: giữ nguyên phân phối.

Lý do: PPO cần log-prob và KL nhất quán trên phân phối đầy đủ của Actor và Reference. Nếu dùng `top_k` hoặc `top_p`, phân phối bị truncate và normalize lại, có thể làm log-prob/KL estimate lệch.

## 13. Tóm Tắt Một Vòng PPO Trong Notebook

```text
1. Tokenize source -> query_tensors
2. Actor sinh response bằng ppo_trainer.generate(...)
3. Decode response -> pred
4. Tính reward bằng compute_reward_breakdown(src, pred, gold)
5. Chuyển reward thành tensor
6. Gọi ppo_trainer.step(query_tensors, response_tensors, rewards)
7. TRL nội bộ tính log_prob, KL, value, advantage, PPO loss
8. Cập nhật Actor model
9. Log mean_reward, KL, ppo_loss, number_recall, hallucination
10. Mỗi 25 step đánh giá validation và lưu best model nếu composite tốt hơn
```
