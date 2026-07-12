# Backend LLMs Inference Optimization

Kho lưu trữ này là bộ công cụ tối ưu inference cho Viettel AI Race 2026, track **LLM Inference Optimization (Phase 1 / online round)**. Mục tiêu là tinh chỉnh một serving stack cho model cố định `Qwen/Qwen3.5-2B` để phục vụ trace yêu cầu với điểm số tốt nhất có thể, trong giới hạn phần cứng và ràng buộc của ban tổ chức.

## Mô tả dự án

Project tập trung vào bài toán hệ thống, không phải bài toán huấn luyện model. Trọng tâm hiện tại là:

- chạy được baseline vLLM/OpenAI-compatible server đúng contract của ban tổ chức;
- replay trace `trace-round1.jsonl` để đo TTFT, TPOT và score;
- xây dựng harness GPQA Diamond làm proxy chất lượng cho mọi thay đổi serving;
- thử các candidate tối ưu hóa ở mức an toàn trước khi đẩy vào vòng benchmark lớn hơn.

Stack chính hiện có:

- Python package theo layout `src/`;
- vLLM OpenAI API server;
- trace replay benchmark bằng `httpx`;
- GPQA evaluation pipeline từ `datasets` và `huggingface-hub`;
- Docker Compose cho baseline và các experiment.

## Trạng thái hiện tại

Hiện repo đang ở giai đoạn **baseline đã xác nhận + harness đã khóa + Experiment 1 đang chạy**.

Các mốc đã có trong repo:

- Organizer baseline đã được submit và accepted, score khoảng `15.19`.
- Local development đang dùng NVIDIA L4, nên các số đo nội bộ chỉ mang tính tương đối, không nên coi là dự đoán trực tiếp cho H200 MIG `1g.18gb`.
- Harness H0.1 đã được chuẩn hóa: 120 HTTP connections, exact streaming usage, dispatch telemetry, trace SHA-256, và readiness ổn định trước khi replay.
- Trace benchmark chuẩn đang dùng `data/trace-round1-diverse-content.jsonl`.
- GPQA Diamond reference hiện là `43 / 120`.
- Experiment 1 là phase đang triển khai; chưa có candidate nào được promote thành kết quả cuối.

Nói ngắn gọn: phần nền tảng để đo lường đã có, còn phần tối ưu hiệu năng vẫn đang được thử nghiệm và đối chiếu bằng trace + GPQA.

## Thành phần chính

- `src/inference_opt/trace`: đọc trace JSONL và dựng trace bundle cho GPQA.
- `src/inference_opt/benchmark`: replay trace, thu TTFT/TPOT, tính ERS và ghi report.
- `src/inference_opt/eval`: tải, chuẩn hóa và chấm GPQA Diamond.
- `src/inference_opt/clients`: health check và readiness cho OpenAI-compatible server.
- `src/inference_opt/serving`: orchestration experiment, sweep candidate và kiểm tra compose override.
- `scripts/`: các CLI mỏng để chạy benchmark, download model, health check và evaluate kết quả.

## Cấu trúc thư mục

- `configs/`: cấu hình serving và benchmark.
- `docker/`: assets cho build image khi cần.
- `src/`: logic Python dùng lại được.
- `scripts/`: entrypoint CLI mỏng.
- `evals/`: entrypoint và tài sản cho GPQA harness.
- `data/`: trace và metadata dataset nhỏ.
- `tests/`: test cho module và script.
- `docs/`: ghi chú kiến trúc, baseline, kết quả và kế hoạch.
- `results/`: output benchmark/eval local.

## Cách chạy nhanh

Thiết lập môi trường Python 3.11+ và cài dependency:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,eval]"
```

Chạy kiểm tra cơ bản:

```powershell
python -m unittest discover -s tests
```

Chạy trace benchmark local:

```powershell
python scripts/run_trace_benchmark.py --trace data/trace-round1-diverse-content.jsonl
```

## Tài liệu liên quan

- [Baseline setup](docs/baseline/setup.md)
- [Baseline results](docs/baseline/results.md)
- [Repository structure](docs/architecture/repo-structure.md)
- [Trace benchmark notes](docs/baseline/trace-benchmark.md)
- [GPQA benchmark notes](docs/baseline/gpqa-benchmark.md)

## Ghi chú trạng thái

README này phản ánh trạng thái hiện có của repo tại thời điểm viết: nền tảng benchmark/eval đã có, còn các thay đổi tối ưu serving vẫn đang ở giai đoạn thử nghiệm và so sánh.



