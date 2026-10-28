.PHONY: help setup ingest run stop restart test eval eval-summary summary eval-taxonomy eval-tier1 eval-tier2 eval-tier2-smoke eval-calc eval-rag benchmark clean

PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)

help: ## Hiển thị trợ giúp các lệnh
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Cài đặt dependencies từ requirements.txt
	pip install -r requirements.txt

ingest: ## Nạp corpus luật vào index RAG engine (idempotent; thêm CLEAN=1 để nạp lại từ đầu)
	$(PYTHON) -m rag_engine.backends.vector_chunking.ingest $(if $(CLEAN),--clean)

run: ## Khởi chạy web chatbot UI tại http://127.0.0.1:7860
	$(PYTHON) frontend/app.py

stop: ## Dừng tiến trình chatbot UI trên cổng 7860
	-lsof -ti tcp:7860 -sTCP:LISTEN | xargs kill -9 2>/dev/null || true

restart: stop run ## Khởi động lại web chatbot UI trên cổng 7860

test: ## Chạy kỹ thuật & frontend test suites (chạy mỗi khi sửa code)
	ENV_FILE=.env.test $(PYTHON) -m pytest $(ARGS)

eval: ## Chạy bộ đánh giá nghiên cứu/paper metrics (chỉ chạy khi yêu cầu)
	ENV_FILE=.env.test $(PYTHON) -m pytest evaluation/ $(ARGS)

eval-summary: ## Xem bảng dashboard tổng hợp Two-Tier evaluation (Accuracy, Latency, Metrics)
	ENV_FILE=.env.test $(PYTHON) -m evaluation.evaluate_summary

summary: eval-summary ## Phím tắt cho eval-summary

eval-taxonomy: ## Hiển thị Bảng 2 Breakdown Taxonomy cho cả 2 backend vector_chunking và page_index
	ENV_FILE=.env.test $(PYTHON) -m evaluation.eval_taxonomy_breakdown

eval-tier1: ## Chạy Tầng 1: Symbolic Engine & Control Confusion Matrix (~2s)
	ENV_FILE=.env.test $(PYTHON) -m pytest evaluation/eval_nav_audit_engine.py evaluation/eval_control_accuracy.py $(ARGS)

eval-tier2: ## Chạy Tầng 2: Agent End-to-End Reasoning Benchmark (40 câu)
	ENV_FILE=.env.test PYTHONWARNINGS=ignore $(PYTHON) -W ignore -m evaluation.evaluate_agent_calculation --all

eval-tier2-smoke: ## Chạy Tầng 2 nhanh (smoke test 2 câu: 1 Calc + 1 RAG)
	ENV_FILE=.env.test PYTHONWARNINGS=ignore $(PYTHON) -W ignore -m evaluation.evaluate_agent_calculation --smoke

eval-calc: ## Chạy Tầng 2: 30 câu kiểm tra tính toán (calculate_decimal)
	ENV_FILE=.env.test PYTHONWARNINGS=ignore $(PYTHON) -W ignore -m evaluation.evaluate_agent_calculation --calc-only

eval-rag: ## Chạy Tầng 2: 10 câu kiểm tra trích xuất luật & RAG Grounding
	ENV_FILE=.env.test PYTHONWARNINGS=ignore $(PYTHON) -W ignore -m evaluation.evaluate_agent_calculation --rag-only

benchmark: ## Chạy benchmark đánh giá đầy đủ cho paper (100 câu hỏi eval_dataset.json)
	ENV_FILE=.env.test $(PYTHON) -m evaluation.run_benchmark --backend vector_chunking

clean: ## Dọn cache Python và pytest
	rm -rf __pycache__ .pytest_cache rag_engine/__pycache__ rag_engine/*/__pycache__ rag_engine/*/*/__pycache__ frontend/__pycache__ frontend/*/__pycache__ frontend/*/*/__pycache__ evaluation/__pycache__ tests/__pycache__

