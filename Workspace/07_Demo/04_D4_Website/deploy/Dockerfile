# syntax=docker/dockerfile:1
# Ảnh chạy website demo trên VPS ARM64 (cũng dựng được trên x86-64). Ngữ cảnh dựng là thư mục Workspace/07_Demo.
# Hai môi trường Python giống laptop: /opt/venv-c (ứng dụng + 5 lõi C2, NeuralHydrology freeze-C) và
# /opt/venv-h (tiến trình suy luận H, freeze-H). venv tạo bằng --copies vì DemoRuntime/h_cpu.py gọi
# Path(python).resolve(): venv thường trên Linux là liên kết mềm => resolve() ra Python gốc, mất thư viện của venv.

FROM python:3.12.10-slim-bookworm AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY 04_D4_Website/deploy/requirements-lock.txt /tmp/requirements-lock.txt
# torch cài trước từ chỉ mục CPU của PyTorch (bản +cpu như laptop), sau đó các gói khóa phiên bản không kéo torch khác
RUN for env in c h; do \
      python -m venv --copies /opt/venv-$env && \
      /opt/venv-$env/bin/pip install torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
      /opt/venv-$env/bin/pip install -r /tmp/requirements-lock.txt || exit 1; \
    done
RUN /opt/venv-c/bin/pip install --no-deps \
      "neuralhydrology @ git+https://github.com/Ticasslo/neuralhydrology.git@db6dc1ffcd8c14994cd80be3ed8ff417002c01a1" && \
    /opt/venv-h/bin/pip install --no-deps \
      "neuralhydrology @ git+https://github.com/Ticasslo/neuralhydrology.git@6f68f471bec255f67d446acd6c6556358ff9bfea" && \
    /opt/venv-c/bin/pip check && /opt/venv-h/bin/pip check

FROM python:3.12.10-slim-bookworm
COPY --from=build /opt /opt
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin demo
COPY 03_D3_LamaHCEExt_Demo/app.py /app/03_D3_LamaHCEExt_Demo/app.py
COPY DemoRuntime/forecast.py DemoRuntime/h_cpu.py /app/DemoRuntime/
COPY 04_D4_Website/server.py /app/04_D4_Website/server.py
COPY 04_D4_Website/static /app/04_D4_Website/static
# Luồng tính theo ghi chú WSGI của D3 (2 OCPU); bộ nhớ đệm của matplotlib, numba ghi vào /tmp (ổ gốc chỉ đọc)
ENV PATH=/opt/venv-c/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE DEMO_THREADS=2 \
    HOME=/tmp MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp \
    DEMO_BUNDLE=/data/bundle DEMO_RAW=/data/raw/LamaH_extended DEMO_PMTILES=/data/basemap.pmtiles \
    DEMO_TERRAIN=/data/terrain.pmtiles DEMO_H_PYTHON=/opt/venv-h/bin/python
USER demo
WORKDIR /app/04_D4_Website
EXPOSE 8000
# Một tiến trình để dùng chung bộ nhớ đệm mô hình, giới hạn lần chạy và một tiến trình con H
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "120", "--graceful-timeout", "30", \
     "--bind", "0.0.0.0:8000", "--access-logfile", "-", "server:from_env()"]
