# =============================================================================
# Chuyển giọng nói buổi họp (video .webm) thành văn bản có mốc thời gian — faster-whisper large-v3
# Notebook Kaggle: Accelerator = GPU T4, Internet = On; Add Input → dataset chứa file .webm.
# Chạy được trên Colab (GPU) nếu đặt file vào /content.
#
# MỤC LỤC
#   Phần 1 — Cài đặt, tìm file, kiểm tra GPU
#   Phần 2 — Chuyển giọng nói thành văn bản, lưu .txt
# =============================================================================


# %% Phần 1 — Cài đặt, tìm file, kiểm tra GPU
# Cài faster-whisper, tìm file âm thanh/video trong thư mục đầu vào, in thời lượng; thử nạp mô hình nhỏ trên GPU
# để phát hiện sớm lỗi thư viện CUDA trước khi chạy mô hình lớn.
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper"], check=True)

import av
import torch
from faster_whisper import WhisperModel

IN_DIRS = [Path("/kaggle/input"), Path("/content")]
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/content")
MEDIA_EXT = {".webm", ".mp4", ".mkv", ".m4a", ".mp3", ".wav", ".ogg"}

media = sorted(p for d in IN_DIRS if d.exists() for p in d.rglob("*") if p.suffix.lower() in MEDIA_EXT)
for p in media:
    with av.open(str(p)) as f:
        dur = f.duration / 1e6 if f.duration else float("nan")
        n_audio = len(f.streams.audio)
    print(f"{p}  |  {p.stat().st_size / 1e6:,.0f} MB  |  {dur / 60:.1f} phút  |  luồng âm thanh: {n_audio}")
if not media:
    print("Không thấy file âm thanh/video nào — kiểm tra Add Input.")

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ")
WhisperModel("tiny", device="cuda", compute_type="float16")
print("Nạp mô hình thử trên GPU: OK")


# %% Phần 2 — Chuyển giọng nói thành văn bản, lưu .txt
# large-v3 tiếng Việt; VAD bỏ đoạn im lặng (giảm câu bịa), không nối ngữ cảnh câu trước (tránh lặp vòng);
# gợi ý thuật ngữ chuyên ngành để nhận đúng từ tiếng Anh. Mỗi dòng: [giờ:phút:giây] câu nói. In tiến độ.
# Tự giải mã âm thanh thành mảng 16 kHz mono vì decode_audio của faster-whisper lỗi với bản PyAV trên Kaggle.
import time

import numpy as np

SAMPLE_RATE = 16000

MODEL_NAME = "large-v3"
LANGUAGE = "vi"
TERMS = ("Mamba, LSTM, GRU, Transformer, dataset, BiasCast, LamaH-CE, NSE, attention, embedding, "
         "bottleneck, fusion, validation, test, training, baseline, sample, lưu lượng, dòng chảy, thủy văn, lưu vực.")


def hms(t: float) -> str:
    """Giây → hh:mm:ss."""
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def load_audio(path: Path) -> np.ndarray:
    """Đọc luồng âm thanh đầu tiên, đổi sang 16 kHz mono float32 trong [-1, 1]."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(str(path)) as f:
        for frame in f.decode(audio=0):
            chunks += [r.to_ndarray().reshape(-1) for r in resampler.resample(frame)]
    chunks += [r.to_ndarray().reshape(-1) for r in resampler.resample(None)]   # phần còn đọng trong bộ đổi tần số
    return np.concatenate(chunks).astype(np.float32) / 32768.0


model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
for src in media:
    out = OUT_DIR / f"{src.stem}_transcript.txt"
    t0 = time.time()
    audio = load_audio(src)
    print(f"Đã giải mã {src.name}: {len(audio) / SAMPLE_RATE / 60:.1f} phút ({time.time() - t0:.0f} s)")
    segments, info = model.transcribe(audio, language=LANGUAGE, beam_size=5, vad_filter=True,
                                      vad_parameters={"min_silence_duration_ms": 500},
                                      condition_on_previous_text=False, initial_prompt=TERMS)
    last_report = 0.0
    with out.open("w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"[{hms(s.start)}] {s.text.strip()}\n")
            if s.end - last_report >= 300:                      # báo tiến độ mỗi 5 phút âm thanh
                last_report = s.end
                print(f"  {hms(s.end)} / {hms(info.duration)}  ({time.time() - t0:.0f} s)")
    print(f"Xong {src.name}: {out}  |  {time.time() - t0:.0f} s")
    print(out.read_text(encoding="utf-8")[:1500])
