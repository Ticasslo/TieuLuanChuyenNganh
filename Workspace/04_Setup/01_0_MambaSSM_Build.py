# =============================================================================
# Bước 0 - biên dịch mamba-ssm và causal-conv1d cho Kaggle T4 (torch của ảnh Kaggle), wheel lưu ở Output notebook
# Kế hoạch: Document/01_Plan/03_Pipeline.md Mục 4.2 (lõi Mamba), 8.2 (rủi ro cài đặt), 9.6 (demo chạy CPU).
# Notebook Kaggle "01_0_MambaSSM_Build": Accelerator = GPU T4 x2 (chỉ dùng 1), Internet = On,
# Persistence = Files only, Environment = Pin to original environment.
# Chạy ba giai đoạn: (1) tương tác Phần 1-9 để kiểm từng bước; (2) Save & Run All cả notebook trong phiên sạch,
# Output của phiên bản đó chứa wheels/ (2 wheel + build_info.json), logs/, src/; (3) phiên mới, gắn Output
# của chính notebook này làm Input, chỉ chạy Phần 9 để xác nhận cài được từ Input.
#
# MỤC LỤC
#   Phần 1 - Kiểm môi trường: Python, torch, GPU, nvcc, CPU, RAM, đĩa, thư viện cần cho mamba_ssm
#   Phần 2 - Thử tìm wheel dựng sẵn khớp môi trường
#   Phần 3 - Biên dịch causal-conv1d chỉ cho sm_75
#   Phần 4 - Biên dịch mamba-ssm chỉ cho sm_75
#   Phần 5 - Cài hai wheel, nạp thử, ghi build_info.json
#   Phần 6 - Kiểm nhân CUDA so với bản PyTorch thuần
#   Phần 7 - Kiểm khối Mamba ở kích thước của đề tài
#   Phần 8 - Thử mambapy với trọng số của khối Mamba
#   Phần 9 - Cell cài dùng lại cho các notebook sau
# =============================================================================


# %% Phần 1 - Kiểm môi trường: Python, torch, GPU, nvcc, CPU, RAM, đĩa, thư viện cần cho mamba_ssm
# Chỉ đọc, không cài gì. In 3 bảng (môi trường, thư viện, điều kiện) rồi kết luận có biên dịch được không.
# Dừng nếu không có nvcc hoặc nvcc khác phiên bản CUDA chính của torch (setup.py cần nvcc khớp torch).
import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import torch
from IPython.display import HTML, display

IS_KAGGLE = Path("/kaggle/working").exists()
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("mamba_build")
WHEEL_DIR = WORK_DIR / "wheels"          # nơi Phần 3-4 ghi wheel, Phần 5 ghi build_info.json
WHEEL_SOURCE = "01-0-mambassm-build"     # slug notebook này; Output của nó gắn làm Input ở giai đoạn 3
TARGET_CC = (7, 5)                       # T4 = sm_75, cũng là kiến trúc duy nhất sẽ biên dịch
NEED_LIBS = ["torch", "triton", "einops", "ninja", "packaging", "setuptools", "wheel",
             "transformers", "huggingface_hub"]
WORK_DIR.mkdir(parents=True, exist_ok=True)


def lib_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def find_nvcc():
    # Tìm nvcc theo CUDA_HOME rồi PATH; trả (đường dẫn, phiên bản "12.8") hoặc (None, None).
    cands = [Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "nvcc", shutil.which("nvcc")]
    for c in cands:
        if c and Path(c).exists():
            out = subprocess.run([str(c), "--version"], capture_output=True, text=True).stdout
            m = re.search(r"release (\d+\.\d+)", out)
            return str(c), (m.group(1) if m else None)
    return None, None


def mem_gb(key):
    # Đọc /proc/meminfo (kB) => GB.
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 1024 ** 2
    return None


def show(title, df):
    display(HTML(f"<h4 style='margin:12px 0 4px'>{title}</h4>"))
    display(df.style.hide(axis="index"))


nvcc_path, nvcc_ver = find_nvcc()
gpu_ok = torch.cuda.is_available()
cc = torch.cuda.get_device_capability(0) if gpu_ok else None
build_date = Path("/etc/build_date").read_text().strip() if Path("/etc/build_date").exists() else "không có"
disk_free = shutil.disk_usage(WORK_DIR).free / 1024 ** 3

ENV = {
    "Python": platform.python_version(),
    "torch": torch.__version__,
    "CUDA của torch": torch.version.cuda,
    "cxx11 ABI của torch": torch._C._GLIBCXX_USE_CXX11_ABI,
    "GPU": torch.cuda.get_device_name(0) if gpu_ok else "không có GPU",
    "Compute capability": f"{cc[0]}.{cc[1]}" if cc else "-",
    "nvcc": f"{nvcc_ver} ({nvcc_path})" if nvcc_path else "không có",
    "CUDA_HOME": os.environ.get("CUDA_HOME", "chưa đặt"),
    "Số CPU": os.cpu_count(),
    "RAM tổng / còn trống (GB)": f"{mem_gb('MemTotal'):.1f} / {mem_gb('MemAvailable'):.1f}",
    "Đĩa trống ở WORK_DIR (GB)": f"{disk_free:.1f}",
    "Ngày dựng ảnh Kaggle": build_date,
}
show("Môi trường", pd.DataFrame({"Mục": list(ENV), "Giá trị": [str(v) for v in ENV.values()]}))

libs = {k: lib_version(k) for k in NEED_LIBS}
show("Thư viện (cần cho biên dịch hoặc cho import mamba_ssm)",
     pd.DataFrame({"Thư viện": list(libs), "Phiên bản": [v or "THIẾU" for v in libs.values()]}))

torch_cuda_major = torch.version.cuda.split(".")[0] if torch.version.cuda else None
try:
    import wheel.bdist_wheel  # setup.py của cả hai repo import đúng đường dẫn này
    bdist_ok = True
except ImportError:
    bdist_ok = False
checks = {
    "Có GPU, compute capability >= 7.5": bool(cc and cc >= TARGET_CC),
    "Có nvcc": nvcc_path is not None,
    "nvcc cùng phiên bản CUDA chính với torch": bool(nvcc_ver and torch_cuda_major
                                                    and nvcc_ver.split(".")[0] == torch_cuda_major),
    "Đủ thư viện": all(libs.values()),
    "Nạp được wheel.bdist_wheel (setup.py cần)": bdist_ok,
}
show("Điều kiện để biên dịch", pd.DataFrame({"Điều kiện": list(checks),
                                             "Đạt": ["ĐẠT" if v else "KHÔNG" for v in checks.values()]}))

if all(checks.values()):
    print("=> Đủ điều kiện biên dịch. Gửi output này, rồi chạy Phần 2.")
else:
    print("=> CHƯA đủ điều kiện: gửi output này để chọn phương án dự phòng (Plans/01 Mục 6), chưa chạy tiếp.")


# %% Phần 2 - Thử tìm wheel dựng sẵn khớp môi trường
# Dựng tên wheel đúng công thức get_wheel_url() trong setup.py của hai repo tại tag dưới đây
# (CUDA 12.x quy về "cu12", torch lấy major.minor, cxx11 ABI của torch, cp<python>, linux_x86_64),
# hỏi GitHub xem có tệp đó không, và liệt kê các bản torch mà release có wheel cp313 cu12.
# Có đủ cả hai wheel => tải về WHEEL_DIR, Phần 3-4 bỏ qua biên dịch; thiếu => biên dịch ở Phần 3-4.
import json
import sys
import urllib.request

PKGS = {  # tên gói trong tên wheel, repo, tag (cùng đợt phát hành 9/5/2026)
    "causal_conv1d": ("Dao-AILab/causal-conv1d", "v1.6.2.post1"),
    "mamba_ssm": ("state-spaces/mamba", "v2.3.2.post1"),
}


def expected_wheel(pkg, tag):
    # Theo get_wheel_url(): ảnh NGC của NVIDIA dùng biến môi trường riêng, Kaggle không thuộc trường hợp đó.
    if os.environ.get("NVIDIA_PRODUCT_NAME", "") == "PyTorch":
        raise RuntimeError("Ảnh NGC: tên wheel tính theo NVIDIA_PYTORCH_VERSION, cần sửa hàm này")
    cuda_major = torch.version.cuda.split(".")[0]
    torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return f"{pkg}-{tag.lstrip('v')}+cu{cuda_major}torch{torch_mm}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl"


def release_assets(repo, tag):
    # Danh sách tên tệp đính kèm của release (GitHub API, không cần token, giới hạn 60 lần/giờ).
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return [a["name"] for a in json.load(r)["assets"]]


rows, found = [], {}
py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
for pkg, (repo, tag) in PKGS.items():
    want = expected_wheel(pkg, tag)
    assets = release_assets(repo, tag)
    same_env = [a for a in assets if f"-{py_tag}-" in a and "+cu12" in a and a.endswith("linux_x86_64.whl")]
    torch_avail = sorted({a.split("torch")[1].split("cxx")[0] for a in same_env}, key=lambda v: tuple(map(int, v.split("."))))
    found[pkg] = want in assets
    rows.append({"Gói": pkg, "Tag": tag, "Wheel cần": want, "Có trên release": "CÓ" if found[pkg] else "KHÔNG",
                 f"torch có wheel ({py_tag}, cu12)": ", ".join(torch_avail)})
show("Wheel dựng sẵn", pd.DataFrame(rows))

if all(found.values()):
    WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    for pkg, (repo, tag) in PKGS.items():
        name = expected_wheel(pkg, tag)
        urllib.request.urlretrieve(f"https://github.com/{repo}/releases/download/{tag}/{name}", WHEEL_DIR / name)
        print(f"Đã tải {name} ({(WHEEL_DIR / name).stat().st_size / 1024 ** 2:.1f} MB)")
    print("=> Có wheel dựng sẵn: Phần 3-4 sẽ bỏ qua biên dịch. Gửi output này.")
else:
    print(f"=> Không có wheel cho torch {torch.__version__}: phải biên dịch ở Phần 3-4. Gửi output này.")


# %% Phần 3 - Biên dịch causal-conv1d chỉ cho sm_75
# Clone repo đúng tag, sửa setup.py để chỉ sinh mã cho sm_75 (bản gốc gán cứng 75/80/87/90/100/120),
# rồi pip wheel --no-build-isolation (dùng torch có sẵn của ảnh). Hàm build_wheel dùng lại ở Phần 4.
# Đã có wheel trong WHEEL_DIR hoặc /kaggle/input (giai đoạn 3) => chép sang WHEEL_DIR, không biên dịch.
import time
import zipfile

SRC_DIR = WORK_DIR / "src"                # mã nguồn clone về
LOG_DIR = WORK_DIR / "logs"               # log biên dịch đầy đủ
INPUT_DIR = Path("/kaggle/input")
MAX_JOBS = os.cpu_count()                 # số tiến trình ninja; RAM 31 GB đủ cho 4 tiến trình nvcc
# setup.py của cả hai repo: dòng chú thích ngay sau khối gán cc_flag; chèn phép gán lại ngay trước nó
PATCH_ANCHOR = "    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI"
PATCH_LINE = '    cc_flag = ["-gencode", "arch=compute_75,code=sm_75"]  # patched: T4 only\n'
FORCE_ENV = {"causal_conv1d": "CAUSAL_CONV1D_FORCE_BUILD", "mamba_ssm": "MAMBA_FORCE_BUILD"}
for d in (WHEEL_DIR, SRC_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def find_wheel(pkg, tag):
    # Wheel tự dựng: <pkg>-<version>-cp313-cp313-linux_x86_64.whl; wheel tải ở Phần 2 có thêm "+cu12torch..."
    # sau version, nên mẫu chừa chỗ cho phần đuôi đó. Tìm ở WHEEL_DIR rồi /kaggle/input.
    pat = f"{pkg}-{tag.lstrip('v')}*-{py_tag}-{py_tag}-linux_x86_64.whl"
    for root in (WHEEL_DIR, INPUT_DIR):
        hits = sorted(root.rglob(pat)) if root.exists() else []
        if hits:
            return hits[0]
    return None


def patch_setup(setup_py):
    # Chèn PATCH_LINE trước PATCH_ANCHOR đúng một lần; báo lỗi nếu không thấy mốc (mã gốc đã đổi).
    text = setup_py.read_text()
    if PATCH_LINE in text:
        return
    if text.count(PATCH_ANCHOR) != 1:
        raise RuntimeError(f"Không thấy đúng 1 mốc sửa trong {setup_py}, đọc lại setup.py trước khi sửa")
    setup_py.write_text(text.replace(PATCH_ANCHOR, PATCH_LINE + PATCH_ANCHOR))


def wheel_archs(whl):
    # Liệt kê kiến trúc GPU có trong các tệp .so của wheel bằng cuobjdump (đi kèm CUDA toolkit).
    archs = set()
    cuobjdump = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "cuobjdump"
    tmp = WORK_DIR / "tmp_so"
    with zipfile.ZipFile(whl) as z:
        for name in [n for n in z.namelist() if n.endswith(".so")]:
            so = Path(z.extract(name, tmp))
            out = subprocess.run([str(cuobjdump), "--list-elf", str(so)], capture_output=True, text=True).stdout
            archs |= set(re.findall(r"sm_\d+", out))
    shutil.rmtree(tmp, ignore_errors=True)
    return sorted(archs)


def clone_src(pkg):
    # Clone repo đúng tag vào SRC_DIR/<tên repo> nếu chưa có setup.py ở đó (thư mục dở dang thì xóa, clone lại).
    repo, tag = PKGS[pkg]
    src = SRC_DIR / repo.split("/")[1]
    if not (src / "setup.py").exists():
        shutil.rmtree(src, ignore_errors=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", tag, f"https://github.com/{repo}.git", str(src)],
                       check=True, capture_output=True)
    return src


def build_wheel(pkg):
    # Trả (đường dẫn wheel, cách có được wheel, số giây biên dịch).
    repo, tag = PKGS[pkg]
    whl = find_wheel(pkg, tag)
    if whl is not None:
        if whl.parent != WHEEL_DIR:
            whl = Path(shutil.copy2(whl, WHEEL_DIR))
        return whl, "có sẵn, không biên dịch", 0.0
    src = clone_src(pkg)
    patch_setup(src / "setup.py")
    flags = [ln.strip() for ln in (src / "setup.py").read_text().splitlines() if "compute_" in ln]
    print(f"Các dòng arch trong setup.py sau khi sửa ({pkg}): dòng cuối là dòng chèn, ghi đè mọi dòng trước")
    for ln in flags:
        print("   ", ln)
    env = {**os.environ, FORCE_ENV[pkg]: "TRUE", "MAX_JOBS": str(MAX_JOBS)}
    log = LOG_DIR / f"build_{pkg}.log"
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "--no-build-isolation", "--no-deps",
                             "-v", "-w", str(WHEEL_DIR)], cwd=src, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    secs = time.time() - t0
    if rc != 0:
        print("\n".join(log.read_text().splitlines()[-40:]))
        raise RuntimeError(f"Biên dịch {pkg} lỗi (mã {rc}), log đầy đủ: {log}")
    whl = find_wheel(pkg, tag)
    if whl is None:
        raise RuntimeError(f"pip báo xong nhưng không thấy wheel {pkg} trong {WHEEL_DIR}, xem {log}")
    # Xóa tệp trung gian của lần dựng (build/, *.egg-info) để Output gọn; giữ mã nguồn và tests/ cho Phần 6
    for junk in [src / "build", *src.glob("*.egg-info")]:
        shutil.rmtree(junk, ignore_errors=True)
    return whl, "biên dịch", secs


def report_wheel(pkg):
    # Dựng (hoặc lấy sẵn) wheel của pkg rồi in bảng kiểm: kiến trúc trong .so, log có kiến trúc khác không.
    whl, how, secs = build_wheel(pkg)
    log = LOG_DIR / f"build_{pkg}.log"
    info = {
        "Wheel": whl.name,
        "Cách có": how,
        "Thời gian biên dịch": f"{secs / 60:.1f} phút",
        "Dung lượng": f"{whl.stat().st_size / 1024 ** 2:.1f} MB",
        "Kiến trúc GPU trong .so": ", ".join(wheel_archs(whl)),
        "Log có -gencode khác sm_75": ("có" if log.exists() and re.search(r"code=sm_(?!75)\d+", log.read_text())
                                       else "không"),
    }
    show(pkg, pd.DataFrame({"Mục": list(info), "Giá trị": [str(v) for v in info.values()]}))
    print("=> Đạt nếu kiến trúc chỉ có sm_75 và log không có -gencode khác. Gửi output này.")
    return whl


cc_whl = report_wheel("causal_conv1d")


# %% Phần 4 - Biên dịch mamba-ssm chỉ cho sm_75
# Cùng cách Phần 3 (build_wheel: clone tag v2.3.2.post1, chèn dòng chỉ sm_75, pip wheel --no-build-isolation).
# setup.py chỉ có một extension selective_scan_cuda gồm 1 tệp .cpp và 9 tệp .cu (fwd/bwd x fp32/fp16/bf16).
mamba_whl = report_wheel("mamba_ssm")


# %% Phần 5 - Cài hai wheel, nạp thử, ghi build_info.json
# Cài bằng --no-deps để pip không đụng torch và không kéo tilelang/quack-kernels (chỉ Mamba-3 dùng, đã bọc try/except).
# Kiểm torch trước/sau giống nhau, nạp hai module CUDA và khối Mamba, rồi ghi build_info.json cạnh wheel
# để notebook sau so torch đang chạy với torch lúc dựng (lệch => phải dựng lại).
import datetime
import hashlib
import importlib

BUILD_INFO = WHEEL_DIR / "build_info.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


torch_before = lib_version("torch")
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", "-q",
                str(cc_whl), str(mamba_whl)], check=True)
importlib.invalidate_caches()
torch_after = lib_version("torch")
if torch_after != torch_before:
    raise RuntimeError(f"torch bị đổi {torch_before} => {torch_after}: không dùng được wheel, kiểm lại lệnh cài")

load = {}
for mod in ["causal_conv1d_cuda", "selective_scan_cuda", "causal_conv1d", "mamba_ssm"]:
    try:
        importlib.import_module(mod)
        load[mod] = "ĐẠT"
    except Exception as e:  # in lỗi thật để biết thiếu gì
        load[mod] = f"LỖI: {type(e).__name__}: {e}"
from mamba_ssm import Mamba  # noqa: E402  khối Mamba (v1) đề tài dùng

load["from mamba_ssm import Mamba"] = "ĐẠT"
show("Nạp module sau khi cài", pd.DataFrame({"Module": list(load), "Kết quả": list(load.values())}))

info = {
    "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "kaggle_image_build_date": build_date,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    "nvcc": nvcc_ver,
    "arch": "sm_75",
    "packages": {pkg: {"repo": repo, "tag": tag} for pkg, (repo, tag) in PKGS.items()},
    "wheels": {w.name: sha256(w) for w in (cc_whl, mamba_whl)},
}
BUILD_INFO.write_text(json.dumps(info, indent=2))
show("build_info.json", pd.DataFrame({"Khóa": list(info), "Giá trị": [str(v) for v in info.values()]}))
print(f"torch trước/sau khi cài: {torch_before} / {torch_after}")
print("=> Đạt nếu mọi dòng nạp module là ĐẠT. Gửi output này.")


# %% Phần 6 - Kiểm nhân CUDA so với bản PyTorch thuần
# Chạy bộ test chính thức của tác giả (trong mã nguồn đã clone ở Phần 3-4) trên wheel vừa cài:
#   (1) mamba tests/ops/test_selective_scan.py: selective_scan_fn và mamba_inner_fn so với bản *_ref thuần PyTorch,
#       xuôi và ngược, ngưỡng của tác giả (fp32: rtol 6e-4, atol 2e-3);
#   (2) causal-conv1d test_causal_conv1d chỉ với fp32 (itype0 = float32, kiểu đề tài huấn luyện), xuôi và ngược;
#   (3) ba test tất định của causal-conv1d: lan truyền ngược chỉ lặp lại y hệt khi bật use_deterministic_algorithms.
# --import-mode=importlib để pytest không chèn thư mục repo vào sys.path (nếu chèn sẽ nạp mamba_ssm chưa biên dịch).
import xml.etree.ElementTree as ET

if lib_version("pytest") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)

SRC = {pkg: clone_src(pkg) for pkg in PKGS}   # bảo đảm có mã nguồn đúng tag (clone lại nếu thiếu)
CC_TEST = SRC["causal_conv1d"] / "tests" / "test_causal_conv1d.py"
SS_TEST = SRC["mamba_ssm"] / "tests" / "ops" / "test_selective_scan.py"
for p in (CC_TEST, SS_TEST):
    print(f"{'có ' if p.exists() else 'THIẾU'}  {p}")
if not (CC_TEST.exists() and SS_TEST.exists()):
    for d in sorted(SRC_DIR.iterdir()):
        print(d, sorted(x.name for x in d.iterdir())[:20])
    raise RuntimeError("Thiếu tệp test trong mã nguồn đã clone, gửi output này")
TEST_GROUPS = {  # tên nhóm: (đối số pytest, số test thu được theo cách đếm tham số trong tệp test)
    "mamba selective_scan + mamba_inner": ([str(SS_TEST)], 20),
    "causal_conv1d fp32 xuôi/ngược": ([f"{CC_TEST}::test_causal_conv1d", "-k", "itype0"], 3456),
    "causal_conv1d tất định": ([f"{CC_TEST}::test_causal_conv1d_bwd_deterministic_reproducible",
                                f"{CC_TEST}::test_causal_conv1d_bwd_deterministic_close_to_default",
                                f"{CC_TEST}::test_causal_conv1d_bwd_default_mode_is_not_reproducible"], 6),
}


def run_pytest(name, args):
    # Chạy pytest trong tiến trình con, đọc kết quả từ tệp junit xml; trả dòng cho bảng.
    xml = LOG_DIR / f"pytest_{re.sub(r'\W+', '_', name)}.xml"
    log = xml.with_suffix(".log")
    with open(log, "w") as f:
        rc = subprocess.run([sys.executable, "-m", "pytest", *args, "-q", "-rfEsx", "-p", "no:cacheprovider",
                             "--import-mode=importlib", f"--junitxml={xml}"],
                            cwd=WORK_DIR, stdout=f, stderr=subprocess.STDOUT).returncode
    s = ET.parse(xml).getroot()
    s = s if s.tag == "testsuite" else s.find("testsuite")
    n = {k: int(s.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
    if rc not in (0, 1) or n["failures"] or n["errors"]:
        print(f"--- {name}: 40 dòng cuối log ({log}) ---")
        print("\n".join(log.read_text().splitlines()[-40:]))
    return {"Nhóm": name, "Số test": n["tests"], "Đạt": n["tests"] - n["failures"] - n["errors"] - n["skipped"],
            "Lỗi": n["failures"] + n["errors"], "Bỏ qua/xfail": n["skipped"],
            "Thời gian (giây)": round(float(s.get("time", 0))), "Mã thoát pytest": rc}


rows = []
for name, (args, n_expect) in TEST_GROUPS.items():
    r = run_pytest(name, args)
    r["Số test dự kiến"] = n_expect
    rows.append(r)
test_df = pd.DataFrame(rows)
show("Test chính thức của tác giả trên wheel tự dựng", test_df)
for name in TEST_GROUPS:
    log = LOG_DIR / f"pytest_{re.sub(r'\W+', '_', name)}.log"
    skips = [ln for ln in log.read_text().splitlines() if ln.startswith(("SKIPPED", "XFAIL"))]
    print(f"{name}: {len(skips)} dòng SKIPPED/XFAIL, ví dụ: {skips[:2]}")
ok = (test_df["Lỗi"] == 0).all() and (test_df["Số test"] == test_df["Số test dự kiến"]).all()
print("=> ĐẠT" if ok else "=> CHƯA ĐẠT", "(đạt khi không có lỗi và số test khớp dự kiến). Gửi output này.")


# %% Phần 7 - Kiểm khối Mamba ở kích thước của đề tài
# Kích thước như Sequential Forecast LSTM của BiasCast: lô 256, 365 bước (364 ngày quá khứ + ngày t), 128 chiều ẩn;
# Mamba v1 nhận (lô, bước, chiều) (mamba_simple.py, forward).
# Kiểm: (a) đường nhanh (mamba_inner_fn gộp nhân) so với đường chậm (causal_conv1d_fn + selective_scan_fn) cùng trọng số;
# (b) không rò rỉ tương lai: đổi đầu vào từ bước K trở đi thì đầu ra trước K không đổi;
# (c) tính tất định của lan truyền ngược, mặc định và khi bật use_deterministic_algorithms (liên quan nhiễu hạt giống A4);
# (d) thời gian xuôi + ngược, bộ nhớ, so với nn.LSTM 128 => 128 (chỉ tham khảo, đo chính thức ở bước đo thời gian).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # cuBLAS cần biến này khi bật chế độ tất định
DEV = "cuda"
B_SZ, SEQ, D_MODEL = 256, 365, 128
K_CUT = 300                     # bước bắt đầu đổi đầu vào ở phép kiểm rò rỉ
N_REPEAT = 3                    # số lần chạy lại ở phép kiểm tất định
REL_TOL = 1e-3                  # sai số lớn nhất / giá trị tuyệt đối lớn nhất, ngưỡng tự đặt cho phép so (a)


def make_mamba(fast):
    torch.manual_seed(0)
    return Mamba(d_model=D_MODEL, use_fast_path=fast).to(DEV)


def fwd_bwd(model, x, g):
    # Một lượt xuôi + ngược; trả đầu ra và đạo hàm theo đầu vào, theo từng tham số.
    model.zero_grad(set_to_none=True)
    xx = x.detach().clone().requires_grad_()
    y = model(xx)
    y.backward(g)
    grads = {"đầu vào": xx.grad.detach().clone()}
    grads.update({n: p.grad.detach().clone() for n, p in model.named_parameters()})
    return y.detach(), grads


def rel_err(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


torch.manual_seed(1)
x = torch.randn(B_SZ, SEQ, D_MODEL, device=DEV)
g = torch.randn(B_SZ, SEQ, D_MODEL, device=DEV)
m_fast, m_slow = make_mamba(True), make_mamba(False)
m_slow.load_state_dict(m_fast.state_dict())

# (a) đường nhanh so với đường chậm
y_f, gr_f = fwd_bwd(m_fast, x, g)
y_s, gr_s = fwd_bwd(m_slow, x, g)
rows = [{"Đại lượng": "đầu ra", "Kích thước": tuple(y_f.shape), "Sai số tuyệt đối lớn nhất": (y_f - y_s).abs().max().item(),
         "Sai số tương đối": rel_err(y_f, y_s)}]
rows += [{"Đại lượng": f"grad {n}", "Kích thước": tuple(gr_f[n].shape),
          "Sai số tuyệt đối lớn nhất": (gr_f[n] - gr_s[n]).abs().max().item(), "Sai số tương đối": rel_err(gr_f[n], gr_s[n])}
         for n in gr_f]
a_df = pd.DataFrame(rows)
a_df["Đạt"] = (a_df["Sai số tương đối"] <= REL_TOL).map({True: "ĐẠT", False: "KHÔNG"})
show(f"(a) Đường nhanh so với đường chậm (đạt khi sai số tương đối <= {REL_TOL:g})", a_df)

# (b) không rò rỉ tương lai
x2 = x.clone()
x2[:, K_CUT:] = torch.randn_like(x2[:, K_CUT:])
with torch.no_grad():
    rows = []
    for name, m in [("đường nhanh", m_fast), ("đường chậm", m_slow)]:
        y1, y2 = m(x), m(x2)
        rows.append({"Đường": name, f"Sai khác lớn nhất trước bước {K_CUT}": (y1[:, :K_CUT] - y2[:, :K_CUT]).abs().max().item(),
                     f"Sai khác lớn nhất từ bước {K_CUT}": (y1[:, K_CUT:] - y2[:, K_CUT:]).abs().max().item()})
b_df = pd.DataFrame(rows)
b_df["Đạt"] = ((b_df.iloc[:, 1] == 0) & (b_df.iloc[:, 2] > 0)).map({True: "ĐẠT", False: "KHÔNG"})
show("(b) Không rò rỉ tương lai (đạt khi trước bước cắt bằng đúng 0, sau bước cắt khác 0)", b_df)

# (c) tất định: chạy lại N_REPEAT lần cùng trọng số, cùng đầu vào
rows = []
for det in (False, True):
    try:
        torch.use_deterministic_algorithms(det)
        runs = [fwd_bwd(m_fast, x, g) for _ in range(N_REPEAT)]
        diff = {n: max((r[1][n] - runs[0][1][n]).abs().max().item() for r in runs[1:]) for n in runs[0][1]}
        out_diff = max((r[0] - runs[0][0]).abs().max().item() for r in runs[1:])
        worst = max(diff, key=diff.get)
        rows.append({"use_deterministic_algorithms": det, "Đầu ra lệch giữa các lần": out_diff,
                     "Grad lệch lớn nhất": diff[worst], "Ở tham số": worst,
                     "Số tham số có grad lệch": sum(v > 0 for v in diff.values())})
    except Exception as e:
        rows.append({"use_deterministic_algorithms": det, "Đầu ra lệch giữa các lần": None,
                     "Grad lệch lớn nhất": None, "Ở tham số": f"LỖI: {type(e).__name__}: {e}", "Số tham số có grad lệch": None})
    finally:
        torch.use_deterministic_algorithms(False)
show(f"(c) Tất định: {N_REPEAT} lần chạy lại xuôi + ngược (0 = lặp lại y hệt)", pd.DataFrame(rows))


# (d) thời gian và bộ nhớ, chỉ tham khảo
def time_fwd_bwd(model, inp, n_warm=3, n_run=10):
    # Trung vị thời gian một lượt xuôi + ngược (ms) và bộ nhớ GPU đỉnh (GB).
    times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(n_warm + n_run):
        model.zero_grad(set_to_none=True)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = model(inp)
        out = out[0] if isinstance(out, tuple) else out
        out.backward(torch.ones_like(out))
        end.record()
        torch.cuda.synchronize()
        if i >= n_warm:
            times.append(start.elapsed_time(end))
    return float(pd.Series(times).median()), torch.cuda.max_memory_allocated() / 1024 ** 3


lstm = torch.nn.LSTM(D_MODEL, D_MODEL, batch_first=True).to(DEV)
rows = []
for name, m in [("Mamba đường nhanh", m_fast), ("Mamba đường chậm", m_slow), ("nn.LSTM 128 => 128", lstm)]:
    ms, mem = time_fwd_bwd(m, x)
    rows.append({"Mô hình": name, "Số tham số": sum(p.numel() for p in m.parameters()),
                 "Xuôi + ngược (ms, trung vị 10 lần)": round(ms, 1), "Bộ nhớ đỉnh (GB)": round(mem, 2)})
show(f"(d) Thời gian một lô ({B_SZ}, {SEQ}, {D_MODEL}) trên {torch.cuda.get_device_name(0)}", pd.DataFrame(rows))
print("=> Đạt khi (a), (b) đều ĐẠT; (c), (d) để ghi nhận. Gửi output này.")


# %% Phần 8 - Thử mambapy với trọng số của khối Mamba
# mambapy 1.2.0 (PyTorch thuần, chạy được trên CPU, cho demo D2-D4): MambaBlock cùng tên tham số
# (in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj) và cùng mặc định (d_state 16, expand 2, d_conv 4,
# dt_rank = ceil(d_model/16), bias False, conv_bias True) với mamba_ssm.Mamba (đọc mambapy/mamba.py bản 1.2.0).
# Nạp state_dict của m_fast (Phần 7) với strict=True, so đầu ra trên GPU và CPU với mamba_ssm trên cùng mẫu.
# pscan của mambapy đệm chuỗi lên lũy thừa 2 (365 => 512) và giữ tensor (lô, bước, 2*d_model, 16), nên dùng lô nhỏ.
MAMBAPY_VERSION = "1.2.0"
N_GPU_CHECK, N_CPU_CHECK = 32, 4        # số mẫu so trên GPU và trên CPU

if lib_version("mambapy") != MAMBAPY_VERSION:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", f"mambapy=={MAMBAPY_VERSION}"], check=True)
from mambapy.mamba import MambaBlock, MambaConfig  # noqa: E402

cfg = MambaConfig(d_model=D_MODEL, n_layers=1)  # n_layers bắt buộc khai báo, MambaBlock không dùng tới
mp_gpu = MambaBlock(cfg).to(DEV)
load_res = mp_gpu.load_state_dict(m_fast.state_dict(), strict=True)
print(f"Nạp state_dict strict=True: thiếu {load_res.missing_keys}, thừa {load_res.unexpected_keys}")
print(f"Cấu hình mambapy: d_inner {cfg.d_inner}, dt_rank {cfg.dt_rank}, d_state {cfg.d_state}, d_conv {cfg.d_conv}")

rows = []
with torch.no_grad():
    ref_gpu = m_fast(x[:N_GPU_CHECK])
    out_gpu = mp_gpu(x[:N_GPU_CHECK])
    rows.append({"So sánh": f"mambapy GPU vs mamba_ssm GPU ({N_GPU_CHECK} mẫu)",
                 "Sai số tuyệt đối lớn nhất": (out_gpu - ref_gpu).abs().max().item(), "Sai số tương đối": rel_err(out_gpu, ref_gpu)})
    mp_cpu = MambaBlock(cfg)
    mp_cpu.load_state_dict({k: v.cpu() for k, v in m_fast.state_dict().items()}, strict=True)
    x_cpu = x[:N_CPU_CHECK].cpu()
    t0 = time.time()
    out_cpu = mp_cpu(x_cpu)
    cpu_secs = time.time() - t0
    ref_cpu = ref_gpu[:N_CPU_CHECK].cpu()
    rows.append({"So sánh": f"mambapy CPU vs mamba_ssm GPU ({N_CPU_CHECK} mẫu)",
                 "Sai số tuyệt đối lớn nhất": (out_cpu - ref_cpu).abs().max().item(), "Sai số tương đối": rel_err(out_cpu, ref_cpu)})
c_df = pd.DataFrame(rows)
c_df["Đạt"] = (c_df["Sai số tương đối"] <= REL_TOL).map({True: "ĐẠT", False: "KHÔNG"})
show(f"mambapy {MAMBAPY_VERSION} dùng lại trọng số mamba_ssm (đạt khi sai số tương đối <= {REL_TOL:g})", c_df)
print(f"Thời gian mambapy trên CPU ({os.cpu_count()} lõi Kaggle), {N_CPU_CHECK} mẫu x {SEQ} bước, chỉ xuôi: {cpu_secs:.2f} giây")
print("=> Đạt khi nạp không thiếu/thừa khóa và hai dòng đều ĐẠT. Gửi output này.")


# %% Phần 9 - Cell cài dùng lại cho các notebook sau
# Tự đủ (không dùng biến của Phần 1-8): chép nguyên cell này làm cell đầu của mọi notebook cần mamba-ssm.
# Notebook đó gắn Output của notebook 01_0_MambaSSM_Build (bản Save Version) làm Input; cell tìm build_info.json
# dưới /kaggle/input (hoặc /kaggle/working/wheels khi chạy ngay trong notebook dựng), kiểm Python/torch/CUDA
# khớp lúc dựng (lệch => dừng, phải dựng lại), kiểm sha256, cài --no-deps, nạp và chạy thử một khối Mamba.
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

WHEEL_ROOTS = [Path("/kaggle/input"), Path("/kaggle/working/wheels")]


def mamba_install():
    infos = [p for root in WHEEL_ROOTS if root.exists() for p in root.rglob("build_info.json")]
    if not infos:
        raise FileNotFoundError("Không thấy build_info.json: gắn Output của notebook 01_0_MambaSSM_Build làm Input")
    if len(infos) > 1:
        print("Thấy nhiều build_info.json, dùng tệp đầu tiên:", *infos, sep="\n  ")
    info_path = infos[0]
    info = json.loads(info_path.read_text())
    now = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda}
    diff = {k: (info[k], v) for k, v in now.items() if info[k] != v}
    if diff:
        raise RuntimeError(f"Môi trường khác lúc dựng wheel (lúc dựng, bây giờ): {diff}. Phải dựng lại ở 01_0_MambaSSM_Build")
    gpu_cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    if gpu_cc != (7, 5):
        raise RuntimeError(f"Wheel chỉ có mã cho {info['arch']} (T4), GPU hiện tại: {gpu_cc}")
    wheels = []
    for name, digest in info["wheels"].items():
        whl = info_path.parent / name
        if hashlib.sha256(whl.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"sha256 của {name} không khớp build_info.json")
        wheels.append(str(whl))
    torch_dist = importlib.metadata.version("torch")  # đọc bản đã cài trên đĩa (torch.__version__ không đổi khi pip cài lại)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *wheels], check=True)
    if importlib.metadata.version("torch") != torch_dist:
        raise RuntimeError("torch bị đổi sau khi cài wheel, khởi động lại phiên và kiểm lại lệnh cài")
    from mamba_ssm import Mamba
    m = Mamba(d_model=16).cuda()
    y = m(torch.randn(2, 8, 16, device="cuda", requires_grad=True))
    y.sum().backward()
    if not torch.isfinite(y).all():
        raise RuntimeError("Khối Mamba chạy thử ra NaN/inf")
    print(f"mamba-ssm đã cài từ {info_path.parent} (dựng {info['built_at']}, torch {info['torch']}, {info['arch']}); "
          f"chạy thử xuôi + ngược đạt")
    return info


MAMBA_BUILD_INFO = mamba_install()
