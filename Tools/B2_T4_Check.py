# B2 - Cổng kiểm trên Kaggle T4 trước khi gộp vào research.
# Chạy nguyên tệp trong một cell tạm, hoặc python B2_T4_Check.py.
# Không huấn luyện B2 chính thức; kiểm thư viện bằng dữ liệu fixture của repo.
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import torch

NH_REPO = 'https://github.com/Ticasslo/neuralhydrology.git'
NH_COMMIT = '2ccaa147a6ec87d0f56fa3e2f11b1bf38079253f'
WORK = Path('/kaggle/working')
if not WORK.is_dir() or not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
    raise RuntimeError('Cần Kaggle T4, Internet On; đây không phải kiểm CPU hoặc huấn luyện B2 chính thức.')
if importlib.util.find_spec('pytest') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pytest'], check=True)

OUT = WORK / 'b2_t4_check'
OUT.mkdir(exist_ok=True)
if (OUT / 'B2_t4_check.json').exists():
    raise RuntimeError('Đã có kết quả T4 trong phiên này; dùng phiên sạch để tránh trộn bằng chứng.')
checkout = Path(tempfile.mkdtemp(prefix='library_', dir=OUT))
subprocess.run(['git', 'clone', '--no-checkout', NH_REPO, str(checkout)], check=True)
subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', NH_COMMIT], check=True)
actual_commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
if actual_commit != NH_COMMIT:
    raise RuntimeError('Source checkout khác commit B2 đã pin.')

env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1', PYTHONPATH=str(checkout))


def run_tests(name, args, expected):
    report = OUT / f'{name}.xml'
    log = OUT / f'{name}.log'
    command = [sys.executable, '-m', 'pytest', *args, '-q', '-p', 'no:cacheprovider',
               '--basetemp', str(OUT / f'tmp_{name}'), '--junitxml', str(report)]
    with log.open('w', encoding='utf-8') as fp:
        process = subprocess.Popen(command, cwd=checkout, env=env, stdout=fp, stderr=subprocess.STDOUT)
        while True:
            try:
                process.wait(timeout=45)
                break
            except subprocess.TimeoutExpired:
                print(f'{name}: đang chạy; log {log.name}', flush=True)
    print(log.read_text(encoding='utf-8', errors='replace')[-4000:])
    if process.returncode or not report.is_file():
        raise RuntimeError(f'{name} không đạt; giữ log và XML để đối chiếu.')
    suites = ET.parse(report).getroot()
    nodes = [suites] if suites.tag == 'testsuite' else list(suites.iter('testsuite'))
    counts = {key: sum(int(node.get(key, 0)) for node in nodes)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=expected, failures=0, errors=0, skipped=0):
        raise RuntimeError(f'{name}: cần đủ {expected} kiểm không skip, thực tế {counts}')
    return counts


# B2 có 14 kiểm kể cả CUDA; sequential_forecast có đủ 43 kiểm.
results = {'b2_and_cores': run_tests('b2_and_cores',
    ['test/test_b2_source_inputs.py', 'test/test_sequential_forecast.py'], 57)}

# Fixture mặc định của bốn kiểm dài dùng CPU. Chỉ đổi bản clone tạm sang cuda:0,
# để bốn phép so DataLoader/GpuBatchLoader chạy huấn luyện và evaluation thật trên T4.
config_path = checkout / 'test/test_configs/daily_regression.test.yml'
original = config_path.read_bytes()
text = original.decode('utf-8')
if text.count('device: cpu') != 1:
    raise RuntimeError('Config fixture đã thay đổi, cần rà lại trước khi đặt CUDA.')
try:
    config_path.write_text(text.replace('device: cpu', 'device: cuda:0'), encoding='utf-8')
    results['four_training_cuda'] = run_tests('four_training_cuda',
        ['test/test_gpu_loader.py', '-k', 'training_and_evaluation_identical'], 4)
finally:
    config_path.write_bytes(original)

result = {'complete': True, 'library_commit': actual_commit, 'device': torch.cuda.get_device_name(0),
          'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
          'results': results, 'long_training_fixture_device': 'cuda:0',
          'fixture_config_restored': config_path.read_bytes() == original,
          'official_B2_training_started': False, 'research_merged': False}
(OUT / 'B2_t4_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json')
(OUT / 'manifest.json').write_text(json.dumps([
    {'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files], indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
print('Đã kiểm T4; gửi B2_t4_check.json, manifest.json và hai log/XML để rà trước khi gộp research.')
