# =============================================================================
# B2 — Bộ nạp và mô hình Extended LamaH-CE của Người 1
# Kaggle: Internet On; CPU đủ dùng, GPU tùy chọn; Input: lamah-ce-ext + output A1.
# A2 chỉ cần khi muốn lấy config từ export; A3 không bắt buộc.
# Bản sửa kiểm độc lập trên fork f229e02; chưa tích hợp cấu hình B1b.
# Kết quả lưu trực tiếp trong b2_output, không đóng ZIP.
# MỤC LỤC
#   Phần 1 — Môi trường, mã nền và Input
#   Phần 2 — Hợp đồng nguồn, biến, pha và đơn vị
#   Phần 3 — Bộ lọc gốc và các ví dụ biết trước
#   Phần 4 — Bản sửa bộ nạp và kiểm nguồn mã
#   Phần 5 — Kiểm thử bộ nạp bằng dữ liệu giả
#   Phần 6 — Dữ liệu 451 ID và đối chiếu A1
#   Phần 7 — Batch thật, masked mean và gradient
#   Phần 8 — Cách (b), counter và kiểm chống rò rỉ
#   Phần 9 — Lưu bằng chứng và trạng thái bàn giao
# =============================================================================

# %% Phần 1.1 - Thiết lập chạy và các đường dẫn
# Chọn Input/Output, mã nền và phạm vi kiểm; không chạy huấn luyện chính thức.
import ast
import copy
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

IS_KAGGLE = Path('/kaggle/working').is_dir()
INPUT_ROOT = Path(os.environ.get('B2_INPUT_ROOT', os.environ.get('LAMAH_ROOT', '/kaggle/input' if IS_KAGGLE else '.'))).resolve()
OUT = Path(os.environ.get('B2_OUTPUT_ROOT', '/kaggle/working/b2_output' if IS_KAGGLE else 'b2_output')).resolve()
OUT.mkdir(parents=True, exist_ok=True)
BASE_COMMIT = 'f229e02bf98cbd979835a4b6cd503d0e3adb8bc3'
BASE_REPO = 'https://github.com/Ticasslo/neuralhydrology.git'
BASE_HASHES = {
    'datasetzoo/lamah.py': '8eaaca354392d7865ce51ee1c8327c80bafb888d830ab4e0f66628f2ad234e09',
    'datasetzoo/basedataset.py': '9da162ecc1cff00783fbc66e85ace386193c3c9f0f7d272429f8c5521720a494',
    'modelzoo/inputlayer.py': 'b89b8eadbebaebe4f01b1bffaf64513c89d69f5726576f32e5a6bd83f4cbcbcb',
    'datautils/utils.py': 'fdd1ca3120a7eb017a7b1348cc18208e726e41edaeef6e11bc7ece96d2038e88',
}
INSTALL_MISSING = True
SOURCE_OVERRIDE = os.environ.get('B2_NH_SOURCE')  # thư mục chứa package neuralhydrology đã checkout đúng mã nền
RAW_OVERRIDE = os.environ.get('B2_RAW_ROOT')
CONFIG_OVERRIDE = os.environ.get('B2_CONFIG')
B1_CONFIG_PATH = os.environ.get('B2_B1_CONFIG')
B1_COMMIT = os.environ.get('B2_B1_COMMIT')
PROBE_IDS = ('55', '295', '324', '600', '623', '742', '758', '800')
RUN_REAL_DATA = os.environ.get('B2_RUN_REAL_DATA', '1') == '1'
PERIODS = {'train': ('2003-01-01', '2009-12-31'), 'validation': ('2010-01-01', '2013-12-31'), 'test': ('2014-01-01', '2017-12-31')}
SOURCE_ORDER = ('ERA5L', 'EOBS', 'MSWEP', 'GLEAM', 'Q', 'ECMWF')
CHECKS, RECEIPTS = [], []
STARTED = time.perf_counter()
print('B2: fork f229e02; kiểm bộ nạp/mô hình; chờ bàn giao B1b để huấn luyện.')
print('Input:', INPUT_ROOT, '| Output:', OUT, '| Python:', platform.python_version())

# %% Phần 1.2 - Chuẩn bị thư viện và xác nhận mã nền
# Cài đúng commit khi bản có sẵn khác nguồn; kiểm hash trước mọi thay đổi B2.
REQUIRED = {'numpy': 'numpy', 'pandas': 'pandas', 'xarray': 'xarray', 'torch': 'torch',
            'scipy': 'scipy', 'numba': 'numba', 'ruamel.yaml': 'ruamel.yaml',
            'tqdm': 'tqdm', 'tensorboard': 'tensorboard'}
loaded_nh = sorted(name for name in sys.modules
                   if name == 'neuralhydrology' or name.startswith('neuralhydrology.'))
if loaded_nh:
    raise RuntimeError('Restart session rồi chạy từ Phần 1.1; phiên hiện còn module NeuralHydrology đã nạp.')
before_versions = {}
for package in REQUIRED.values():
    try:
        before_versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        before_versions[package] = None
missing = []
for module, package in REQUIRED.items():
    try:
        found = importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        found = False
    if not found:
        missing.append(package)
if missing:
    if not INSTALL_MISSING:
        raise RuntimeError(f'Cần cài các gói trước khi chạy: {missing}')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
    importlib.invalidate_caches()

installed_commit = None
source_mode = 'source_override' if SOURCE_OVERRIDE else 'pinned_pip_install'
if SOURCE_OVERRIDE:
    BASE_PACKAGE = Path(SOURCE_OVERRIDE).resolve()
    if BASE_PACKAGE.name != 'neuralhydrology':
        BASE_PACKAGE = BASE_PACKAGE / 'neuralhydrology'
else:
    try:
        distribution = metadata.distribution('neuralhydrology')
    except metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        direct_url = json.loads(distribution.read_text('direct_url.json') or '{}')
        installed_commit = direct_url.get('vcs_info', {}).get('commit_id')
    if installed_commit != BASE_COMMIT:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
                        f'git+{BASE_REPO}@{BASE_COMMIT}'], check=True)
        importlib.invalidate_caches()
        distribution = metadata.distribution('neuralhydrology')
        direct_url = json.loads(distribution.read_text('direct_url.json') or '{}')
        installed_commit = direct_url.get('vcs_info', {}).get('commit_id')
    if installed_commit != BASE_COMMIT:
        raise RuntimeError(f'Commit thư viện đã cài khác mã yêu cầu: {installed_commit}')
    BASE_PACKAGE = Path(distribution.locate_file('neuralhydrology')).resolve()
if not (BASE_PACKAGE / '__init__.py').is_file():
    raise FileNotFoundError(f'Không thấy package NeuralHydrology tại {BASE_PACKAGE}')

import numpy as np
import pandas as pd
import torch
import xarray as xr
from ruamel.yaml import YAML
from IPython.display import Markdown, display

torch.set_num_threads(int(os.environ.get('B2_CPU_THREADS', '2')))


def b2_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def b2_json(path, value):
    # Metadata không chứa NaN; số liệu thiếu được giữ trong bảng CSV.
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding='utf-8')
    RECEIPTS.append(Path(path))


def b2_table(title, frame, limit=35):
    display(Markdown(f'**{title}**'))
    data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    with pd.option_context('display.max_columns', None, 'display.max_colwidth', 100, 'display.max_rows', limit):
        display(data.style.hide(axis='index').format(precision=6, na_rep='-'))


def b2_save(name, frame):
    path = OUT / name
    frame.to_csv(path, index=False, encoding='utf-8-sig')
    RECEIPTS.append(path)
    return path


def b2_check(name, operation, group):
    try:
        operation()
        row = {'group': group, 'case': name, 'passed': True, 'detail': ''}
    except Exception as exc:
        row = {'group': group, 'case': name, 'passed': False, 'detail': f'{type(exc).__name__}: {exc}'}
    CHECKS[:] = [previous for previous in CHECKS if (previous['group'], previous['case']) != (group, name)]
    CHECKS.append(row)
    b2_save('B2_checks.csv', pd.DataFrame(CHECKS))
    return row['passed']


def b2_require(group):
    rows = [r for r in CHECKS if r['group'] == group]
    b2_table(group, rows)
    if not rows or not all(r['passed'] for r in rows):
        raise AssertionError(f'Nhóm kiểm {group} chưa đạt; xem B2_checks.csv và output phía trên.')


# Kiểm thêm Config và bộ nạp vector hóa, hai điểm nối của fork f229e02.
BASE_HASHES.update({'utils/config.py': '50efadb8859edbf2821c98bc425f515c46510db69d003680de677d2041c12dad', 'datautils/gpu_loader.py': '54c93fc11e8d83f0a74305032c080154dde03c0408f88277d141a09fcce4e322'})
BASE_TEXT, source_rows = {}, []
for relative, expected in BASE_HASHES.items():
    path = BASE_PACKAGE / relative
    if not path.is_file():
        raise FileNotFoundError(f'Thiếu tệp mã nền: {relative}')
    source = path.read_text(encoding='utf-8-sig')
    digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
    if digest != expected:
        raise RuntimeError(f'Mã nền khác f229e02: {relative}, SHA={digest}. Chưa áp bản sửa B2.')
    BASE_TEXT[relative] = source
    source_rows.append({'file': relative, 'sha256_lf': digest, 'passed': True})
VERSIONS = {name: metadata.version(name) for name in REQUIRED.values()}
changed_versions = {name: {'before': old, 'after': VERSIONS[name]} for name, old in before_versions.items()
                    if old is not None and VERSIONS[name] != old}
if changed_versions:
    raise RuntimeError(f'pip đã đổi gói có sẵn; cần kiểm môi trường trước khi tiếp tục: {changed_versions}')
if 'def data_on_gpu(' not in BASE_TEXT['utils/config.py'] or 'class GpuBatchLoader:' not in BASE_TEXT['datautils/gpu_loader.py']:
    raise RuntimeError('Mã nền chưa có giao diện data_on_gpu đã kiểm ở bước 0.')
b2_json(OUT / 'B2_environment.json', {'python': platform.python_version(), 'packages': VERSIONS,
        'base_commit': BASE_COMMIT, 'installed_commit': installed_commit, 'source_mode': source_mode,
        'base_package': str(BASE_PACKAGE), 'source_hashes_lf': BASE_HASHES,
        'source_files_verified': len(source_rows), 'existing_package_versions_changed': changed_versions,
        'device_available': 'cuda' if torch.cuda.is_available() else 'cpu', 'official_training_started': False})
b2_table('1.2. Môi trường', [{'package': k, 'version': v} for k, v in VERSIONS.items()])
b2_table('1.2. Mã nền f229e02', source_rows)
print('Phần 1.2 đạt:', len(source_rows), 'tệp khớp SHA-256; chưa áp thay đổi B2 hoặc bắt đầu huấn luyện.')

# %% Phần 1.3 - Tìm raw, danh sách ID, config và bảng A1
# Chỉ kiểm cấu trúc, header và danh tính nguồn; chưa nạp chuỗi ngày hoặc fit scaler.
b2_json(OUT / 'B2_input_sources.json', {'complete': False, 'stage': 'input_discovery'})
if not INPUT_ROOT.is_dir():
    raise FileNotFoundError(f'Không thấy thư mục Input: {INPUT_ROOT}')
raw_candidates, id_candidates, configs, a1_files = [], [], [], []
skip_dirs = {'.git', '.cache', '__pycache__', 'node_modules', '.venv', OUT.name,
             'synthetic_fixture', 'synthetic_logs', 'probe_logs', 'library_candidate', 'library_base'}


def b2_input_path(path):
    try:
        return Path(path).relative_to(INPUT_ROOT).as_posix()
    except ValueError:
        return str(path)


def b2_walk_error(error):
    raise error


for directory, dirs, files in os.walk(INPUT_ROOT, onerror=b2_walk_error):
    dirs[:] = sorted(d for d in dirs if d not in skip_dirs)
    scan_root = Path(directory)
    if all((scan_root / part).is_dir() for part in
           ('D_gauges/2_timeseries/daily', 'A_basins_total_upstrm/2_timeseries/daily')):
        raw_candidates.append(scan_root.resolve())
    if 'basins_filtered.txt' in files:
        id_candidates.append((scan_root / 'basins_filtered.txt').resolve())
    configs.extend((scan_root / name).resolve() for name in files if name in ('config.yml', 'config.yaml'))
    a1_files.extend((scan_root / name).resolve() for name in files if name.startswith('A1_P7_') and name.endswith('.csv'))
raw_candidates, id_candidates = sorted(set(raw_candidates)), sorted(set(id_candidates))
configs, a1_files = sorted(set(configs)), sorted(set(a1_files))
if RAW_OVERRIDE:
    RAW = Path(RAW_OVERRIDE).resolve()
else:
    canonical = [path for path in raw_candidates if path.name == 'LamaH_extended']
    choices = canonical or raw_candidates
    if len(choices) != 1:
        raise RuntimeError(f'Cần đúng một nguồn LamaH-CE gốc; tìm thấy: {[b2_input_path(p) for p in choices]}')
    RAW = choices[0]
required_raw = ('A_basins_total_upstrm/2_timeseries/daily', 'D_gauges/2_timeseries/daily',
                'A_basins_total_upstrm/1_attributes/Catchment_attributes.csv', 'D_gauges/1_attributes/Gauge_attributes.csv')
if (not all((RAW / name).is_dir() for name in required_raw[:2])
        or not all((RAW / name).is_file() for name in required_raw[2:])):
    raise FileNotFoundError(f'Nguồn raw thiếu timeseries/attributes LamaH-CE: {RAW}')
if not id_candidates:
    raise FileNotFoundError('Thiếu basins_filtered.txt; gắn dataset lamah-ce-ext có kết quả tác giả.')
id_sets = []
for path in id_candidates:
    tokens = path.read_text(encoding='utf-8-sig').split()
    if not tokens or any(not token.isascii() or not token.isdecimal() or int(token) <= 0 for token in tokens):
        raise ValueError(f'ID không hợp lệ trong {b2_input_path(path)}')
    normalized = tuple(str(int(token)) for token in tokens)
    if len(normalized) != 451 or len(set(normalized)) != 451:
        raise ValueError(f'{b2_input_path(path)} cần đúng 451 ID duy nhất sau chuẩn hóa.')
    id_sets.append(normalized)
if any(set(ids) != set(id_sets[0]) for ids in id_sets):
    raise ValueError('Các bản basins_filtered.txt có tập ID khác nhau; không tự chọn nguồn.')
original_id_path = (RAW.parent / 'Experiments' / 'basins_filtered.txt').resolve()
if original_id_path in id_candidates:
    ID_SOURCE = original_id_path
elif len(set(id_sets)) == 1:
    ID_SOURCE = id_candidates[0]
else:
    raise ValueError('Các bản basins_filtered.txt khác thứ tự; cần nguồn Experiments gốc để giữ thứ tự tác giả.')
IDS = id_sets[id_candidates.index(ID_SOURCE)]
missing_raw = []
for basin in IDS:
    for group in ('A_basins_total_upstrm', 'D_gauges'):
        if not (RAW / group / '2_timeseries/daily' / f'ID_{basin}.csv').is_file():
            missing_raw.append(f'{group}/ID_{basin}.csv')
if missing_raw:
    raise FileNotFoundError(f'Thiếu {len(missing_raw)} tệp của 451 ID: {missing_raw[:12]}')
forcing_header = list(pd.read_csv(RAW / f'A_basins_total_upstrm/2_timeseries/daily/ID_{IDS[0]}.csv', sep=';', nrows=0).columns)
gauge_header = list(pd.read_csv(RAW / f'D_gauges/2_timeseries/daily/ID_{IDS[0]}.csv', sep=';', nrows=0).columns)
if not {'YYYY', 'MM', 'DD'}.issubset(forcing_header) or not {'YYYY', 'MM', 'DD', 'qmean', 'qmax'}.issubset(gauge_header):
    raise ValueError('Header forcing/gauge không khớp schema LamaH-CE theo ngày.')


def b2_input_names(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [name for item in value.values() for name in b2_input_names(item)]
    if isinstance(value, (list, tuple)):
        return [name for item in value for name in b2_input_names(item)]
    raise TypeError(f'Danh sách biến trong config có kiểu không hỗ trợ: {type(value).__name__}')


for explicit in (CONFIG_OVERRIDE, B1_CONFIG_PATH):
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Không thấy config được khai báo: {path}')
        configs.append(path)
yaml = YAML(typ='safe')
config_rows, CONFIGS = [], {}
for path in sorted(set(configs)):
    cfg = yaml.load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(cfg, dict):
        raise ValueError(f'Config phải là ánh xạ khóa/giá trị: {b2_input_path(path)}')
    key = str(path)
    CONFIGS[key] = cfg
    inputs = b2_input_names(cfg.get('hindcast_inputs'))
    config_rows.append({'path': key, 'source': b2_input_path(path), 'model': cfg.get('model'),
        'dataset': cfg.get('dataset'), 'seq_length': cfg.get('seq_length'),
        'Q_inputs': ' | '.join(name for name in inputs if name in ('qmean', 'qmax', 'qmean_mmd', 'qmax_mmd', 'qmax_past')),
        'sha256': b2_hash(path)})
if not CONFIGS:
    raise FileNotFoundError('Thiếu config.yml; gắn dataset lamah-ce-ext có Experiments hoặc Output A2.')
if not any(cfg.get('model') == 'sequential_forecast_lstm' and
           {'qmean', 'qmean_mmd'}.intersection(b2_input_names(cfg.get('hindcast_inputs'))) for cfg in CONFIGS.values()):
    raise ValueError('Chưa tìm thấy config Sequential Forecast LSTM có Q quá khứ.')
required_a1 = {'A1_P7_sample_contract_groups.csv':
               {'contract_id', 'dynamic_names', 'hindcast_names', 'forecast_names', 'target_names', 'contract_json'},
               'A1_P7_sample_counts.csv': {'contract_id', 'period', 'sequence_total_days'}}
a1_inventory = pd.DataFrame([{'file': str(p), 'source': b2_input_path(p), 'name': p.name,
                              'bytes': p.stat().st_size, 'sha256': b2_hash(p)} for p in a1_files],
                             columns=['file', 'source', 'name', 'bytes', 'sha256'])
for name, columns in required_a1.items():
    selected = a1_inventory.loc[a1_inventory.name.eq(name)]
    if selected.empty:
        raise FileNotFoundError(f'Thiếu {name}; gắn Output 01_A1_LamaHCEExt_Explore, phần Tables.')
    if selected.sha256.nunique() != 1:
        raise ValueError(f'Có nhiều bản A1 khác nội dung cho {name}; giữ đúng nguồn A1 cần đối chiếu.')
    header = set(pd.read_csv(selected.iloc[0]['file'], nrows=0, encoding='utf-8-sig').columns)
    if not columns.issubset(header):
        raise ValueError(f'{name} thiếu cột: {sorted(columns - header)}')
id_inventory = pd.DataFrame([{'file': b2_input_path(p), 'IDs': len(ids), 'sha256': b2_hash(p),
                              'selected': p == ID_SOURCE, 'same_order_as_selected': ids == IDS}
                             for p, ids in zip(id_candidates, id_sets)])
b2_save('B2_input_configs.csv', pd.DataFrame(config_rows))
b2_save('B2_A1_inventory.csv', a1_inventory)
b2_save('B2_basin_list_inventory.csv', id_inventory)
b2_json(OUT / 'B2_input_sources.json', {'complete': True, 'raw': str(RAW), 'basin_file': str(ID_SOURCE),
        'basin_file_sha256': b2_hash(ID_SOURCE), 'basin_ids': IDS, 'selected_basin_files_checked': 902,
        'config_files': {row['path']: row['sha256'] for row in config_rows},
        'A1_required': {name: a1_inventory.loc[a1_inventory.name.eq(name), 'sha256'].iloc[0] for name in required_a1},
        'A1_notebook_version_expected': 3, 'A1_version_verified_from_metadata': False,
        'B1b_integration_verified': False, 'official_training_started': False})
b2_table('1.3. Input tìm được', [{'raw': b2_input_path(RAW), 'IDs': len(IDS),
    'forcing_files_of_selected_IDs': len(IDS), 'gauge_files_of_selected_IDs': len(IDS),
    'configs': len(CONFIGS), 'A1_P7_CSVs': len(a1_files)}])
b2_table('1.3. Danh sách lưu vực', id_inventory)
b2_table('1.3. Cấu hình tìm được', pd.DataFrame(config_rows)[['source', 'model', 'dataset', 'seq_length', 'Q_inputs']])
b2_table('1.3. Hai bảng A1 bắt buộc', a1_inventory.loc[a1_inventory.name.isin(required_a1), ['source', 'bytes', 'sha256']])
print('Phần 1.3 đạt: đủ 451 ID, 902 tệp chuỗi ngày và hai bảng A1; chưa chọn cấu hình B1b hoặc fit scaler.')

# %% Phần 2.1 - Hàm nhóm nguồn và kiểm hợp đồng cấu hình
# Chốt tên biến và pha được phép; chưa chọn config, dựng Q, fit scaler hoặc nạp mô hình.
B2_Q_INPUTS = ('qmean_mmd', 'qmax_mmd')
B2_GROUP_SPEC = [
    {'phase': 'hindcast', 'source': 'ERA5L', 'expected_variables': 21},
    {'phase': 'hindcast', 'source': 'EOBS', 'expected_variables': 7},
    {'phase': 'hindcast', 'source': 'MSWEP', 'expected_variables': 1},
    {'phase': 'hindcast', 'source': 'GLEAM', 'expected_variables': 2},
    {'phase': 'hindcast', 'source': 'Q', 'expected_variables': 2},
    {'phase': 'forecast', 'source': 'ECMWF', 'expected_variables': 5},
]


def b2_flat(value):
    if value is None:
        return []
    if isinstance(value, str):
        if not value or value.strip() != value:
            raise ValueError('Tên biến không được rỗng hoặc có khoảng trắng ở hai đầu.')
        return [value]
    if isinstance(value, dict):
        return [name for group in value.values() for name in b2_flat(group)]
    if isinstance(value, (list, tuple)):
        return [name for group in value for name in b2_flat(group)]
    raise TypeError(f'Danh sách biến có kiểu không hỗ trợ: {type(value).__name__}')


def b2_source(variable):
    if not isinstance(variable, str) or not variable:
        raise ValueError('Tên biến phải là chuỗi không rỗng.')
    if variable in B2_Q_INPUTS:
        return 'Q'
    if variable in ('qmean', 'qmax', 'qmax_past'):
        raise ValueError('Đầu vào Q của B2 phải là qmean_mmd và qmax_mmd; qmax là nhãn riêng.')
    for prefix in SOURCE_ORDER:
        if prefix != 'Q' and variable.startswith(prefix + '_') and len(variable) > len(prefix) + 1:
            return prefix
    raise ValueError(f'Chưa ánh xạ nguồn cho biến {variable}; không tự suy đơn vị hoặc nguồn.')


def b2_variable_list(cfg):
    values = cfg.as_dict() if hasattr(cfg, 'as_dict') else cfg
    if not isinstance(values, dict):
        raise TypeError('Cấu hình phải là dict hoặc Config của NeuralHydrology.')
    return list(dict.fromkeys(b2_flat(values.get('target_variables')) + b2_flat(values.get('dynamic_inputs'))))


def b2_groups(variables):
    variables = b2_flat(variables)
    if len(variables) != len(set(variables)):
        raise ValueError('Biến trùng trong một pha.')
    grouped = {source: [] for source in SOURCE_ORDER}
    for variable in variables:
        grouped[b2_source(variable)].append(variable)
    return [grouped[source] for source in SOURCE_ORDER if grouped[source]]


def b2_contract(cfg):
    values = cfg.as_dict() if hasattr(cfg, 'as_dict') else cfg
    if not isinstance(values, dict):
        raise TypeError('Cấu hình phải là dict hoặc Config của NeuralHydrology.')
    for legacy in ('b2_data_contract', 'b2_hybrid_forecast'):
        if legacy in values:
            raise ValueError(f'Khóa prototype {legacy} không thuộc cấu hình B2 trên B1b.')
    if b2_flat(values.get('use_frequencies')) not in ([], ['1D']):
        raise ValueError('Notebook B2 chỉ kiểm một tần suất ngày 1D.')
    for key in ('dynamic_inputs', 'hindcast_inputs', 'forecast_inputs'):
        if isinstance(values.get(key), dict):
            raise ValueError('Hợp đồng B2 dùng danh sách biến/nhóm cho một tần suất ngày, không dùng dict tần suất.')
    dynamic = b2_flat(values.get('dynamic_inputs'))
    hindcast = b2_flat(values.get('hindcast_inputs'))
    forecast = b2_flat(values.get('forecast_inputs'))
    for key, variables in (('dynamic_inputs', dynamic), ('hindcast_inputs', hindcast), ('forecast_inputs', forecast)):
        if not variables or len(variables) != len(set(variables)):
            raise ValueError(f'{key} phải không rỗng và không trùng biến.')
    if set(dynamic) != set(hindcast + forecast):
        raise ValueError('dynamic_inputs phải đúng hợp của các biến dùng ở hai pha, không có biến thừa hoặc thiếu.')
    if set(hindcast) & set(forecast):
        raise ValueError('Hợp đồng B2 tách nguồn hindcast và forecast; không dùng cùng biến ở hai pha.')
    targets = b2_flat(values.get('target_variables'))
    if targets != ['qmax'] or set(targets) & set(dynamic):
        raise ValueError('Nhãn phải là qmax, tách khỏi hai đầu vào Q mm/ngày để giữ warmup.')
    hindcast_sources = [b2_source(variable) for variable in hindcast]
    if 'ECMWF' in hindcast_sources:
        raise ValueError('Hindcast B2 chỉ nhận ERA5L/EOBS/MSWEP/GLEAM và Q.')
    if {variable for variable in hindcast if b2_source(variable) == 'Q'} != set(B2_Q_INPUTS):
        raise ValueError('Hindcast B2 phải chứa đủ qmean_mmd và qmax_mmd.')
    if any(b2_source(variable) != 'ECMWF' for variable in forecast):
        raise ValueError('Forecast B2 chỉ nhận ECMWF HRES, không nhận Q hoặc tái phân tích tương lai.')
    overlap = values.get('forecast_overlap')
    if overlap is not None and (type(overlap) is not int or overlap != 0):
        raise ValueError('Sequential Forecast B2 yêu cầu forecast_overlap=0 hoặc None.')
    length, horizon, predicted = (values.get(key) for key in ('seq_length', 'forecast_seq_length', 'predict_last_n'))
    if (type(length) is not int or type(horizon) is not int or type(predicted) is not int
            or not 1 <= horizon <= 7 or length <= horizon or predicted != horizon):
        raise ValueError('Cần seq_length > forecast_seq_length, hạn 1–7 và predict_last_n bằng số hạn.')
    mode = values.get('forecast_input_mode', 'all')
    if mode not in ('all', 'first_day'):
        raise ValueError('forecast_input_mode chỉ nhận all hoặc first_day.')
    if horizon > 1 and mode != 'first_day':
        raise ValueError('LamaH-CE chỉ có HRES hạn một ngày; kiểm nhiều hạn phải che các ngày sau bằng first_day.')
    if mode == 'first_day' and (values.get('nan_handling_method') != 'masked_mean'
                                or values.get('timestep_counter') is not True):
        raise ValueError('Cách (b) cần forecast_input_mode=first_day, masked_mean và timestep_counter=True.')
    return dynamic, hindcast, forecast


b2_save('B2_group_spec.csv', pd.DataFrame(B2_GROUP_SPEC))
b2_json(OUT / 'B2_expected_contract.json', {'group_spec': B2_GROUP_SPEC, 'Q_input_names': B2_Q_INPUTS,
        'Q_unit': 'mm/day_area_gov', 'Q_allowed_phase': 'hindcast', 'Q_latest_day_expected': 't-1',
        'target': 'qmax', 'forecast_source': 'ECMWF', 'forecast_input_modes': ['all', 'first_day'],
        'scaler_policy': 'inherit_B1b_f229e02', 'selected_config_verified': False,
        'B1b_integration_verified': False, 'official_training_started': False})
b2_table('2.1. Nhóm nguồn kỳ vọng (chưa đối chiếu config)', pd.DataFrame(B2_GROUP_SPEC))
b2_table('2.1. Quy tắc Q và dự báo', [
    {'item': 'Q inputs', 'rule': 'qmean_mmd, qmax_mmd; đã là mm/ngày, không đổi đơn vị lần hai'},
    {'item': 'Q availability', 'rule': 'Chỉ hindcast, cửa sổ phải kết thúc t-1; kiểm ngày thật ở phần sau'},
    {'item': 'Target', 'rule': 'qmax tách khỏi đầu vào để không xóa Q quá khứ trong warmup'},
    {'item': 'Forecast', 'rule': 'Chỉ ECMWF HRES; cách (b) dùng first_day + masked_mean + counter'},
    {'item': 'Normalization', 'rule': 'Giữ giao thức B1b/f229e02; không dùng thay đổi scaler của prototype'},
])
print('Phần 2.1 đạt: đã định nghĩa hợp đồng và hàm kiểm; chưa chọn config hoặc xác nhận dữ liệu thực tế.')

# %% Phần 2.2 - Chọn config nền và tạo hợp đồng kiểm B2
# Chọn mốc tác giả nhúng đơn giản, so bản sao A2 và tạo template; chưa nhận bàn giao B1b.
AUTHOR_RUN_NAME = 'Sequential_Forecast_LSTM_RA_with_q_Simple_Embedding'
RUNTIME_CONFIG_KEYS = {'data_dir', 'run_dir', 'train_dir', 'img_log_dir', 'train_basin_file',
    'validation_basin_file', 'test_basin_file', 'device', 'num_workers', 'verbose', 'experiment_name',
    'commit_hash', 'package_version', 'number_of_basins'}
AUTHOR_EXPECTED = {'model': 'sequential_forecast_lstm', 'dataset': 'lamah_a', 'seed': 111,
    'hidden_size': 128, 'batch_size': 256, 'epochs': 30, 'learning_rate': 0.001,
    'output_dropout': 0.3, 'initial_forget_bias': 3, 'seq_length': 365,
    'forecast_seq_length': 1, 'predict_last_n': 1, 'early_stopping_patience': 5,
    'early_stopping_min_delta': 0.005, 'early_stopping_metric': 'NSE', 'loss': 'NSE',
    'clip_gradient_norm': 1, 'target_variables': ['qmax'], 'optimizer': 'Adam',
    'head': 'regression', 'output_activation': 'linear'}


def b2_config_date(value):
    if value is None:
        raise ValueError('Config thiếu ngày bắt đầu/kết thúc kỳ.')
    if isinstance(value, str) and '/' in value:
        return pd.to_datetime(value, format='%d/%m/%Y')
    return pd.Timestamp(value)


def b2_config_value(key, value):
    # Giống Config._parse_config ở f229e02: chuỗi "None" chỉ là giá trị rỗng ở khóa đường dẫn.
    if key.endswith(('_dir', '_path', '_file', '_files')):
        if value is None or value == 'None':
            return None
        if isinstance(value, list):
            return [Path(element) for element in value]
        return Path(value)
    return value


def b2_scientific_signature(cfg):
    selected = {key: b2_config_value(key, copy.deepcopy(value))
                for key, value in cfg.items() if key not in RUNTIME_CONFIG_KEYS}
    for period in PERIODS:
        for suffix in ('_start_date', '_end_date'):
            key = period + suffix
            selected[key] = b2_config_date(cfg.get(key)).isoformat()
    return json.dumps(selected, sort_keys=True, ensure_ascii=False, default=str)


def b2_swap_q(variables):
    variables = b2_flat(variables)
    if variables.count('qmean') != 1 or any(name in variables for name in B2_Q_INPUTS + ('qmax', 'qmax_past')):
        raise ValueError('Mốc tác giả phải có đúng qmean m³/s và chưa có hai cột Q của B1b.')
    position = variables.index('qmean')
    return variables[:position] + list(B2_Q_INPUTS) + variables[position + 1:]


simple_paths = sorted(Path(path) for path, cfg in CONFIGS.items()
    if cfg.get('model') == 'sequential_forecast_lstm'
    and Path(path).parent.name in (AUTHOR_RUN_NAME, 'SeqLSTM_Q')
    and 'qmean' in b2_flat(cfg.get('hindcast_inputs')))
if not simple_paths:
    raise FileNotFoundError('Không thấy config SeqLSTM có Q nhúng đơn giản của tác giả hoặc bản sao A2.')
author_paths = [path for path in simple_paths if path.parent.name == AUTHOR_RUN_NAME]
if not author_paths:
    raise FileNotFoundError('Cần config tác giả trong Experiments để đối chiếu bản sao A2; không thay bằng nhúng Complex.')
canonical = ID_SOURCE.parent / 'Sequential_Forecast_LSTM' / 'Sequential_Forecast_LSTM_with_q' / AUTHOR_RUN_NAME / 'config.yml'
if canonical in author_paths:
    AUTHOR_CONFIG_PATH = canonical
elif len(author_paths) == 1:
    AUTHOR_CONFIG_PATH = author_paths[0]
else:
    raise ValueError('Có nhiều nguồn config tác giả; cần nguồn Experiments đi cùng danh sách lưu vực.')
AUTHOR_CONFIG = copy.deepcopy(CONFIGS[str(AUTHOR_CONFIG_PATH)])
author_checks = [{'key': key, 'expected': str(expected), 'observed': str(AUTHOR_CONFIG.get(key)),
                  'passed': AUTHOR_CONFIG.get(key) == expected} for key, expected in AUTHOR_EXPECTED.items()]
for key in ('dynamics_embedding', 'statics_embedding'):
    embedding = AUTHOR_CONFIG.get(key)
    expected_embedding = {'type': 'fc', 'hiddens': [16], 'activation': 'tanh', 'dropout': 0.0}
    author_checks.append({'key': key, 'expected': str(expected_embedding), 'observed': str(embedding),
                          'passed': embedding == expected_embedding})
scheduler = AUTHOR_CONFIG.get('lr_scheduler', {})
author_checks.append({'key': 'lr_scheduler', 'expected': 'CosineAnnealingLR, T_max=30, eta_min=1e-5',
    'observed': str(scheduler), 'passed': isinstance(scheduler, dict)
        and str(scheduler.get('type')).lower() == 'cosineannealinglr'
        and scheduler.get('T_max') == 30 and scheduler.get('eta_min') == 1e-5})
if not all(row['passed'] for row in author_checks):
    b2_table('2.2. Kiểm mốc tác giả', pd.DataFrame(author_checks))
    raise ValueError('Config được nhận diện chưa khớp mốc tác giả SeqLSTM có Q nhúng đơn giản.')
for period, (start, end) in PERIODS.items():
    for suffix, expected in (('_start_date', start), ('_end_date', end)):
        if b2_config_date(AUTHOR_CONFIG.get(period + suffix)) != pd.Timestamp(expected):
            raise ValueError(f'Config tác giả khác kỳ {period}{suffix} đã cố định.')
if AUTHOR_CONFIG.get('forecast_overlap') not in (None, 0):
    raise ValueError('Config tác giả có forecast_overlap khác zero.')
for key in ('lagged_features', 'autoregressive_inputs', 'evolving_attributes', 'mass_inputs',
            'dynamic_conceptual_inputs', 'additional_feature_files', 'hydroatlas_attributes',
            'per_basin_train_periods_file', 'per_basin_validation_periods_file', 'per_basin_test_periods_file',
            'use_basin_id_encoding', 'nan_handling_method', 'nan_step_probability', 'nan_sequence_probability'):
    value = AUTHOR_CONFIG.get(key)
    if b2_config_value(key, value):
        raise ValueError(f'Mốc tác giả có thiết lập ngoài hợp đồng hiện tại: {key}={value!r}')
reference_signature = b2_scientific_signature(AUTHOR_CONFIG)
copy_rows = []
for path in simple_paths:
    candidate = CONFIGS[str(path)]
    matched = b2_scientific_signature(candidate) == reference_signature
    copy_rows.append({'source': b2_input_path(path), 'role': 'author' if path in author_paths else 'A2_copy',
                      'sha256': b2_hash(path), 'scientific_match': matched})
if not all(row['scientific_match'] for row in copy_rows):
    b2_table('2.2. Đối chiếu config nguồn', pd.DataFrame(copy_rows))
    raise ValueError('Các bản config nhúng đơn giản khác thiết lập khoa học; chưa chọn nguồn.')
config_path = Path(CONFIG_OVERRIDE).resolve() if CONFIG_OVERRIDE else AUTHOR_CONFIG_PATH
if config_path not in simple_paths:
    raise ValueError('Config được chỉ định không thuộc các bản SeqLSTM có Q nhúng đơn giản đã đối chiếu.')
BASE_CONFIG = copy.deepcopy(CONFIGS[str(config_path)])
BASE_HINDCAST = b2_flat(BASE_CONFIG.get('hindcast_inputs'))
FORECAST = b2_flat(BASE_CONFIG.get('forecast_inputs'))
base_dynamic = b2_flat(BASE_CONFIG.get('dynamic_inputs'))
STATIC = b2_flat(BASE_CONFIG.get('static_attributes'))
if (len(BASE_HINDCAST) != 32 or BASE_HINDCAST.count('qmean') != 1 or len(FORECAST) != 5
        or base_dynamic != BASE_HINDCAST + FORECAST or len(set(base_dynamic)) != 37
        or len(STATIC) != 33 or len(set(STATIC)) != 33):
    raise ValueError('Mốc tác giả cần 32/5/33 biến, không trùng và dynamic đúng thứ tự hindcast + forecast.')
past = [variable for variable in BASE_HINDCAST if variable != 'qmean']
B1B_EXPECTED_CONFIG = copy.deepcopy(BASE_CONFIG)
B1B_EXPECTED_CONFIG.update({'dynamic_inputs': b2_swap_q(base_dynamic), 'hindcast_inputs': b2_swap_q(BASE_HINDCAST),
                           'early_stopping_patience': 30, 'data_on_gpu': True})
HINDCAST = b2_flat(B1B_EXPECTED_CONFIG['hindcast_inputs'])
GROUPS_H, GROUPS_F = b2_groups(HINDCAST), b2_groups(FORECAST)
B2_CONFIG_TEMPLATE = copy.deepcopy(B1B_EXPECTED_CONFIG)
B2_CONFIG_TEMPLATE.update({'dynamic_inputs': GROUPS_H + GROUPS_F, 'hindcast_inputs': GROUPS_H,
    'forecast_inputs': GROUPS_F, 'nan_handling_method': 'masked_mean', 'nan_step_probability': 0.0,
    'nan_sequence_probability': 0.0, 'timestep_counter': False, 'forecast_input_mode': 'all'})
dynamic_checked, hindcast_checked, forecast_checked = b2_contract(B2_CONFIG_TEMPLATE)
if hindcast_checked != HINDCAST or forecast_checked != FORECAST:
    raise ValueError('Nhóm nguồn đã làm thay thứ tự biến trong một pha.')
EXPECTED_GROUPS = {row['source']: row['expected_variables'] for row in B2_GROUP_SPEC}
actual_groups = {b2_source(group[0]): len(group) for group in GROUPS_H + GROUPS_F}
if actual_groups != EXPECTED_GROUPS or len(dynamic_checked) != 38:
    raise ValueError(f'Nhóm nguồn không khớp hợp đồng: {actual_groups}')
allowed_changes = {'dynamic_inputs', 'hindcast_inputs', 'forecast_inputs', 'nan_handling_method',
                   'nan_step_probability', 'nan_sequence_probability', 'timestep_counter', 'forecast_input_mode'}
if any(B2_CONFIG_TEMPLATE.get(key) != value for key, value in B1B_EXPECTED_CONFIG.items() if key not in allowed_changes):
    raise ValueError('Template B2 đã đổi thiết lập ngoài phạm vi nhóm nguồn/masked mean.')
q_source_columns = dict(zip(B2_Q_INPUTS, ('qmean', 'qmax')))
GROUP_MAP = pd.DataFrame([{'phase': phase, 'group': b2_source(variable), 'variable': variable,
    'unit': 'mm/day_area_gov' if variable in B2_Q_INPUTS else 'from_raw_dataset',
    'last_available_date': 't-1' if phase == 'hindcast' else 't (lead 1)',
    'source_column': q_source_columns.get(variable, variable)}
    for phase, variables in [('hindcast', HINDCAST), ('forecast', FORECAST)] for variable in variables])
for row, path in zip(copy_rows, simple_paths):
    row['selected'] = path == config_path
b2_save('B2_config_source_check.csv', pd.DataFrame(copy_rows))
b2_save('B2_author_config_check.csv', pd.DataFrame(author_checks))
b2_save('B2_variable_contract.csv', GROUP_MAP)
b2_json(OUT / 'B2_base_config.json', BASE_CONFIG)
b2_json(OUT / 'B2_B1b_expected_config.json', B1B_EXPECTED_CONFIG)
b2_json(OUT / 'B2_config_template.json', B2_CONFIG_TEMPLATE)
b2_json(OUT / 'B2_source_contract.json', {'library_base_commit': BASE_COMMIT,
    'author_config_source': str(AUTHOR_CONFIG_PATH), 'config_source': str(config_path),
    'config_sha256': b2_hash(config_path), 'scientific_signature_sha256': hashlib.sha256(reference_signature.encode('utf-8')).hexdigest(),
    'author_config_identity_verified': True, 'simple_source_copies_scientifically_equal': True,
    'derived_B2_input_contract_verified': True,
    'raw': str(RAW), 'basin_ids': IDS, 'Q_inputs': B2_Q_INPUTS, 'Q_unit_expected': 'mm/day_area_gov',
    'Q_origin': 'D_gauges; B1b conversion once; hindcast through t-1',
    'B1b_handoff_verified': False, 'Q_columns_verified_against_raw': False,
    'template_only': True, 'runtime_paths_prepared': False, 'official_training_started': False,
    'planned_seeds': [111, 222, 333], 'periods': PERIODS})
b2_table('2.2. Đối chiếu config nguồn', pd.DataFrame(copy_rows))
b2_table('2.2. Kiểm mốc tác giả', pd.DataFrame(author_checks))
b2_table('2.2. Nhóm nguồn thực tế từ config và quy tắc B1b', GROUP_MAP.groupby(
    ['phase', 'group'], sort=False).size().rename('variables').reset_index())
b2_table('2.2. Hợp đồng biến', GROUP_MAP)
b2_table('2.2. Phạm vi template', [
    {'item': 'training protocol', 'value': '30 epoch; best validation NSE; data_on_gpu=True'},
    {'item': 'B2 architecture', 'value': 'Giữ hidden_size=128 và hai embedding FC 16 chiều; thêm nhóm nguồn và masked mean'},
    {'item': 'Q inputs', 'value': 'qmean_mmd, qmax_mmd; chưa kiểm cột dữ liệu thật'},
    {'item': 'B1b handoff', 'value': 'Chưa xác nhận; config kỳ vọng được suy ra từ mốc và quy tắc đã công bố'},
    {'item': 'training', 'value': 'Chưa chuẩn bị đường dẫn chạy, chưa fit scaler và chưa huấn luyện'},
])
print('Phần 2.2 đạt: mốc 32/5/33; template B2 có 33/5/33 biến, 38 đầu vào động; chưa xác nhận bàn giao B1b.')

# %% Phần 2.3 - Đọc diện tích, thuộc tính và cố định đường dẫn
# Kiểm đúng tập ID/thứ tự đã chọn; chưa đọc chuỗi ngày, đổi Q hoặc fit scaler.
import csv
from neuralhydrology.datasetzoo.lamah import load_lamah_attributes as b2_nh_attributes

b2_json(OUT / 'B2_attribute_contract.json', {'complete': False, 'stage': 'attribute_and_path_checks'})
ATTRIBUTE_PATHS = (RAW / 'A_basins_total_upstrm/1_attributes/Catchment_attributes.csv',
                   RAW / 'D_gauges/1_attributes/Gauge_attributes.csv')


def b2_id(value):
    token = str(value)
    if not token.isascii() or not token.isdecimal() or int(token) <= 0:
        raise ValueError(f'ID không phải số nguyên dương: {value!r}')
    return str(int(token))


def b2_file_map(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Thiếu thư mục chuỗi ngày: {folder}')
    result = {}
    for path in sorted(folder.glob('ID_*.csv')):
        if not path.is_file():
            raise ValueError(f'Đường dẫn CSV không phải tệp: {path}')
        key = b2_id(path.stem[3:])
        if key in result:
            raise ValueError(f'ID trùng tệp sau chuẩn hóa: {key}, {result[key].name}, {path.name}')
        result[key] = path
    return result


def b2_attributes(raw):
    tables = []
    for path in (Path(raw) / 'A_basins_total_upstrm/1_attributes/Catchment_attributes.csv',
                 Path(raw) / 'D_gauges/1_attributes/Gauge_attributes.csv'):
        if not path.is_file():
            raise FileNotFoundError(f'Thiếu tệp thuộc tính: {path}')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream, delimiter=';'), [])
        if header.count('ID') != 1 or len(header) != len(set(header)):
            raise ValueError(f'Header thuộc tính thiếu ID hoặc trùng cột: {path.name}')
        frame = pd.read_csv(path, sep=';', dtype={'ID': str}).set_index('ID')
        frame.index = frame.index.map(b2_id)
        if frame.index.has_duplicates:
            raise ValueError(f'ID thuộc tính trùng sau chuẩn hóa: {path.name}')
        missing = [basin for basin in IDS if basin not in frame.index]
        if missing:
            raise ValueError(f'{path.name} thiếu {len(missing)} ID đã chọn: {missing[:12]}')
        tables.append(frame)
    result = pd.concat(tables, axis=1)
    if result.columns.has_duplicates:
        raise ValueError('Tên thuộc tính trùng giữa tệp lưu vực và tệp trạm.')
    return result


ATTRIBUTES = b2_attributes(RAW).loc[list(IDS)]
missing_fields = [field for field in ['area_gov', *STATIC] if field not in ATTRIBUTES.columns]
if missing_fields:
    raise ValueError(f'Thiếu trường diện tích/thuộc tính theo config: {missing_fields}')
AREAS = pd.to_numeric(ATTRIBUTES['area_gov'], errors='raise')
if not (np.isfinite(AREAS) & AREAS.gt(0)).all():
    bad = AREAS.index[~(np.isfinite(AREAS) & AREAS.gt(0))].tolist()
    raise ValueError(f'area_gov không hữu hạn/dương ở ID: {bad[:12]}')
STATIC_RAW = ATTRIBUTES[STATIC].apply(pd.to_numeric, errors='raise').astype('float64')
bad_static = ~np.isfinite(STATIC_RAW.to_numpy()) | STATIC_RAW.eq(-999).to_numpy()
if bad_static.any():
    rows, columns = np.where(bad_static)
    examples = [(str(STATIC_RAW.index[row]), STATIC[column]) for row, column in zip(rows[:12], columns[:12])]
    raise ValueError(f'Thuộc tính tĩnh thiếu/vô cực/-999 tại (ID, trường): {examples}')
# Đối chiếu giá trị với chính bộ đọc mã nền; không sắp lại danh sách STATIC theo thứ tự tensor.
library_attributes = b2_nh_attributes(RAW, sub_dataset=BASE_CONFIG['dataset'], basins=list(IDS))
fields = ['area_gov', *STATIC]
if (list(library_attributes.index) != list(IDS)
        or not np.array_equal(ATTRIBUTES[fields].to_numpy(dtype=float),
                              library_attributes[fields].to_numpy(dtype=float))):
    raise ValueError('Diện tích/thuộc tính hoặc thứ tự ID khác bộ đọc LamaH của mã nền.')
FORCING_FILES = b2_file_map(RAW / 'A_basins_total_upstrm/2_timeseries/daily')
Q_FILES = b2_file_map(RAW / 'D_gauges/2_timeseries/daily')
for label, files in [('forcing', FORCING_FILES), ('gauge', Q_FILES)]:
    missing = [basin for basin in IDS if basin not in files]
    if missing:
        raise FileNotFoundError(f'Thiếu {len(missing)} tệp {label} cho ID: {missing[:12]}')
    noncanonical = [files[basin].name for basin in IDS if files[basin].name != f'ID_{basin}.csv']
    if noncanonical:
        raise ValueError(f'Tên tệp {label} khác quy tắc bộ đọc thư viện: {noncanonical[:12]}')
FORCING_FILES = {basin: FORCING_FILES[basin] for basin in IDS}
Q_FILES = {basin: Q_FILES[basin] for basin in IDS}
attribute_sources = pd.DataFrame([{'file': path.name, 'sha256': b2_hash(path),
    'selected_IDs': len(IDS), 'path': b2_input_path(path)} for path in ATTRIBUTE_PATHS])
static_checks = pd.DataFrame([{'variable': field, 'IDs': len(IDS),
    'finite': int(np.isfinite(STATIC_RAW[field]).sum()), 'missing_code_minus999': int(STATIC_RAW[field].eq(-999).sum()),
    'min': STATIC_RAW[field].min(), 'median': STATIC_RAW[field].median(), 'max': STATIC_RAW[field].max()}
    for field in STATIC])
basin_attributes = STATIC_RAW.copy()
basin_attributes.insert(0, 'area_gov', AREAS)
basin_attributes.index.name = 'basin_id'
file_index = pd.DataFrame([{'basin_id': basin, 'forcing_file': str(FORCING_FILES[basin]),
                          'gauge_file': str(Q_FILES[basin])} for basin in IDS])
b2_save('B2_attribute_sources.csv', attribute_sources)
b2_save('B2_static_field_check.csv', static_checks)
b2_save('B2_basin_attributes.csv', basin_attributes.reset_index())
b2_save('B2_data_file_index.csv', file_index)
b2_json(OUT / 'B2_attribute_contract.json', {'complete': True, 'stage': 'attribute_and_path_checks',
    'IDs': len(IDS), 'static_fields': len(STATIC), 'basin_order_preserved': True,
    'area_field': 'area_gov', 'area_unit': 'km2', 'library_attribute_values_matched': True,
    'Q_conversion_reference': 'Q_m3_s / (area_gov_km2 * 1e6) * 1000 * 86400',
    'forcing_files_indexed': len(FORCING_FILES), 'gauge_files_indexed': len(Q_FILES),
    'daily_series_values_checked': False, 'Q_converted': False, 'scaler_fitted': False,
    'B1b_handoff_verified': False, 'official_training_started': False})
b2_table('2.3. Diện tích và thuộc tính', [{'IDs': len(IDS), 'static_fields': len(STATIC),
    'area_min_km2': AREAS.min(), 'area_median_km2': AREAS.median(), 'area_max_km2': AREAS.max(),
    'forcing_files': len(FORCING_FILES), 'gauge_files': len(Q_FILES), 'library_values_matched': True}])
b2_table('2.3. Nguồn thuộc tính', attribute_sources)
b2_table('2.3. Kiểm 33 thuộc tính theo config', static_checks)
print(f'Phần 2.3 đạt: {len(IDS)} ID, {len(STATIC)} thuộc tính, area_gov dương/hữu hạn; '
      'khớp bộ đọc thư viện; chưa đọc chuỗi ngày, đổi Q hoặc fit scaler.')

# %% Phần 3.1 - Đọc hàm lọc gốc và xác lập quy tắc đối chiếu
# Chỉ bỏ decorator JIT khi tách hàm; chưa áp bản sửa bộ nạp hoặc chạy dữ liệu thật.
validate_source = BASE_TEXT['datasetzoo/basedataset.py']
validate_source_hash = hashlib.sha256(validate_source.encode('utf-8')).hexdigest()
if validate_source_hash != BASE_HASHES['datasetzoo/basedataset.py']:
    raise RuntimeError('Nguồn bộ lọc khác tệp basedataset.py đã kiểm ở Phần 1.2.')
base_ast = ast.parse(validate_source)
validate_nodes = [n for n in base_ast.body if isinstance(n, ast.FunctionDef) and n.name == '_validate_samples']
if len(validate_nodes) != 1:
    raise ValueError('Mã nền cần đúng một hàm _validate_samples ở cấp module.')
node = copy.deepcopy(validate_nodes[0])
validate_arguments = ['x_d', 'x_s', 'y', 'seq_length', 'predict_last_n', 'frequency_maps']
if [argument.arg for argument in node.args.args] != validate_arguments:
    raise ValueError('Chữ ký _validate_samples khác giao diện f229e02 đã đối chiếu.')
validate_function_hash = hashlib.sha256(ast.get_source_segment(validate_source, node).encode('utf-8')).hexdigest()
node.decorator_list = []
namespace = {'np': np, 'prange': range, 'List': list}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<pinned_NH_validate_samples>', 'exec'), namespace)
ORIGINAL_VALIDATE = namespace['_validate_samples']


def b2_filter(x, y, length, *, masked=False, statics=None, training=True, predict_last_n=1):
    # Wrapper cho ví dụ một tần suất: x_s là thuộc tính biến thiên theo thời gian, không phải vector tĩnh toàn lưu vực.
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError('Fixture cần x[time, features], y[time] có cùng số ngày.')
    for name, value in [('length', length), ('predict_last_n', predict_last_n)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} phải là số nguyên dương.')
    if predict_last_n > length:
        raise ValueError('predict_last_n không được lớn hơn length.')
    if not isinstance(masked, (bool, np.bool_)) or not isinstance(training, (bool, np.bool_)):
        raise ValueError('masked/training phải là bool.')
    if statics is not None:
        statics = np.asarray(statics, dtype=float)
        if statics.ndim != 2 or len(statics) != len(y):
            raise ValueError('statics của fixture cần dạng [time, features], cùng số ngày với y.')
    return ORIGINAL_VALIDATE(x_d=[x] if training and not masked else None,
        x_s=[statics] if training and statics is not None else None,
        y=[y[:, None]] if training else None, seq_length=[length], predict_last_n=[predict_last_n],
        frequency_maps=[np.arange(len(y))]).astype(bool)


def b2_assert(condition, message='Kiểm không đạt'):
    if not condition:
        raise AssertionError(message)


filter_rules = pd.DataFrame([
    {'mode': 'train_original', 'dynamic_inputs': 'Reject any NaN in full window',
     'target': 'Reject all-NaN prediction window', 'time_varying_attributes': 'Reject any NaN at sample end',
     'history': 'Require full sequence'},
    {'mode': 'train_B2_fixture', 'dynamic_inputs': 'Allow NaN; x_d=None in validator',
     'target': 'Reject all-NaN prediction window', 'time_varying_attributes': 'Reject any NaN at sample end',
     'history': 'Require full sequence'},
    {'mode': 'validation_test', 'dynamic_inputs': 'Not checked when building lookup',
     'target': 'Not required when building lookup', 'time_varying_attributes': 'Not checked when building lookup',
     'history': 'Require full sequence'},
])
filter_reference = {'library_base_commit': BASE_COMMIT, 'source': 'datasetzoo/basedataset.py',
    'source_sha256_lf': validate_source_hash, 'function': ORIGINAL_VALIDATE.__name__,
    'function_sha256_lf': validate_function_hash, 'arguments': validate_arguments,
    'body_preserved': True, 'execution_mode': 'isolated_python_without_JIT',
    'fixture_scope': 'single_daily_frequency_single_target', 'rules': filter_rules.to_dict('records'),
    'fixed_basin_attributes': 'Checked separately by BaseDataset; section 2.3 checked raw selected attributes',
    'B2_loader_patch_applied': False, 'real_lookup_verified': False, 'official_training_started': False}
b2_json(OUT / 'B2_filter_reference.json', filter_reference)
b2_save('B2_filter_rules.csv', filter_rules)
b2_table('3.1. Nguồn hàm lọc', [{'commit': BASE_COMMIT[:7], 'function': ORIGINAL_VALIDATE.__name__,
    'source_sha256_lf': validate_source_hash, 'function_sha256_lf': validate_function_hash,
    'body_preserved': True, 'execution': 'Python; bỏ decorator JIT'}])
b2_table('3.1. Quy tắc lập lookup để đối chiếu', filter_rules)
print('Phần 3.1 đạt: đã tách nguyên thân hàm _validate_samples từ mã nền; '
      'mới định nghĩa wrapper đối chiếu, chưa kiểm ví dụ 3.2 hoặc lookup dữ liệu thật.')

# %% Phần 3.2 - Kiểm các mặt nạ giữ và loại mẫu biết trước
# Dữ liệu giả với kỳ vọng viết độc lập; kiểm wrapper, chưa xác nhận bộ nạp B2 thật.
b2_json(OUT / 'B2_filter_fixture_summary.json', {'complete': False, 'stage': 'synthetic_filter_fixtures'})
CHECKS[:] = [row for row in CHECKS if row['group'] != 'fixture_filter']
fixture_x = np.ones((7, 2), dtype=float)
fixture_y = np.arange(1, 8, dtype=float)
fixture_x[3, 0] = np.nan
fixture_y[5] = np.nan
fixture_static = np.ones((7, 1))
fixture_static[6] = np.nan
fixture_multi_y = np.array([1., np.nan, 2., np.nan, np.nan, 4., np.nan])
fixture_past_static = np.ones((7, 1))
fixture_past_static[3] = np.nan
fixture_arrays = [fixture_x, fixture_y, fixture_static, fixture_multi_y, fixture_past_static]
fixture_snapshots = [array.copy() for array in fixture_arrays]
old_mask = b2_filter(fixture_x, fixture_y, 3)
new_mask = b2_filter(fixture_x, fixture_y, 3, masked=True)
static_mask = b2_filter(fixture_x, fixture_y, 3, masked=True, statics=fixture_static)
eval_mask = b2_filter(fixture_x, fixture_y, 3, masked=True, statics=fixture_static, training=False)
fixture_cases = [
    ('original_exact_mask', old_mask, [0, 0, 1, 0, 0, 0, 1], 3, 1),
    ('masked_exact_mask', new_mask, [0, 0, 1, 1, 1, 0, 1], 3, 1),
    ('time_varying_static_exact_mask', static_mask, [0, 0, 1, 1, 1, 0, 0], 3, 1),
    ('evaluation_exact_mask', eval_mask, [0, 0, 1, 1, 1, 1, 1], 3, 1),
    ('all_dynamic_missing_masked', b2_filter(np.full((7, 2), np.nan), np.ones(7), 3, masked=True),
     [0, 0, 1, 1, 1, 1, 1], 3, 1),
    ('all_dynamic_missing_original', b2_filter(np.full((7, 2), np.nan), np.ones(7), 3),
     [0, 0, 0, 0, 0, 0, 0], 3, 1),
    ('all_targets_missing_train', b2_filter(np.ones((7, 2)), np.full(7, np.nan), 3, masked=True),
     [0, 0, 0, 0, 0, 0, 0], 3, 1),
    ('all_data_missing_evaluation', b2_filter(np.full((7, 2), np.nan), np.full(7, np.nan), 3,
        statics=np.full((7, 1), np.nan), training=False), [0, 0, 1, 1, 1, 1, 1], 3, 1),
    ('insufficient_history', b2_filter(np.ones((2, 2)), np.ones(2), 3, masked=True), [0, 0], 3, 1),
    ('multi_day_partial_targets', b2_filter(np.ones((7, 2)), fixture_multi_y, 3,
        masked=True, predict_last_n=2), [0, 0, 1, 1, 0, 1, 1], 3, 2),
    ('time_varying_static_at_sample_end', b2_filter(np.ones((7, 2)), np.ones(7), 3,
        masked=True, statics=fixture_past_static), [0, 0, 1, 0, 1, 1, 1], 3, 1),
]
fixture_mask_rows = []
for case, actual, expected, length, horizon in fixture_cases:
    expected = np.asarray(expected, dtype=bool)
    expected_text = ''.join(str(int(value)) for value in expected)
    observed_text = ''.join(str(int(value)) for value in actual)
    passed = b2_check(case, lambda a=actual, e=expected, name=case: b2_assert(np.array_equal(a, e),
        f'{name}: expected={e.astype(int).tolist()}, observed={a.astype(int).tolist()}'), 'fixture_filter')
    fixture_mask_rows.append({'case': case, 'length': length, 'predict_last_n': horizon,
        'expected_mask': expected_text, 'observed_mask': observed_text, 'passed': passed})
b2_check('fixture_arrays_preserved', lambda: b2_assert(all(np.array_equal(before, after, equal_nan=True)
    for before, after in zip(fixture_snapshots, fixture_arrays)), 'Wrapper đã sửa mảng fixture đầu vào.'), 'fixture_filter')
fixture_daily = pd.DataFrame({'index': range(7), 'input_has_NaN': np.isnan(fixture_x).any(axis=1),
    'label': fixture_y, 'time_varying_static_has_NaN': np.isnan(fixture_static).any(axis=1),
    'original': old_mask, 'masked_mean': new_mask, 'with_time_varying_static_check': static_mask, 'evaluation': eval_mask})
b2_save('B2_filter_fixture_days.csv', fixture_daily)
b2_save('B2_filter_fixture_masks.csv', pd.DataFrame(fixture_mask_rows))
b2_table('3.2. Ví dụ 7 ngày giả; True là giữ mẫu', fixture_daily)
b2_require('fixture_filter')
fixture_check_count = sum(row['group'] == 'fixture_filter' for row in CHECKS)
b2_json(OUT / 'B2_filter_fixture_summary.json', {'complete': True, 'synthetic_data_only': True,
    'source_sha256_lf': validate_source_hash, 'function_sha256_lf': validate_function_hash,
    'mask_cases': len(fixture_cases), 'checks_passed': fixture_check_count,
    'partial_target_rule': 'retain_if_any_target_in_prediction_window_is_not_NaN',
    'fixture_arrays_preserved': True, 'B2_loader_patch_applied': False,
    'real_lookup_verified': False, 'official_training_started': False})
print(f'Phần 3.2 đạt: {fixture_check_count}/{fixture_check_count} kiểm, {len(fixture_cases)} mặt nạ đúng kỳ vọng; '
      'chỉ dữ liệu giả và wrapper; chưa áp patch B2 hoặc kiểm lookup dữ liệu thật.')

# %% Phần 4.1 - Đọc ngày và kiểm Q tham chiếu trên một ID
# Đổi Q tham chiếu từ D_gauges bằng hàm mã nền; không sửa bộ nạp hoặc tệp forcing.
from neuralhydrology.datasetzoo.lamah import load_lamah_discharge as b2_nh_discharge
from neuralhydrology.datasetzoo.lamah import _normalize_discharge as b2_nh_q_normalize

b2_json(OUT / 'B2_daily_reference_summary.json', {'complete': False, 'stage': 'single_basin_daily_reference'})


def b2_read_daily(path, columns):
    path = Path(path)
    columns = b2_flat(columns)
    date_columns = ['YYYY', 'MM', 'DD']
    if not columns or len(columns) != len(set(columns)) or set(columns).intersection(date_columns):
        raise ValueError('Danh sách biến ngày cần khác rỗng, duy nhất và không chứa YYYY/MM/DD.')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream, delimiter=';'), [])
    if len(header) != len(set(header)) or not set(date_columns + columns).issubset(header):
        raise ValueError(f'Header ngày trùng hoặc thiếu cột yêu cầu: {path.name}')
    raw = pd.read_csv(path, sep=';', usecols=date_columns + columns)
    if raw.empty:
        raise ValueError(f'Tệp ngày không có dòng dữ liệu: {path.name}')
    parts = raw[['YYYY', 'MM', 'DD']].apply(pd.to_numeric, errors='raise')
    date_values = parts.to_numpy(dtype=float)
    if not np.isfinite(date_values).all() or not np.equal(date_values, np.floor(date_values)).all():
        raise ValueError(f'Thành phần ngày thiếu, vô hạn hoặc không nguyên: {path.name}')
    dates = pd.DatetimeIndex(pd.to_datetime(parts.astype('int64').rename(
        columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}), errors='raise'), name='date')
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f'Ngày trùng hoặc không tăng: {path}')
    values = raw[columns].apply(pd.to_numeric, errors='raise')
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError(f'Giá trị vô cực: {path}')
    values.index = dates
    return values


def b2_q_reference(data_dir, basin, area):
    # Chỉ đọc Q m³/s từ gauge; tuyệt đối không lấy cột forcing *_mmd để đổi đơn vị.
    basin, area = b2_id(basin), float(area)
    if not np.isfinite(area) or area <= 0:
        raise ValueError(f'area_gov không hợp lệ: {basin}')
    gauge_path = Path(data_dir) / 'D_gauges/2_timeseries/daily' / f'ID_{basin}.csv'
    gauge = b2_read_daily(gauge_path, ['qmean', 'qmax'])
    reference, rows = pd.DataFrame(index=gauge.index), []
    for variable, source in zip(B2_Q_INPUTS, ('qmean', 'qmax')):
        cleaned = gauge[source].where(gauge[source].ge(0))
        library = b2_nh_discharge(Path(data_dir), basin, temporal_resolution='1D',
                                  normalize_discharge=False, target_name=source)[source]
        if (not library.index.equals(gauge.index)
                or not np.array_equal(cleaned.to_numpy(), library.to_numpy(), equal_nan=True)):
            raise ValueError(f'Q gốc/lịch hoặc quy tắc Q âm khác mã nền: {basin}/{source}')
        reference[variable] = b2_nh_q_normalize(ser=library, area=area, temporal_resolution='1D')
        if np.isinf(reference[variable].to_numpy()).any():
            raise ValueError(f'Q tham chiếu vô cực sau đổi đơn vị: {basin}/{source}')
        rows.append({'variable': variable, 'source': source, 'unit': 'mm/day_area_gov',
            'raw_negative': int(gauge[source].lt(0).sum()), 'raw_missing': int(gauge[source].isna().sum()),
            'reference_finite': int(np.isfinite(reference[variable]).sum()), 'raw_cleaning_matched_library': True})
    reference['qmax'] = reference['qmax_mmd'].copy()
    rows.append(dict(rows[1], variable='qmax'))
    return reference, pd.DataFrame(rows)


daily_probe_basin = IDS[0]
daily_probe_weather = [name for name in HINDCAST + FORECAST if name not in B2_Q_INPUTS]
daily_probe_forcing = b2_read_daily(FORCING_FILES[daily_probe_basin], daily_probe_weather)
daily_probe_q, daily_probe_q_check = b2_q_reference(RAW, daily_probe_basin, AREAS.loc[daily_probe_basin])
daily_probe_aligned_q = daily_probe_q.reindex(daily_probe_forcing.index)
daily_probe_sources = pd.DataFrame([{'kind': kind, 'basin_id': daily_probe_basin,
    'rows': len(frame), 'first_date': frame.index.min().strftime('%Y-%m-%d'),
    'last_date': frame.index.max().strftime('%Y-%m-%d'), 'sha256': b2_hash(path)}
    for kind, frame, path in [('forcing', daily_probe_forcing, FORCING_FILES[daily_probe_basin]),
                              ('gauge', daily_probe_q, Q_FILES[daily_probe_basin])]])
daily_probe_q_check['aligned_finite'] = [int(np.isfinite(daily_probe_aligned_q[name]).sum())
                                      for name in daily_probe_q_check['variable']]
b2_save('B2_daily_probe_sources.csv', daily_probe_sources)
b2_save('B2_daily_probe_Q_check.csv', daily_probe_q_check)
b2_json(OUT / 'B2_daily_reference_summary.json', {'complete': True, 'probe_ID': daily_probe_basin,
    'library_base_commit': BASE_COMMIT,
    'IDs_read': 1, 'weather_fields': len(daily_probe_weather), 'forcing_rows': len(daily_probe_forcing),
    'gauge_rows': len(daily_probe_q), 'area_gov_km2': float(AREAS.loc[daily_probe_basin]),
    'Q_reference_from_raw_gauge': True, 'Q_negative_rule': 'negative_to_NaN_as_f229e02',
    'conversion': 'library _normalize_discharge; daily; area_gov', 'Q_inputs': list(B2_Q_INPUTS),
    'target': 'qmax; separate column, same values as qmax_mmd before warmup handling',
    'alignment': 'reindex gauge to forcing calendar; missing stays NaN',
    'raw_files_modified': False, 'B1b_handoff_verified': False, 'B1b_forcing_columns_checked': False,
    'B2_loader_patch_applied': False, 'real_lookup_verified': False, 'scaler_fitted': False,
    'official_training_started': False})
b2_table('4.1. Nguồn chuỗi ngày của một ID', daily_probe_sources)
b2_table('4.1. Q tham chiếu mm/ngày từ D_gauges', daily_probe_q_check)
print(f'Phần 4.1 đạt: đọc chuỗi ngày ID {daily_probe_basin}, {len(daily_probe_weather)} biến khí tượng và Q tham chiếu; '
      'chưa xác nhận cột forcing B1b, áp patch hoặc kiểm lookup thật.')

# %% Phần 4.2 - Khóa chuẩn hóa và std nhãn theo mã nền B1b
# Bảo vệ các phương thức gốc; chưa fit scaler hoặc áp patch B2.
from neuralhydrology.datasetzoo.basedataset import BaseDataset as B2_BASE_DATASET

b2_json(OUT / 'B2_normalization_contract.json', {'complete': False, 'stage': 'normalization_source_and_config_checks'})
NORMALIZATION_METHODS = ('_load_or_create_xarray_dataset', '_load_combined_attributes',
                         '_setup_normalization', '_calculate_per_basin_std', '_load_data')


def b2_normalization_signatures(source):
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'BaseDataset']
    if len(classes) != 1:
        raise ValueError('Cần đúng một lớp BaseDataset để kiểm chuẩn hóa.')
    signatures = {}
    for name in NORMALIZATION_METHODS:
        methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(methods) != 1:
            raise ValueError(f'Phương thức chuẩn hóa thiếu hoặc trùng: {name}')
        signatures[name] = hashlib.sha256(ast.dump(methods[0], include_attributes=False).encode('utf-8')).hexdigest()
    return signatures


normalization_source = BASE_TEXT['datasetzoo/basedataset.py']
normalization_source_hash = hashlib.sha256(normalization_source.encode('utf-8')).hexdigest()
if normalization_source_hash != BASE_HASHES['datasetzoo/basedataset.py']:
    raise RuntimeError('Nguồn kiểm chuẩn hóa khác basedataset.py đã chốt ở Phần 1.2.')
NORMALIZATION_SIGNATURES = b2_normalization_signatures(normalization_source)


def b2_require_normalization_preserved(candidate_source):
    observed = b2_normalization_signatures(candidate_source)
    changed = [name for name in NORMALIZATION_METHODS if observed[name] != NORMALIZATION_SIGNATURES[name]]
    if changed:
        raise ValueError(f'B2 đã thay phương thức cần giữ nguyên của B1b/f229e02: {changed}')
    return True


normalization_method_rows = []
for name in NORMALIZATION_METHODS:
    loaded_source = textwrap.dedent(inspect.getsource(getattr(B2_BASE_DATASET, name)))
    loaded_node = ast.parse(loaded_source).body[0]
    loaded_hash = hashlib.sha256(ast.dump(loaded_node, include_attributes=False).encode('utf-8')).hexdigest()
    matched = loaded_hash == NORMALIZATION_SIGNATURES[name]
    normalization_method_rows.append({'method': name, 'ast_sha256': NORMALIZATION_SIGNATURES[name],
                                      'loaded_method_matched': matched, 'B2_action': 'keep unchanged'})
if not all(row['loaded_method_matched'] for row in normalization_method_rows):
    raise RuntimeError('Phương thức chuẩn hóa đang nạp khác mã nền; chưa tiếp tục tích hợp B2.')
normalization_keys = ['dataset', 'target_variables', 'static_attributes', 'custom_normalization', 'loss',
    'clip_targets_to_zero', 'use_frequencies', 'seq_length', 'forecast_seq_length', 'predict_last_n',
    'forecast_overlap', 'train_data_file']
normalization_keys += [period + suffix for period in PERIODS for suffix in ('_start_date', '_end_date')]
normalization_keys += [f'per_basin_{period}_periods_file' for period in PERIODS]
normalization_config_rows = [{'key': key, 'reference': str(B1B_EXPECTED_CONFIG.get(key)),
    'B2_template': str(B2_CONFIG_TEMPLATE.get(key)),
    'passed': b2_config_value(key, B1B_EXPECTED_CONFIG.get(key)) == b2_config_value(key, B2_CONFIG_TEMPLATE.get(key))}
    for key in normalization_keys]
if not all(row['passed'] for row in normalization_config_rows):
    changed = [row['key'] for row in normalization_config_rows if not row['passed']]
    raise ValueError(f'Template B2 khác thiết lập chuẩn hóa/phạm vi B1b kỳ vọng: {changed}')
normalization_rules = pd.DataFrame([
    {'item': 'Dynamic feature scaler', 'rule': 'Train xarray includes warmup; mean/std skip NaN, ddof=0'},
    {'item': 'Warmup target', 'rule': 'Only target qmax is NaN before period start; Q input columns stay separate'},
    {'item': 'Static attributes', 'rule': 'Keep alphabetical tensor order and original mean/sample std (ddof=1)'},
    {'item': 'Validation/test feature scaler', 'rule': 'Reuse supplied train feature scaler; no refit'},
    {'item': 'Per-basin NSE target std', 'rule': 'Keep nanstd(ddof=0) of targets in the currently loaded period'},
    {'item': 'B2 change scope', 'rule': 'Grouping/masked mean/sample filter; normalization methods unchanged'},
])
b2_save('B2_normalization_methods.csv', pd.DataFrame(normalization_method_rows))
b2_save('B2_normalization_config_check.csv', pd.DataFrame(normalization_config_rows))
b2_json(OUT / 'B2_normalization_contract.json', {'complete': True, 'library_base_commit': BASE_COMMIT,
    'basedataset_sha256_lf': normalization_source_hash, 'protected_method_ast_sha256': NORMALIZATION_SIGNATURES,
    'loaded_methods_matched': True, 'expected_B1b_template_config_keys_matched': len(normalization_keys),
    'rules': normalization_rules.to_dict('records'), 'custom_scaler_or_target_std_patch': False,
    'source_and_template_checks_only': True, 'scaler_fitted': False, 'raw_normalization_verified': False,
    'B1b_handoff_verified': False, 'B2_loader_patch_applied': False, 'official_training_started': False})
b2_table('4.2. Phương thức chuẩn hóa phải giữ nguyên', normalization_method_rows)
b2_table('4.2. Quy tắc chuẩn hóa và std nhãn của mã nền', normalization_rules)
print(f'Phần 4.2 đạt: {len(NORMALIZATION_METHODS)} phương thức gốc và {len(normalization_keys)} khóa template khớp; '
      'đã định nghĩa kiểm bảo toàn chuẩn hóa; chưa fit scaler hoặc áp patch B2.')

# %% Phần 4.3 - Hàm chung che forecast và chuyển counter cho mẫu hoặc batch
# Kiểm tensor giả; chưa nối hàm vào bộ nạp, GpuBatchLoader hoặc InputLayer.
from types import SimpleNamespace

b2_json(OUT / 'B2_forecast_helper_summary.json', {'complete': False, 'stage': 'forecast_helper_fixtures'})
CHECKS[:] = [row for row in CHECKS if row['group'] != 'forecast_input_helper']


def prepare_forecast_inputs(data, cfg):
    if cfg.nan_handling_method == 'masked_mean' and cfg.timestep_counter:
        for phase in ('hindcast', 'forecast'):
            key, counter = f'x_d_{phase}', f'{phase}_counter'
            if key not in data or counter not in data.get('x_d', {}):
                raise ValueError('Masked-mean counters require single-frequency hindcast/forecast inputs.')
            data[key][counter] = data['x_d'][counter].clone()
    if cfg.forecast_input_mode == 'first_day':
        if (cfg.nan_handling_method != 'masked_mean' or not cfg.timestep_counter
                or cfg.forecast_overlap or not cfg.hindcast_inputs_flattened):
            raise ValueError('first_day requires masked_mean, timestep_counter and zero-overlap forecast inputs.')
        features = cfg.forecast_inputs_flattened
        if not features or any(not name.startswith('ECMWF_') for name in features):
            raise ValueError('first_day permits only ECMWF HRES forecast features.')
        data['x_d_forecast'] = dict(data['x_d_forecast'])
        for feature in features:
            # Hỗ trợ mẫu [time, feature] và batch [batch, time, feature].
            values = data['x_d_forecast'][feature].clone()
            values[..., 1:, :] = torch.nan
            data['x_d_forecast'][feature] = values
    return data


forecast_fixture_horizon = 7
forecast_fixture_cfg = SimpleNamespace(nan_handling_method='masked_mean', timestep_counter=True,
    forecast_input_mode='first_day', forecast_overlap=0, hindcast_inputs_flattened=list(HINDCAST),
    forecast_inputs_flattened=list(FORECAST))
forecast_fixture_source = {
    'x_d': {'hindcast_counter': torch.zeros(4, 1),
            'forecast_counter': torch.arange(1, forecast_fixture_horizon + 1, dtype=torch.float32).unsqueeze(-1)},
    'x_d_hindcast': {name: torch.full((4, 1), float(index + 1)) for index, name in enumerate(HINDCAST)},
    'x_d_forecast': {name: (torch.arange(forecast_fixture_horizon, dtype=torch.float32) + 100 + index * 10).unsqueeze(-1)
                     for index, name in enumerate(FORECAST)},
}
forecast_fixture_snapshots = {key: {name: value.clone() for name, value in items.items()}
                              for key, items in forecast_fixture_source.items()}
# Sao các dictionary nhưng giữ tensor nguồn để kiểm việc clone trước khi che.
forecast_fixture_sample = prepare_forecast_inputs({key: dict(items) for key, items in forecast_fixture_source.items()}, forecast_fixture_cfg)
forecast_fixture_batch_source = {key: {name: value.unsqueeze(0).repeat(2, 1, 1)
    for name, value in items.items()} for key, items in forecast_fixture_source.items()}
forecast_fixture_batch = prepare_forecast_inputs({key: dict(items) for key, items in forecast_fixture_batch_source.items()}, forecast_fixture_cfg)
b2_check('sample_first_day_preserved', lambda: b2_assert(all(torch.equal(forecast_fixture_sample['x_d_forecast'][name][0],
    forecast_fixture_source['x_d_forecast'][name][0]) for name in FORECAST)), 'forecast_input_helper')
b2_check('sample_later_days_masked', lambda: b2_assert(all(torch.isnan(forecast_fixture_sample['x_d_forecast'][name][1:]).all()
    for name in FORECAST)), 'forecast_input_helper')
b2_check('batch_time_axis_masked', lambda: b2_assert(all(
    torch.equal(forecast_fixture_batch['x_d_forecast'][name][:, 0], forecast_fixture_batch_source['x_d_forecast'][name][:, 0])
    and torch.isnan(forecast_fixture_batch['x_d_forecast'][name][:, 1:]).all() for name in FORECAST)), 'forecast_input_helper')
b2_check('hindcast_values_unchanged', lambda: b2_assert(all(torch.equal(forecast_fixture_sample['x_d_hindcast'][name],
    forecast_fixture_source['x_d_hindcast'][name]) for name in HINDCAST)), 'forecast_input_helper')
b2_check('sample_and_batch_match', lambda: b2_assert(all(torch.allclose(
    forecast_fixture_batch[f'x_d_{phase}'][name][row], forecast_fixture_sample[f'x_d_{phase}'][name], equal_nan=True)
    for phase in ('hindcast', 'forecast') for name in forecast_fixture_sample[f'x_d_{phase}'] for row in range(2))), 'forecast_input_helper')
b2_check('counter_values_and_clones', lambda: b2_assert(all(torch.equal(
    forecast_fixture_sample[f'x_d_{phase}'][f'{phase}_counter'], forecast_fixture_source['x_d'][f'{phase}_counter'])
    and forecast_fixture_sample[f'x_d_{phase}'][f'{phase}_counter'].data_ptr() != forecast_fixture_source['x_d'][f'{phase}_counter'].data_ptr()
    for phase in ('hindcast', 'forecast'))), 'forecast_input_helper')
b2_check('source_tensors_preserved', lambda: b2_assert(all(torch.equal(value, forecast_fixture_snapshots[key][name])
    for key, items in forecast_fixture_source.items() for name, value in items.items())
    and all(torch.equal(value, forecast_fixture_source[key][name].unsqueeze(0).repeat(2, 1, 1))
    for key, items in forecast_fixture_batch_source.items() for name, value in items.items())), 'forecast_input_helper')
forecast_fixture_all_cfg = copy.copy(forecast_fixture_cfg)
forecast_fixture_all_cfg.forecast_input_mode, forecast_fixture_all_cfg.timestep_counter = 'all', False
forecast_fixture_all = {key: dict(items) for key, items in forecast_fixture_source.items()}
forecast_fixture_all_result = prepare_forecast_inputs(forecast_fixture_all, forecast_fixture_all_cfg)
b2_check('all_mode_counter_off_unchanged', lambda: b2_assert(forecast_fixture_all_result is forecast_fixture_all
    and all(torch.equal(value, forecast_fixture_source[key][name])
            for key, items in forecast_fixture_all_result.items() for name, value in items.items())), 'forecast_input_helper')
b2_check('official_B2_template_unchanged', lambda: b2_assert(B2_CONFIG_TEMPLATE['forecast_input_mode'] == 'all'
    and B2_CONFIG_TEMPLATE['timestep_counter'] is False and B2_CONFIG_TEMPLATE['forecast_seq_length'] == 1), 'forecast_input_helper')
forecast_fixture_days = pd.DataFrame({'lead': range(1, forecast_fixture_horizon + 1),
    'before': forecast_fixture_source['x_d_forecast'][FORECAST[0]].squeeze(-1).tolist(),
    'after': forecast_fixture_sample['x_d_forecast'][FORECAST[0]].squeeze(-1).tolist(),
    'counter': forecast_fixture_sample['x_d_forecast']['forecast_counter'].squeeze(-1).tolist()})
b2_save('B2_forecast_helper_days.csv', forecast_fixture_days)
b2_table(f'4.3. Tensor giả 7 hạn; ví dụ {FORECAST[0]}', forecast_fixture_days)
b2_require('forecast_input_helper')
forecast_helper_check_count = sum(row['group'] == 'forecast_input_helper' for row in CHECKS)
b2_json(OUT / 'B2_forecast_helper_summary.json', {'complete': True, 'synthetic_data_only': True,
    'helper': 'prepare_forecast_inputs', 'checks_passed': forecast_helper_check_count,
    'sample_shape': '[time, feature]', 'batch_shape': '[batch, time, feature]',
    'counter_rule': 'copy into phase dictionaries; InputLayer integration appends after embedding',
    'mask_rule': 'first_day keeps lead 1; later ECMWF values become NaN on clones',
    'source_tensors_preserved': True, 'official_B2_template_changed': False,
    'loader_or_model_integration_verified': False, 'B2_loader_patch_applied': False,
    'real_lookup_verified': False, 'official_training_started': False})
print(f'Phần 4.3 đạt: {forecast_helper_check_count}/{forecast_helper_check_count} kiểm helper trên tensor giả; '
      'mẫu/batch khớp, tensor nguồn giữ nguyên; chưa tích hợp bộ nạp hoặc mô hình.')

# %% Phần 4.4 - Tạo bản ứng viên đúng patch B2 trên f229e02
# Patch đóng băng từ nhánh B2; không thay bộ nạp Q, scaler hoặc phương thức std nhãn.
import difflib

B2_LIBRARY_PATCH = ('--- a/neuralhydrology/datasetzoo/lamah.py\n'
 '+++ b/neuralhydrology/datasetzoo/lamah.py\n'
 '@@ -91,18 +91,10 @@\n'
 ' \n'
 '     @staticmethod\n'
 '     def _get_list_of_all_variables(cfg: Config) -> List[str]:\n'
 '-        all_variables = []\n'
 '-        if isinstance(cfg.target_variables, dict):\n'
 '-            for val in cfg.target_variables.values():\n'
 '-                all_variables = all_variables + val\n'
 '-        else:\n'
 '-            all_variables = all_variables + cfg.target_variables\n'
 '-        if isinstance(cfg.dynamic_inputs, dict):\n'
 '-            for val in cfg.dynamic_inputs.values():\n'
 '-                all_variables = all_variables + val\n'
 '-        else:\n'
 '-            all_variables = all_variables + cfg.dynamic_inputs\n'
 '-        return all_variables\n'
 '+        targets = cfg.target_variables\n'
 '+        if isinstance(targets, dict):\n'
 '+            targets = [variable for group in targets.values() for variable in group]\n'
 '+        return list(dict.fromkeys(targets + cfg.dynamic_inputs_flattened))\n'
 ' \n'
 '     def _load_basin_data(self, basin: str) -> pd.DataFrame:\n'
 '         """Load input and output data from text files."""\n'
 '--- a/neuralhydrology/datasetzoo/basedataset.py\n'
 '+++ b/neuralhydrology/datasetzoo/basedataset.py\n'
 '@@ -208,7 +208,8 @@\n'
 "             sample['x_one_hot'] = torch.nn.functional.one_hot(torch.tensor(self.id_to_int[basin]),\n"
 '                                                               '
 'num_classes=len(self.id_to_int)).to(torch.float32)\n'
 ' \n'
 '-        return sample\n'
 '+        from neuralhydrology.datautils.forecast_inputs import prepare_forecast_inputs\n'
 '+        return prepare_forecast_inputs(sample, self.cfg)\n'
 ' \n'
 '     def _add_nan_streaks(self, x_d: dict[str, torch.Tensor], groups: list[list[str]]) -> dict[str, '
 'torch.Tensor]:\n'
 '         """Samples NaN streaks for each feature group."""\n'
 '@@ -618,7 +619,7 @@\n'
 '                 # manually unroll the dicts into lists to make sure the order of frequencies is '
 'consistent.\n'
 '                 # during inference, we want all samples with sufficient history (even if input is '
 'NaN), so\n'
 '                 # we pass x_d, x_s, y as None.\n'
 '-                if self.is_train:\n'
 '+                if self.is_train and self.cfg.nan_handling_method != "masked_mean":\n'
 '                     x_d_validate = [np.concatenate([v for v in x_d[freq].values()], axis=-1)\n'
 '                                     for freq in self.frequencies]\n'
 '                 else:\n'
 '@@ -639,6 +640,11 @@\n'
 "                     raise ValueError('Autoregressive inputs are not supported for datasets with "
 "multiple frequencies.')\n"
 '                 x_d[self.frequencies[0]].update({col: df_resampled[[col]].values\n'
 '                                                  for col in self.cfg.autoregressive_inputs})\n'
 '+\n'
 '+            # Static attributes are never repaired by dynamic masked-mean embeddings.\n'
 '+            if self.is_train and self.cfg.nan_handling_method == "masked_mean":\n'
 '+                if basin in self._attributes and not torch.isfinite(self._attributes[basin]).all():\n'
 '+                    flag[:] = 0\n'
 ' \n'
 '             valid_samples = np.argwhere(flag == 1)\n'
 '             for f in valid_samples:\n'
 '--- a/neuralhydrology/modelzoo/inputlayer.py\n'
 '+++ b/neuralhydrology/modelzoo/inputlayer.py\n'
 '@@ -68,7 +68,7 @@\n'
 '                 self._dynamic_inputs = [dynamic_inputs]\n'
 '             else:\n'
 '                 self._dynamic_inputs = dynamic_inputs\n'
 '-            if cfg.timestep_counter:\n'
 '+            if cfg.timestep_counter and cfg.nan_handling_method != "masked_mean":\n'
 '                 # Add timestep counter to each feature group.\n'
 "                 if self.embedding_type == 'hindcast':\n"
 "                     self._dynamic_inputs = [group + ['hindcast_counter'] for group in "
 'self._dynamic_inputs]\n'
 '@@ -145,6 +145,12 @@\n'
 '         else:\n'
 "             self.dynamics_embedding_p_dropout = cfg.dynamics_embedding['dropout']\n"
 ' \n'
 '+        # Append time independently of source availability, including the all-missing case.\n'
 "+        self._separate_counter = cfg.timestep_counter and cfg.nan_handling_method == 'masked_mean'\n"
 '+        if self._separate_counter:\n'
 "+            if embedding_type not in ('hindcast', 'forecast'):\n"
 "+                raise ValueError('Masked-mean timestep counters require a forecast model.')\n"
 '+            self.dynamics_output_size += 1\n'
 '         self.output_size = self.dynamics_output_size + self.statics_output_size + '
 'self._num_autoregression_inputs\n'
 '         self.cfg = cfg\n'
 ' \n'
 '@@ -233,6 +239,10 @@\n'
 '             # transpose to [seq_length, batch_size, n_features]\n'
 '             x_d = torch.cat([data[self._x_d_key][k] for k in itertools.chain(*features)], '
 'dim=-1).transpose(0, 1)\n'
 '             dynamics_out = self.dynamics_embeddings[0](x_d)\n'
 '+\n'
 '+        if self._separate_counter:\n'
 "+            counter = data[self._x_d_key][f'{self.embedding_type}_counter'].transpose(0, 1)\n"
 '+            dynamics_out = torch.cat([dynamics_out, counter], dim=-1)\n'
 ' \n'
 '         if not concatenate_output:\n'
 '             ret_val = dynamics_out, statics_out\n'
 '--- a/neuralhydrology/datautils/gpu_loader.py\n'
 '+++ b/neuralhydrology/datautils/gpu_loader.py\n'
 '@@ -136,7 +136,8 @@\n'
 '             batch["x_s"] = self._attributes[basin]\n'
 '         if self._stds is not None:\n'
 '             batch["per_basin_target_stds"] = self._stds[basin]\n'
 '-        return batch\n'
 '+        from neuralhydrology.datautils.forecast_inputs import prepare_forecast_inputs\n'
 '+        return prepare_forecast_inputs(batch, self.cfg)\n'
 ' \n'
 '     def _add_nan_streaks(self, block: torch.Tensor, keys: List[str], name: str) -> torch.Tensor:\n'
 '         # same distribution as BaseDataset._add_nan_streaks (per sample and feature group), drawn on '
 'the device\n'
 '--- a/neuralhydrology/utils/config.py\n'
 '+++ b/neuralhydrology/utils/config.py\n'
 '@@ -441,6 +441,14 @@\n'
 '         return self._cfg.get("forecast_overlap", None)\n'
 ' \n'
 '     @property\n'
 '+    def forecast_input_mode(self) -> str:\n'
 '+        # all: existing one-day inputs; first_day: only HRES from the first forecast day.\n'
 "+        mode = self._cfg.get('forecast_input_mode', 'all')\n"
 "+        if mode not in ('all', 'first_day'):\n"
 "+            raise ValueError('forecast_input_mode must be all or first_day.')\n"
 '+        return mode\n'
 '+\n'
 '+    @property\n'
 '     def forecast_seq_length(self) -> int:\n'
 '         return self._cfg.get("forecast_seq_length", None)\n'
 ' \n'
 '--- /dev/null\n'
 '+++ b/neuralhydrology/datautils/forecast_inputs.py\n'
 '@@ -0,0 +1,25 @@\n'
 '+# Shared by the regular dataset and the vectorized batch loader.\n'
 '+import torch\n'
 '+\n'
 '+\n'
 '+def prepare_forecast_inputs(data, cfg):\n'
 "+    if cfg.nan_handling_method == 'masked_mean' and cfg.timestep_counter:\n"
 "+        for phase in ('hindcast', 'forecast'):\n"
 "+            key, counter = f'x_d_{phase}', f'{phase}_counter'\n"
 "+            if key not in data or counter not in data.get('x_d', {}):\n"
 "+                raise ValueError('Masked-mean counters require single-frequency hindcast/forecast "
 "inputs.')\n"
 "+            data[key][counter] = data['x_d'][counter].clone()\n"
 "+    if cfg.forecast_input_mode == 'first_day':\n"
 "+        if (cfg.nan_handling_method != 'masked_mean' or not cfg.timestep_counter\n"
 '+                or cfg.forecast_overlap or not cfg.hindcast_inputs_flattened):\n'
 "+            raise ValueError('first_day requires masked_mean, timestep_counter and zero-overlap "
 "forecast inputs.')\n"
 '+        features = cfg.forecast_inputs_flattened\n'
 "+        if not features or any(not name.startswith('ECMWF_') for name in features):\n"
 "+            raise ValueError('first_day permits only ECMWF HRES forecast features.')\n"
 "+        data['x_d_forecast'] = dict(data['x_d_forecast'])\n"
 '+        for feature in features:\n'
 '+            # Both [time, feature] samples and [batch, time, feature] batches.\n'
 "+            values = data['x_d_forecast'][feature].clone()\n"
 '+            values[..., 1:, :] = torch.nan\n'
 "+            data['x_d_forecast'][feature] = values\n"
 '+    return data\n'
 '--- /dev/null\n'
 '+++ b/neuralhydrology/datautils/source_groups.py\n'
 '@@ -0,0 +1,42 @@\n'
 '+import copy\n'
 '+\n'
 '+from neuralhydrology.utils.config import Config\n'
 '+\n'
 '+\n'
 '+def lamah_source_groups(variables):\n'
 '+    # Stable source order; keep the original variable order within each source.\n'
 "+    order = ('ERA5L', 'EOBS', 'MSWEP', 'GLEAM', 'Q', 'ECMWF')\n"
 '+    groups = {source: [] for source in order}\n'
 '+    if len(variables) != len(set(variables)):\n'
 "+        raise ValueError('Duplicate variables in a model phase.')\n"
 '+    for variable in variables:\n'
 "+        source = 'Q' if variable in ('qmean_mmd', 'qmax_mmd') else variable.split('_', 1)[0]\n"
 '+        if source not in groups:\n'
 "+            raise ValueError(f'Unknown LamaH source: {variable}')\n"
 '+        groups[source].append(variable)\n'
 '+    return [groups[source] for source in order if groups[source]]\n'
 '+\n'
 '+\n'
 '+def configure_lamah_b2(cfg: Config) -> Config:\n'
 '+    # B2 adds only grouping, masked mean and its sample filter to the B1b configuration.\n'
 "+    if (cfg.model != 'sequential_forecast_lstm' or cfg.target_variables != ['qmax']\n"
 '+            or cfg.forecast_seq_length != 1 or cfg.forecast_overlap\n'
 '+            or cfg.seq_length != 365 or cfg.predict_last_n != 1\n'
 '+            or cfg.epochs != 30 or cfg.early_stopping_patience != 30):\n'
 "+        raise ValueError('B2 requires the one-day, 365-step, 30-epoch B1b protocol.')\n"
 '+    hindcast = lamah_source_groups(cfg.hindcast_inputs_flattened)\n'
 '+    forecast = lamah_source_groups(cfg.forecast_inputs_flattened)\n'
 '+    expected = (21, 7, 1, 2, 2)\n'
 '+    if (tuple(map(len, hindcast)) != expected or len(forecast) != 1 or len(forecast[0]) != 5\n'
 "+            or not all(name.startswith('ECMWF_') for name in forecast[0])\n"
 "+            or set(hindcast[-1]) != {'qmean_mmd', 'qmax_mmd'}):\n"
 "+        raise ValueError('B1b inputs must contain ERA5L/EOBS/MSWEP/GLEAM/Q and five ECMWF "
 "variables.')\n"
 '+    flattened = [variable for group in hindcast + forecast for variable in group]\n'
 '+    if set(flattened) != set(cfg.dynamic_inputs_flattened):\n'
 "+        raise ValueError('B1b dynamic inputs differ from its hindcast/forecast phases.')\n"
 '+    result = copy.deepcopy(cfg)\n'
 '+    result.update_config(dict(dynamic_inputs=hindcast + forecast, hindcast_inputs=hindcast,\n'
 "+                              forecast_inputs=forecast, nan_handling_method='masked_mean',\n"
 '+                              nan_step_probability=0., nan_sequence_probability=0.,\n'
 "+                              timestep_counter=False, forecast_input_mode='all'))\n"
 '+    return result\n')


def b2_apply_source_patch(base_sources, patch_text):
    result = dict(base_sources)
    allowed = set(base_sources) | {'datautils/forecast_inputs.py', 'datautils/source_groups.py'}
    lines, pos, seen = patch_text.splitlines(keepends=True), 0, set()
    while pos < len(lines):
        if not lines[pos].startswith('--- ') or pos + 1 >= len(lines) or not lines[pos + 1].startswith('+++ b/neuralhydrology/'):
            raise ValueError('Header patch B2 không hợp lệ.')
        name = lines[pos + 1].strip().removeprefix('+++ b/neuralhydrology/')
        if name not in allowed or name in seen:
            raise ValueError(f'Tệp patch ngoài phạm vi hoặc trùng: {name}')
        seen.add(name)
        current, pos = result.get(name, ''), pos + 2
        while pos < len(lines) and lines[pos].startswith('@@ '):
            pos += 1
            old, new = [], []
            while pos < len(lines) and not lines[pos].startswith(('@@ ', '--- ')):
                line = lines[pos]
                if line[:1] not in (' ', '+', '-'):
                    raise ValueError('Dòng patch không hợp lệ.')
                if line[0] in (' ', '-'):
                    old.append(line[1:])
                if line[0] in (' ', '+'):
                    new.append(line[1:])
                pos += 1
            old, new = ''.join(old), ''.join(new)
            if not old:
                if current:
                    raise ValueError(f'Không ghi đè module mới đã có trong mã nền: {name}')
                current = new
            elif current.count(old) != 1:
                raise ValueError(f'Anchor patch không duy nhất/không khớp: {name}')
            else:
                current = current.replace(old, new, 1)
        compile(current, name, 'exec')
        result[name] = current
    return result


for relative, expected in BASE_HASHES.items():
    if hashlib.sha256(BASE_TEXT[relative].encode('utf-8')).hexdigest() != expected:
        raise RuntimeError(f'Nguồn nền thay đổi trước khi tạo patch: {relative}')
patched = b2_apply_source_patch(BASE_TEXT, B2_LIBRARY_PATCH)
b2_require_normalization_preserved(patched['datasetzoo/basedataset.py'])
b2_assert(patched['datautils/utils.py'] == BASE_TEXT['datautils/utils.py'], 'Không thêm định dạng scaler riêng.')
patch_sha256 = hashlib.sha256(B2_LIBRARY_PATCH.encode('utf-8')).hexdigest()
CANDIDATE_ROOT = OUT / 'library_candidate' / patch_sha256[:12]
CANDIDATE_PACKAGE = CANDIDATE_ROOT / 'neuralhydrology'
shutil.copytree(BASE_PACKAGE, CANDIDATE_PACKAGE, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
candidate_rows = []
for relative, source in patched.items():
    path = CANDIDATE_PACKAGE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8', newline='\n')
    candidate_rows.append({'file': relative, 'changed': source != BASE_TEXT.get(relative), 'sha256_lf': b2_hash(path)})
(OUT / 'B2_loader_candidate.patch').write_text(B2_LIBRARY_PATCH, encoding='utf-8', newline='\n')
RECEIPTS.append(OUT / 'B2_loader_candidate.patch')
b2_json(OUT / 'B2_candidate_source.json', {'base_commit': BASE_COMMIT, 'B1b_integrated': False,
    'candidate_root': str(CANDIDATE_ROOT), 'patch_sha256': patch_sha256,
    'files': {row['file']: row['sha256_lf'] for row in candidate_rows},
    'normalization_methods_preserved': True, 'official_training_started': False})
b2_table('4.4. Source ứng viên B2, chưa gộp research', candidate_rows)
print('Phần 4.4 đạt: tạo source và patch; chuẩn hóa gốc giữ nguyên; chưa merge hoặc huấn luyện.')

# %% Phần 4.5 - Nạp bản ứng viên và kiểm cấu hình B1b sang B2
# Nạp lại package từ bản sao; nguồn cài đặt và fork chưa bị ghi đè.
for name in list(sys.modules):
    if name == 'neuralhydrology' or name.startswith('neuralhydrology.'):
        del sys.modules[name]
sys.path[:] = [p for p in sys.path if not str(p).startswith(str(OUT / 'library_candidate'))]
sys.path.insert(0, str(CANDIDATE_ROOT))
importlib.invalidate_caches()
from neuralhydrology.utils.config import Config
from neuralhydrology.datasetzoo.lamah import LamaH
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.sequential_forecast_lstm import SequentialForecastLSTM
from neuralhydrology.datautils.gpu_loader import GpuBatchLoader
from neuralhydrology.datautils.forecast_inputs import prepare_forecast_inputs as candidate_forecast_inputs
from neuralhydrology.datautils.source_groups import configure_lamah_b2

loaded_rows = []
for obj in (Config, LamaH, BaseDataset, InputLayer, SequentialForecastLSTM, GpuBatchLoader, candidate_forecast_inputs, configure_lamah_b2):
    path = Path(inspect.getfile(obj)).resolve()
    if not path.is_relative_to(CANDIDATE_PACKAGE.resolve()):
        raise RuntimeError(f'Nạp nhầm source: {path}')
    loaded_rows.append({'object': obj.__name__, 'file': path.relative_to(CANDIDATE_PACKAGE).as_posix(), 'candidate': True})
for relative, source in patched.items():
    b2_assert(b2_hash(CANDIDATE_PACKAGE / relative) == hashlib.sha256(source.encode('utf-8')).hexdigest())
constructed = configure_lamah_b2(Config(copy.deepcopy(B1B_EXPECTED_CONFIG)))
b2_assert(constructed.as_dict() == Config(copy.deepcopy(B2_CONFIG_TEMPLATE)).as_dict(), 'Builder khác template đã kiểm.')
b2_json(OUT / 'B2_candidate_loaded.json', {'loaded_from_candidate': True, 'builder_matches_template': True,
    'B1b_handoff_verified': False, 'official_training_started': False})
b2_table('4.5. Module đang nạp', loaded_rows)
print('Phần 4.5 đạt: loader, model và builder nạp từ cùng source ứng viên.')

# %% Phần 5.1 - Bộ dữ liệu giả theo đúng schema LamaH
# Dựng hai lưu vực có số biết trước; kết quả phần này không phải kết quả thực nghiệm.
FIXTURE = OUT / 'synthetic_fixture'
fixture_dates = pd.date_range('2002-12-20', '2003-03-01', name='date')
for relative in ('A_basins_total_upstrm/1_attributes', 'D_gauges/1_attributes',
                 'A_basins_total_upstrm/2_timeseries/daily', 'D_gauges/2_timeseries/daily'):
    (FIXTURE / relative).mkdir(parents=True, exist_ok=True)
pd.DataFrame({'ID': ['1', '2'], 'elev_mean': [100., 300.]}).to_csv(
    FIXTURE / 'A_basins_total_upstrm/1_attributes/Catchment_attributes.csv', sep=';', index=False)
pd.DataFrame({'ID': ['1', '2'], 'area_gov': [43.2, 86.4]}).to_csv(
    FIXTURE / 'D_gauges/1_attributes/Gauge_attributes.csv', sep=';', index=False)
(FIXTURE / 'basins.txt').write_text('1\n2\n', encoding='utf-8')
for basin in (1, 2):
    day = np.arange(len(fixture_dates), dtype=float)
    raw_qmean = 2. + basin + day * .03
    raw_qmax = raw_qmean + .5 + .1 * np.cos(day)
    raw_qmean[fixture_dates == pd.Timestamp('2003-01-03')] = np.nan
    raw_qmax[fixture_dates == pd.Timestamp('2003-01-07')] = np.nan
    calendar = {'YYYY': fixture_dates.year, 'MM': fixture_dates.month, 'DD': fixture_dates.day}
    pd.DataFrame({**calendar, 'ERA5L_a': 1. + day + basin, 'ECMWF_a': np.sin(day / 4.) + basin,
                  'qmean_mmd': raw_qmean * (86.4 / (43.2 if basin == 1 else 86.4)),
                  'qmax_mmd': raw_qmax * (86.4 / (43.2 if basin == 1 else 86.4))}).to_csv(FIXTURE / f'A_basins_total_upstrm/2_timeseries/daily/ID_{basin}.csv', sep=';', index=False)
    pd.DataFrame({**calendar, 'qmean': raw_qmean, 'qmax': raw_qmax}).to_csv(
        FIXTURE / f'D_gauges/2_timeseries/daily/ID_{basin}.csv', sep=';', index=False)


def b2_fixture_config(**changes):
    cfg = {'experiment_name': 'B2_synthetic_only', 'run_dir': str(OUT / 'synthetic_logs'),
        'train_dir': str(OUT / 'synthetic_logs/train_data'),
        'train_basin_file': str(FIXTURE / 'basins.txt'), 'validation_basin_file': str(FIXTURE / 'basins.txt'),
        'test_basin_file': str(FIXTURE / 'basins.txt'), 'data_dir': str(FIXTURE), 'dataset': 'lamah_a',
        'train_start_date': '01/01/2003', 'train_end_date': '12/01/2003',
        'validation_start_date': '13/01/2003', 'validation_end_date': '20/01/2003',
        'test_start_date': '21/01/2003', 'test_end_date': '28/01/2003',
        'dynamic_inputs': [['ERA5L_a'], ['qmean_mmd', 'qmax_mmd'], ['ECMWF_a']],
        'hindcast_inputs': [['ERA5L_a'], ['qmean_mmd', 'qmax_mmd']], 'forecast_inputs': [['ECMWF_a']],
        'target_variables': ['qmax'], 'static_attributes': ['elev_mean'], 'seq_length': 5,
        'predict_last_n': 1, 'forecast_seq_length': 1, 'forecast_overlap': 0,
        'model': 'sequential_forecast_lstm', 'hidden_size': 8, 'head': 'regression',
        'output_activation': 'linear', 'initial_forget_bias': 3., 'output_dropout': 0.,
        'dynamics_embedding': {'type': 'fc', 'hiddens': [4], 'activation': 'tanh', 'dropout': 0.},
        'statics_embedding': {'type': 'fc', 'hiddens': [2], 'activation': 'tanh', 'dropout': 0.},
        'nan_handling_method': 'masked_mean', 'timestep_counter': False, 'loss': 'NSE',
        'batch_size': 4, 'num_workers': 0, 'verbose': 0, 'save_train_data': False,
        'forecast_input_mode': 'all', 'data_on_gpu': True, 'device': 'cpu', 'epochs': 1,
        'learning_rate': .001, 'optimizer': 'Adam', 'seed': 111}
    cfg.update(changes)
    b2_contract(cfg)
    return Config(cfg)


SYN_CFG = b2_fixture_config()
SYN_TRAIN = LamaH(SYN_CFG, is_train=True, period='train')
SYN_SCALER = copy.deepcopy(SYN_TRAIN.scaler)
SYN_EVAL = LamaH(SYN_CFG, is_train=False, period='validation', scaler=copy.deepcopy(SYN_SCALER))
print('5.1. Dataset giả đã qua LamaH/BaseDataset thật:', len(SYN_TRAIN), 'train;', len(SYN_EVAL), 'validation.')

# %% Phần 5.2 - Kiểm Q, warmup và chuẩn hóa gốc trên fixture
# Tính kỳ vọng độc lập từ CSV và so cả scaler lẫn std theo kỳ.
def b2_test_raw_q():
    frame = SYN_TRAIN._load_basin_data('1')
    raw = pd.read_csv(FIXTURE / 'D_gauges/2_timeseries/daily/ID_1.csv', sep=';')
    np.testing.assert_allclose(frame.qmax, raw.qmax.to_numpy() * 2., rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(frame.qmax_mmd, frame.qmax, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(frame.qmean_mmd, raw.qmean.to_numpy() * 2., rtol=1e-12, equal_nan=True)


def b2_test_first_window():
    sample = SYN_TRAIN[0]
    dates = pd.DatetimeIndex(sample['date'])
    b2_assert(dates[0] == pd.Timestamp('2002-12-28') and dates[-1] == pd.Timestamp('2003-01-01'))
    b2_assert(sample['x_d_hindcast']['qmax_mmd'].shape == (4, 1))
    b2_assert(torch.isfinite(sample['x_d_hindcast']['qmax_mmd']).all())
    b2_assert(torch.isnan(sample['y'][:-1]).all())


def b2_test_train_scaler():
    arrays = []
    for basin in ('1', '2'):
        frame = SYN_TRAIN._load_basin_data(basin).loc['2002-12-28':'2003-01-12'].copy()
        frame.loc[frame.index < '2003-01-01', 'qmax'] = np.nan
        arrays.append(frame.astype('float32'))
    for variable in SYN_CFG.dynamic_inputs_flattened + ['qmax']:
        values = np.concatenate([frame[variable].to_numpy() for frame in arrays])
        np.testing.assert_allclose(float(SYN_SCALER['xarray_feature_center'][variable]), np.nanmean(values), rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(float(SYN_SCALER['xarray_feature_scale'][variable]), np.nanstd(values), rtol=2e-6, atol=1e-7)


def b2_test_period_std():
    for ds, start, end in [(SYN_TRAIN, '2003-01-01', '2003-01-12'), (SYN_EVAL, '2003-01-13', '2003-01-20')]:
        for basin in ('1','2'):
            values = ds._load_basin_data(basin).loc[start:end, 'qmax'].to_numpy(dtype='float32')
            np.testing.assert_allclose(ds._per_basin_target_stds[basin].numpy(), [[np.nanstd(values)]], rtol=1e-6)


def b2_test_grouped_variables():
    variables = LamaH._get_list_of_all_variables(SYN_CFG)
    b2_assert(len(variables) == len(set(variables)) and all(isinstance(v, str) for v in variables))
    b2_assert('qmax_mmd' in variables and 'qmax' in variables)


for name, op in [('Q_converted_once_separate_target', b2_test_raw_q), ('first_window_and_warmup', b2_test_first_window),
                 ('baseline_scaler_includes_dynamic_warmup', b2_test_train_scaler), ('target_std_preserves_period_behavior', b2_test_period_std),
                 ('nested_groups_flattened', b2_test_grouped_variables)]:
    b2_check(name, op, 'synthetic_loader')

# %% Phần 5.3 - Kiểm lookup thực, chế độ cũ và nhãn thiếu
# So lookup của BaseDataset với phép đếm trực tiếp bằng các ngày raw đã biết.
def b2_test_lookup():
    targets = [pd.Timestamp('2003-01-01') + pd.Timedelta(days=i) for i in range(12) if i != 6]
    actual = []
    for basin, indices in SYN_TRAIN.lookup_table.values():
        actual.append((basin, pd.Timestamp(SYN_TRAIN._dates[basin][SYN_TRAIN.frequencies[0]][indices[0]])))
    expected = {(basin, date) for basin in ('1', '2') for date in targets}
    b2_assert(set(actual) == expected, 'Lookup B2 không đúng các ngày nhãn hữu hạn đã biết.')
    b2_assert(any(date == pd.Timestamp('2003-01-04') for basin, date in actual), 'Mẫu có qmean hôm trước thiếu phải được giữ.')


def b2_test_no_nan_handler():
    cfg = b2_fixture_config(nan_handling_method=None)
    ds = LamaH(cfg, is_train=True, period='train', scaler=copy.deepcopy(SYN_SCALER))
    expected = set()
    for basin in ('1', '2'):
        frame = ds._load_basin_data(basin).loc['2002-12-28':'2003-01-12']
        mask = b2_filter(frame[cfg.dynamic_inputs_flattened].to_numpy(), frame.qmax.to_numpy(), 5)
        expected.update((basin, frame.index[i]) for i in np.flatnonzero(mask))
    actual = {(b, pd.Timestamp(ds._dates[b][ds.frequencies[0]][idx[0]])) for b, idx in ds.lookup_table.values()}
    b2_assert(actual == expected)


def b2_test_eval_lookup():
    b2_assert(len(SYN_EVAL) == 16)
    before = {k: v.copy(deep=True) if hasattr(v, 'copy') else copy.deepcopy(v) for k, v in SYN_SCALER.items()}
    for key in ('xarray_feature_center', 'xarray_feature_scale'):
        xr.testing.assert_identical(SYN_EVAL.scaler[key], before[key])


def b2_test_repeated_sample():
    a = SYN_EVAL[0]
    source = a['x_d_hindcast']['ERA5L_a'].clone()
    _ = SYN_EVAL[1]
    b = SYN_EVAL[0]
    torch.testing.assert_close(source, b['x_d_hindcast']['ERA5L_a'], rtol=0, atol=0, equal_nan=True)
    b2_assert(torch.equal(a['y'], b['y']) or torch.allclose(a['y'], b['y'], equal_nan=True))


for name, op in [('lookup_keeps_input_missing_rejects_label_missing', b2_test_lookup),
                 ('nan_handler_off_matches_original_filter', b2_test_no_nan_handler),
                 ('evaluation_calendar_and_scaler_unchanged', b2_test_eval_lookup),
                 ('repeated_overlapping_samples', b2_test_repeated_sample)]:
    b2_check(name, op, 'synthetic_loader')
b2_require('synthetic_loader')

# %% Phần 5.4 - Kiểm lớp nhúng, gradient và counter
# Dùng InputLayer thật, trọng số khởi tạo giả; không công bố điểm dự báo từ kiểm này.
def b2_batch_copy(batch):
    return copy.deepcopy(batch)


def b2_gradient_probe(model, batch):
    model.train()
    result = model(batch)['y_hat'][:, -1:, :]
    target = batch['y'][:, -1:, :]
    good = torch.isfinite(target)
    if not good.any():
        raise ValueError('Batch kiểm gradient không có nhãn hữu hạn.')
    loss = ((result[good] - target[good]) ** 2).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    b2_assert(torch.isfinite(result).all() and torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads))
    return float(loss.detach())


def b2_test_embedding_missing():
    torch.manual_seed(111)
    layer = InputLayer(SYN_CFG, embedding_type='hindcast')
    batch = BaseDataset.collate_fn([SYN_TRAIN[0], SYN_TRAIN[1]])
    all_missing = b2_batch_copy(batch)
    for value in all_missing['x_d_hindcast'].values():
        value[:] = torch.nan
    result = layer(all_missing, concatenate_output=False)[0]
    b2_assert(torch.equal(result, torch.zeros_like(result)), 'Fallback mọi nhóm thiếu phải bằng 0 trước static/counter.')
    partial = b2_batch_copy(batch)
    partial['x_d_hindcast']['qmean_mmd'][:] = torch.nan
    b2_assert(torch.isfinite(layer(partial)).all())


def b2_test_counter():
    cfg = b2_fixture_config(timestep_counter=True)
    before = copy.deepcopy(cfg.as_dict())
    first, second = InputLayer(cfg, 'forecast'), InputLayer(cfg, 'forecast')
    b2_assert(cfg.as_dict() == before, 'InputLayer không được sửa danh sách nhóm trong config.')
    b2_assert(len(first._dynamic_inputs) == len(second._dynamic_inputs) == 1)
    ds = LamaH(cfg, is_train=False, period='validation', scaler=copy.deepcopy(SYN_SCALER))
    batch = BaseDataset.collate_fn([ds[0], ds[1]])
    batch['x_d_forecast']['ECMWF_a'][:] = torch.nan
    embedded = first(batch, concatenate_output=False)[0]
    b2_assert(torch.isfinite(embedded).all())
    torch.testing.assert_close(embedded[..., -1:], batch['x_d_forecast']['forecast_counter'].transpose(0, 1))
    b2_assert(torch.equal(embedded[..., :-1], torch.zeros_like(embedded[..., :-1])))


def b2_test_forward_backward():
    torch.manual_seed(111)
    model = SequentialForecastLSTM(SYN_CFG)
    batch = BaseDataset.collate_fn([SYN_TRAIN[i] for i in range(min(4, len(SYN_TRAIN)))])
    b2_gradient_probe(model, batch)


def b2_test_counter_off_legacy_layer():
    namespace = {'__name__': 'b2_original_inputlayer'}
    exec(compile(BASE_TEXT['modelzoo/inputlayer.py'], '<original_InputLayer>', 'exec'), namespace)
    legacy = namespace['InputLayer'](SYN_CFG, 'hindcast')
    candidate = InputLayer(SYN_CFG, 'hindcast')
    candidate.load_state_dict(legacy.state_dict())
    batch = BaseDataset.collate_fn([SYN_TRAIN[0], SYN_TRAIN[1]])
    legacy.eval(); candidate.eval()
    torch.testing.assert_close(candidate(batch), legacy(batch), rtol=0, atol=0, equal_nan=True)


for name, op in [('missing_group_and_all_missing_fallback', b2_test_embedding_missing),
                 ('counter_is_separate_and_config_immutable', b2_test_counter),
                 ('forward_backward_finite', b2_test_forward_backward),
                 ('counter_off_same_weights_same_output', b2_test_counter_off_legacy_layer)]:
    b2_check(name, op, 'synthetic_model')

def b2_compare_batch_paths(ds, device='cpu'):
    n = min(4, len(ds))
    regular = BaseDataset.collate_fn([ds[i] for i in range(n)])
    vector = next(iter(GpuBatchLoader(ds, batch_size=n, device=device, shuffle=False, with_dates=True)))
    for key in regular:
        if key == 'date':
            np.testing.assert_array_equal(regular[key], vector[key])
        elif isinstance(regular[key], dict):
            b2_assert(regular[key].keys() == vector[key].keys())
            for name in regular[key]:
                torch.testing.assert_close(regular[key][name], vector[key][name].cpu(), rtol=0, atol=0, equal_nan=True)
        else:
            torch.testing.assert_close(regular[key], vector[key].cpu(), rtol=0, atol=0, equal_nan=True)
    return True


b2_check('regular_vectorized_CPU_batches_equal', lambda: b2_compare_batch_paths(SYN_TRAIN), 'synthetic_model')
GPU_PROBE_STATUS = 'not_run_no_CUDA'
if torch.cuda.is_available():
    b2_check('regular_vectorized_CUDA_batches_equal', lambda: b2_compare_batch_paths(SYN_TRAIN, 'cuda'), 'synthetic_model')
    GPU_PROBE_STATUS = 'passed' if CHECKS[-1]['passed'] else 'failed'
b2_require('synthetic_model')

# %% Phần 5.5 - Từ chối hợp đồng sai và can thiệp ngoài kỳ fit
# Kiểm lỗi đầu vào và bảo đảm thay dữ liệu ngoài train không làm đổi scaler.
def b2_expect_error(operation, expected=(ValueError, TypeError)):
    try:
        operation()
    except expected:
        return
    raise AssertionError('Đầu vào sai chưa bị từ chối.')


def b2_test_bad_contracts():
    raw = copy.deepcopy(SYN_CFG.as_dict())
    raw['forecast_inputs'] = [['qmean_mmd']]
    b2_expect_error(lambda: b2_contract(raw))
    raw = copy.deepcopy(SYN_CFG.as_dict())
    raw['dynamic_inputs'].append(['qmax'])
    b2_expect_error(lambda: b2_contract(raw))
    b2_expect_error(lambda: b2_source('unknown_feature'))
    b2_expect_error(lambda: b2_groups(['ERA5L_a', 'ERA5L_a']))


def b2_test_bad_area():
    b2_expect_error(lambda: b2_q_reference(FIXTURE, '1', 0.))


def b2_test_outside_train_intervention():
    from unittest.mock import patch
    import neuralhydrology.datasetzoo.lamah as module
    read_forcing, read_q = module.load_lamah_forcing, module.load_lamah_discharge
    def change_forcing(*args, **kwargs):
        frame = read_forcing(*args, **kwargs).copy()
        outside = (frame.index < '2002-12-28') | (frame.index > '2003-01-12')
        frame.loc[outside, ['ERA5L_a', 'ECMWF_a', 'qmean_mmd', 'qmax_mmd']] += 100000.
        return frame
    def change_q(*args, **kwargs):
        frame = read_q(*args, **kwargs).copy()
        outside = (frame.index < '2002-12-28') | (frame.index > '2003-01-12')
        frame.loc[outside, ['qmean', 'qmax']] += 100000.
        return frame
    with patch.object(module, 'load_lamah_forcing', side_effect=change_forcing), patch.object(module, 'load_lamah_discharge', side_effect=change_q):
        modified = LamaH(SYN_CFG, is_train=True, period='train')
    for key in ('xarray_feature_center', 'xarray_feature_scale'):
        xr.testing.assert_identical(modified.scaler[key], SYN_SCALER[key])


for name, operation in [('invalid_contracts_rejected', b2_test_bad_contracts),
                        ('nonpositive_area_rejected', b2_test_bad_area),
                        ('outside_train_intervention_does_not_change_fit', b2_test_outside_train_intervention)]:
    b2_check(name, operation, 'input_contract')
b2_require('input_contract')

# %% Phần 6.1 - Hàm đếm cửa sổ độc lập và thống kê gộp train
# Dùng tổng tích lũy boolean; không sao chép hàng triệu cửa sổ và không gọi bộ lọc NH.
def b2_window_complete(finite, length):
    mask = np.asarray(finite, dtype=bool)
    if mask.ndim != 1 or length < 1:
        raise ValueError('Cần mask 1D và length dương.')
    prefix = np.r_[0, np.cumsum(~mask, dtype=np.int64)]
    result = np.zeros(len(mask), dtype=bool)
    if len(mask) >= length:
        result[length - 1:] = (prefix[length:] - prefix[:-length]) == 0
    return result


def b2_window_masks(frame, length, old_hindcast, hindcast, forecast):
    dates = pd.DatetimeIndex(frame.index)
    if dates.empty or dates.tz is not None or not dates.equals(pd.date_range(dates[0], dates[-1], freq='D')):
        raise ValueError('Đếm cửa sổ cần lịch ngày liên tục đã reindex, không bỏ ngày thiếu.')
    labels = np.isfinite(frame['qmax'].to_numpy())
    enough = np.arange(len(frame)) >= length - 1
    old_all = np.isfinite(frame[list(dict.fromkeys(old_hindcast + forecast))].to_numpy()).all(axis=1)
    new_all = np.isfinite(frame[list(dict.fromkeys(hindcast + forecast))].to_numpy()).all(axis=1)
    old_h = b2_window_complete(np.isfinite(frame[old_hindcast].to_numpy()).all(axis=1), length - 1)
    new_h = b2_window_complete(np.isfinite(frame[hindcast].to_numpy()).all(axis=1), length - 1)
    f = np.isfinite(frame[forecast].to_numpy()).all(axis=1)
    return {'original_A2': b2_window_complete(old_all, length) & labels,
            'phase_complete_A2': np.r_[False, old_h[:-1]] & f & labels & enough,
            'original_B1b_inputs': b2_window_complete(new_all, length) & labels,
            'phase_complete_B1b_inputs': np.r_[False, new_h[:-1]] & f & labels & enough,
            'masked_mean_B2': enough & labels, 'evaluation_lookup': enough}


def b2_test_window_oracle():
    rng = np.random.default_rng(222)
    mask = rng.random(60) > .1
    for length in (1, 5, 30, 80):
        expected = np.array([i >= length - 1 and mask[i-length+1:i+1].all() for i in range(len(mask))])
        np.testing.assert_array_equal(b2_window_complete(mask, length), expected)
    leap = pd.DataFrame(1., index=pd.date_range('2004-02-27', '2004-03-02'),
                        columns=['ERA5L_a', 'ECMWF_a', 'qmean_mmd', 'qmax_mmd', 'qmax'])
    result = b2_window_masks(leap, 3, ['ERA5L_a', 'qmean_mmd'], ['ERA5L_a', 'qmean_mmd', 'qmax_mmd'], ['ECMWF_a'])
    b2_assert(result['masked_mean_B2'].sum() == 3)
    b2_expect_error(lambda: b2_window_masks(leap.drop(pd.Timestamp('2004-02-29')), 3,
        ['ERA5L_a', 'qmean_mmd'], ['ERA5L_a', 'qmean_mmd', 'qmax_mmd'], ['ECMWF_a']))


b2_check('rolling_mask_matches_explicit_windows', b2_test_window_oracle, 'count_oracle')
b2_require('count_oracle')

# %% Phần 6.2 - Đếm 451 ID và tính scaler bằng nguyên phương thức NH
# Bộ nhớ ánh xạ chứa train và warmup; chỉ dựng forcing phụ cho các ID probe.
COUNT_ROWS, YEAR_ROWS, Q_AUDIT, TARGET_STDS = [], [], [], {}
COUNT_MASKS, REAL_SCALERS, REAL_COUNTS = {}, {}, pd.DataFrame()
PROBE_BASINS = list(dict.fromkeys([*IDS[:2], *(b for b in PROBE_IDS if b in IDS)]))
FEATURES = sorted(set(HINDCAST + FORECAST + ['qmax']))
LENGTHS = (365, 730)
CALENDAR = pd.date_range(pd.Timestamp(PERIODS['train'][0]) - pd.Timedelta(days=max(LENGTHS)-1),
                         PERIODS['test'][1], name='date')
SCALER_BASINS = tuple(sorted(IDS))
REFERENCE_DIR = OUT / '_scratch/reference_input'
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)


def b2_write_probe_forcing(basin, reference_q, expected_frame):
    # Giữ chữ của các cột khí tượng gốc; chỉ thêm hai cột Q mới theo cùng cách B1b.
    original = pd.read_csv(FORCING_FILES[basin], sep=';', dtype=str, keep_default_na=False)
    dates = pd.DatetimeIndex(pd.to_datetime(original[['YYYY','MM','DD']].astype(int).rename(
        columns={'YYYY':'year','MM':'month','DD':'day'})))
    original.index = dates
    dates = pd.date_range(dates.min(), dates.max(), name='date')
    original = original.reindex(dates, fill_value='')
    original['YYYY'], original['MM'], original['DD'] = dates.year, dates.month, dates.day
    for name in B2_Q_INPUTS:
        values = reference_q[name].reindex(dates).to_numpy()
        original[name] = ['' if np.isnan(value) else format(float(value), '.17g') for value in values]
    destination = REFERENCE_DIR / 'A_basins_total_upstrm/2_timeseries/daily' / f'ID_{basin}.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    original.to_csv(destination, sep=';', index=False)
    gauge_destination = REFERENCE_DIR / 'D_gauges/2_timeseries/daily' / f'ID_{basin}.csv'
    gauge_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Q_FILES[basin], gauge_destination)
    restored = b2_read_daily(destination, HINDCAST + FORECAST).reindex(CALENDAR)
    np.testing.assert_array_equal(restored.to_numpy(dtype='float32'), expected_frame[HINDCAST + FORECAST].to_numpy(dtype='float32'))


if RUN_REAL_DATA:
    for relative in ('A_basins_total_upstrm/1_attributes/Catchment_attributes.csv', 'D_gauges/1_attributes/Gauge_attributes.csv'):
        target = REFERENCE_DIR / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RAW / relative, target)
    fit_dates = pd.date_range(CALENDAR[0], PERIODS['train'][1], name='date')
    memory_path = OUT / '_scratch/train_reference_float32.npy'
    train_memory = np.lib.format.open_memmap(memory_path, mode='w+', dtype='float32',
        shape=(len(IDS), len(fit_dates), len(FEATURES)))
    scaler_row = {basin: i for i, basin in enumerate(SCALER_BASINS)}
    for ordinal, basin in enumerate(IDS, 1):
        forcing = b2_read_daily(FORCING_FILES[basin], past + FORECAST).reindex(CALENDAR)
        reference_q, audit = b2_q_reference(RAW, basin, AREAS.loc[basin])
        for name in reference_q:
            forcing[name] = reference_q[name].reindex(CALENDAR)
        original_qmean = b2_read_daily(Q_FILES[basin], ['qmean'])['qmean']
        forcing['qmean'] = original_qmean.where(original_qmean.ge(0)).reindex(CALENDAR)
        audit['basin_id'] = basin
        Q_AUDIT.extend(audit.to_dict('records'))
        train_values = forcing.reindex(fit_dates)[FEATURES].astype('float32')
        train_values.loc[train_values.index < pd.Timestamp(PERIODS['train'][0]), 'qmax'] = np.nan
        if np.isinf(train_values.to_numpy()).any():
            raise ValueError(f'Giá trị vượt phạm vi float32 tại ID {basin}.')
        train_memory[scaler_row[basin]] = train_values.to_numpy()
        train_labels = train_values.qmax.to_numpy()
        TARGET_STDS[basin] = [float(np.nanstd(train_labels)) if np.count_nonzero(~np.isnan(train_labels)) > 1 else float('nan')]
        for length in LENGTHS:
            masks = b2_window_masks(forcing, length, BASE_HINDCAST, HINDCAST, FORECAST)
            for period, (start, end) in PERIODS.items():
                in_period = (CALENDAR >= start) & (CALENDAR <= end)
                row = {'basin_id': basin, 'period': period, 'sequence_total_days': length,
                    'calendar_days': int(in_period.sum()), 'label_present_days': int(forcing.loc[in_period,'qmax'].notna().sum())}
                row.update({name: int((mask & in_period).sum()) for name, mask in masks.items()})
                COUNT_ROWS.append(row)
                for year in sorted(set(CALENDAR[in_period].year)):
                    yearly = in_period & (CALENDAR.year == year)
                    YEAR_ROWS.append({'basin_id':basin, 'period':period, 'sequence_total_days':length,
                        'year':int(year), 'calendar_days':int(yearly.sum()),
                        **{name:int((mask & yearly).sum()) for name,mask in masks.items()}})
                if basin in PROBE_BASINS:
                    COUNT_MASKS[(basin,period,length)] = {name:mask[in_period].copy() for name,mask in masks.items()}
        if basin in PROBE_BASINS:
            b2_write_probe_forcing(basin, reference_q, forcing)
        if ordinal % 50 == 0 or ordinal == len(IDS):
            print(f'Phần 6.2: đọc/đếm {ordinal}/{len(IDS)} ID.', flush=True)
    train_memory.flush()
    for length in LENGTHS:
        start = pd.Timestamp(PERIODS['train'][0]) - pd.Timedelta(days=length-1)
        offset = fit_dates.get_loc(start)
        data = xr.Dataset({name: (('basin','date'), train_memory[:,offset:,i]) for i,name in enumerate(FEATURES)},
                          coords={'basin':list(SCALER_BASINS),'date':fit_dates[offset:]})
        holder = SimpleNamespace(cfg=Config(copy.deepcopy(B2_CONFIG_TEMPLATE)), scaler={})
        BaseDataset._setup_normalization(holder, data)
        static = STATIC_RAW.loc[list(SCALER_BASINS), sorted(STATIC)]
        holder.scaler.update(attribute_means=static.mean(), attribute_stds=static.std())
        scale = holder.scaler['xarray_feature_scale'].to_array().to_numpy()
        if not np.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError(f'Scaler mã nền có scale không hữu hạn/dương: L={length}')
        REAL_SCALERS[length] = copy.deepcopy(holder.scaler)
        holder.cfg.update_config({'train_dir': OUT / f'train_scalers/L{length}/train_data'})
        BaseDataset._dump_scaler(holder)
        RECEIPTS.append(holder.cfg.train_dir / 'train_data_scaler.yml')
        rows = [{'variable':name, 'center':float(holder.scaler['xarray_feature_center'][name]),
                 'scale':float(holder.scaler['xarray_feature_scale'][name]), 'sequence_total_days':length,
                 'dynamic_fit_start':str(start.date()), 'target_fit_start':PERIODS['train'][0],
                 'fit_end':PERIODS['train'][1], 'method':'BaseDataset._setup_normalization unchanged'} for name in FEATURES]
        b2_save(f'B2_train_scaler_L{length}.csv', pd.DataFrame(rows))
        del holder, data
    del train_memory
    if not memory_path.resolve().is_relative_to((OUT/'_scratch').resolve()):
        raise ValueError('Đường dẫn tệp tạm ngoài thư mục scratch của notebook.')
    memory_path.unlink()
    REAL_SCALER = REAL_SCALERS[365]
    REAL_COUNTS = pd.DataFrame(COUNT_ROWS)
    numeric = [c for c in REAL_COUNTS if c not in ('basin_id','period','sequence_total_days')]
    COUNT_SUMMARY = REAL_COUNTS.groupby(['period','sequence_total_days'],sort=False)[numeric].sum().reset_index()
    b2_save('B2_sample_counts_by_basin.csv',REAL_COUNTS)
    b2_save('B2_sample_counts_by_year_basin.csv',pd.DataFrame(YEAR_ROWS))
    b2_save('B2_sample_counts.csv',COUNT_SUMMARY)
    b2_save('B2_Q_source_audit.csv',pd.DataFrame(Q_AUDIT))
    b2_save('B2_train_target_std.csv',pd.DataFrame([{'basin_id':b,'sigma_mm_day':v[0]} for b,v in TARGET_STDS.items()]))
    b2_json(OUT/'B2_reference_input.json',{'derived_from_raw':True,'B1b_handoff_verified':False,
        'prepared_IDs':PROBE_BASINS,'raw_IDs_counted':len(IDS),'model_input_float32_roundtrip_verified':True,
        'reference_only':True,'source_raw':str(RAW),'source_library_commit':BASE_COMMIT})
    b2_table('6.2. Số mẫu toàn tập; B1b_inputs là hợp đồng, chưa là run huấn luyện',COUNT_SUMMARY)
else:
    REAL_SCALER = None
    print('Phần 6.2: not_run, chưa kiểm dữ liệu thật.')

# %% Phần 6.3 - Đối chiếu CSV A1 theo đúng hợp đồng biến và pha
# So phép đếm A2 cũ; không ép số mẫu B2 phải bằng số complete-case A1.
def b2_read_unique_a1(name):
    matches = sorted(p for p in a1_files if p.name == name)
    if not matches:
        return None
    digests = {b2_hash(path) for path in matches}
    if len(digests) != 1:
        raise ValueError(f'Nhiều bản A1 khác nội dung: {name}')
    return pd.read_csv(matches[0], dtype={'basin_id': str, 'contract_id': str})


A1_COMPARISON = []
groups_a1 = b2_read_unique_a1('A1_P7_sample_contract_groups.csv')
counts_a1 = b2_read_unique_a1('A1_P7_sample_counts.csv')
if RUN_REAL_DATA and groups_a1 is not None and counts_a1 is not None:
    def split_names(text):
        return set(str(text).split(' | ')) if pd.notna(text) else set()
    required = {'contract_id', 'dynamic_names', 'hindcast_names', 'forecast_names', 'target_names', 'contract_json'}
    if not required.issubset(groups_a1):
        raise ValueError(f'A1 contract groups thiếu cột: {required-set(groups_a1)}')
    def a1_scope_matches(row):
        spec = json.loads(row.contract_json)
        if (spec.get('dataset') != 'lamah_a' or spec.get('predict_last_n') != 1
                or spec.get('forecast_seq_length') != 1 or (spec.get('forecast_overlap') or 0) != 0
                or set(b2_flat(spec.get('static_attributes'))) != set(STATIC)):
            return False
        for field in ('use_frequencies', 'lagged_features', 'evolving_attributes', 'autoregressive_inputs', 'mass_inputs', 'dynamic_conceptual_inputs'):
            if spec.get(field):
                return False
        for period, (start, end) in PERIODS.items():
            for suffix, expected in (('_start_date', start), ('_end_date', end)):
                value = spec.get(period + suffix)
                if value is None:
                    return False
                date = pd.to_datetime(value, format='%d/%m/%Y') if isinstance(value, str) and '/' in value else pd.Timestamp(value)
                if date != pd.Timestamp(expected):
                    return False
        return (split_names(row.dynamic_names) == set(BASE_HINDCAST + FORECAST)
            and split_names(row.hindcast_names) == set(BASE_HINDCAST)
            and split_names(row.forecast_names) == set(FORECAST) and split_names(row.target_names) == {'qmax'})
    match = groups_a1.apply(a1_scope_matches, axis=1)
    matched_contracts = groups_a1.loc[match, 'contract_id'].tolist()
    for contract in matched_contracts:
        for _, a1 in counts_a1.loc[counts_a1.contract_id.eq(contract)].iterrows():
            scope = COUNT_SUMMARY.loc[(COUNT_SUMMARY.period == a1['period']) &
                                      (COUNT_SUMMARY.sequence_total_days == int(a1.sequence_total_days))]
            if len(scope) != 1 or int(a1.get('basins', 451)) != 451:
                continue
            for old_name, new_name in [('label_present_days', 'label_present_days'),
                    ('original_complete_with_label_days', 'original_A2'),
                    ('phase_complete_with_label_days', 'phase_complete_A2')]:
                if old_name not in a1:
                    continue
                expected, actual = int(a1[old_name]), int(scope.iloc[0][new_name])
                A1_COMPARISON.append({'contract_id': contract, 'period': a1['period'],
                    'sequence_total_days': int(a1.sequence_total_days), 'metric': old_name,
                    'A1_count': expected, 'B2_recount_old_contract': actual, 'difference': actual-expected,
                    'status': 'counts_match_source_identity_not_proven' if expected == actual else 'review_required'})
    if not A1_COMPARISON:
        A1_COMPARISON.append({'status': 'not_run_no_matching_contract', 'detail': 'Có CSV nhưng chưa có hợp đồng biến/pha khớp.'})
else:
    A1_COMPARISON.append({'status': 'not_run_missing_A1_tables_or_real_data', 'detail': 'Cần A1_P7_sample_contract_groups.csv và A1_P7_sample_counts.csv.'})
b2_save('B2_A1_count_comparison.csv', pd.DataFrame(A1_COMPARISON))
b2_table('6.3. Đối chiếu A1', A1_COMPARISON)
A1_COUNTS_OK = bool(A1_COMPARISON) and all(r['status'] == 'counts_match_source_identity_not_proven' for r in A1_COMPARISON)

# %% Phần 6.4 - Hồ sơ thiếu train và bất biến của phép đếm
# Giữ 451 ID; đếm nhãn, mẫu, lookup và lý do không có mẫu riêng nhau.
REAL_COUNT_CHECKS_OK = False
if RUN_REAL_DATA:
    def check_count_invariants():
        b2_assert(REAL_COUNTS.groupby(['period', 'sequence_total_days']).size().eq(451).all())
        b2_assert((REAL_COUNTS.masked_mean_B2 <= REAL_COUNTS.label_present_days).all())
        b2_assert((REAL_COUNTS.phase_complete_A2 >= REAL_COUNTS.original_A2).all())
        b2_assert((REAL_COUNTS.masked_mean_B2 >= REAL_COUNTS.phase_complete_B1b_inputs).all())
        for period in ('validation', 'test'):
            b2_assert(COUNT_SUMMARY.loc[COUNT_SUMMARY.period.eq(period), 'evaluation_lookup'].eq(len(pd.date_range(*PERIODS[period])) * len(IDS)).all())
    b2_check('calendar_labels_and_superset_relations', check_count_invariants, 'real_counts')
    profiles = REAL_COUNTS.loc[REAL_COUNTS.basin_id.isin(PROBE_IDS)].copy()
    b2_save('B2_profile_coverage.csv', profiles)
    b2_table('6.4. Hồ sơ train, khung 365', profiles.loc[profiles.period.eq('train') & profiles.sequence_total_days.eq(365)])
    b2_require('real_counts')
    REAL_COUNT_CHECKS_OK = True

# %% Phần 7.1 - Cấu hình probe thật và nguồn Q tham chiếu
# Chỉ giảm hidden_size cho mô hình ngẫu nhiên; cấu hình huấn luyện B2 vẫn giữ template.
probe_file = OUT/'probe_basins.txt'
probe_file.write_text('\n'.join(PROBE_BASINS)+'\n',encoding='utf-8')
RECEIPTS.append(probe_file)


def b2_real_config(length=365, horizon=1):
    cfg=copy.deepcopy(B2_CONFIG_TEMPLATE)
    cfg.update(data_dir=str(REFERENCE_DIR),run_dir=str(OUT/f'probe_logs/L{length}_H{horizon}'),
        train_dir=str(OUT/f'probe_logs/L{length}_H{horizon}/train_data'),experiment_name='B2_diagnostic_only',
        train_data_file=None,num_workers=0,batch_size=8,save_train_data=False,verbose=0,device='cpu',
        seq_length=length+horizon-1,forecast_seq_length=horizon,predict_last_n=horizon,
        timestep_counter=horizon>1,forecast_input_mode='first_day' if horizon>1 else 'all',hidden_size=16)
    cfg.pop('autoregressive_inputs',None)
    for period in PERIODS:
        cfg[period+'_basin_file']=str(probe_file)
        cfg[f'per_basin_{period}_periods_file']=None
    b2_contract(cfg)
    return Config(cfg)


probe_cfg=b2_real_config()
path=OUT/'B2_probe_config.yml'
if path.exists():
    path.unlink()
probe_cfg.dump_config(OUT,filename=path.name)
RECEIPTS.append(path)
b2_json(OUT/'B2_probe_settings.json',{'IDs':PROBE_BASINS,'source':'derived raw Q reference, not B1b handoff',
    'random_weights':True,'diagnostic_hidden_size':16,'official_template_hidden_size':B2_CONFIG_TEMPLATE['hidden_size'],
    'optimizer_steps':0,'B1b_handoff_verified':False})
print('Phần 7.1: probe trên dữ liệu tham chiếu; không phải huấn luyện chính thức.')

# %% Phần 7.2 - Lookup thật so với phép đếm độc lập ở 365 và 730
# Tạo LamaH thực cho tập hồ sơ nhỏ bằng scaler của 451 ID; không fit lại trên tập nhỏ.
LOOKUP_AUDIT, WINDOW_EXAMPLES, REAL_DATASETS = [], [], {}
REAL_LOADER_OK = False
if RUN_REAL_DATA:
    for length in (365, 730):
        cfg = b2_real_config(length=length)
        for period in PERIODS:
            ds = LamaH(cfg, is_train=period == 'train', period=period, scaler=copy.deepcopy(REAL_SCALERS[length]))
            frequency = ds.frequencies[0]
            actual = {basin: set() for basin in PROBE_BASINS}
            for basin, indices in ds.lookup_table.values():
                actual[basin].add(pd.Timestamp(ds._dates[basin][frequency][indices[0]]))
            dates = pd.date_range(*PERIODS[period])
            for basin in PROBE_BASINS:
                if (basin, period, length) not in COUNT_MASKS:
                    raise ValueError(f'Thiếu mask đối chứng cho {basin}.')
                mask_name = 'masked_mean_B2' if period == 'train' else 'evaluation_lookup'
                expected = set(dates[COUNT_MASKS[(basin, period, length)][mask_name]])
                ok = actual[basin] == expected
                LOOKUP_AUDIT.append({'basin_id': basin, 'period': period, 'sequence_total_days': length,
                    'expected': len(expected), 'actual': len(actual[basin]), 'same_dates': ok})
                if not ok:
                    b2_save('B2_real_lookup_audit.csv', pd.DataFrame(LOOKUP_AUDIT))
                    raise AssertionError(f'Lookup thật khác bộ đếm độc lập: {basin}, {period}, L={length}')
            sample = ds[0]
            sample_dates = pd.DatetimeIndex(sample['date'])
            WINDOW_EXAMPLES.append({'period': period, 'sequence_total_days': length,
                'first_date': sample_dates[0], 'hindcast_last': sample_dates[-2], 'target': sample_dates[-1],
                'hindcast_steps': len(sample['x_d_hindcast'][HINDCAST[0]]),
                'forecast_steps': len(sample['x_d_forecast'][FORECAST[0]])})
            if length == 365:
                REAL_DATASETS[period] = ds
            else:
                del ds
        print(f'7.2. Đã đối chiếu ngày lookup thật L={length}.', flush=True)
    REAL_LOADER_OK = True
    b2_save('B2_real_lookup_audit.csv', pd.DataFrame(LOOKUP_AUDIT))
    b2_save('B2_real_window_examples.csv', pd.DataFrame(WINDOW_EXAMPLES))
    b2_table('7.2. Ví dụ cửa sổ ngày thật', WINDOW_EXAMPLES)
    b2_table('7.2. Lookup thật: tổng hợp, vẫn giữ CSV từng ID', pd.DataFrame(LOOKUP_AUDIT).groupby(
        ['period', 'sequence_total_days']).agg(IDs=('basin_id', 'size'), expected=('expected', 'sum'), actual=('actual', 'sum'), all_dates_match=('same_dates', 'all')).reset_index())
else:
    print('7.2. not_run: chưa kiểm lookup từ dữ liệu thật.')

# %% Phần 7.3 - Batch thật, loader vector hóa và loss NH
# Forward/backward bằng trọng số ngẫu nhiên; không tạo optimizer hoặc chạy epoch.
REAL_MODEL_OK=False
REAL_MODEL_REPORT=[]
if RUN_REAL_DATA and REAL_LOADER_OK:
    from neuralhydrology.training import get_loss_obj
    ds=REAL_DATASETS['train']
    b2_compare_batch_paths(ds)
    torch.manual_seed(111)
    real_model=SequentialForecastLSTM(probe_cfg)
    batch=BaseDataset.collate_fn([ds[i] for i in range(min(8,len(ds)))])
    prediction=real_model(batch)
    loss,_=get_loss_obj(probe_cfg)(prediction,batch)
    real_model.zero_grad(set_to_none=True)
    loss.backward()
    grads=[p.grad for p in real_model.parameters() if p.grad is not None]
    pred=prediction['y_hat'][:,-1:]
    REAL_MODEL_OK=bool(torch.isfinite(pred).all() and torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads))
    if not REAL_MODEL_OK:
        raise AssertionError('Batch thật có dự báo/loss/gradient không hữu hạn.')
    REAL_MODEL_REPORT=[{'role':'diagnostic_random_weights','batch_size':len(batch['y']),
        'loss_name':'NH get_loss_obj(NSE)','diagnostic_loss_not_skill':float(loss.detach()),
        'prediction_shape':list(pred.shape),'gradients_finite':True,'ordinary_vectorized_equal':True,
        'optimizer_steps':0,'hidden_size':16,'device':'cpu'}]
    b2_save('B2_real_model_probe.csv',pd.DataFrame(REAL_MODEL_REPORT))
    b2_table('7.3. Probe mô hình trên batch thật',REAL_MODEL_REPORT)
else:
    print('Phần 7.3: not_run, chưa có loader thật được kiểm.')

# %% Phần 8.1 - Cửa sổ nhiều hạn và ngày phát hành t
# Giữ 4 ngày hindcast trong fixture; horizon 7 làm tổng khung 11, không giữ nhầm tổng khung 5.
HYBRID_CFG = b2_fixture_config(seq_length=11, forecast_seq_length=7, predict_last_n=7,
                              timestep_counter=True, forecast_input_mode='first_day')
HYBRID_DS = LamaH(HYBRID_CFG, is_train=False, period='validation', scaler=copy.deepcopy(SYN_SCALER))
hybrid_sample = HYBRID_DS[0]
hybrid_dates = pd.DatetimeIndex(hybrid_sample['date'])
ORIGIN = hybrid_dates[-7]
HORIZON_TABLE = pd.DataFrame({'release_date': [ORIGIN] * 7, 'target_date': hybrid_dates[-7:],
    'lead_index': range(1, 8), 'ECMWF_available_from_this_release': [True] + [False] * 6,
    'last_Q_input_date': [hybrid_dates[-8]] * 7})
b2_save('B2_hybrid_calendar_example.csv', HORIZON_TABLE)
b2_table('8.1. Lịch giả minh họa cách (b)', HORIZON_TABLE)

# %% Phần 8.2 - Kiểm che forecast, counter và tính bất biến của dữ liệu lưu chung
# Cơ chế che chạy trong __getitem__; không chỉ che thủ công batch trước khi gọi mô hình.
def b2_test_hybrid_values():
    sample = HYBRID_DS[0]
    values = sample['x_d_forecast']['ECMWF_a']
    b2_assert(torch.isfinite(values[0]).all() and torch.isnan(values[1:]).all())
    torch.testing.assert_close(sample['x_d_forecast']['forecast_counter'], torch.arange(1., 8.).unsqueeze(-1))
    b2_assert(torch.equal(sample['x_d_hindcast']['hindcast_counter'], torch.zeros((4, 1))))
    basin, idx = HYBRID_DS.lookup_table[0]
    original = HYBRID_DS._x_d[basin][HYBRID_DS.frequencies[0]]['ECMWF_a'][idx[0]-6:idx[0]+1]
    b2_assert(torch.isfinite(original).all(), 'Việc lấy mẫu không được che mảng nguồn dùng chung.')


def b2_test_hybrid_overlap():
    a = copy.deepcopy(HYBRID_DS[0])
    _ = HYBRID_DS[min(1, len(HYBRID_DS)-1)]
    b = HYBRID_DS[0]
    for phase in ('hindcast', 'forecast'):
        for name in a[f'x_d_{phase}']:
            torch.testing.assert_close(a[f'x_d_{phase}'][name], b[f'x_d_{phase}'][name], rtol=0, atol=0, equal_nan=True)


def b2_test_hybrid_forward():
    torch.manual_seed(333)
    model = SequentialForecastLSTM(HYBRID_CFG)
    batch = BaseDataset.collate_fn([HYBRID_DS[0]])
    model.eval()
    with torch.no_grad():
        predictions = model(batch)['y_hat'][:, -7:, :]
    b2_assert(predictions.shape == (1, 7, 1) and torch.isfinite(predictions).all())


for name, operation in [('hybrid_mask_and_counters_exact', b2_test_hybrid_values),
                        ('hybrid_no_source_mutation', b2_test_hybrid_overlap),
                        ('seven_step_forward_finite', b2_test_hybrid_forward)]:
    b2_check(name, operation, 'hybrid_forecast')


b2_check('hybrid_ordinary_vectorized_batches_equal', lambda: b2_compare_batch_paths(HYBRID_DS), 'hybrid_forecast')

# %% Phần 8.3 - Can thiệp dữ liệu tương lai và đối chiếu đầu ra
# Thay Q/tái phân tích từ t và HRES sau t trong đường đọc của loader; không sửa Input.
from unittest.mock import patch as mock_patch
import neuralhydrology.datasetzoo.lamah as lamah_module


def b2_test_causality():
    original_forcing = lamah_module.load_lamah_forcing
    original_discharge = lamah_module.load_lamah_discharge

    def change_forcing(*args, **kwargs):
        frame = original_forcing(*args, **kwargs).copy()
        frame.loc[frame.index >= ORIGIN, ['ERA5L_a','qmean_mmd','qmax_mmd']] += 10000.
        frame.loc[frame.index > ORIGIN, 'ECMWF_a'] += 20000.
        return frame

    def change_discharge(*args, **kwargs):
        frame = original_discharge(*args, **kwargs).copy()
        for variable in ('qmean', 'qmax'):
            if variable in frame:
                frame.loc[frame.index >= ORIGIN, variable] += 30000.
        return frame

    with mock_patch.object(lamah_module, 'load_lamah_forcing', side_effect=change_forcing), \
         mock_patch.object(lamah_module, 'load_lamah_discharge', side_effect=change_discharge):
        changed = LamaH(HYBRID_CFG, is_train=False, period='validation', scaler=copy.deepcopy(SYN_SCALER))
        sample_changed = changed[0]
    first, second = BaseDataset.collate_fn([HYBRID_DS[0]]), BaseDataset.collate_fn([sample_changed])
    for key in ('x_d_hindcast', 'x_d_forecast'):
        for variable in first[key]:
            torch.testing.assert_close(first[key][variable], second[key][variable], rtol=0, atol=0, equal_nan=True)
    torch.manual_seed(111)
    model = SequentialForecastLSTM(HYBRID_CFG).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(first)['y_hat'], model(second)['y_hat'], rtol=0, atol=0)
    b2_assert(not torch.allclose(first['y'], second['y'], equal_nan=True), 'Phải thực sự can thiệp nhãn tương lai trong ca kiểm.')


def b2_test_valid_forecast_is_used():
    original_forcing = lamah_module.load_lamah_forcing
    def change_release(*args, **kwargs):
        frame = original_forcing(*args, **kwargs).copy()
        frame.loc[frame.index == ORIGIN, 'ECMWF_a'] += 100.
        return frame
    with mock_patch.object(lamah_module, 'load_lamah_forcing', side_effect=change_release):
        changed = LamaH(HYBRID_CFG, is_train=False, period='validation', scaler=copy.deepcopy(SYN_SCALER))[0]
    b2_assert(not torch.equal(changed['x_d_forecast']['ECMWF_a'][0], HYBRID_DS[0]['x_d_forecast']['ECMWF_a'][0]))


b2_check('future_raw_intervention_same_inputs_same_predictions', b2_test_causality, 'hybrid_forecast')
b2_check('release_day_forecast_remains_active', b2_test_valid_forecast_is_used, 'hybrid_forecast')
b2_require('hybrid_forecast')

# %% Phần 8.4 - Kiểm lưu/đọc scaler và batch nhiều hạn trên raw
# Giữ định dạng scaler NH; std nhãn vẫn theo kỳ đang nạp, batch nhiều hạn chỉ là probe.
def b2_test_scaler_roundtrip():
    from neuralhydrology.datautils.utils import load_scaler
    restored = load_scaler(SYN_CFG.run_dir)
    for key in ('xarray_feature_center', 'xarray_feature_scale'):
        for variable in SYN_SCALER[key]:
            np.testing.assert_allclose(restored[key][variable], SYN_SCALER[key][variable], rtol=1e-6, atol=1e-7, equal_nan=True)
    ev = LamaH(SYN_CFG, is_train=False, period='test', scaler=restored)
    b2_assert(len(ev) > 0 and torch.isfinite(ev[0]['x_d_hindcast']['ERA5L_a']).all())


b2_check('NH_scaler_yaml_roundtrip_unchanged_format', b2_test_scaler_roundtrip, 'scaler_roundtrip')
b2_require('scaler_roundtrip')
REAL_HYBRID_OK = False
if RUN_REAL_DATA and REAL_LOADER_OK:
    cfg = b2_real_config(horizon=7)
    cfg._cfg['validation_end_date'] = pd.Timestamp('2010-01-21')
    ds = LamaH(cfg, is_train=False, period='validation', basin=IDS[0], scaler=copy.deepcopy(REAL_SCALER))
    b2_compare_batch_paths(ds)
    sample = ds[0]
    dates = pd.DatetimeIndex(sample['date'])
    for variable in FORECAST:
        b2_assert(torch.isnan(sample['x_d_forecast'][variable][1:]).all())
    torch.manual_seed(222)
    model = SequentialForecastLSTM(cfg).eval()
    with torch.no_grad():
        prediction = model(BaseDataset.collate_fn([sample]))['y_hat'][:, -7:, :]
    REAL_HYBRID_OK = bool(torch.isfinite(prediction).all())
    if not REAL_HYBRID_OK:
        raise AssertionError('Probe nhiều hạn trên raw chưa hữu hạn.')
    real_hybrid_rows = [{'basin_id': IDS[0], 'origin': dates[-7],
        'hindcast_last': dates[-8], 'last_target': dates[-1], 'hindcast_steps': len(dates)-7,
        'forecast_steps': 7, 'HRES_from_future_releases_used': False, 'prediction_finite': REAL_HYBRID_OK}]
    b2_save('B2_real_hybrid_probe.csv',pd.DataFrame(real_hybrid_rows))
    b2_table('8.4. Cửa sổ nhiều hạn thật',real_hybrid_rows)

# %% Phần 9.1 - Trạng thái kiểm độc lập và điểm chờ B1b
# Sự hiện diện của config không chứng minh bàn giao; chưa merge hoặc huấn luyện.
B1_REVIEW={'provided_commit':B1_COMMIT,'provided_config':B1_CONFIG_PATH,
    'candidate_base_commit':BASE_COMMIT,'integrated':False,'status':'waiting_for_B1b_handoff_and_review'}
if B1_CONFIG_PATH:
    path=Path(B1_CONFIG_PATH)
    if not path.is_file():
        raise FileNotFoundError(f'Không thấy config B1b được cung cấp: {path}')
    received=yaml.load(path.read_text(encoding='utf-8'))
    B1_REVIEW.update(config_sha256=b2_hash(path),status='config_received_not_yet_verified',
                     hindcast_inputs=b2_flat(received.get('hindcast_inputs')))
b2_json(OUT/'B2_B1_review.json',B1_REVIEW)
ALL_SYNTHETIC_OK=bool(CHECKS) and all(row['passed'] for row in CHECKS)
B2_LOADER_INDEPENDENT_OK=bool(ALL_SYNTHETIC_OK and REAL_COUNT_CHECKS_OK and REAL_LOADER_OK
    and REAL_MODEL_OK and REAL_HYBRID_OK and A1_COUNTS_OK)
B2_READY_FOR_TRAINING=False
STATUS=[{'item':item,'status':'pass' if ok else 'review_or_not_run','evidence':evidence} for item,ok,evidence in [
    ('synthetic_reference_loader_model',ALL_SYNTHETIC_OK,'B2_checks.csv'),
    ('451_ID_raw_counts',REAL_COUNT_CHECKS_OK,'B2_sample_counts.csv'),
    ('A1_comparison',A1_COUNTS_OK,'B2_A1_count_comparison.csv'),
    ('real_lookup_365_730',REAL_LOADER_OK,'B2_real_lookup_audit.csv'),
    ('real_model_gradient_and_vectorized_batch',REAL_MODEL_OK,'B2_real_model_probe.csv'),
    ('hybrid_raw_data_probe',REAL_HYBRID_OK,'B2_real_hybrid_probe.csv')]]
STATUS.extend([{'item':'B1b_handoff','status':B1_REVIEW['status'],'evidence':'B2_B1_review.json'},
    {'item':'CUDA_optional_probe','status':GPU_PROBE_STATUS,'evidence':'B2_checks.csv'},
    {'item':'merge_into_research','status':'pending_review_and_tests','evidence':'B2_loader_candidate.patch'},
    {'item':'official_three_seed_training','status':'waiting_for_B1b_and_pinned_library_commit','evidence':''}])
b2_save('B2_handoff_status.csv',pd.DataFrame(STATUS))
b2_table('9.1. Kết quả và điểm chờ',STATUS)

# %% Phần 9.2 - Lưu manifest, tổng kết và điểm dừng
# Kết thúc kiểm độc lập; các bước huấn luyện chờ dữ liệu/config B1b đã đối chiếu.
summary={'complete':True,'scope':'independent_loader_model_checks_not_training',
    'base_commit':BASE_COMMIT,'candidate_patch_sha256':patch_sha256,
    'checks_run':len(CHECKS),'checks_passed':sum(row['passed'] for row in CHECKS),
    'real_data_run':RUN_REAL_DATA,'A1_counts_ok':A1_COUNTS_OK,
    'B2_LOADER_INDEPENDENT_OK':B2_LOADER_INDEPENDENT_OK,'B2_READY_FOR_TRAINING':B2_READY_FOR_TRAINING,
    'B1b_integrated':False,'official_training_started':False,'optimizer_steps':0,
    'full_451_raw_counting':REAL_COUNT_CHECKS_OK,'full_451_loader_instantiation':False,
    'real_loader_probe_IDs':PROBE_BASINS,'GPU_probe_status':GPU_PROBE_STATUS,
    'executed_in_Kaggle':IS_KAGGLE,'elapsed_seconds':time.perf_counter()-STARTED}
b2_json(OUT/'B2_summary.json',summary)
b2_json(OUT/'environment.json',{'base_commit':BASE_COMMIT,'candidate_patch_sha256':patch_sha256,
    'packages':VERSIONS,'input_sources':str(OUT/'B2_input_sources.json'),'B1b_handoff_verified':False})
paths=set(RECEIPTS)|{CANDIDATE_PACKAGE/name for name in patched}
rows=[]
for path in sorted(paths):
    if not path.is_file() or not path.is_relative_to(OUT):
        continue
    relative=path.relative_to(OUT).as_posix()
    if relative.startswith(('_scratch/','synthetic_fixture/','synthetic_logs/','probe_logs/')) or path.name in ('manifest.csv','B2_output_manifest.csv'):
        continue
    rows.append({'file':relative,'bytes':path.stat().st_size,'sha256':b2_hash(path)})
b2_save('manifest.csv',pd.DataFrame(rows))
b2_save('B2_output_manifest.csv',pd.DataFrame(rows))
b2_table('9.2. Tổng kết kiểm độc lập',[{'checks_passed':summary['checks_passed'],'checks_run':summary['checks_run'],
    'raw_IDs_counted':len(IDS) if REAL_COUNT_CHECKS_OK else 0,'probe_IDs':len(PROBE_BASINS),
    'independent_checks_ok':B2_LOADER_INDEPENDENT_OK,'ready_for_training':False,'optimizer_steps':0}])
print('Hoàn thành notebook kiểm độc lập; output:',OUT)
print('B2_LOADER_INDEPENDENT_OK =',B2_LOADER_INDEPENDENT_OK)
print('B2_READY_FOR_TRAINING = False: chờ bàn giao B1b, đối chiếu cấu hình/dữ liệu và commit thư viện đã duyệt.')
