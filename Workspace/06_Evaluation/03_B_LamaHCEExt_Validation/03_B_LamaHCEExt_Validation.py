# Validation bước B - T4, Internet On để cài đúng commit; không huấn luyện.
# Tên Kaggle: 03_B_LamaHCEExt_Validation. Output: /kaggle/working/b_validation.
# Add Input, nguyên Output, Version 1 của 15 notebook sau và Dataset lamah-ce-ext:
# 04_B1b_LamaHCEExt_Train_Seed111, 04_B1b_LamaHCEExt_Train_Seed222, 04_B1b_LamaHCEExt_Train_Seed333.
# 06_B2_LamaHCEExt_Train_Seed111, 06_B2_LamaHCEExt_Train_Seed222, 06_B2_LamaHCEExt_Train_Seed333.
# 08_B3_LamaHCEExt_Train_P10S12_Seed111, 08_B3_LamaHCEExt_Train_P10S12_Seed222, 08_B3_LamaHCEExt_Train_P10S12_Seed333.
# 08_B3_LamaHCEExt_Train_P05S05_Seed111, 08_B3_LamaHCEExt_Train_P05S05_Seed222, 08_B3_LamaHCEExt_Train_P05S05_Seed333.
# 05_B1_LamaHCEExt_Compare, 07_B2_LamaHCEExt_Compare, 09_B3_LamaHCEExt_Compare.
# Phần 1: môi trường; 2: nguồn/checkpoint; 3: lịch/nhãn; 4: kịch bản; 5: suy luận;
# Phần 6: giao ngày chung và bảng bàn giao; 7: hình; 8: môi trường/manifest.
# S1 xuất riêng Q1/Q3/Q7. B1b điền chỉ ô bị che: LOCF từ phần còn nhìn thấy trong mẫu,
# không có giá trị quá khứ thì khí hậu cùng tháng-ngày chỉ từ train, rồi trung bình train.
# Không chọn B*, không dùng kết quả test; chỉ đọc metadata nguồn Compare để khóa danh tính.

# %% Phần 1 - Môi trường suy luận và commit cố định
# Giữ mã nguồn NeuralHydrology nguyên trạng; dùng cùng commit B2/B3, kiểm tái lập B1b ở S0.
import copy
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

NH_COMMIT = '2ccaa147a6ec87d0f56fa3e2f11b1bf38079253f'
B1_COMMIT = 'f229e02'
NOTEBOOK = '03_B_LamaHCEExt_Validation'
INPUT_ROOT = Path('/kaggle/input')
WORK_DIR = Path('/kaggle/working')
OUT_DIR = WORK_DIR / 'b_validation'

if __name__ == '__main__':
    if not WORK_DIR.exists():
        raise RuntimeError('Notebook chính thức chạy trên Kaggle T4; kiểm cục bộ dùng các hàm riêng trên dữ liệu giả.')
    missing = []
    for package in ('ruamel.yaml', 'tensorboard', 'PyYAML'):
        try:
            metadata.version(package)
        except metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', *missing])
    try:
        installed = json.loads(metadata.distribution('neuralhydrology').read_text('direct_url.json'))['vcs_info']['commit_id']
    except (metadata.PackageNotFoundError, TypeError, KeyError):
        installed = None
    if installed != NH_COMMIT:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps',
                               f'git+https://github.com/Ticasslo/neuralhydrology.git@{NH_COMMIT}'])

import numpy as np
import pandas as pd
import torch
import yaml
import xarray as xr
from IPython.display import HTML, display
from neuralhydrology.datautils.gpu_loader import GpuBatchLoader
from neuralhydrology.datautils.utils import load_basin_file, load_scaler
from neuralhydrology.datasetzoo.lamah import LamaH, load_lamah_attributes, load_lamah_discharge, _normalize_discharge
from neuralhydrology.modelzoo import get_model
from neuralhydrology.evaluation.metrics import nse as nh_nse
from neuralhydrology.utils.config import Config

SEEDS = (111, 222, 333)
STEPS = ('B1b', 'B2', 'B3_P10S12', 'B3_P05S05')
SCENARIOS = ('S0', 'S1_Q1', 'S1_Q3', 'S1_Q7', 'S2', 'S3', 'S4')
Q_INPUTS = ('qmean_mmd', 'qmax_mmd')
TRAINING_VERSION = 1
OBS_ATOL = OBS_RTOL = 1e-5
NSE_TOL = 1e-5
IMPUTATION = 'masked_cells_only_LOCFin_visible_window_then_train_month_day_then_train_feature_mean'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def show(title, frame, digits=5):
    display(HTML(f'<h4>{title}</h4>'))
    display(frame.style.hide(axis='index').format(precision=digits, na_rep='-'))


def inside(root, relative):
    root = Path(root).resolve()
    # Posix paths are the portable paths recorded by the Kaggle source manifests.
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError(f'Đường dẫn nguồn không hợp lệ: {relative}')
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f'Đường dẫn ngoài nguồn Input: {relative}')
    return path


def unique_file(root, name):
    paths = sorted(Path(root).rglob(name))
    if len(paths) != 1:
        raise ValueError(f'Cần đúng một {name}, thấy {len(paths)}; kiểm lại danh sách Add Input.')
    return paths[0]


def verify_files(root, manifest_path, required):
    frame = pd.read_csv(manifest_path)
    column = 'file' if 'file' in frame else 'tệp'
    if column not in frame or 'sha256' not in frame or frame[column].duplicated().any():
        raise ValueError(f'Manifest thiếu cột hoặc trùng đường dẫn: {manifest_path}')
    frame = frame.set_index(column)
    # Hash only files actually consumed; no need to read or score test prediction pickles.
    for path in required:
        relative = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
        if relative not in frame.index or not re.fullmatch('[0-9a-f]{64}', str(frame.loc[relative, 'sha256'])):
            raise ValueError(f'Manifest thiếu/hash sai: {relative}')
        checked = inside(root, relative)
        if not checked.is_file() or sha256(checked) != frame.loc[relative, 'sha256']:
            raise ValueError(f'Hash tệp nguồn không khớp: {relative}')
        if 'bytes' in frame and checked.stat().st_size != int(frame.loc[relative, 'bytes']):
            raise ValueError(f'Kích thước tệp nguồn không khớp: {relative}')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ids_of(values):
    ids = tuple(str(int(x)) for x in values)
    if len(set(ids)) != len(ids):
        raise ValueError('Danh sách lưu vực bị trùng.')
    return ids


def scalar_date(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError('Chỉ hỗ trợ một khoảng ngày liên tục cho mỗi kỳ.')
        value = value[0]
    dayfirst = not isinstance(value, str) or re.match(r'^\d{4}-\d{2}-\d{2}', value) is None
    return pd.Timestamp(pd.to_datetime(value, dayfirst=dayfirst)).normalize()


def nse_by_id(obs, sim, mask=None):
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(sim)
    if mask is not None:
        valid &= mask
    count = valid.sum(axis=0)
    center = np.divide(np.where(valid, obs, 0).sum(axis=0), count,
                       out=np.zeros(obs.shape[1]), where=count > 0)
    denominator = np.where(valid, (obs-center)**2, 0).sum(axis=0)
    numerator = np.where(valid, (obs-sim)**2, 0).sum(axis=0)
    result = np.full(obs.shape[1], np.nan)
    np.divide(numerator, denominator, out=result, where=(count > 1) & (denominator > 0))
    return 1-result, count


def native_library_nse(obs, sim):
    # S0 reproduction must use the source Tester's arithmetic AND dtype. Independent
    # float64 scoring remains nse_by_id; float32 rounding can exceed 1e-5 for very negative NSE.
    obs, sim = np.asarray(obs), np.asarray(sim)
    if not (np.isfinite(obs) & np.isfinite(sim)).any():
        return float('nan')
    observed = xr.DataArray(obs, dims=['datetime'])
    predicted = xr.DataArray(sim, dims=['datetime'])
    return nh_nse(observed, xr.where(predicted < 0, 0, predicted))


if __name__ == '__main__':
    vcs = json.loads(metadata.distribution('neuralhydrology').read_text('direct_url.json'))
    if vcs['vcs_info']['commit_id'] != NH_COMMIT:
        raise ValueError('Không nạp đúng commit thư viện đã khóa.')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Chọn GPU T4 rồi Run All; đây là suy luận, không cần chạy train lại.')
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise RuntimeError('Output đã có tệp; dùng phiên sạch để không trộn các lần chạy.')
    (OUT_DIR / 'contracts').mkdir(parents=True, exist_ok=True)
    show('Môi trường suy luận', pd.DataFrame([
        {'package': name, 'version': metadata.version(name)} for name in
        ('numpy', 'pandas', 'xarray', 'torch', 'scipy', 'numba', 'ruamel.yaml', 'tensorboard')]))

# %% Phần 2 - Khóa nguồn Version 1, checkpoint và cấu hình của 12 run
# SHA lấy từ Compare Version 1 đã kiểm trong Evaluation Version 8; không dùng số test để chọn.
COMPARE_PINS = {
    'B1': ('b1_summary.json', '8290a961e2b65d4d97426b6b0c9ca7172df0f311593a169e43984ae38321adf8',
           '8ee3102ee1fbcacacebf94afc5b04801a9d2c6ab59b1c38d699a58b2e6d3d2bb'),
    'B2': ('b2_summary.json', 'b8c4ac9a3f3ef68c33711f4af8e575f7620c706e7f64fedb03c34b630a47973c',
           '65d8821e7a1d3977f43441df262296ff803d621c4cc10ea7ffd115d1150a0cc6'),
    'B3': ('b3_summary.json', '5905009431072896bf2c0d04be5a739f6de1cb2ce63fb9cf0d66930a5796ced1',
           '6b1353ac18b5ee37d8b4ad3ea8371705dfff1e92ac94507ce107201981a20a43')}


def compare_validation_values(step, summary):
    if step == 'B1b':
        return summary['B1']['seed_val_nse_median']['B1b']
    if step == 'B2':
        return summary['B2']['seed_val_nse_median']
    return summary['B3']['mask_levels'][step.removeprefix('B3_')]['seed_val_nse_median']


def validate_compare_source(step, summary, environment):
    # B1 Compare V1 predates the complete/step fields used by B2/B3. Authenticate its
    # own schema instead of treating an absent, never-exported flag as a failed run.
    if step == 'B1':
        if (environment.get('notebook') != '05_B1_LamaHCEExt_Compare'
                or not str(environment.get('neuralhydrology_commit_of_runs', '')).startswith(B1_COMMIT)
                or set(summary.get('contracts', {})) != {f'{s}_{seed}' for s in ('B1a', 'B1b') for seed in SEEDS}):
            raise ValueError('Compare B1 không đúng schema Version 1: notebook/commit/sáu hợp đồng.')
        epochs = summary.get('saved_epoch', {}).get('B1b', {})
        values = summary.get('seed_val_nse_median', {}).get('B1b', {})
        origins = environment.get('inputs', {}).get('B1b', {})
        version_key = 'version'
    else:
        if (summary.get('complete') is not True or summary.get('step') != step
                or summary.get('library_commit') != NH_COMMIT
                or environment.get('notebook') != {'B2': '07_B2_LamaHCEExt_Compare', 'B3': '09_B3_LamaHCEExt_Compare'}[step]
                or environment.get('library_commit_of_runs') != NH_COMMIT):
            raise ValueError(f'Compare {step} chưa hoàn thành hoặc sai notebook/commit.')
        if step == 'B3':
            levels = summary.get('mask_levels', {})
            if set(levels) != {'P10S12', 'P05S05'}:
                raise ValueError('Compare B3 thiếu hoặc thừa mức che.')
            groups = [(levels[level].get('saved_epoch', {}), levels[level].get('seed_val_nse_median', {}),
                       environment.get('inputs', {}).get('B3', {}).get(level, {})) for level in levels]
        else:
            groups = [(summary.get('saved_epoch', {}), summary.get('seed_val_nse_median', {}),
                       environment.get('inputs', {}).get('B2', {}))]
        version_key = 'version_expected'
    if step == 'B1':
        groups = [(epochs, values, origins)]
    seed_keys = {str(seed) for seed in SEEDS}
    for epochs, values, origins in groups:
        if set(epochs) != seed_keys or set(values) != seed_keys or set(origins) != seed_keys:
            raise ValueError(f'Compare {step} thiếu hoặc trùng ba seed validation.')
        for seed in seed_keys:
            if (type(epochs[seed]) is not int or not 1 <= epochs[seed] <= 30
                    or isinstance(values[seed], bool) or not isinstance(values[seed], (int, float))
                    or not np.isfinite(values[seed]) or type(origins[seed].get(version_key)) is not int
                    or origins[seed][version_key] != TRAINING_VERSION):
                raise ValueError(f'Compare {step} seed {seed}: epoch/validation/Version không hợp lệ.')


def compare_sources(input_root):
    summaries, environments, paths = {}, {}, []
    for step, (name, summary_hash, env_hash) in COMPARE_PINS.items():
        path = unique_file(input_root, name)
        env = path.parent / 'environment.json'
        if sha256(path) != summary_hash or sha256(env) != env_hash:
            raise ValueError(f'Nguồn Compare {step} khác Version 1 đã bàn giao.')
        verify_files(path.parent, path.parent/'manifest.csv', [path, env])
        summaries[step], environments[step] = read_json(path), read_json(env)
        validate_compare_source(step, summaries[step], environments[step])
        paths.extend([path, env, path.parent/'manifest.csv'])
    thresholds = [summaries['B1']['bstar_threshold_val_nse_sd'],
                  summaries['B2']['B1b_validation_comparison']['B0_Bstar_threshold'],
                  summaries['B3']['bstar_threshold_val_nse_sd']]
    if not all(np.isfinite(x) and x > 0 and x == thresholds[0] for x in thresholds):
        raise ValueError('Các nguồn không thống nhất ngưỡng nhiễu validation B0.')
    return summaries, environments, paths, thresholds[0]


def source_spec(step, seed, summaries, environments, candidates):
    if step == 'B1b':
        # B1 Compare V1 does not store per-run environment hashes. B2's pinned environment
        # carries that handoff hash, so authenticate the intermediate B2 environment first.
        ep, env = candidates[('B2', seed)]
        if sha256(ep) != environments['B2']['inputs']['B2'][str(seed)]['environment_sha256']:
            raise ValueError('Environment B2 dùng làm khóa nguồn B1b không đúng hash Compare.')
        origin = env['B1b_source']
        if origin['seed'] != seed or origin['version'] != 1 or not origin['library_commit'].startswith(B1_COMMIT):
            raise ValueError('B2 không tham chiếu đúng B1b cùng seed Version 1.')
        return origin['environment_sha256'], summaries['B1']['saved_epoch']['B1b'][str(seed)]
    if step == 'B2':
        origin = environments['B2']['inputs']['B2'][str(seed)]
        return origin['environment_sha256'], summaries['B2']['saved_epoch'][str(seed)]
    level = step.removeprefix('B3_')
    origin = environments['B3']['inputs']['B3'][level][str(seed)]
    return origin['environment_sha256'], summaries['B3']['mask_levels'][level]['saved_epoch'][str(seed)]


def load_sources(input_root, summaries, environments):
    candidates = {}
    for ep in Path(input_root).rglob('environment.json'):
        if ep.parent.name not in ('b1b_output', 'b2_output', 'b3_output'):
            continue
        env = read_json(ep)
        step = {'b1b_output': 'B1b', 'b2_output': 'B2'}.get(ep.parent.name, 'B3_'+str(env.get('mask_level')))
        key = step, env.get('seed')
        if key in candidates or step not in STEPS or key[1] not in SEEDS:
            raise ValueError(f'Run nguồn trùng/ngoài hợp đồng: {key}')
        candidates[key] = ep, env
    if set(candidates) != {(step, seed) for step in STEPS for seed in SEEDS}:
        raise ValueError(f'Cần đủ 12 Output train; đang thấy {sorted(candidates)}.')
    records, consumed = [], []
    reference = None
    for (step, seed), (ep, env) in sorted(candidates.items()):
        expected_hash, epoch = source_spec(step, seed, summaries, environments, candidates)
        if sha256(ep) != expected_hash:
            raise ValueError(f'{step}/{seed}: environment không khớp Compare Version 1.')
        root = ep.parent.parent
        run = inside(root, env['run_dir'])
        config = run/'config.yml'
        scaler = run/'train_data/train_data_scaler.yml'
        best = run/'best_model.pt'
        verify_files(root, ep.parent/'manifest.csv', [config, scaler, best])
        if sha256(config) != env['config_sha256']:
            raise ValueError(f'{step}/{seed}: config khác hash environment.')
        raw = yaml.safe_load(config.read_text(encoding='utf-8'))
        cfg = Config(config)
        expected_commit = B1_COMMIT if step == 'B1b' else NH_COMMIT
        if not str(env['neuralhydrology_commit']).startswith(expected_commit):
            raise ValueError(f'{step}/{seed}: commit train không đúng.')
        if (raw['seed'] != seed or raw['model'] != 'sequential_forecast_lstm' or raw['dataset'] != 'lamah_a'
                or raw['seq_length'] != 365 or raw['forecast_seq_length'] != 1 or raw.get('forecast_overlap')
                or raw.get('timestep_counter') or raw.get('forecast_input_mode', 'all') != 'all'
                or raw['target_variables'] != ['qmax'] or 'qmax' not in raw.get('clip_targets_to_zero', [])
                or raw['epochs'] != 30 or raw['early_stopping_patience'] != 30
                or cfg.additional_feature_files or raw.get('evolving_attributes')
                or raw.get('lagged_features') or raw.get('autoregressive_inputs')):
            raise ValueError(f'{step}/{seed}: cấu hình ngoài hợp đồng một ngày của bước B.')
        if (len(cfg.hindcast_inputs_flattened) != 33 or len(cfg.forecast_inputs_flattened) != 5
                or not set(Q_INPUTS).issubset(cfg.hindcast_inputs_flattened)
                or any(not v.startswith('ECMWF_') for v in cfg.forecast_inputs_flattened)):
            raise ValueError('Nhóm biến nguồn không đúng hợp đồng.')
        if step != 'B1b' and (cfg.nan_handling_method != 'masked_mean'
                             or list(map(len, cfg.hindcast_inputs)) != [21, 7, 1, 2, 2]):
            raise ValueError('B2/B3 thiếu nhúng theo nguồn/masked mean.')
        if step == 'B1b' and cfg.nan_handling_method:
            raise ValueError('B1b phải giữ mô hình không masked mean.')
        signature = {k: raw.get(k) for k in ('train_start_date', 'train_end_date', 'validation_start_date',
                    'validation_end_date', 'static_attributes', 'target_variables', 'seq_length', 'predict_last_n')}
        if reference is not None and signature != reference:
            raise ValueError('Các run khác kỳ dữ liệu/thuộc tính/nhãn.')
        reference = signature
        # Selected-epoch file and 30 validation CSVs establish the saved best epoch without test data.
        metrics = []
        for e in range(1, 31):
            p = run/f'validation/model_epoch{e:03d}/validation_metrics.csv'
            table = pd.read_csv(p, index_col=0)
            table.index = ids_of(table.index)
            if len(table) != 451 or not np.isfinite(table.NSE.median()):
                raise ValueError(f'{step}/{seed}: validation epoch {e} thiếu ID/NSE.')
            metrics.append(float(table.NSE.median()))
            consumed.append(p)
        if int(np.argmax(metrics))+1 != epoch or epoch != env['summary']['saved_epoch']:
            raise ValueError(f'{step}/{seed}: epoch không phải cực đại validation trong 30 epoch.')
        if not np.isclose(metrics[epoch-1], env['summary']['val_nse_median_best_model'], atol=1e-9, rtol=1e-9):
            raise ValueError('NSE validation không khớp summary train.')
        if not np.isclose(metrics[epoch-1], compare_validation_values(step, summaries)[str(seed)], atol=1e-9, rtol=1e-9):
            raise ValueError(f'{step}/{seed}: NSE validation khác Compare đã khóa hash.')
        a = torch.load(best, map_location='cpu', weights_only=True)
        b = torch.load(run/f'model_epoch{epoch:03d}.pt', map_location='cpu', weights_only=True)
        if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
            raise ValueError('Trọng số best_model.pt không trùng epoch đã chọn.')
        records.append(dict(step=step, seed=seed, env=env, run=run, config=raw, epoch=epoch,
                            environment_sha256=sha256(ep), config_sha256=sha256(config),
                            scaler_sha256=sha256(scaler), checkpoint_sha256=sha256(best),
                            native_validation_NSE=metrics[epoch-1]))
        consumed.extend([ep, ep.parent/'manifest.csv', config, scaler, best, run/f'model_epoch{epoch:03d}.pt'])
    return records, consumed


if __name__ == '__main__':
    SUMMARIES, COMPARE_ENVS, CONSUMED, B0_THRESHOLD = compare_sources(INPUT_ROOT)
    SOURCES, paths = load_sources(INPUT_ROOT, SUMMARIES, COMPARE_ENVS)
    CONSUMED.extend(paths)
    SOURCE_TABLE = pd.DataFrame([{k: r[k] for k in ('step', 'seed', 'epoch', 'environment_sha256',
        'config_sha256', 'scaler_sha256', 'checkpoint_sha256', 'native_validation_NSE')} for r in SOURCES])
    SOURCE_TABLE['training_version'] = TRAINING_VERSION
    SOURCE_TABLE.to_csv(OUT_DIR/'B_validation_sources.csv', index=False)
    show('12 checkpoint cố định theo validation; không chọn lại epoch', SOURCE_TABLE)

# %% Phần 3 - Dữ liệu gốc, lịch validation và khí hậu chỉ từ train
# Dùng nguyên LamaH/BaseDataset và scaler đã lưu. Chỉ bổ sung hai cột Q như notebook B1b/B2/B3.
def month_day_index(dates):
    dates = pd.DatetimeIndex(dates)
    # A leap reference year keeps March-December aligned across leap/non-leap training years.
    return pd.to_datetime('2000-'+dates.strftime('%m-%d')).dayofyear.to_numpy()-1


def fit_train_climatology(frame, start, end, columns):
    train = frame.loc[start:end, list(columns)].copy()
    if train.empty or train.index.max() > end:
        raise ValueError('Không có chuỗi train để tính khí hậu ngày.')
    return train.groupby(month_day_index(train.index)).mean().reindex(range(366))


class ValidationLamaH(LamaH):
    # Add mm/day columns before BaseDataset selects variables; normalization/lookup remain inherited.
    frames = {}

    def _load_basin_data(self, basin):
        frame = super()._load_basin_data(basin)
        for target, original in zip(Q_INPUTS, ('qmean', 'qmax')):
            raw = load_lamah_discharge(self.cfg.data_dir, basin, temporal_resolution='1D',
                                       normalize_discharge=False, target_name=original)[original]
            frame[target] = _normalize_discharge(ser=raw, area=self._basin_area.loc[basin], temporal_resolution='1D')
        # No test observations/inputs enter the validation data or the climatology.
        frame = frame.loc[:scalar_date(self.cfg.validation_end_date)].copy()
        self.frames[basin] = frame
        return frame


def find_raw(input_root):
    candidates = [p.parent.parent.parent for p in Path(input_root).rglob('Catchment_attributes.csv')
                  if p.parent.name == '1_attributes' and p.parent.parent.name == 'A_basins_total_upstrm']
    candidates = [p for p in candidates if (p/'D_gauges/1_attributes/Gauge_attributes.csv').is_file()]
    if len(candidates) != 1:
        raise ValueError(f'Cần đúng một Dataset Extended LamaH-CE gốc, thấy {len(candidates)}.')
    return candidates[0]


def runtime_config(record, raw_dir, basin_file):
    raw = copy.deepcopy(record['config'])
    for key in ('run_dir', 'train_dir', 'img_log_dir', 'train_data_file'):
        raw.pop(key, None)
    raw.update(data_dir=str(raw_dir), train_basin_file=str(basin_file), validation_basin_file=str(basin_file),
               test_basin_file=str(basin_file), device='cuda:0', verbose=0, num_workers=0,
               nan_step_probability=0., nan_sequence_probability=0., cache_validation_data=False)
    return Config(raw)


if __name__ == '__main__':
    RAW_DIR = find_raw(INPUT_ROOT)
    basin_files = sorted(INPUT_ROOT.rglob('basins_filtered.txt'))
    if not basin_files:
        raise ValueError('Dataset thiếu basins_filtered.txt của tác giả.')
    BASIN_FILE = basin_files[0]
    IDS = ids_of(load_basin_file(BASIN_FILE))
    if len(IDS) != 451 or any(ids_of(load_basin_file(p)) != IDS for p in basin_files):
        raise ValueError('Danh sách/thứ tự 451 ID không thống nhất.')
    cfg0 = Config(SOURCES[0]['config'])
    TRAIN_START, TRAIN_END = scalar_date(cfg0.train_start_date), scalar_date(cfg0.train_end_date)
    VAL_START, VAL_END = scalar_date(cfg0.validation_start_date), scalar_date(cfg0.validation_end_date)
    if not TRAIN_START <= TRAIN_END < VAL_START <= VAL_END or VAL_END >= scalar_date(cfg0.test_start_date):
        raise ValueError('Kỳ train/validation không tách khỏi test.')
    DATES = pd.date_range(VAL_START, VAL_END, freq='D')
    AREA = load_lamah_attributes(RAW_DIR, sub_dataset='lamah_a').loc[list(IDS), 'area_gov'].to_numpy(float)
    if not np.isfinite(AREA).all() or (AREA <= 0).any():
        raise ValueError('area_gov không dương/hữu hạn.')
    OBS, PERSISTENCE = (np.full((len(DATES), len(IDS)), np.nan) for _ in range(2))
    for j, basin in enumerate(IDS):
        q = load_lamah_discharge(RAW_DIR, basin, temporal_resolution='1D', normalize_discharge=False,
                                 target_name='qmax')['qmax']
        q = _normalize_discharge(ser=q, area=AREA[j], temporal_resolution='1D')
        OBS[:, j] = q.reindex(DATES).to_numpy()
        PERSISTENCE[:, j] = q.reindex(DATES-pd.Timedelta(days=1)).to_numpy()
    CONSUMED.extend(basin_files)
    CONSUMED.extend([RAW_DIR/'A_basins_total_upstrm/1_attributes/Catchment_attributes.csv',
                     RAW_DIR/'D_gauges/1_attributes/Gauge_attributes.csv'])
    for basin in IDS:
        CONSUMED.extend([RAW_DIR/f'A_basins_total_upstrm/2_timeseries/daily/ID_{basin}.csv',
                         RAW_DIR/f'D_gauges/2_timeseries/daily/ID_{basin}.csv'])
    np.savez_compressed(OUT_DIR/'B_validation_context.npz', dates=DATES.to_numpy(dtype='datetime64[D]'),
        basin_ids=np.array(IDS), area_gov_km2=AREA, obs_mm_day=OBS, persistence_mm_day=PERSISTENCE)
    show('Lịch và ngữ cảnh validation', pd.DataFrame([dict(IDs=len(IDS), days=len(DATES),
        first_date=str(VAL_START.date()), last_date=str(VAL_END.date()),
        train_first=str(TRAIN_START.date()), train_last=str(TRAIN_END.date()),
        paired_obs_persistence=int((np.isfinite(OBS)&np.isfinite(PERSISTENCE)).sum()))]))

# %% Phần 4 - Che theo thời điểm phát hành; điền nhân quả cho B1b
# Mask chạy trên bản sao của từng mẫu/batch sau chuẩn hóa, không sửa chuỗi lưu hoặc nhãn.
def scenario_lengths(scenario, names, length):
    if scenario not in SCENARIOS or length < 7:
        raise ValueError('Kịch bản/độ dài cửa sổ không hợp lệ.')
    result = {}
    for name in names:
        n = 0
        if name in Q_INPUTS:
            if scenario.startswith('S1_Q'):
                n = int(scenario[-1])
            elif scenario == 'S2':
                n = length
            elif scenario == 'S4':
                n = 1
        if scenario in ('S3', 'S4'):
            if name.startswith('ERA5L_'):
                n = 5
            elif name.startswith('GLEAM_'):
                n = length
        if n:
            result[name] = n
    return result


def causal_fill(original, mask, climatology):
    # The last finite VISIBLE input at/before each date; no bfill or values from masked cells.
    visible = original.masked_fill(mask, torch.nan)
    indices = torch.arange(original.shape[1], device=original.device)[None, :, None].expand_as(original)
    last = torch.where(torch.isfinite(visible), indices, -1).cummax(dim=1).values
    past = visible.gather(1, last.clamp_min(0))
    fallback = torch.where(torch.isfinite(climatology), climatology, torch.zeros_like(climatology))
    filled = torch.where(last >= 0, past, fallback)
    return torch.where(mask, filled, original)


def scenario_batch(batch, scenario, impute=False, climatology=None):
    # Nested dictionaries and modified tensors are new; immutable values can be shared safely.
    changed = dict(batch)
    history = batch['x_d_hindcast']
    changed['x_d_hindcast'] = dict(history)
    length = next(iter(history.values())).shape[1]
    for name, n in scenario_lengths(scenario, history, length).items():
        original = history[name]
        mask = torch.zeros_like(original, dtype=torch.bool)
        mask[:, -n:, :] = True
        if impute:
            if climatology is None or name not in climatology:
                raise ValueError(f'Thiếu khí hậu train cho {name}.')
            changed['x_d_hindcast'][name] = causal_fill(original, mask, climatology[name])
        else:
            changed['x_d_hindcast'][name] = original.masked_fill(mask, torch.nan)
    return changed


def climatology_batch(climate, dates, scaler, names, device):
    dates = np.asarray(dates)[:, :-1]
    calendar = month_day_index(dates.ravel()).reshape(dates.shape)
    values = {}
    for name in names:
        mean = float(scaler['xarray_feature_center'][name].item())
        scale = float(scaler['xarray_feature_scale'][name].item())
        if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f'Scaler train không hữu hạn/dương: {name}')
        raw = climate[name].to_numpy()[calendar]
        values[name] = torch.as_tensor((raw-mean)/scale, dtype=torch.float32, device=device)[..., None]
    return values


def audit_batch_dates(batch):
    dates = np.asarray(batch['date']).astype('datetime64[D]')
    if dates.ndim != 2 or dates.shape[1] != 365 or not (np.diff(dates, axis=1) == np.timedelta64(1, 'D')).all():
        raise ValueError('Mẫu không có lịch ngày liên tục 365 bước.')
    if next(iter(batch['x_d_hindcast'].values())).shape[1] != 364:
        raise ValueError('Q/hindcast phải kết thúc t-1, 364 bước; forecast chỉ ngày t.')
    if next(iter(batch['x_d_forecast'].values())).shape[1] != 1:
        raise ValueError('Pha ECMWF HRES không đúng một ngày.')
    return dates[:, -1]


if __name__ == '__main__':
    names = cfg0.hindcast_inputs_flattened
    if sum(v.startswith('ERA5L_') for v in names) != 21 or sum(v.startswith('GLEAM_') for v in names) != 2:
        raise ValueError('Không nhận đủ biến ERA5L/GLEAM để che nguồn.')
    show('Kịch bản suy luận sau chuẩn hóa', pd.DataFrame([
        dict(scenario=s, masked_columns=len(scenario_lengths(s, names, 364)),
             masked_day_features=sum(scenario_lengths(s, names, 364).values()),
             forecast_changed=False, B1b_policy=IMPUTATION if s != 'S0' else 'unchanged') for s in SCENARIOS]))

# %% Phần 5 - Suy luận 12 checkpoint trên 7 kịch bản, kiểm lại S0
# Không optimizer/backward; model.eval() tắt che ngẫu nhiên và dropout. Nhãn/scaler giữ nguyên.
def infer_basin(model, dataset, scaler, step, climate, device):
    loader = GpuBatchLoader(dataset, batch_size=dataset.cfg.batch_size, device=device,
                            shuffle=False, with_dates=True)
    predictions = {s: [] for s in SCENARIOS}
    dates_all, observed = [], []
    scale = scaler['xarray_feature_scale'][['qmax']].to_array().values
    center = scaler['xarray_feature_center'][['qmax']].to_array().values
    climate_names = sorted(set().union(*(scenario_lengths(s, dataset.cfg.hindcast_inputs_flattened, 364) for s in SCENARIOS)))
    with torch.no_grad():
        for batch in loader:
            dates_all.append(audit_batch_dates(batch))
            observed.append((batch['y'][:, -1:, :].cpu().numpy()*scale+center).reshape(-1))
            cli = climatology_batch(climate, batch['date'], scaler, climate_names, device) if step == 'B1b' else None
            for scenario in SCENARIOS:
                data = scenario_batch(batch, scenario, impute=step == 'B1b', climatology=cli)
                data = model.pre_model_hook(data, is_train=False)
                pred = model(data)['y_hat'][:, -1:, :].cpu().numpy()*scale+center
                if np.isinf(pred).any():
                    raise ValueError('Dự báo vô cực; không tiếp tục chấm.')
                predictions[scenario].append(pred.reshape(-1))
    return np.concatenate(dates_all), np.concatenate(observed), {s: np.concatenate(v) for s, v in predictions.items()}


def export_contract(record, scenario, raw_predictions, dates, ids, area, obs, persistence, output):
    meta = dict(run=f"SeqLSTM_Q_{record['step']}_seed{record['seed']}", period='validation', scenario=scenario,
                lead=1, forecast_horizon=1, seed=record['seed'], unit='mm/day_area_gov', clip_to_zero=True,
                config_sha256=record['config_sha256'], library_commit=record['env']['neuralhydrology_commit'],
                inference_config_sha256=record.get('inference_config_sha256'),
                inference_library_commit=NH_COMMIT, training_notebook_version=1, selected_epoch=record['epoch'],
                checkpoint_sha256=record['checkpoint_sha256'], scaler_sha256=record['scaler_sha256'],
                imputation_policy=IMPUTATION if record['step'] == 'B1b' and scenario != 'S0' else 'none',
                scenario_family='S1' if scenario.startswith('S1_') else scenario,
                validation_only=True, optimizer_steps=0)
    path = output/'contracts'/f"{meta['run']}_validation_{scenario}_lead1.npz"
    dates = np.asarray(dates, dtype='datetime64[D]')
    np.savez_compressed(path, dates=dates, origin_dates=dates, persistence_source_dates=dates-np.timedelta64(1, 'D'),
        basin_ids=np.array(ids), area_gov_km2=area, obs_mm_day=obs, persistence_mm_day=persistence,
        sim_raw_mm_day=raw_predictions, sim_mm_day=np.maximum(raw_predictions, 0),
        metadata_json=json.dumps(meta, ensure_ascii=False, sort_keys=True))
    with np.load(path, allow_pickle=False) as data:
        if (json.loads(str(data['metadata_json'].item())) != meta
                or not np.array_equal(data['sim_raw_mm_day'], raw_predictions, equal_nan=True)
                or not np.array_equal(data['sim_mm_day'], np.maximum(raw_predictions, 0), equal_nan=True)
                or not np.array_equal(data['obs_mm_day'], obs, equal_nan=True)):
            raise ValueError('NPZ đọc ngược khác dữ liệu trước khi ghi.')
    return path


if __name__ == '__main__':
    NATIVE_AUDIT, CONTRACT_ROWS, TIMING = [], [], []
    for record in SOURCES:
        started = time.perf_counter()
        step, seed = record['step'], record['seed']
        cfg = runtime_config(record, RAW_DIR, BASIN_FILE)
        runtime_dir = OUT_DIR/'configs'
        runtime_dir.mkdir(exist_ok=True)
        runtime_name = f'{step}_seed{seed}.yml'
        cfg.dump_config(runtime_dir, filename=runtime_name)
        record['inference_config_sha256'] = sha256(runtime_dir/runtime_name)
        scaler = load_scaler(record['run'])
        model = get_model(cfg).to('cuda:0')
        model.load_state_dict(torch.load(record['run']/'best_model.pt', map_location='cuda:0', weights_only=True), strict=True)
        model.eval()
        original_metrics = pd.read_csv(record['run']/f"validation/model_epoch{record['epoch']:03d}/validation_metrics.csv", index_col=0)
        original_metrics.index = ids_of(original_metrics.index)
        predictions = {s: np.full(OBS.shape, np.nan) for s in SCENARIOS}
        for j, basin in enumerate(IDS):
            dataset = ValidationLamaH(cfg, is_train=False, period='validation', basin=basin, scaler=copy.deepcopy(scaler))
            if dataset.is_train or dataset._compute_scaler:
                raise ValueError('Dataset đang định fit lại scaler hoặc chạy ở chế độ train.')
            climate = fit_train_climatology(ValidationLamaH.frames.pop(basin), TRAIN_START, TRAIN_END,
                                            cfg.hindcast_inputs_flattened) if step == 'B1b' else None
            ValidationLamaH.frames.pop(basin, None)
            dates, own_obs, raw = infer_basin(model, dataset, scaler, step, climate, torch.device('cuda:0'))
            if not np.array_equal(dates, DATES.to_numpy(dtype='datetime64[D]')):
                raise ValueError(f'{step}/{seed}/{basin}: lookup không phủ đúng lịch validation.')
            reference = OBS[:, j]
            finite = np.isfinite(reference)
            if (not np.array_equal(np.isfinite(own_obs), finite)
                    or not np.allclose(own_obs[finite], reference[finite], atol=OBS_ATOL, rtol=OBS_RTOL)):
                raise ValueError(f'{step}/{seed}/{basin}: nhãn khôi phục khác Q gốc.')
            native = native_library_nse(own_obs, raw['S0'])
            independent, _ = nse_by_id(own_obs[:, None], np.maximum(raw['S0'], 0)[:, None])
            expected = float(original_metrics.loc[basin, 'NSE'])
            if not np.isclose(native, expected, atol=NSE_TOL, rtol=0, equal_nan=True):
                raise ValueError(f'{step}/{seed}/{basin}: S0 NSE {native} khác nguồn {expected}; dừng trước bàn giao.')
            NATIVE_AUDIT.append(dict(step=step, seed=seed, basin_id=basin, expected_NSE=expected,
                reproduced_NSE=native, independent_float64_NSE=float(independent[0]),
                observation_dtype=str(own_obs.dtype), prediction_dtype=str(raw['S0'].dtype),
                passed=True, scaler_sha256=record['scaler_sha256']))
            for scenario in SCENARIOS:
                predictions[scenario][:, j] = raw[scenario]
            del dataset
            if (j+1) % 50 == 0 or j+1 == len(IDS):
                print(f'{step} seed {seed}: {j+1}/{len(IDS)} ID, 7 kịch bản; S0 khớp nguồn.')
        for scenario, raw in predictions.items():
            if step != 'B1b' and not np.isfinite(raw).all():
                raise ValueError(f'{step}/{seed}/{scenario}: masked mean không cho dự báo hữu hạn toàn lịch.')
            path = export_contract(record, scenario, raw, DATES, IDS, AREA, OBS, PERSISTENCE, OUT_DIR)
            CONTRACT_ROWS.append(dict(step=step, seed=seed, scenario=scenario, file=path.relative_to(OUT_DIR).as_posix(),
                finite_predictions=int(np.isfinite(raw).sum()), negative_raw=int((raw < 0).sum()),
                checkpoint_sha256=record['checkpoint_sha256'], passed=True))
        if sha256(record['run']/'train_data/train_data_scaler.yml') != record['scaler_sha256']:
            raise ValueError('Scaler nguồn thay đổi sau suy luận.')
        TIMING.append(dict(step=step, seed=seed, seconds=time.perf_counter()-started))
        pd.DataFrame(NATIVE_AUDIT).to_csv(OUT_DIR/'B_validation_S0_reproduction.csv', index=False)
        pd.DataFrame(CONTRACT_ROWS).to_csv(OUT_DIR/'B_validation_contract_check.csv', index=False)
        pd.DataFrame(TIMING).to_csv(OUT_DIR/'B_validation_timing.csv', index=False)
        del model, scaler, predictions
        gc.collect()
        torch.cuda.empty_cache()
    show('Đã xuất 84 NPZ validation; 5412 đối chiếu NSE S0 từng ID', pd.DataFrame(CONTRACT_ROWS))

# %% Phần 6 - Giao ngày chung và bảng validation bàn giao B*
# Xuất riêng S1_Q1/Q3/Q7. Bảng family cân đều S1/S2/S3/S4, không tự chọn cấu hình/seed.
def score_contracts(output, obs, persistence, ids):
    paths = sorted((output/'contracts').glob('*.npz'))
    expected = {(step, seed, scenario) for step in STEPS for seed in SEEDS for scenario in SCENARIOS}
    context_mask = np.isfinite(obs) & np.isfinite(persistence)
    common = context_mask.copy()
    identities, metadata_rows = set(), []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata_json'].item()))
            step = meta['run'].removeprefix('SeqLSTM_Q_').removesuffix(f"_seed{meta['seed']}")
            identity = step, meta['seed'], meta['scenario']
            if identity in identities or identity not in expected or meta['period'] != 'validation':
                raise ValueError('Hợp đồng validation trùng/ngoài 84 trường hợp.')
            identities.add(identity)
            if (tuple(data['basin_ids']) != tuple(ids) or not np.array_equal(data['obs_mm_day'], obs, equal_nan=True)
                    or not np.array_equal(data['persistence_mm_day'], persistence, equal_nan=True)):
                raise ValueError('Ngữ cảnh khác nhau giữa các hợp đồng validation.')
            common &= np.isfinite(data['sim_mm_day'])
            metadata_rows.append((path, identity))
    if identities != expected or not common.any():
        raise ValueError('Thiếu hợp đồng hoặc giao ngày chung rỗng.')
    details, summaries = [], []
    for path, (step, seed, scenario) in metadata_rows:
        with np.load(path, allow_pickle=False) as data:
            sim = data['sim_mm_day']
            for scope, mask in [('native', None), ('common_all_runs_and_scenarios', common)]:
                nse, counts = nse_by_id(obs, sim, mask)
                if not np.isfinite(nse).any():
                    raise ValueError('Không có NSE hữu hạn để bàn giao.')
                summaries.append(dict(step=step, seed=seed, scenario=scenario, scope=scope,
                    NSE_median=float(np.nanmedian(nse)), NSE_finite_IDs=int(np.isfinite(nse).sum()),
                    scored_day_IDs=int(counts.sum())))
                details.extend(dict(step=step, seed=seed, scenario=scenario, scope=scope, basin_id=b,
                                    n=int(n), NSE=float(v)) for b, n, v in zip(ids, counts, nse))
    return common, pd.DataFrame(summaries), pd.DataFrame(details)


def family_scores(seed_table):
    common = seed_table.loc[seed_table.scope.eq('common_all_runs_and_scenarios')].copy()
    common['family'] = common.scenario.map(lambda x: 'S1' if x.startswith('S1_') else x)
    family = common.groupby(['step', 'seed', 'family'], as_index=False).NSE_median.mean()
    robust = family.loc[family.family.ne('S0')].groupby(['step', 'seed'], as_index=False).NSE_median.mean()
    robust = robust.rename(columns={'NSE_median': 'S1_S4_family_balanced_NSE'})
    return family, robust


if __name__ == '__main__':
    COMMON, SEED_TABLE, BY_BASIN = score_contracts(OUT_DIR, OBS, PERSISTENCE, IDS)
    FAMILY, ROBUST = family_scores(SEED_TABLE)
    SEED_TABLE.to_csv(OUT_DIR/'B_validation_seed_table.csv', index=False)
    BY_BASIN.to_csv(OUT_DIR/'B_validation_basin_NSE.csv', index=False)
    FAMILY.to_csv(OUT_DIR/'B_validation_family_scores.csv', index=False)
    ROBUST.to_csv(OUT_DIR/'B_validation_robustness_by_seed.csv', index=False)
    baseline = SEED_TABLE.loc[SEED_TABLE.scenario.eq('S0'), ['step', 'seed', 'scope', 'NSE_median']]
    DELTA = SEED_TABLE.merge(baseline.rename(columns={'NSE_median': 'S0_NSE_median'}), on=['step', 'seed', 'scope'], validate='many_to_one')
    DELTA['NSE_change_from_S0'] = DELTA.NSE_median-DELTA.S0_NSE_median
    DELTA.to_csv(OUT_DIR/'B_validation_changes.csv', index=False)
    STATS = SEED_TABLE.groupby(['step', 'scenario', 'scope']).NSE_median.agg(['mean', 'std']).reset_index()
    STATS.to_csv(OUT_DIR/'B_validation_seed_stats.csv', index=False)
    GATE = SOURCE_TABLE.groupby('step').native_validation_NSE.agg(['mean', 'std']).reset_index()
    GATE['gap_from_best_native_S0'] = GATE['mean'].max()-GATE['mean']
    GATE['B0_threshold'] = B0_THRESHOLD
    GATE['within_B0_noise_native_S0'] = GATE.gap_from_best_native_S0 <= B0_THRESHOLD
    GATE.to_csv(OUT_DIR/'B_validation_S0_gate.csv', index=False)
    COVERAGE = pd.DataFrame([dict(runs=len(SOURCES), scenario_variants=len(SCENARIOS), contracts=84,
        common_day_IDs=int(COMMON.sum()), IDs_with_days=int(COMMON.any(axis=0).sum()), IDs_retained=len(IDS))])
    COVERAGE.to_csv(OUT_DIR/'B_validation_coverage.csv', index=False)
    np.savez_compressed(OUT_DIR/'B_validation_common_mask.npz', mask=COMMON, dates=DATES.to_numpy(dtype='datetime64[D]'), basin_ids=np.array(IDS))
    show('S0 nguồn đã tái lập; điều kiện nằm trong nhiễu B0, chưa chọn B*', GATE)
    show('Validation trên giao ngày chung, trung bình và SD mẫu qua ba seed', STATS.loc[STATS.scope.eq('common_all_runs_and_scenarios')])
    show('Phạm vi giao chung', COVERAGE)
    print('S1_Q1/Q3/Q7 giữ riêng. Family-balanced: trung bình ba mức S1 trước, sau đó cân đều S1/S2/S3/S4.')
    print('Đây là bảng bàn giao; người phụ trách B* rà soát chính sách điền B1b/trọng số kịch bản trước freeze-C.')

# %% Phần 7 - Hình suy giảm validation theo kịch bản
# Điểm là trung bình ba median lưu vực; thanh sai số là SD mẫu qua seed, không phải khoảng tin cậy.
def draw_validation(stats, output, period_label):
    import matplotlib.pyplot as plt
    table = stats.loc[stats.scope.eq('common_all_runs_and_scenarios')]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ('#0072B2', '#009E73', '#D55E00', '#CC79A7')
    for step, color in zip(STEPS, colors):
        part = table.loc[table.step.eq(step)].set_index('scenario').reindex(SCENARIOS)
        ax.errorbar(range(len(SCENARIOS)), part['mean'], yerr=part['std'], marker='o', capsize=3,
                    color=color, label=step.replace('_', ' '))
    ax.set_xticks(range(len(SCENARIOS)), SCENARIOS)
    ax.set(xlabel='Inference scenario', ylabel='Median basin NSE (mean across three seeds)',
           title='(a) Shared daily mask; error bars = sample SD across seeds')
    ax.grid(alpha=.25)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.suptitle(f'B validation robustness | {period_label} | fixed validation-selected checkpoints')
    fig.tight_layout()
    path = output/'B_validation_Scenarios.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    return fig, path


if __name__ == '__main__':
    fig, figure_path = draw_validation(STATS, OUT_DIR, f'{VAL_START.year}-{VAL_END.year}')
    display(fig)
    print(f'Hình: NSE trên cùng {int(COMMON.sum()):,} ngày–ID; điểm cao hơn là tốt hơn; '
          'thanh là SD qua ba seed. S1 là ba độ dài mất Q, không phải ba kịch bản độc lập khi cân theo họ.')

# %% Phần 8 - Ghi môi trường, kiểm hoàn thành và manifest
# Chỉ complete khi đủ 84 NPZ, 5412 đối chiếu S0 và đúng nguồn; không ghi đã chọn B*.
if __name__ == '__main__':
    INPUT_MANIFEST = pd.DataFrame([dict(path=str(p), bytes=p.stat().st_size, sha256=sha256(p))
        for p in sorted(set(CONSUMED))])
    INPUT_MANIFEST.to_csv(OUT_DIR/'B_validation_input_manifest.csv', index=False)
    if len(NATIVE_AUDIT) != 12*451 or not all(r['passed'] for r in NATIVE_AUDIT) or len(CONTRACT_ROWS) != 84:
        raise ValueError('Chưa đủ đối chiếu S0/hợp đồng để kết luận hoàn thành.')
    SUMMARY = dict(complete=True, notebook=NOTEBOOK, period='validation', first_date=str(VAL_START.date()),
        last_date=str(VAL_END.date()), steps=list(STEPS), seeds=list(SEEDS), scenarios=list(SCENARIOS),
        contracts=84, S0_basin_checks=12*451, common_day_IDs=int(COMMON.sum()),
        IDs_with_days=int(COMMON.any(axis=0).sum()), B0_bstar_threshold=B0_THRESHOLD,
        imputation_B1b=IMPUTATION, original_missing_outside_scenario_preserved=True,
        normalization_refit=False, train_climatology_period=[str(TRAIN_START.date()), str(TRAIN_END.date())],
        family_aggregation='mean_S1_Q1_Q3_Q7_then_equal_mean_S1_S2_S3_S4',
        selection_status='handoff_only_Bstar_owner_must_review_policy_and_apply_predeclared_rule',
        Bstar_selected=False, test_predictions_read=False, optimizer_steps=0,
        inference_library_commit=NH_COMMIT, source_versions=1, training_rerun=False)
    (OUT_DIR/'b_validation_summary.json').write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=2), encoding='utf-8')
    ENV = dict(notebook=NOTEBOOK, python=platform.python_version(), neuralhydrology_commit=NH_COMMIT,
        libs={name: metadata.version(name) for name in ('numpy', 'pandas', 'xarray', 'torch', 'scipy', 'numba', 'ruamel.yaml')},
        gpu=torch.cuda.get_device_name(0), compare_pins=COMPARE_PINS, protocol=SUMMARY)
    (OUT_DIR/'environment.json').write_text(json.dumps(ENV, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = pd.DataFrame([dict(file=p.relative_to(OUT_DIR).as_posix(), bytes=p.stat().st_size, sha256=sha256(p))
        for p in sorted(OUT_DIR.rglob('*')) if p.is_file() and p.name != 'manifest.csv'])
    manifest.to_csv(OUT_DIR/'manifest.csv', index=False)
    verify_files(OUT_DIR, OUT_DIR/'manifest.csv', [OUT_DIR/r for r in manifest.file])
    show('Tệp output', manifest.assign(sha256=manifest.sha256.str[:12]))
    print(json.dumps(SUMMARY, ensure_ascii=False, indent=2))
    print('Hoàn thành suy luận validation. Save Version; gửi b_validation_summary.json và các CSV B_validation_*.')
    print('Giữ toàn bộ b_validation/contracts, context, common_mask và manifest trong Output để chấm/kiểm độc lập.')
