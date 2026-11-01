# B3 Compare - hai mức che, mỗi mức ba seed; CPU, Internet Off.
# Add Input: Output của đúng tám notebook sau, gắn nguyên Output nguồn:
# 1. 08_B3_LamaHCEExt_Train_P10S12_Seed111 - Version 1 đã Save.
# 2. 08_B3_LamaHCEExt_Train_P10S12_Seed222 - Version 1 đã Save.
# 3. 08_B3_LamaHCEExt_Train_P10S12_Seed333 - Version 1 đã Save.
# 4. 08_B3_LamaHCEExt_Train_P05S05_Seed111 - Version 1 đã Save.
# 5. 08_B3_LamaHCEExt_Train_P05S05_Seed222 - Version 1 đã Save.
# 6. 08_B3_LamaHCEExt_Train_P05S05_Seed333 - Version 1 đã Save.
# 7. 07_B2_LamaHCEExt_Compare - Version 1.
# 8. 02_A3_LamaHCEExt_Benchmarks - Version 1.
# Mục lục: 1 nguồn/config/hash; 2 tổng hợp; 3 NPZ; 4 hình; 5 summary/manifest.
# %% Phần 1 - Kiểm sáu run B3 và nguồn B2/A3
import hashlib
import json
import os
import pickle
import platform
from importlib import metadata
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from IPython.display import HTML,display



def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    root=Path(root).resolve()
    path=(root/relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f'Đường dẫn nằm ngoài nguồn Input: {relative}')
    return path


def check_manifest(root, manifest_path, required):
    manifest=pd.read_csv(manifest_path)
    col='file' if 'file' in manifest else 'tệp'
    if col not in manifest or 'sha256' not in manifest or manifest[col].duplicated().any():
        raise ValueError(f'Manifest sai cột/trùng đường dẫn: {manifest_path}')
    known=set(manifest[col])
    for path in required:
        relative=path.resolve().relative_to(Path(root).resolve()).as_posix()
        if relative not in known:
            raise ValueError(f'Manifest thiếu tệp được đọc: {relative}')
    for relative, digest in zip(manifest[col],manifest.sha256):
        path=inside(root,relative)
        if not path.is_file() or sha256(path)!=digest:
            raise ValueError(f'Không khớp manifest: {relative}')
    return manifest


NH_COMMIT='2ccaa147a6ec87d0f56fa3e2f11b1bf38079253f'
INPUT_ROOT=Path(os.environ.get('B3_INPUT_ROOT','/kaggle/input' if Path('/kaggle/working').exists() else 'b3_compare_input'))
WORK_DIR=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('b3_compare_work')
OUT_DIR=WORK_DIR/'b3_compare'
if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
    raise RuntimeError('Thư mục Compare đã có output; dùng phiên sạch để không trộn kết quả.')
OUT_DIR.mkdir(parents=True,exist_ok=True)
SEEDS=[111,222,333]
MASKS={'P10S12':(.1,.12),'P05S05':(.05,.05)}
RUN_KEYS=[(level,s) for level in MASKS for s in SEEDS]
SOURCE_NOTEBOOKS={key:f'08_B3_LamaHCEExt_Train_{key[0]}_Seed{key[1]}' for key in RUN_KEYS}
A3_VERSION=B2_COMPARE_VERSION=1
TRAINING_VERSION_EXPECTED=1  # Cả sáu nguồn đã Save Version 1.

def show(title,frame,digits=4):
    display(HTML(f'<h4>{title}</h4>'))
    display(frame.style.hide(axis='index').format(precision=digits,na_rep='-'))

def unique_file(root,name):
    hits=sorted(root.rglob(name))
    if len(hits)!=1:
        raise RuntimeError(f'Cần đúng một {name}, thấy {len(hits)} nguồn.')
    return hits[0]

def read_metrics(path):
    frame=pd.read_csv(path,index_col=0)
    frame.index=[str(int(b)) for b in frame.index]
    if frame.index.has_duplicates:
        raise ValueError(f'ID trùng trong {path}')
    return frame

B2_SUMMARY_PATH=unique_file(INPUT_ROOT,'b2_summary.json')
B2_DIR=B2_SUMMARY_PATH.parent
B2_ENV_PATH=B2_DIR/'environment.json'
check_manifest(B2_DIR,B2_DIR/'manifest.csv',[B2_SUMMARY_PATH,B2_ENV_PATH,B2_DIR/'B2_seed_table.csv'])
B2_SUMMARY=json.loads(B2_SUMMARY_PATH.read_text(encoding='utf-8'))
B2_COMPARE_ENV=json.loads(B2_ENV_PATH.read_text(encoding='utf-8'))
if (B2_SUMMARY.get('complete') is not True or B2_SUMMARY.get('library_commit')!=NH_COMMIT
        or B2_COMPARE_ENV.get('notebook')!='07_B2_LamaHCEExt_Compare'
        or B2_COMPARE_ENV.get('library_commit_of_runs')!=NH_COMMIT):
    raise ValueError('Nguồn B2 Compare không đúng Version 1 đã bàn giao.')
B2_SEEDS=pd.read_csv(B2_DIR/'B2_seed_table.csv')
if sorted(B2_SEEDS.seed.tolist())!=SEEDS:
    raise ValueError('B2 Compare thiếu hoặc trùng seed.')
B0_THRESHOLD=float(B2_SUMMARY['B1b_validation_comparison']['B0_Bstar_threshold'])
if not np.isfinite(B0_THRESHOLD) or B0_THRESHOLD<=0:
    raise ValueError('Thiếu ngưỡng validation B0.')
B2_BASE={}
for seed in SEEDS:
    source=B2_COMPARE_ENV['inputs']['B2'][str(seed)]
    if source['version_expected']!=1 or source['notebook']!=f'06_B2_LamaHCEExt_Train_Seed{seed}':
        raise ValueError('B2 Compare dùng nguồn train khác Version 1.')
    path=inside(B2_DIR,B2_SUMMARY['contracts'][str(seed)])
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata_json'].item()))
    if meta.get('seed')!=seed or meta.get('library_commit')!=NH_COMMIT:
        raise ValueError('Hợp đồng mốc B2 khác nguồn/seed.')
    B2_BASE[seed]={'environment_sha256':source['environment_sha256'],'config_sha256':meta['config_sha256']}

ROOTS,ENVS,SUMMARY,CURVES,TEST_METRICS,BEST,run_dirs=({} for _ in range(7))
source_rows=[]
science=packages=None
for ep in sorted(INPUT_ROOT.rglob('b3_output/environment.json')):
    e=json.loads(ep.read_text(encoding='utf-8'))
    key=(e.get('mask_level'),e.get('seed'))
    if key not in RUN_KEYS or key in ENVS:
        raise ValueError(f'Output B3 trùng hoặc ngoài sáu run dự kiến: {key}')
    level,seed=key
    root=ep.parent.parent
    run=inside(root,e['run_dir'])
    config_path=run/'config.yml'
    summary_path=ep.parent/f"summary_{e['experiment_name']}.json"
    curve_path=ep.parent/f"curve_{e['experiment_name']}.csv"
    audit_path=ep.parent/f"evaluation_checks_{e['experiment_name']}.csv"
    preflight_path=ep.parent/'B3_mask_preflight.csv'
    check_manifest(root,ep.parent/'manifest.csv',[ep,config_path,summary_path,curve_path,audit_path,preflight_path,
        ep.parent/'B3_config_delta.csv',run/'best_model.pt',run/'test/best_model/test_results.p',
        run/'test/best_model/test_metrics.csv'])
    if (e.get('notebook')!=SOURCE_NOTEBOOKS[key] or e.get('step')!='B3'
            or e.get('experiment_name')!=f'B3_{level}_SeqLSTM_Q_seed{seed}'
            or e.get('neuralhydrology_commit')!=NH_COMMIT or sha256(config_path)!=e['config_sha256']):
        raise ValueError(f'{key}: danh tính/config B3 sai.')
    if (e.get('q_inputs')!=['qmean_mmd','qmax_mmd'] or any(e.get('protocol',{}).get(k)!=v for k,v in {
            'epochs':30,'early_stopping_patience':30,'data_on_gpu':True,'test_every_epoch':True,'deterministic':False}.items())):
        raise ValueError(f'{key}: giao thức train/evaluation không khớp B2.')
    source=e['B2_source']
    if (source.get('version')!=1 or source.get('seed')!=seed or source.get('library_commit')!=NH_COMMIT
            or source.get('notebook')!=f'06_B2_LamaHCEExt_Train_Seed{seed}'
            or any(source.get(k)!=v for k,v in B2_BASE[seed].items())):
        raise ValueError(f'{key}: nguồn B2 khác Compare Version 1 hiện gắn.')
    c=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    p_step,p_sequence=MASKS[level]
    masking={'nan_step_probability':p_step,'nan_sequence_probability':p_sequence,'training_only':True}
    if (e.get('masking')!=masking or c.get('nan_step_probability')!=p_step
            or c.get('nan_sequence_probability')!=p_sequence or c.get('nan_handling_method')!='masked_mean'
            or c.get('timestep_counter',False) or c.get('forecast_input_mode','all')!='all'
            or c.get('model')!='sequential_forecast_lstm' or c.get('seq_length')!=365
            or c.get('forecast_seq_length')!=1 or c.get('forecast_overlap',0)
            or c.get('epochs')!=30 or c.get('early_stopping_patience')!=30 or c.get('data_on_gpu') is not True
            or list(map(len,c.get('hindcast_inputs',[])))!=[21,7,1,2,2]
            or list(map(len,c.get('forecast_inputs',[])))!=[5]):
        raise ValueError(f'{key}: giao thức B3 sai.')
    current={k:v for k,v in c.items() if k not in {'seed','experiment_name','run_dir','train_dir','img_log_dir',
        'data_dir','train_basin_file','validation_basin_file','test_basin_file','nan_step_probability','nan_sequence_probability'}}
    if science is None:
        science,packages=current,e['libs']
    if current!=science or e['libs']!=packages:
        raise ValueError('Sáu run khác cấu hình khoa học/môi trường ngoài seed và hai xác suất.')
    audit=pd.read_csv(audit_path)
    preflight=pd.read_csv(preflight_path)
    delta=pd.read_csv(ep.parent/'B3_config_delta.csv').set_index('key')
    required_flags=('source_group_masks_shared','stored_inputs_and_targets_unchanged','evaluation_masking_disabled',
                    'forward_backward_optimizer_finite','passed')
    if (audit.empty or not audit['passed'].eq(True).all() or len(preflight)!=1
            or any(not preflight[k].eq(True).all() for k in required_flags)
            or preflight.iloc[0]['device']!='cuda:0' or preflight.iloc[0]['probe_optimizer_steps']!=1
            or set(delta.index)!={'nan_step_probability','nan_sequence_probability'}
            or not delta['B2'].eq(0).all() or delta.loc['nan_step_probability','B3']!=p_step
            or delta.loc['nan_sequence_probability','B3']!=p_sequence):
        raise ValueError(f'{key}: kiểm checkpoint/preflight/config delta chưa đạt.')
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    curve=pd.read_csv(curve_path)
    metrics=read_metrics(run/'test/best_model/test_metrics.csv')
    if (summary!=e['summary'] or summary.get('seed')!=seed or summary.get('mask_level')!=level
            or summary.get('step')!='B3' or summary.get('nan_step_probability')!=p_step
            or summary.get('nan_sequence_probability')!=p_sequence
            or not summary.get('evaluation_checks_passed') or summary.get('B2_source')!=source
            or summary.get('n_epochs')!=30 or not summary.get('ready_for_evaluation')
            or summary.get('saved_epoch')!=summary.get('true_best_epoch')
            or summary.get('library_commit')!=NH_COMMIT or curve.epoch.tolist()!=list(range(1,31))
            or not np.isfinite(curve['NSE trung vị validation (tệp)']).all()
            or int(curve.loc[curve['NSE trung vị validation (tệp)'].idxmax(),'epoch'])!=summary['saved_epoch']
            or not np.isclose(curve.set_index('epoch').loc[summary['saved_epoch'],'NSE trung vị validation (tệp)'],
                              summary['val_nse_median_best_model'],rtol=1e-9,atol=1e-9)
            or not np.isclose(metrics.NSE.median(),summary['test_nse_median_best_model'],rtol=1e-9,atol=1e-9)):
        raise ValueError(f'{key}: curve/summary/checkpoint chưa khớp.')
    ROOTS[key],ENVS[key],SUMMARY[key],CURVES[key],TEST_METRICS[key]=root,e,summary,curve,metrics
    BEST[key],run_dirs[key]=summary['saved_epoch'],run
    source_rows.append({'mask_level':level,'seed':seed,'notebook':SOURCE_NOTEBOOKS[key],
                       'version_expected':TRAINING_VERSION_EXPECTED,'environment_sha256':sha256(ep),
                       'B2_environment_sha256':source['environment_sha256']})
if set(ENVS)!=set(RUN_KEYS):
    raise RuntimeError(f'Cần đủ sáu run B3; còn thiếu {sorted(set(RUN_KEYS)-set(ENVS))}')
SOURCE_TABLE=pd.DataFrame(source_rows)
SOURCE_TABLE.to_csv(OUT_DIR/'B3_input_sources.csv',index=False)
show('Nguồn sáu run B3 và nguồn B2 cùng seed',SOURCE_TABLE)
A3_PATHS={name:unique_file(INPUT_ROOT,name) for name in ['A3_persistence_predictions.npz','A3_Q_source_window.npz']}
if B2_COMPARE_ENV['inputs']['A3']['version']!=A3_VERSION:
    raise ValueError('Nguồn A3 của Compare B2 sai Version.')
for name,path in A3_PATHS.items():
    if sha256(path)!=B2_COMPARE_ENV['inputs']['A3']['files'][name]:
        raise ValueError(f'A3 khác nguồn Compare B2: {name}')
cfg0=yaml.safe_load((run_dirs[RUN_KEYS[0]]/'config.yml').read_text(encoding='utf-8'))
TEST_YEARS=f"{str(cfg0['test_start_date'])[-4:]}-{str(cfg0['test_end_date'])[-4:]}"
ENV={'python':platform.python_version(),'numpy':metadata.version('numpy'),'pandas':metadata.version('pandas')}
b2_identity={'notebook':'07_B2_LamaHCEExt_Compare','version':B2_COMPARE_VERSION,
             'summary_sha256':sha256(B2_SUMMARY_PATH),'environment_sha256':sha256(B2_ENV_PATH)}
def run_dir(key):
    return run_dirs[key]
print('Sáu run B3 khớp nguồn, config, mask preflight và checkpoint theo validation.')

# %% Phần 2 - Tổng hợp hai mức che và đối chiếu B2
# SD mẫu qua ba seed; test chỉ báo cáo. Chưa có validation S1-S4 để chọn B*.
rows=[{'mask_level':level,'seed':seed,'selected_epoch':BEST[(level,seed)],
       'nan_step_probability':MASKS[level][0],'nan_sequence_probability':MASKS[level][1],
       'validation_NSE_median':SUMMARY[(level,seed)]['val_nse_median_best_model'],
       'test_NSE_median':SUMMARY[(level,seed)]['test_nse_median_best_model'],
       'test_KGE_median':SUMMARY[(level,seed)]['test_kge_median_best_model'],
       'train_hours':SUMMARY[(level,seed)]['train_hours']} for level,seed in RUN_KEYS]
SEED_TABLE=pd.DataFrame(rows)
SEED_TABLE.to_csv(OUT_DIR/'B3_seed_table.csv',index=False)
metric_columns=['validation_NSE_median','test_NSE_median','test_KGE_median','train_hours']
STATS=SEED_TABLE.melt(id_vars=['mask_level','seed'],value_vars=metric_columns,var_name='metric')
STATS=STATS.groupby(['mask_level','metric'],sort=False)['value'].agg(['mean','std']).reset_index()
STATS.to_csv(OUT_DIR/'B3_seed_stats.csv',index=False)
show('Hai mức che, mỗi mức ba seed',SEED_TABLE)
show('Trung bình và SD mẫu qua seed',STATS)
baseline=float(B2_SEEDS.validation_NSE_median.mean())
if not np.isclose(baseline,np.mean(list(B2_SUMMARY['seed_val_nse_median'].values())),rtol=1e-9,atol=1e-9):
    raise ValueError('Bảng seed B2 khác summary validation.')
DELTAS=pd.DataFrame([{'comparison':f'B3 {level} - B2','B2_validation_NSE_mean':baseline,
    'B3_validation_NSE_mean':float(SEED_TABLE.loc[SEED_TABLE.mask_level.eq(level),'validation_NSE_median'].mean()),
    'validation_NSE_mean_difference':float(SEED_TABLE.loc[SEED_TABLE.mask_level.eq(level),'validation_NSE_median'].mean())-baseline,
    'B0_Bstar_threshold':B0_THRESHOLD,'common_mask_evaluation_completed':False,'Bstar_selected':False} for level in MASKS])
DELTAS.to_csv(OUT_DIR/'B3_B2_reported_summary.csv',index=False)
show('Validation S0 so B2, chưa chọn B*',DELTAS)

# %% Phần 3 - Dựng NPZ theo hợp đồng bộ đánh giá và kiểm lại
# Mỗi mức/hạt giống một tệp SeqLSTM_Q_B3_<mức>_seed{N}_test_S0_lead1.npz theo hợp đồng ở Workspace/06_Evaluation/
# 01_Evaluation_LamaHCEExt.md Mục 7, cùng cách dựng A4 (04_A4_LamaHCEExt_Compare.py Phần 5, đã qua bộ đánh giá Version 4):
# sim thô lấy từ test_results.p của best_model.pt (["1D"]["xr"]["qmax_sim"], time_step = 0, chưa chặn âm - tester.py dòng
# 319-322 chỉ chặn âm khi tính chỉ số), chặn âm theo clip_targets_to_zero của config.yml. obs, persistence, diện tích, thứ
# tự ID lấy nguyên từ ngữ cảnh A3 (như read_context của bộ đánh giá, Phần 8.1) vì bộ đánh giá đòi bằng nhau từng bit giữa
# các run cùng nhóm (check_same_context). Khác A4: library_commit là commit B3 đã gộp research (2ccaa14), selected_epoch là epoch
# best_model.pt sau 30 epoch. Kiểm lại: (1) đọc ngược tệp theo đúng các điều kiện của read_contract (Phần 8.3) và
# check_same_context; (2) NSE từng lưu vực tính lại khớp test_metrics.csv của thư viện; (3) obs riêng của run trùng mặt nạ
# và giá trị với obs A3.
CONTRACT_DIR = OUT_DIR / "contracts"
CONTRACT_DIR.mkdir(exist_ok=True)
CONTRACT_KEYS = ("run", "period", "scenario", "lead", "forecast_horizon", "seed", "unit", "config_sha256",
                 "library_commit", "clip_to_zero")
NSE_TOL = 1e-5  # NSE tính lại (float64) so với test_metrics.csv của thư viện
OBS_RTOL = OBS_ATOL = 1e-5  # mm/ngày: cho phép sai số chuẩn hóa/khôi phục float32 của thư viện
TEST_DATES = pd.date_range(pd.to_datetime(str(cfg0["test_start_date"]), dayfirst=True),
                           pd.to_datetime(str(cfg0["test_end_date"]), dayfirst=True))


def basin_id(value):
    return str(int(str(value).strip()))


def contract_name(key):
    return f"SeqLSTM_Q_B3_{key[0]}_seed{key[1]}"


# (1) Ngữ cảnh A3 - như read_context của bộ đánh giá
with np.load(A3_PATHS["A3_persistence_predictions.npz"], allow_pickle=False) as d:
    CTX_DATES = pd.DatetimeIndex(d["dates"]).as_unit("ns")
    CTX_IDS = tuple(basin_id(x) for x in d["basin_ids"])
    CTX_OBS = np.array(d["obs_source_mm_day"], dtype=float)
    CTX_PERS = np.array(d["persistence_qmax_mm_day"], dtype=float)
    src_ok = np.array_equal(d["source_dates"], d["dates"] - np.timedelta64(1, "D"))
with np.load(A3_PATHS["A3_Q_source_window.npz"], allow_pickle=False) as d:
    q_ids = tuple(basin_id(x) for x in d["basin_ids"])
    q_dates = pd.DatetimeIndex(d["dates"]).as_unit("ns")
    CTX_AREA = np.array(d["area_gov_km2"], dtype=float)
    q = np.array(d["qmax_numeric_m3_s"], dtype=float)
q[q < 0] = np.nan
physical = q * 86.4 / CTX_AREA[None, :]
ctx_checks = [(f"Lịch test {TEST_YEARS} theo config.yml của B3", CTX_DATES.equals(TEST_DATES.as_unit("ns"))),
              ("Ngày nguồn persistence = ngày đích - 1", src_ok),
              ("ID nguồn Q trùng ID persistence", q_ids == CTX_IDS),
              ("Lịch nguồn Q bắt đầu trước kỳ test 1 ngày",
               q_dates.equals(pd.date_range(TEST_DATES[0] - pd.Timedelta(days=1), TEST_DATES[-1]).as_unit("ns"))),
              ("obs, persistence khớp Q gốc / diện tích",
               np.allclose(CTX_OBS, physical[1:], rtol=1e-12, atol=1e-12, equal_nan=True)
               and np.allclose(CTX_PERS, physical[:-1], rtol=1e-12, atol=1e-12, equal_nan=True)),
              ("Đủ 451 ID", len(CTX_IDS) == 451)]
show(f"Ngữ cảnh A3 ({len(CTX_DATES)} ngày x {len(CTX_IDS)} lưu vực)",
     pd.DataFrame([{"mục": a, "đạt": "ĐẠT" if b else "KHÔNG"} for a, b in ctx_checks]))
if not all(b for _, b in ctx_checks):
    raise RuntimeError(f"Phần 3 CHƯA đạt: ngữ cảnh A3 sai - kiểm Input A3 Version {A3_VERSION}")


def check_contract(path):
    # Các điều kiện của read_contract (Phần 8.3) + check_same_context (Phần 2.1) với ngữ cảnh A3; trả danh sách lỗi.
    errs = []
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"].item()))
        if any(k not in meta for k in CONTRACT_KEYS) or meta.get("unit") != "mm/day_area_gov":
            errs.append("thiếu metadata hoặc sai đơn vị")
        if not isinstance(meta.get("clip_to_zero"), bool):
            errs.append("clip_to_zero không phải bool")
        lead, hz = meta.get("lead"), meta.get("forecast_horizon")
        if not (isinstance(lead, int) and isinstance(hz, int) and 1 <= lead <= hz <= 7):
            errs.append("sai lead/forecast_horizon")
        dates = np.array(d["dates"], dtype="datetime64[D]")
        if (not np.array_equal(d["origin_dates"], dates - np.timedelta64(lead - 1, "D"))
                or not np.array_equal(d["persistence_source_dates"], dates - np.timedelta64(lead, "D"))):
            errs.append("sai origin/persistence_source_dates")
        raw, sim = np.array(d["sim_raw_mm_day"], dtype=float), np.array(d["sim_mm_day"], dtype=float)
        if not np.array_equal(sim, np.maximum(raw, 0) if meta.get("clip_to_zero") else raw, equal_nan=True):
            errs.append("sim khác chính sách chặn âm")
        if np.isinf(sim).any() or np.isinf(raw).any() or sim.shape != (len(CTX_DATES), len(CTX_IDS)):
            errs.append("sim sai kích thước hoặc có vô cực")
        if (not pd.DatetimeIndex(dates).as_unit("ns").equals(CTX_DATES)
                or tuple(basin_id(x) for x in d["basin_ids"]) != CTX_IDS
                or not np.array_equal(d["area_gov_km2"], CTX_AREA)):
            errs.append("khác lịch, ID hoặc diện tích ngữ cảnh A3")
        for k, ref in (("obs_mm_day", CTX_OBS), ("persistence_mm_day", CTX_PERS)):
            if not np.array_equal(d[k], ref, equal_nan=True):
                errs.append(f"{k} khác ngữ cảnh A3")
    return errs


def nse_basin(obs, sim):
    ok = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[ok], sim[ok]
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2) if len(o) > 1 and np.sum((o - o.mean()) ** 2) > 0 else np.nan


# (2) Dựng và kiểm từng hạt giống
rows, CONTRACT_PATHS = [], {}
target = CTX_DATES.to_numpy().astype("datetime64[D]")
for key in RUN_KEYS:
    level,s=key
    cfg_path = run_dir(key) / "config.yml"
    clip = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig")).get("clip_targets_to_zero", []) or []
    clip = [clip] if isinstance(clip, str) else clip
    # test_results.p là pickle do chính notebook hạt giống ghi, sha256 đã khớp manifest ở Phần 1 => nạp được.
    with (run_dir(key) / "test" / "best_model" / "test_results.p").open("rb") as fp:
        saved = pickle.load(fp)
    keys = {basin_id(k): k for k in saved}
    if len(keys) != len(saved) or set(keys) != set(CTX_IDS):
        raise RuntimeError(f"Hạt giống {s}: ID trong test_results.p thiếu, thừa hoặc trùng")
    native, raw = np.full_like(CTX_OBS, np.nan), np.full_like(CTX_OBS, np.nan)
    for j, b in enumerate(CTX_IDS):
        xr_ = saved[keys[b]]["1D"]["xr"]
        if (not np.array_equal(xr_["time_step"].values, [0])
                or not pd.DatetimeIndex(xr_["date"].values).as_unit("ns").equals(CTX_DATES)):
            raise RuntimeError(f"Hạt giống {s}, lưu vực {b}: time_step khác [0] hoặc lịch khác kỳ test")
        native[:, j] = xr_["qmax_obs"].transpose("date", "time_step").values[:, 0]
        raw[:, j] = xr_["qmax_sim"].transpose("date", "time_step").values[:, 0]
    sim = np.maximum(raw, 0) if "qmax" in clip else raw.copy()
    meta = {"run": contract_name(key), "period": "test", "scenario": "S0", "lead": 1, "forecast_horizon": 1, "seed": s,
            "step":"B3","mask_level":level,
            "nan_step_probability":MASKS[level][0],"nan_sequence_probability":MASKS[level][1],
            "unit": "mm/day_area_gov", "config_sha256": sha256(cfg_path), "library_commit": NH_COMMIT,
            "clip_to_zero": "qmax" in clip, "selected_epoch": BEST[key],
            "training_notebook_metadata": ENVS[key]['notebook'],
            "source": f"{SOURCE_NOTEBOOKS[key]} Version {TRAINING_VERSION_EXPECTED}, "
                      f"{ENVS[key]['run_dir']}/test/best_model/test_results.p"}
    path = CONTRACT_DIR / f"{contract_name(key)}_test_S0_lead1.npz"
    np.savez_compressed(path, dates=target, origin_dates=target, persistence_source_dates=target - np.timedelta64(1, "D"),
                        basin_ids=np.array(CTX_IDS), area_gov_km2=CTX_AREA, obs_mm_day=CTX_OBS,
                        persistence_mm_day=CTX_PERS, sim_raw_mm_day=raw, sim_mm_day=sim, metadata_json=json.dumps(meta))
    CONTRACT_PATHS[key] = path

    # Kiểm: hợp đồng, NSE so với thư viện, obs riêng của run so với A3
    errs = check_contract(path)
    lib = TEST_METRICS[key]["NSE"].reindex(list(CTX_IDS)).to_numpy(dtype=float)
    mine = np.array([nse_basin(native[:, j], sim[:, j]) for j in range(len(CTX_IDS))])
    both = np.isfinite(lib) & np.isfinite(mine)
    nse_diff = float(np.max(np.abs(lib[both] - mine[both]))) if both.any() else np.nan
    nan_same = np.array_equal(np.isfinite(lib), np.isfinite(mine))
    mask_same = np.array_equal(np.isfinite(native), np.isfinite(CTX_OBS))
    values_same = np.allclose(native, CTX_OBS, rtol=OBS_RTOL, atol=OBS_ATOL, equal_nan=True)
    both_obs = np.isfinite(native) & np.isfinite(CTX_OBS)
    canonical_nse = np.array([nse_basin(CTX_OBS[:, j], sim[:, j]) for j in range(len(CTX_IDS))])
    rows.append({"mask_level":level,"hạt giống": s, "tệp": path.name, "MB": path.stat().st_size / 1e6, "selected_epoch": BEST[key],
                 "clip_to_zero": meta["clip_to_zero"], "dự báo âm trước chặn": int((raw < 0).sum()),
                 "ngày-lưu vực có sim": int(np.isfinite(sim).sum()), "lỗi hợp đồng": "; ".join(errs) or "-",
                 "max |ΔNSE| so với thư viện": nse_diff, "NaN NSE trùng thư viện": nan_same,
                 "mặt nạ obs trùng A3": mask_same,
                 "giá trị obs khớp A3 trong sai số float32": values_same,
                 "max |obs run - obs A3|": float(np.max(np.abs(native[both_obs] - CTX_OBS[both_obs]))),
                 "NSE trung vị obs run (đối chiếu thư viện)": float(np.nanmedian(mine)),
                 "NSE trung vị từ NPZ (obs A3)": float(np.nanmedian(canonical_nse)),
                 "đạt": not errs and nse_diff <= NSE_TOL and nan_same and mask_same and values_same})
CONTRACT_TABLE = pd.DataFrame(rows)
show("Hợp đồng NPZ cho bộ đánh giá (kiểm đọc ngược và NSE tính lại)", CONTRACT_TABLE, digits=6)
if not CONTRACT_TABLE["đạt"].all():
    raise RuntimeError(f"Phần 3 CHƯA đạt (cột 'đạt' = False ở trên; ngưỡng NSE {NSE_TOL})")
print(f"=> Đạt. 6 tệp ở {CONTRACT_DIR}; bộ đánh giá đọc bằng --extra-contract. Chạy Phần 4.")


# %% Phần 4 - Đường học theo hai mức che
# Hai hàng tương ứng hai mức che; sao là checkpoint validation, đường test chỉ theo dõi.
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(14,8),squeeze=False)
for row,level in enumerate(MASKS):
    for seed in SEEDS:
        key=(level,seed)
        curve=CURVES[key]
        axes[row,0].plot(curve.epoch,curve['NSE trung vị validation (tệp)'],label=f'Seed {seed}')
        axes[row,0].scatter([BEST[key]],[SUMMARY[key]['val_nse_median_best_model']],marker='*',s=90)
        axes[row,1].plot(curve.epoch,curve['NSE trung vị test'],label=f'Seed {seed}')
    for col in range(2):
        label='Validation: checkpoint selection' if col==0 else 'Test: reporting only'
        axes[row,col].set(title=f'({chr(97+2*row+col)}) {level} - {label}',xlabel='Epoch',ylabel='Median NSE')
        axes[row,col].grid(alpha=.25)
    axes[row,1].legend(loc='upper left',bbox_to_anchor=(1.02,1))
fig.suptitle('B3 - training with source masking, two levels and three seeds each')
fig.tight_layout()
fig.savefig(OUT_DIR/'B3_Seeds.png',dpi=150,bbox_inches='tight')
plt.show()
print(f'Hình: {len(MASKS)} mức che, mỗi mức {len(SEEDS)} seed; '
      'cột trái chọn checkpoint theo validation, cột phải theo dõi test; giữ toàn bộ điểm âm.')

# %% Phần 5 - Summary và manifest
# Hai cấu hình được báo cáo đầy đủ, không xếp hạng hoặc chọn B* bằng test.
result={'complete':True,'step':'B3','scope':'test_S0_lead1_two_mask_levels_three_seeds_each',
        'library_commit':NH_COMMIT,'B2_source':b2_identity,'bstar_threshold_val_nse_sd':B0_THRESHOLD,
        'mask_levels':{},'Bstar_selected':False,'operational_scenarios_run':False,'multi_lead_run':False}
for level,probs in MASKS.items():
    table=SEED_TABLE.loc[SEED_TABLE.mask_level.eq(level)]
    result['mask_levels'][level]={'nan_step_probability':probs[0],'nan_sequence_probability':probs[1],
        'training_versions_expected':{s:TRAINING_VERSION_EXPECTED for s in SEEDS},
        'saved_epoch':{s:BEST[(level,s)] for s in SEEDS},
        'seed_val_nse_median':dict(zip(table.seed,table.validation_NSE_median)),
        'seed_test_nse_median':dict(zip(table.seed,table.test_NSE_median)),
        'contracts':{s:CONTRACT_PATHS[(level,s)].relative_to(OUT_DIR).as_posix() for s in SEEDS}}
(OUT_DIR/'b3_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
environment={'notebook':'09_B3_LamaHCEExt_Compare','env':ENV,'library_commit_of_runs':NH_COMMIT,
    'inputs':{'B3':{level:{s:{'notebook':SOURCE_NOTEBOOKS[(level,s)],'version_expected':TRAINING_VERSION_EXPECTED,
        'environment_sha256':sha256(ROOTS[(level,s)]/'b3_output/environment.json')} for s in SEEDS} for level in MASKS},
        'B2_Compare':b2_identity,'A3':{'version':A3_VERSION,'files':{name:sha256(path) for name,path in A3_PATHS.items()}}},
    'Bstar_selected':False}
(OUT_DIR/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
CONTRACT_TABLE.to_csv(OUT_DIR/'B3_contract_check.csv',index=False)
files=sorted(p for p in OUT_DIR.rglob('*') if p.is_file() and p.name!='manifest.csv')
manifest=pd.DataFrame([{'file':p.relative_to(OUT_DIR).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files])
manifest.to_csv(OUT_DIR/'manifest.csv',index=False)
check_manifest(OUT_DIR,OUT_DIR/'manifest.csv',files)
show('Output B3 Compare',manifest.assign(sha256=manifest.sha256.str[:12]))
print('Hoàn thành Compare B3 test/S0/hạn 1: sáu NPZ. Save Version rồi bàn giao cho Evaluation; '
      'còn validation S0-S4 để chọn B*, không huấn luyện lại hoặc chọn seed theo test.')
