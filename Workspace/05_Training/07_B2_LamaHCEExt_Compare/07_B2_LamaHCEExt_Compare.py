# B2 Compare - CPU, Input ba Output B2 + B1 Compare Version 1 + A3 Version 1.
# Mục lục: 1 nguồn/hash/config; 2 ba seed và mốc B1b; 3 NPZ; 4 hình; 5 summary/manifest.
# %% Phần 1 - Đọc ba Output B2, ngữ cảnh A3 và Compare B1 Version 1
# CPU. Ba notebook train 06_B2 đã Save Version 1; giữ tên nhúng cũ nếu environment chưa đổi số.
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
INPUT_ROOT=Path(os.environ.get('B2_INPUT_ROOT','/kaggle/input' if Path('/kaggle/working').exists() else 'b2_compare_input'))
WORK_DIR=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('b2_compare_work')
OUT_DIR=WORK_DIR/'b2_compare'
OUT_DIR.mkdir(parents=True,exist_ok=True)
SEEDS=[111,222,333]
VERSIONS={s:1 for s in SEEDS}  # cả ba bản Save Version 1 đã được xác nhận ngày 31/10/2026
SOURCE_NOTEBOOKS={s:f'06_B2_LamaHCEExt_Train_Seed{s}' for s in SEEDS}
A3_VERSION=1
B1_COMPARE_VERSION=1
B1B_COMMIT='f229e02bf98cbd979835a4b6cd503d0e3adb8bc3'

def show(title,frame,digits=4):
    display(HTML(f'<h4>{title}</h4>'))
    display(frame.style.hide(axis='index').format(precision=digits,na_rep='-'))
def unique_file(root,name):
    hits=sorted(root.rglob(name))
    if len(hits)!=1:raise RuntimeError(f'Cần đúng một {name}, thấy {len(hits)} nguồn.')
    return hits[0]
def read_metrics(path):
    frame=pd.read_csv(path,index_col=0)
    frame.index=[str(int(b)) for b in frame.index]
    if frame.index.has_duplicates:raise ValueError(f'ID trùng trong {path}')
    return frame

# B1 Compare có manifest tương đối từ b1_compare/, không từ /kaggle/working.
B1_SUMMARY_PATH=unique_file(INPUT_ROOT,'b1_summary.json')
B1_DIR=B1_SUMMARY_PATH.parent
B1_COMPARE_ENV_PATH=B1_DIR/'environment.json'
B1_MANIFEST=check_manifest(B1_DIR,B1_DIR/'manifest.csv',[B1_SUMMARY_PATH,B1_COMPARE_ENV_PATH])
B1_COMPARE_ENV=json.loads(B1_COMPARE_ENV_PATH.read_text(encoding='utf-8'))
B1_SUMMARY=json.loads(B1_SUMMARY_PATH.read_text(encoding='utf-8'))
if B1_COMPARE_ENV.get('notebook')!='05_B1_LamaHCEExt_Compare' or B1_COMPARE_ENV.get('neuralhydrology_commit_of_runs')!=B1B_COMMIT:
    raise ValueError('Nguồn B1 Compare không đúng bàn giao.')
for s in SEEDS:
    source=B1_COMPARE_ENV['inputs']['B1b'][str(s)]
    if source['version']!=1 or source['notebook']!=f'04_B1b_LamaHCEExt_Train_Seed{s}':
        raise ValueError('Nguồn B1b trong Compare khác Version 1.')
ROOTS,ENVS,SUMMARY,CURVES,TEST_METRICS=({'B2':{}} for _ in range(5))
EXP,BEST,run_dirs={}, {}, {}
source_rows=[]
versions=VERSIONS
science,packages=None,None
seen={}
for ep in sorted(INPUT_ROOT.rglob('b2_output/environment.json')):
    e=json.loads(ep.read_text(encoding='utf-8'))
    s=e.get('seed')
    if s in seen:raise ValueError(f'Trùng Output B2 seed {s}')
    seen[s]=(ep,e)
if sorted(seen)!=SEEDS:raise RuntimeError(f'Cần Output đúng ba seed B2 {SEEDS}; thấy {sorted(seen)}')
for s in SEEDS:
    ep,e=seen[s]
    root=ep.parent.parent
    run=inside(root,e['run_dir'])
    summary_path=ep.parent/f"summary_{e['experiment_name']}.json"
    curve_path=ep.parent/f"curve_{e['experiment_name']}.csv"
    config_path=run/'config.yml'
    check_manifest(root,ep.parent/'manifest.csv',[ep,summary_path,curve_path,config_path,
        run/'best_model.pt',run/'test/best_model/test_results.p',run/'test/best_model/test_metrics.csv'])
    # Bản đã chạy có thể còn tên 07 trong environment, trước khi sửa thứ tự thành 06.
    # Giữ nguyên environment/hash nguồn; tên notebook Kaggle đã xác nhận là SOURCE_NOTEBOOKS.
    accepted_metadata_names={SOURCE_NOTEBOOKS[s],f'07_B2_LamaHCEExt_Train_Seed{s}'}
    if (e.get('seed')!=s or e.get('step')!='B2' or e.get('neuralhydrology_commit')!=NH_COMMIT
            or e.get('notebook') not in accepted_metadata_names or sha256(config_path)!=e['config_sha256']):
        raise ValueError(f'Seed {s}: danh tính/config B2 không đúng.')
    source_rows.append({'seed':s,'notebook':SOURCE_NOTEBOOKS[s],'version':VERSIONS[s],
        'notebook_in_environment':e['notebook'],'metadata_name_matches_source':e['notebook']==SOURCE_NOTEBOOKS[s],
        'environment_sha256':sha256(ep)})
    t4_digest=e.get('T4_check_sha256','')
    if not isinstance(t4_digest,str) or len(t4_digest)!=64 or any(c not in '0123456789abcdef' for c in t4_digest):
        raise ValueError(f'Seed {s}: thiếu hash kết quả kiểm T4 của phiên train.')
    source=e['B1b_source']
    if source.get('seed')!=s or source.get('version')!=1 or source.get('library_commit')!=B1B_COMMIT:
        raise ValueError(f'Seed {s}: nguồn cấu hình không phải B1b đã bàn giao.')
    c=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    current={k:v for k,v in c.items() if k not in {'seed','experiment_name','run_dir','train_dir','img_log_dir',
        'data_dir','train_basin_file','validation_basin_file','test_basin_file'}}
    if science is None:science,packages=current,e['libs']
    if current!=science or e['libs']!=packages:raise ValueError('Ba seed khác cấu hình khoa học/môi trường.')
    if (c.get('nan_handling_method')!='masked_mean' or c.get('nan_step_probability',0) or c.get('nan_sequence_probability',0)
        or c.get('timestep_counter',False) or c['epochs']!=30 or c['early_stopping_patience']!=30 or not c['data_on_gpu']):
        raise ValueError('Không phải giao thức B2.')
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    curve=pd.read_csv(curve_path)
    metrics=read_metrics(run/'test/best_model/test_metrics.csv')
    if (summary!=e['summary'] or summary.get('seed')!=s or summary.get('n_epochs')!=30
        or summary.get('saved_epoch')!=summary.get('true_best_epoch') or summary.get('library_commit')!=NH_COMMIT
        or not summary.get('ready_for_evaluation') or curve.epoch.tolist()!=list(range(1,31))
        or not np.isfinite(curve['NSE trung vị validation (tệp)']).all()
        or int(curve.loc[curve['NSE trung vị validation (tệp)'].idxmax(),'epoch'])!=summary['saved_epoch']
        or abs(float(curve.set_index('epoch').loc[summary['saved_epoch'],'NSE trung vị validation (tệp)'])-summary['val_nse_median_best_model'])>1e-9
        or abs(float(metrics.NSE.median())-summary['test_nse_median_best_model'])>1e-9):
        raise ValueError(f'Seed {s}: curve/summary/checkpoint/chỉ số chưa khớp.')
    ROOTS['B2'][s],ENVS['B2'][s],SUMMARY['B2'][s]=root,e,summary
    CURVES['B2'][s],TEST_METRICS['B2'][s]=curve,metrics
    EXP[s],BEST[s],run_dirs[s]=e['experiment_name'],summary['saved_epoch'],run
RUNS={'B2':{'commit':NH_COMMIT,'notebook':'06_B2_LamaHCEExt_Train_Seed{}','versions':VERSIONS}}
SOURCE_TABLE=pd.DataFrame(source_rows)
SOURCE_TABLE.to_csv(OUT_DIR/'B2_input_sources.csv',index=False)
show('Nguồn B2 đã lưu, giữ nguyên tên nhúng trong environment',SOURCE_TABLE)
A3_PATHS={name:unique_file(INPUT_ROOT,name) for name in ['A3_persistence_predictions.npz','A3_Q_source_window.npz']}
for name,path in A3_PATHS.items():
    if sha256(path)!=B1_COMPARE_ENV['inputs']['a3']['files'][name]['sha256']:
        raise ValueError(f'A3 khác nguồn đã dùng B1 Compare: {name}')
if B1_COMPARE_ENV['inputs']['a3']['version']!=A3_VERSION:raise ValueError('Sai Version A3.')
cfg0=yaml.safe_load((run_dirs[111]/'config.yml').read_text(encoding='utf-8'))
TEST_YEARS=f"{str(cfg0['test_start_date'])[-4:]}-{str(cfg0['test_end_date'])[-4:]}"
ENV={'python':platform.python_version(),'numpy':metadata.version('numpy'),'pandas':metadata.version('pandas')}
b1_identity={'notebook':'05_B1_LamaHCEExt_Compare','version':B1_COMPARE_VERSION,
    'summary_sha256':sha256(B1_SUMMARY_PATH),'environment_sha256':sha256(B1_COMPARE_ENV_PATH)}
def run_dir(seed):return run_dirs[seed]
print('Ba seed B2 khớp manifest, cấu hình, checkpoint; nguồn B1 Compare Version 1 đã kiểm hash.')


# %% Phần 2 - Tổng hợp ba seed
# Chỉ chọn checkpoint theo validation. Test để báo cáo; không lấy std B2 thay ngưỡng B0.
rows = [{'seed':s,'selected_epoch':BEST[s],
         'validation_NSE_median':SUMMARY['B2'][s]['val_nse_median_best_model'],
         'test_NSE_median':SUMMARY['B2'][s]['test_nse_median_best_model'],
         'test_KGE_median':SUMMARY['B2'][s]['test_kge_median_best_model'],
         'train_hours':SUMMARY['B2'][s]['train_hours']} for s in SEEDS]
SEED_TABLE = pd.DataFrame(rows)
SEED_TABLE.to_csv(OUT_DIR/'B2_seed_table.csv',index=False)
STATS = SEED_TABLE.drop(columns=['seed','selected_epoch']).agg(['mean','std']).T.reset_index(names='metric')
STATS.to_csv(OUT_DIR/'B2_seed_stats.csv',index=False)
show('Ba seed B2, checkpoint theo validation',SEED_TABLE)
show('Trung bình và độ lệch chuẩn mẫu qua ba seed',STATS)


# B1b là mốc đã bàn giao; hiệu của trung bình ba median chưa phải hiệu trên giao ngày chung.
baseline=B1_SUMMARY['step_mean_sd']['B1b']
delta={'comparison':'B2 - B1b','B1b_validation_NSE_mean':baseline['NSE trung vị validation'][0],
    'B2_validation_NSE_mean':float(SEED_TABLE.validation_NSE_median.mean()),
    'validation_NSE_mean_difference':float(SEED_TABLE.validation_NSE_median.mean()-baseline['NSE trung vị validation'][0]),
    'B0_Bstar_threshold':B1_SUMMARY['bstar_threshold_val_nse_sd'],
    'common_mask_evaluation_completed':False,'Bstar_selected':False}
pd.DataFrame([delta]).to_csv(OUT_DIR/'B2_B1b_reported_summary.csv',index=False)
show('Đối chiếu validation với B1b, chưa chọn B*',pd.DataFrame([delta]))


# %% Phần 3 - Dựng NPZ theo hợp đồng bộ đánh giá và kiểm lại
# Mỗi hạt giống một tệp SeqLSTM_Q_B2_seed{N}_test_S0_lead1.npz theo hợp đồng ở Workspace/06_Evaluation/
# 01_Evaluation_LamaHCEExt.md Mục 7, cùng cách dựng A4 (04_A4_LamaHCEExt_Compare.py Phần 5, đã qua bộ đánh giá Version 4):
# sim thô lấy từ test_results.p của best_model.pt (["1D"]["xr"]["qmax_sim"], time_step = 0, chưa chặn âm - tester.py dòng
# 319-322 chỉ chặn âm khi tính chỉ số), chặn âm theo clip_targets_to_zero của config.yml. obs, persistence, diện tích, thứ
# tự ID lấy nguyên từ ngữ cảnh A3 (như read_context của bộ đánh giá, Phần 8.1) vì bộ đánh giá đòi bằng nhau từng bit giữa
# các run cùng nhóm (check_same_context). Khác A4: library_commit là commit B2 đã gộp research (2ccaa14), selected_epoch là epoch
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


def contract_name(seed):
    return f"SeqLSTM_Q_B2_seed{seed}"


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
ctx_checks = [(f"Lịch test {TEST_YEARS} theo config.yml của B2", CTX_DATES.equals(TEST_DATES.as_unit("ns"))),
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
for s in SEEDS:
    cfg_path = run_dir(s) / "config.yml"
    clip = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig")).get("clip_targets_to_zero", []) or []
    clip = [clip] if isinstance(clip, str) else clip
    # test_results.p là pickle do chính notebook hạt giống ghi, sha256 đã khớp manifest ở Phần 1 => nạp được.
    with (run_dir(s) / "test" / "best_model" / "test_results.p").open("rb") as fp:
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
    meta = {"run": contract_name(s), "period": "test", "scenario": "S0", "lead": 1, "forecast_horizon": 1, "seed": s,
            "unit": "mm/day_area_gov", "config_sha256": sha256(cfg_path), "library_commit": RUNS["B2"]["commit"],
            "clip_to_zero": "qmax" in clip, "selected_epoch": BEST[s],
            "training_notebook_metadata": ENVS['B2'][s]['notebook'],
            "source": f"{RUNS['B2']['notebook'].format(s)} Version {RUNS['B2']['versions'][s]}, "
                      f"{ENVS['B2'][s]['run_dir']}/test/best_model/test_results.p"}
    path = CONTRACT_DIR / f"{contract_name(s)}_test_S0_lead1.npz"
    np.savez_compressed(path, dates=target, origin_dates=target, persistence_source_dates=target - np.timedelta64(1, "D"),
                        basin_ids=np.array(CTX_IDS), area_gov_km2=CTX_AREA, obs_mm_day=CTX_OBS,
                        persistence_mm_day=CTX_PERS, sim_raw_mm_day=raw, sim_mm_day=sim, metadata_json=json.dumps(meta))
    CONTRACT_PATHS[s] = path

    # Kiểm: hợp đồng, NSE so với thư viện, obs riêng của run so với A3
    errs = check_contract(path)
    lib = TEST_METRICS["B2"][s]["NSE"].reindex(list(CTX_IDS)).to_numpy(dtype=float)
    mine = np.array([nse_basin(native[:, j], sim[:, j]) for j in range(len(CTX_IDS))])
    both = np.isfinite(lib) & np.isfinite(mine)
    nse_diff = float(np.max(np.abs(lib[both] - mine[both]))) if both.any() else np.nan
    nan_same = np.array_equal(np.isfinite(lib), np.isfinite(mine))
    mask_same = np.array_equal(np.isfinite(native), np.isfinite(CTX_OBS))
    values_same = np.allclose(native, CTX_OBS, rtol=OBS_RTOL, atol=OBS_ATOL, equal_nan=True)
    both_obs = np.isfinite(native) & np.isfinite(CTX_OBS)
    canonical_nse = np.array([nse_basin(CTX_OBS[:, j], sim[:, j]) for j in range(len(CTX_IDS))])
    rows.append({"hạt giống": s, "tệp": path.name, "MB": path.stat().st_size / 1e6, "selected_epoch": BEST[s],
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
print(f"=> Đạt. 3 tệp ở {CONTRACT_DIR}; bộ đánh giá đọc bằng --extra-contract. Chạy Phần 4.")


# %% Phần 4 - Hình đường học và kết quả theo seed
# Không cắt điểm NSE âm; chú giải ngoài hình. Chưa so B* hoặc S1-S4 khi chưa có NPZ.
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13,4.5))
for s in SEEDS:
    curve = CURVES['B2'][s]
    axes[0].plot(curve.epoch,curve['NSE trung vị validation (tệp)'],label=f'Seed {s}')
    axes[0].scatter([BEST[s]],[SUMMARY['B2'][s]['val_nse_median_best_model']],marker='*',s=90)
    axes[1].plot(curve.epoch,curve['NSE trung vị test'],label=f'Seed {s}')
axes[0].set(title='(a) Validation: checkpoint selection',xlabel='Epoch',ylabel='Median NSE')
axes[1].set(title='(b) Test: reporting only',xlabel='Epoch',ylabel='Median NSE')
for ax in axes:
    ax.grid(alpha=.25)
axes[1].legend(loc='upper left',bbox_to_anchor=(1.02,1))
fig.suptitle('B2 - source embeddings and masked mean, three seeds')
fig.tight_layout()
fig.savefig(OUT_DIR/'B2_Seeds.png',dpi=150,bbox_inches='tight')
plt.show()
print('Hình: (a) NSE validation chọn checkpoint, dấu sao là epoch đã chọn; '
      '(b) NSE test chỉ theo dõi, không dùng chọn epoch hay B*.')


# %% Phần 5 - Tóm tắt và manifest bàn giao bộ đánh giá
# NPZ test/S0/hạn 1 đã kiểm ở Phần 3; validation/S1-S4 là bước tiếp theo để chọn B*.
result = {'complete':True,'step':'B2','scope':'test_S0_lead1_three_seeds',
          'library_commit':NH_COMMIT,'training_versions_expected':versions,'saved_epoch':BEST,
          'seed_val_nse_median':dict(zip(SEED_TABLE.seed,SEED_TABLE.validation_NSE_median)),
          'seed_test_nse_median':dict(zip(SEED_TABLE.seed,SEED_TABLE.test_NSE_median)),
          'seed_test_nse_mean':float(SEED_TABLE.test_NSE_median.mean()),
          'seed_test_nse_sd':float(SEED_TABLE.test_NSE_median.std(ddof=1)),
          'contracts':{s:CONTRACT_PATHS[s].relative_to(OUT_DIR).as_posix() for s in SEEDS},
          'B1b_source':b1_identity,'B1b_validation_comparison':delta,'Bstar_selected':False,'operational_scenarios_run':False}
(OUT_DIR/'b2_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
environment = {'notebook':'07_B2_LamaHCEExt_Compare','env':ENV,'library_commit_of_runs':NH_COMMIT,
               'inputs':{'B2':{s:{'notebook':SOURCE_NOTEBOOKS[s],'version_expected':VERSIONS[s],
                    'notebook_in_environment':ENVS['B2'][s]['notebook'],
                    'environment_sha256':sha256(ROOTS['B2'][s]/'b2_output/environment.json')} for s in SEEDS},
                    'B1_Compare':b1_identity,'A3':{'version':A3_VERSION,'files':{n:sha256(p) for n,p in A3_PATHS.items()}}},'Bstar_selected':False}
(OUT_DIR/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
CONTRACT_TABLE.to_csv(OUT_DIR/'B2_contract_check.csv',index=False)
files = sorted(p for p in OUT_DIR.rglob('*') if p.is_file() and p.name != 'manifest.csv')
manifest = pd.DataFrame([{'file':p.relative_to(OUT_DIR).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files])
manifest.to_csv(OUT_DIR/'manifest.csv',index=False)
show('Output B2 Compare',manifest.assign(sha256=manifest.sha256.str[:12]))
print('Hoàn thành Compare B2 test/S0/hạn 1. Chưa chọn B*: cần validation và S1-S4 theo pipeline.')
