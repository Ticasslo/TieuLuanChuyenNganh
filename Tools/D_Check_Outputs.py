from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FREEZE_C = "db6dc1ffcd8c14994cd80be3ed8ff417002c01a1"
D_SPECS = {
    "lstm_L2_x1_lr1e-3": ("lstm", 84106),
    "mamba_L2_x1_lr1e-3": ("mamba", 85039),
    "s4d_L4_x1_lr1e-3": ("s4d", 84449),
}
SEEDS = (111, 222, 333)
C2_SEED_TABLE = ROOT / "Workspace/05_Training/13_C2_LamaHCEExt_Compare/Tables/C2_seed_table.csv"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_costs(summary):
    prev = summary.get("previous_session") or {}
    sessions = 2 if prev else 1
    hours = float(summary["train_hours"]) + float(prev.get("train_hours") or 0.0)
    mibs = [float(x) for x in (summary.get("gpu_mib_max"), prev.get("gpu_mib_max")) if x is not None]
    return sessions, hours, (max(mibs) if mibs else np.nan)


def load_c2_reference():
    if not C2_SEED_TABLE.is_file():
        return {}
    df = pd.read_csv(C2_SEED_TABLE)
    ref = {}
    for _, row in df.iterrows():
        ref[(str(row["conf"]), int(row["seed"]))] = (
            float(row["val_nse_median"]),
            float(row["test_nse_median"]),
        )
    return ref


def audit_one_env(env_path):
    env = json.loads(env_path.read_text(encoding="utf-8-sig"))
    summary = env.get("summary") or {}
    conf = summary.get("configuration")
    seed = summary.get("seed")
    if conf not in D_SPECS or seed not in SEEDS:
        return None
    seed = int(seed)
    expected_core, expected_params = D_SPECS[conf]
    errors = []
    if env.get("neuralhydrology_commit") != FREEZE_C:
        errors.append(f"commit={env.get('neuralhydrology_commit')} != {FREEZE_C[:7]}")
    if summary.get("core") != expected_core:
        errors.append(f"core={summary.get('core')} != {expected_core}")
    if summary.get("n_params") != expected_params:
        errors.append(f"n_params={summary.get('n_params')} != {expected_params}")
    if summary.get("seq_length") != 730:
        errors.append(f"seq_length={summary.get('seq_length')} != 730")

    c_out = env_path.parent
    manifest_path = c_out / "manifest.csv"
    checked_files = 0
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        file_col = "file" if "file" in manifest.columns else "tệp"
        base = c_out.parent if c_out.name == "c_output" else c_out
        for _, mrow in manifest.iterrows():
            rel = mrow[file_col]
            candidate = base / rel
            if not candidate.is_file() and rel.startswith("c_output/"):
                candidate = c_out / rel.removeprefix("c_output/")
            if candidate.is_file():
                checked_files += 1
                actual_sha = sha256(candidate)
                if actual_sha != mrow["sha256"]:
                    errors.append(f"SHA lệch: {candidate.name}")
    else:
        errors.append("thiếu manifest.csv")

    exp = summary.get("experiment_name", f"SF_{conf}_D_seed{seed}")
    curve_candidates = sorted(c_out.glob(f"curve_{exp}.csv"))
    if summary.get("status") == "xong":
        if len(curve_candidates) == 1:
            curve = pd.read_csv(curve_candidates[0])
            if len(curve) != 30 or list(curve["epoch"]) != list(range(1, 31)):
                errors.append(f"curve có {len(curve)} epoch (cần 30)")
            else:
                best_idx = int(curve["val_nse"].astype(float).idxmax())
                best_ep = int(curve.loc[best_idx, "epoch"])
                if best_ep != int(summary.get("saved_epoch", -1)):
                    errors.append(f"saved_epoch={summary.get('saved_epoch')} != argmax curve={best_ep}")
                if not np.isclose(float(curve.loc[best_idx, "val_nse"]), float(summary.get("val_nse", np.nan)), atol=1e-6):
                    errors.append("val_nse lệch curve")
                if not np.isclose(float(curve.loc[best_idx, "test_nse"]), float(summary.get("test_nse", np.nan)), atol=1e-6):
                    errors.append("test_nse lệch curve")
        else:
            errors.append("thiếu curve_*.csv")

    sessions, hours, gpu_mib = run_costs(summary)
    is_cont = str(env.get("notebook", "")).endswith("_Cont")
    return dict(
        conf=conf,
        seed=seed,
        notebook=env.get("notebook"),
        is_cont=is_cont,
        status=summary.get("status"),
        last_epoch=summary.get("last_epoch"),
        saved_epoch=summary.get("saved_epoch"),
        val_nse=summary.get("val_nse"),
        test_nse=summary.get("test_nse"),
        sessions=sessions,
        train_hours=hours,
        gpu_mib_max=gpu_mib,
        grad_guard_enabled=bool(summary.get("grad_guard_enabled", False)),
        grad_guard_skipped=int(summary.get("grad_guard_skipped_steps") or 0),
        checked_files=checked_files,
        errors=errors,
        path=str(env_path),
    )


def check_directory(target_dir):
    target = Path(target_dir)
    if not target.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {target}")
    found = {}
    for env_path in sorted(target.rglob("environment.json")):
        rec = audit_one_env(env_path)
        if rec is None:
            continue
        key = (rec["conf"], rec["seed"])
        prev = found.get(key)
        # Ưu tiên bản đã xong hoặc bản _Cont có last_epoch cao hơn.
        if prev is None or (rec["status"] == "xong" and prev["status"] != "xong") or (
            (rec["last_epoch"] or 0) > (prev["last_epoch"] or 0)
        ):
            found[key] = rec

    c2_ref = load_c2_reference()
    rows = []
    for conf in D_SPECS:
        for seed in SEEDS:
            rec = found.get((conf, seed))
            c2_val, c2_test = c2_ref.get((conf, seed), (np.nan, np.nan))
            if rec is None:
                rows.append(dict(
                    configuration=conf, seed=seed, status="chưa có", notebook=f"16_D_Train_{conf}_Seed{seed}",
                    epochs="0/30", saved_epoch=None, val_730=np.nan, test_730=np.nan,
                    c2_test_365=c2_test, delta_test_730_365=np.nan,
                    sessions=0, train_hours=np.nan, gpu_mib=np.nan,
                    grad_skips=0, sha_checked=0, audit="chờ tải Output",
                ))
            else:
                t730 = float(rec["test_nse"]) if rec["test_nse"] is not None else np.nan
                v730 = float(rec["val_nse"]) if rec["val_nse"] is not None else np.nan
                next_action = (
                    f"CHẠY 16_D_Train_{conf}_Seed{seed}_Cont.py" if rec["status"] == "cần phiên 2"
                    else ("ĐẠT" if not rec["errors"] and rec["status"] == "xong" else "; ".join(rec["errors"]))
                )
                rows.append(dict(
                    configuration=conf, seed=seed, status=rec["status"], notebook=rec["notebook"],
                    epochs=f"{rec['last_epoch']}/30", saved_epoch=rec["saved_epoch"],
                    val_730=round(v730, 5) if np.isfinite(v730) else None,
                    test_730=round(t730, 5) if np.isfinite(t730) else None,
                    c2_test_365=round(c2_test, 5) if np.isfinite(c2_test) else None,
                    delta_test_730_365=round(t730 - c2_test, 5) if np.isfinite(t730) and np.isfinite(c2_test) else None,
                    sessions=rec["sessions"],
                    train_hours=round(rec["train_hours"], 2) if np.isfinite(rec["train_hours"]) else None,
                    gpu_mib=round(rec["gpu_mib_max"], 1) if np.isfinite(rec["gpu_mib_max"]) else None,
                    grad_skips=rec["grad_guard_skipped"],
                    sha_checked=rec["checked_files"],
                    audit=next_action,
                ))
    table = pd.DataFrame(rows)
    done_6 = sum(
        r["status"] == "xong" and r["audit"] == "ĐẠT"
        for r in rows if r["configuration"] in ("lstm_L2_x1_lr1e-3", "mamba_L2_x1_lr1e-3")
    )
    done_9 = sum(r["status"] == "xong" and r["audit"] == "ĐẠT" for r in rows)
    return table, done_6, done_9


def main():
    parser = argparse.ArgumentParser(description="Kiểm nhanh tiến độ và mã băm Output Kaggle của bước D (16_D_Train_*).")
    parser.add_argument("directory", nargs="?", default=".", help="Thư mục chứa các Output 16_D_Train_* đã tải về")
    args = parser.parse_args()
    table, done_6, done_9 = check_directory(args.directory)
    print(table.to_string(index=False))
    print("-" * 96)
    print(f"Tiến độ đạt chuẩn: {done_6}/6 run bắt buộc (LSTM + Mamba) | {done_9}/9 run đầy đủ (+ S4D).")
    if done_9 == 9:
        print("SẴN SÀNG chạy 17_D_LamaHCEExt_Compare với đủ 9 cặp (LSTM + Mamba + S4D).")
    elif done_6 == 6:
        print("ĐÃ ĐỦ 6 run LSTM + Mamba: có thể chạy 17_D_LamaHCEExt_Compare (chế độ 6 cặp) nếu cắt S4D theo Mục 8.3, hoặc chờ S4D xong.")


if __name__ == "__main__":
    main()
