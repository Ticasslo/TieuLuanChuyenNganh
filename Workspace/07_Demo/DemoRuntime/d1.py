def run_d1(input_root, output):
    import torch
    from time import perf_counter

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "D1_summary.json"
    write_json(summary_path, {"complete": False})
    context, compare, records = compare_inputs(input_root)
    sources = c2_train_sources(input_root, records)
    raw_dir = locate_raw(input_root)
    basin_rows, run_rows, examples = [], [], []
    for record in records:
        run = record_run(record)
        ref, _ = c2_forecast(record, context)
        started = perf_counter()
        predictor = DemoForecaster(sources[run]["best_model.pt"].parent, raw_dir, ref.ids, ref.area, output,
                                   n_params=record["cfg"]["n_params"])
        records_start = len(basin_rows)
        try:
            for j, basin in enumerate(ref.ids):
                series = predictor.predict_basin(basin)
                prediction, audit = audit_prediction(series, ref, j)
                basin_rows.append(dict(run=run, basin_id=basin, passed=True, **audit))
                if j in (0, len(ref.ids)//2, len(ref.ids)-1):
                    i = len(ref.dates)//2
                    day = str(ref.dates[i].date())
                    actual = predictor.predict(basin, day)[0]
                    compare_arrays(np.array([actual["qmax_raw_mm_day"]]), prediction[i:i+1], PRED_ATOL, PRED_RTOL)
                    examples.append(dict(run=run, **actual))
        finally:
            predictor.close()
        elapsed = perf_counter() - started
        run_rows.append(dict(run=run, configuration=record["conf"], seed=record["seed"], epoch=record["epoch"],
                             core=record["cfg"]["core"], n_params=record["cfg"]["n_params"], horizon=1, days=len(ref.dates),
                             IDs=len(basin_rows)-records_start, seconds=elapsed, passed=True,
                             source_contract_sha256=sha256(record["contract"]),
                             **{f"{Path(name).name}_sha256": sha256(file) for name, file in sources[run].items()}))
        del predictor
        torch.cuda.empty_cache()
        pd.DataFrame(basin_rows).to_csv(output / "D1_reproduction.csv", index=False)
        pd.DataFrame(run_rows).to_csv(output / "D1_run_check.csv", index=False)
        print(f"{len(run_rows)}/24 {run}: đạt đối chiếu {len(ref.ids)} lưu vực ({elapsed:.1f}s).", flush=True)
    pd.DataFrame(examples).to_csv(output / "D1_examples.csv", index=False)
    d_root, d_records, d_sources = d_demo_sources(input_root, context, compare)
    d_basin_rows, d_run_rows, d_examples = [], [], []
    for d_rec in d_records:
        run = d_rec["run"]
        ref = d_rec["forecast"]
        started = perf_counter()
        predictor = DemoForecaster(d_sources[run]["best_model.pt"].parent, raw_dir, ref.ids, ref.area, output,
                                   n_params=d_rec["n_params"])
        records_start = len(d_basin_rows)
        try:
            for j, basin in enumerate(ref.ids):
                series = predictor.predict_basin(basin)
                prediction, audit = audit_prediction(series, ref, j)
                d_basin_rows.append(dict(run=run, basin_id=basin, passed=True, **audit))
                if j in (0, len(ref.ids)//2, len(ref.ids)-1):
                    i = len(ref.dates)//2
                    day = str(ref.dates[i].date())
                    actual = predictor.predict(basin, day)[0]
                    compare_arrays(np.array([actual["qmax_raw_mm_day"]]), prediction[i:i+1], PRED_ATOL, PRED_RTOL)
                    d_examples.append(dict(run=run, **actual))
        finally:
            predictor.close()
        elapsed = perf_counter() - started
        d_run_rows.append(dict(run=run, configuration=d_rec["conf"], seed=d_rec["seed"], epoch=d_rec["epoch"],
                               core=d_rec["core"], n_params=d_rec["n_params"], seq_length=730, horizon=1,
                               days=len(ref.dates), IDs=len(d_basin_rows)-records_start, seconds=elapsed, passed=True,
                               source_contract_sha256=sha256(d_rec["contract"]),
                               **{f"{Path(name).name}_sha256": sha256(file) for name, file in d_sources[run].items()}))
        del predictor
        torch.cuda.empty_cache()
        print(f"D {len(d_run_rows)}/{len(d_records)} {run}: đạt đối chiếu {len(ref.ids)} lưu vực ({elapsed:.1f}s).", flush=True)
    extra_names = ()
    if d_run_rows:
        pd.DataFrame(d_basin_rows).to_csv(output / "D1_D_reproduction.csv", index=False)
        pd.DataFrame(d_run_rows).to_csv(output / "D1_D_run_check.csv", index=False)
        pd.DataFrame(d_examples).to_csv(output / "D1_D_examples.csv", index=False)
        extra_names = ("D1_D_run_check.csv", "D1_D_reproduction.csv", "D1_D_examples.csv")
    summary = dict(complete=True, notebook="01_D1_LamaHCEExt_Inference", library_commit=FREEZE_C,
                   C2_compare_manifest_sha256=sha256(compare/"manifest.csv"), C2_compare_version_expected=1,
                   runs=len(run_rows), checks=len(basin_rows), IDs=len(context["ids"]), days=len(context["dates"]),
                   atol=PRED_ATOL, rtol=PRED_RTOL, optimizer_steps=0,
                   max_raw_abs_error=max(row["max_raw_abs_error"] for row in basin_rows),
                   max_clip_abs_error=max(row["max_clip_abs_error"] for row in basin_rows),
                   default_demo_run=C2_DEFAULT)
    if d_run_rows:
        summary.update(
            D_evaluated=True, D_compare_manifest_sha256=sha256(d_root/"manifest.csv"),
            D_runs=len(d_run_rows), D_checks=len(d_basin_rows),
            D_max_raw_abs_error=max(row["max_raw_abs_error"] for row in d_basin_rows),
            D_max_clip_abs_error=max(row["max_clip_abs_error"] for row in d_basin_rows),
        )
    write_json(summary_path, summary)
    env = output / "environment.json"
    write_json(env, dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                         torch=torch.__version__, torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                         neuralhydrology_commit=FREEZE_C, C2_train_version_expected=1, A3_version_expected=1,
                         D_evaluated=bool(d_run_rows)))
    names = ("D1_summary.json", "D1_run_check.csv", "D1_reproduction.csv", "D1_examples.csv", *extra_names, "environment.json")
    export_manifest(output, [output/name for name in names])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Gửi D1_summary.json, D1_run_check.csv, D1_reproduction.csv, D1_examples.csv, environment.json, manifest.csv"
          + (" (kèm D1_D_*.csv)." if d_run_rows else "."))


if __name__ == "__main__":
    import torch

    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Chọn Kaggle T4 cho D1.")
    root = Path("/kaggle/input")
    install_mamba_cuda(root)
    install_library(FREEZE_C)
    run_d1(root, Path("/kaggle/working/d1_output"))
