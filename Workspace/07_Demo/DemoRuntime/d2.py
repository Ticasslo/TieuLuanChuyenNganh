def run_d2(input_root, output):
    from concurrent.futures import ThreadPoolExecutor
    from time import perf_counter
    import os
    import threading

    import psutil
    import torch

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "D2_summary.json"
    write_json(summary_path, {"complete": False})
    context, compare, records = compare_inputs(input_root)
    winners = c2_winners(records)
    latest = unique_file(input_root, "D1_summary.json")
    files, _ = verified_manifest(latest.parent)
    value = read_json(files["D1_summary.json"])
    d1 = pd.read_csv(files["D1_run_check.csv"]).set_index("run")
    basin_audit = pd.read_csv(files["D1_reproduction.csv"], dtype={"basin_id": str})
    expected = set(itertools.product((record_run(r) for r in records), context["ids"]))
    if (value.get("complete") is not True or value.get("runs") != 24
            or value.get("checks") != 24*len(context["ids"]) or value.get("library_commit") != FREEZE_C
            or value.get("C2_compare_manifest_sha256") != sha256(compare/"manifest.csv")
            or d1.index.has_duplicates or set(d1.index) != {record_run(r) for r in records}
            or not d1["passed"].eq(True).all()
            or len(basin_audit) != len(expected)
            or set(basin_audit[["run", "basin_id"]].itertuples(index=False, name=None)) != expected
            or not basin_audit["passed"].eq(True).all()):
        raise ValueError("Gắn bản D1 đã hoàn thành trên cùng bộ C2.")
    sources = c2_train_sources(input_root, winners)
    raw = locate_raw(input_root)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    process = psutil.Process()
    results = []
    for record in winners:
        run = record_run(record)
        ref, _ = c2_forecast(record, context)
        audit = d1.loc[run]
        if (audit.source_contract_sha256 != sha256(record["contract"])
                or any(audit[f"{Path(name).name}_sha256"] != sha256(file) for name, file in sources[run].items())):
            raise ValueError(f"Input của D2 khác D1: {run}.")
        start_rss = process.memory_info().rss
        rss = [start_rss]
        stop = threading.Event()
        timings, errors = [], []
        row = dict(run=run, configuration=record["conf"], core=record["cfg"]["core"],
                   seed=record["seed"], seq_length=365, horizon=1,
                   checkpoint_sha256=sha256(sources[run]["best_model.pt"]), status="verified", error=None)
        predictor = None

        def measure_rss():
            while not stop.wait(.01):
                rss[0] = max(rss[0], process.memory_info().rss)

        with ThreadPoolExecutor(max_workers=1) as pool:
            task = pool.submit(measure_rss)
            try:
                before = perf_counter()
                predictor = DemoForecaster(sources[run]["best_model.pt"].parent, raw, ref.ids, ref.area, output,
                                           device="cpu", n_params=record["cfg"]["n_params"])
                row["load_seconds"] = perf_counter()-before
                # Kiểm CPU trên ba lưu vực trải theo diện tích, 12 ngày đều trên trục test.
                for j in np.argsort(ref.area)[[0, len(ref.ids)//2, -1]]:
                    basin = ref.ids[j]
                    for i in np.linspace(0, len(ref.dates)-1, 12, dtype=int):
                        started = perf_counter()
                        prediction = predictor.predict(basin, ref.dates[i])[0]
                        timings.append((perf_counter()-started)*1000)
                        errors.append(compare_arrays(np.array([prediction["qmax_raw_mm_day"]]), ref.raw[i:i+1, j],
                                                     PRED_ATOL, PRED_RTOL))
            except (ValueError, RuntimeError, ImportError) as exc:
                row.update(status="replay_only_numeric_or_runtime", error=str(exc))
            finally:
                stop.set()
                task.result()
                if predictor is not None:
                    predictor.close()
                del predictor
        times = np.asarray(timings)
        row.update(checks=len(errors), max_abs=max(errors, default=None),
                   first_ms=float(times[0]) if len(times) else None,
                   warm_p50_ms=float(np.median(times.reshape(3, 12)[:, 1:])) if len(times)==36 else None,
                   warm_p95_ms=float(np.percentile(times.reshape(3, 12)[:, 1:], 95)) if len(times)==36 else None,
                   process_peak_rss_MiB=rss[0]/2**20, sampled_rss_growth_MiB=(rss[0]-start_rss)/2**20)
        results.append(row)
        print(f"{run}: {row['status']}, p95={row['warm_p95_ms']} ms.", flush=True)
    table = output / "D2_CPU_benchmark.csv"
    pd.DataFrame(results).to_csv(table, index=False)
    d_root, d_records, d_sources = d_demo_sources(input_root, context, compare)
    d_results, extra_paths = [], []
    if d_records:
        d1_d = (pd.read_csv(files["D1_D_run_check.csv"]).set_index("run")
                if "D1_D_run_check.csv" in files else None)
        c2_by_conf = {r["configuration"]: r for r in results}
        for d_rec in d_records:
            run = d_rec["run"]
            ref = d_rec["forecast"]
            if d1_d is not None:
                if run not in d1_d.index:
                    raise ValueError(f"D1_D_run_check.csv thiếu run {run}.")
                d_audit = d1_d.loc[run]
                if (not bool(d_audit["passed"]) or d_audit.source_contract_sha256 != sha256(d_rec["contract"])
                        or any(d_audit[f"{Path(name).name}_sha256"] != sha256(file)
                               for name, file in d_sources[run].items())):
                    raise ValueError(f"Input D của D2 khác D1: {run}.")
            start_rss = process.memory_info().rss
            rss = [start_rss]
            stop = threading.Event()
            timings, errors = [], []
            row = dict(run=run, configuration=d_rec["conf"], core=d_rec["core"],
                       seed=d_rec["seed"], seq_length=730, horizon=1,
                       checkpoint_sha256=sha256(d_sources[run]["best_model.pt"]), status="verified", error=None)
            predictor = None

            def measure_d_rss():
                while not stop.wait(.01):
                    rss[0] = max(rss[0], process.memory_info().rss)

            with ThreadPoolExecutor(max_workers=1) as pool:
                task = pool.submit(measure_d_rss)
                try:
                    before = perf_counter()
                    predictor = DemoForecaster(d_sources[run]["best_model.pt"].parent, raw, ref.ids, ref.area, output,
                                               device="cpu", n_params=d_rec["n_params"])
                    row["load_seconds"] = perf_counter()-before
                    numeric_error = None
                    for j in np.argsort(ref.area)[[0, len(ref.ids)//2, -1]]:
                        basin = ref.ids[j]
                        for i in np.linspace(0, len(ref.dates)-1, 12, dtype=int):
                            started = perf_counter()
                            prediction = predictor.predict(basin, ref.dates[i])[0]
                            timings.append((perf_counter()-started)*1000)
                            actual = np.array([prediction["qmax_raw_mm_day"]])
                            ref_slice = ref.raw[i:i+1, j]
                            errors.append(float(np.abs(actual - ref_slice).max()))
                            try:
                                compare_arrays(actual, ref_slice, PRED_ATOL, PRED_RTOL)
                            except ValueError as exc:
                                if numeric_error is None:
                                    numeric_error = str(exc)
                    if numeric_error is not None:
                        row.update(status="replay_only_numeric_or_runtime", error=numeric_error)
                except (ValueError, RuntimeError, ImportError) as exc:
                    row.update(status="replay_only_numeric_or_runtime", error=str(exc))
                finally:
                    stop.set()
                    task.result()
                    if predictor is not None:
                        predictor.close()
                    del predictor
            times = np.asarray(timings)
            p50 = float(np.median(times.reshape(3, 12)[:, 1:])) if len(times) == 36 else None
            p95 = float(np.percentile(times.reshape(3, 12)[:, 1:], 95)) if len(times) == 36 else None
            c2_match = c2_by_conf.get(d_rec["conf"], {})
            c2_p50, c2_p95 = c2_match.get("warm_p50_ms"), c2_match.get("warm_p95_ms")
            row.update(
                checks=len(errors), max_abs=max(errors, default=None),
                first_ms=float(times[0]) if len(times) else None,
                warm_p50_ms=p50, warm_p95_ms=p95,
                c2_365_warm_p50_ms=c2_p50, c2_365_warm_p95_ms=c2_p95,
                p50_ratio_730_vs_365=(p50 / c2_p50 if p50 and c2_p50 else None),
                p95_ratio_730_vs_365=(p95 / c2_p95 if p95 and c2_p95 else None),
                process_peak_rss_MiB=rss[0]/2**20, sampled_rss_growth_MiB=(rss[0]-start_rss)/2**20,
            )
            d_results.append(row)
            print(f"D {run}: {row['status']}, p95={row['warm_p95_ms']} ms (tỉ số 730/365={row['p95_ratio_730_vs_365']}).", flush=True)
        d_table = output / "D2_D_CPU_benchmark.csv"
        pd.DataFrame(d_results).to_csv(d_table, index=False)
        extra_paths.append(d_table)
    summary = dict(complete=True, notebook="02_D2_LamaHCEExt_CPU", runs=len(results), tests_per_run=36,
                   numeric_passed=sum(row["status"] == "verified" for row in results),
                   threads=torch.get_num_threads(), library_commit=FREEZE_C, D1_manifest_sha256=sha256(latest.parent/"manifest.csv"),
                   C2_compare_manifest_sha256=sha256(compare/"manifest.csv"), atol=PRED_ATOL, rtol=PRED_RTOL,
                   latency_status="review_measured_p95", memory_scope="process_RSS_sampled_10ms")
    if d_results:
        summary.update(
            D_evaluated=True, D_compare_manifest_sha256=sha256(d_root/"manifest.csv"),
            D_runs=len(d_results), D_numeric_passed=sum(row["status"] == "verified" for row in d_results),
        )
    write_json(summary_path, summary)
    env = output / "environment.json"
    write_json(env, dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                         mambapy=metadata.version("mambapy"), neuralhydrology_commit=FREEZE_C,
                         D_evaluated=bool(d_results)))
    export_manifest(output, [table, *extra_paths, summary_path, env])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Gửi D2_summary.json, D2_CPU_benchmark.csv, environment.json, manifest.csv"
          + (" (kèm D2_D_CPU_benchmark.csv)." if d_results else "."))


if __name__ == "__main__":
    install_library(FREEZE_C, mamba_cpu=True)
    run_d2(Path("/kaggle/input"), Path("/kaggle/working/d2_output"))
