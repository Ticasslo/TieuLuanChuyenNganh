from collections import OrderedDict
from bisect import bisect_left
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import threading

import numpy as np
import pandas as pd


PRED_ATOL, PRED_RTOL = 2e-5, 1e-6
HRES = ("ECMWF_t2m", "ECMWF_d2m", "ECMWF_ssrd", "ECMWF_tp", "ECMWF_e")


def qmax_m3_s(mm_day, area_gov):
    return mm_day * area_gov / 86.4


def compare_arrays(actual, reference, atol, rtol):
    if actual.shape != reference.shape or not np.array_equal(np.isfinite(actual), np.isfinite(reference)):
        raise ValueError("Kích thước hoặc mặt nạ hữu hạn khác dự báo nguồn.")
    if not np.allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=True) or np.isinf(actual).any():
        raise ValueError(f"Dự báo khác nguồn, max abs={np.nanmax(np.abs(actual-reference)):.8g}")
    delta = np.abs(actual[np.isfinite(actual)] - reference[np.isfinite(reference)])
    return float(delta.max(initial=0))


def on_cpu_mamba(cfg):
    from mambapy.mamba import MambaBlock, MambaConfig
    from neuralhydrology.modelzoo.sequential_forecast import SequentialForecast
    from neuralhydrology.modelzoo.sf_cores import ResidualCore, mamba_expand

    # Dựng phần nhúng và head qua cùng lớp của fork; thay duy nhất lõi Mamba CUDA.
    shell = copy.deepcopy(cfg)
    shell.update_config({"sf_core": "s4d", "sf_layers": 0})
    model = SequentialForecast(shell)

    def block(width):
        config = MambaConfig(d_model=width, n_layers=1, d_state=cfg.sf_mamba_d_state,
                             d_conv=cfg.sf_mamba_d_conv, use_cuda=False)
        config.d_inner = int(width * mamba_expand(cfg, width))
        return MambaBlock(config)

    model.core = ResidualCore(cfg, model.hindcast_embedding_net.output_size, block)
    model.cfg = cfg
    return model


class DemoForecaster:
    def __init__(self, run, raw_dir, ids, areas, scratch, device="cuda:0", *, n_params=None):
        import torch
        from neuralhydrology.datautils.utils import load_scaler
        from neuralhydrology.datasetzoo.lamah import (
            LamaH, load_lamah_attributes, load_lamah_discharge, load_lamah_forcing, _normalize_discharge,
        )
        from neuralhydrology.modelzoo import get_model
        from neuralhydrology.utils.config import Config

        class DemoLamaH(LamaH):
            def _load_basin_data(self, basin):
                frame = super()._load_basin_data(basin)
                for name in ("qmean", "qmax"):
                    flow = load_lamah_discharge(self.cfg.data_dir, basin, temporal_resolution="1D",
                                                normalize_discharge=False, target_name=name)[name]
                    flow = _normalize_discharge(ser=flow, area=self._basin_area.loc[basin],
                                                temporal_resolution="1D").reindex(frame.index)
                    # C2/H/D đọc lại CSV với định dạng %.17g khi dựng q*_mmd.
                    frame[f"{name}_mmd"] = pd.read_csv(StringIO(flow.to_csv(index=False, header=["q"],
                                                                         float_format="%.17g")))["q"].to_numpy()
                return frame

        self.torch, self.dataset_type = torch, DemoLamaH
        self.load_forcing = load_lamah_forcing
        self.device = torch.device(device)
        ids = tuple(map(str, ids))
        areas = np.asarray(areas)
        if (len(ids) != len(areas) or len(set(ids)) != len(ids)
                or not np.isfinite(areas).all() or np.any(areas <= 0)):
            raise ValueError("ID hoặc diện tích lưu vực không hợp lệ.")
        self.areas = dict(zip(ids, areas))
        attrs = load_lamah_attributes(Path(raw_dir), sub_dataset="lamah_a").loc[list(ids), "area_gov"]
        if not np.array_equal(attrs.to_numpy(), areas):
            raise ValueError("Diện tích lưu vực của dữ liệu demo khác A3.")
        run = Path(run)
        cfg = Config(run / "config.yml")
        self.length, self.horizon = cfg.seq_length, cfg.forecast_seq_length
        self.mode = cfg.forecast_input_mode
        checks = (
            ("model", cfg.model, cfg.model == "sequential_forecast"),
            ("dataset", cfg.dataset, cfg.dataset == "lamah_a"),
            ("seq_length/forecast_seq_length", (self.length, self.horizon),
             (self.length, self.horizon) in ((365, 1), (371, 7), (730, 1))),
            ("predict_last_n", cfg.predict_last_n, cfg.predict_last_n == self.horizon),
            ("forecast_overlap", cfg.forecast_overlap, cfg.forecast_overlap in (None, 0)),
            ("target_variables", cfg.target_variables, cfg.target_variables == ["qmax"]),
            ("clip_targets_to_zero", cfg.clip_targets_to_zero, "qmax" in cfg.clip_targets_to_zero),
            ("additional_feature_files", cfg.additional_feature_files, not cfg.additional_feature_files),
            ("use_basin_id_encoding", cfg.use_basin_id_encoding, not cfg.use_basin_id_encoding),
        )
        invalid = [f"{key}={value!r}" for key, value, valid in checks if not valid]
        if invalid:
            raise ValueError(f"{run.name}: config ngoài giao thức C2/H/D ({', '.join(invalid)}).")
        self.scratch = TemporaryDirectory(dir=scratch)
        basins = Path(self.scratch.name) / "basins.txt"
        basins.write_text("\n".join(ids) + "\n", encoding="utf-8")
        cfg.update_config(dict(data_dir=Path(raw_dir), train_basin_file=basins,
                               validation_basin_file=basins, test_basin_file=basins, device=device))
        self.cfg = cfg
        self.dates = pd.date_range(self.cfg.test_start_date, self.cfg.test_end_date)
        self.forecast_features = tuple(self.cfg.forecast_inputs_flattened)
        self.accepts_hres = set(self.forecast_features) == set(HRES) and self.mode in ("all", "first_day")
        self.scaler = load_scaler(run)
        self.centers = self.scaler["xarray_feature_center"]
        self.scales = self.scaler["xarray_feature_scale"]
        if self.device.type == "cpu" and self.cfg.sf_core == "mamba":
            self.model = on_cpu_mamba(self.cfg)
        else:
            self.model = get_model(self.cfg)
        self.model.to(self.device)
        weights = torch.load(run / "best_model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        if n_params is not None and sum(p.numel() for p in self.model.parameters() if p.requires_grad) != n_params:
            raise ValueError("Số tham số khác bản bàn giao.")
        self.model.eval()
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def close(self):
        self.scratch.cleanup()

    def _dataset(self, basin):
        if basin not in self.areas:
            raise ValueError(f"Lưu vực ngoài tập đánh giá: {basin}")
        if basin not in self.cache:
            dataset = self.dataset_type(self.cfg, is_train=False, period="test", basin=basin,
                                        scaler=copy.deepcopy(self.scaler))
            if dataset.is_train or dataset._compute_scaler:
                raise ValueError("Demo cần dùng scaler train đã lưu.")
            self.cache[basin] = dataset
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(basin)
        return self.cache[basin]

    def _forward(self, batch):
        dates = np.asarray(batch["date"]).astype("datetime64[D]")
        if (dates.ndim != 2 or dates.shape[1] != self.length or not np.all(np.diff(dates, axis=1) == np.timedelta64(1, "D"))
                or any(v.shape[1] != self.length-self.horizon for v in batch["x_d_hindcast"].values())
                or any(v.shape[1] != self.horizon for v in batch["x_d_forecast"].values())):
            raise ValueError("Lịch hoặc độ dài các pha khác config đã lưu.")
        with self.torch.inference_mode():
            predictions = self.model(self.model.pre_model_hook(batch, is_train=False))["y_hat"]
        values = predictions[:, -self.horizon:, 0].cpu().numpy()
        obs = batch["y"][:, -self.horizon:, 0].cpu().numpy()
        # Giữ dtype của phép hoàn nguyên trong Tester (scaler có thể lưu float64).
        scale = self.scales[["qmax"]].to_array().values
        center = self.centers[["qmax"]].to_array().values
        return dates[:, -self.horizon:], values * scale + center, obs * scale + center

    def predict_basin(self, basin, batch_size=None):
        from neuralhydrology.datautils.gpu_loader import GpuBatchLoader

        with self.lock:
            loader = GpuBatchLoader(self._dataset(str(basin)), batch_size=batch_size or self.cfg.batch_size,
                                    device=self.device, shuffle=False, with_dates=True)
            return tuple(np.concatenate(chunks) for chunks in zip(*(self._forward(batch) for batch in loader)))

    def _window(self, basin, origin_date):
        day = pd.Timestamp(origin_date)
        last = day + pd.Timedelta(days=self.horizon - 1)
        if day not in self.dates or last not in self.dates:
            raise ValueError("Cửa sổ dự báo phải nằm trong kỳ test.")
        dataset = self._dataset(basin)
        daily = dataset._dates[basin]["1D"]
        last_date = last.to_datetime64()
        # H bắt đầu mẫu khi đủ bảy ngày đích trong kỳ test. Chỉ mục lookup khác chỉ số lịch.
        def end_date(index):
            _, positions = dataset.lookup_table[index]
            return daily[positions[0]]

        index = bisect_left(range(len(dataset)), last_date, key=end_date)
        if index == len(dataset) or end_date(index) != last_date:
            raise ValueError("Ngày này không có cửa sổ hoàn chỉnh trong lookup.")
        batch = dataset.collate_fn([dataset[index]])
        for key, value in batch.items():
            if self.torch.is_tensor(value):
                batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                batch[key] = {k: v.to(self.device) for k, v in value.items()}
        if np.asarray(batch["date"])[0, -self.horizon] != day.to_datetime64():
            raise ValueError("Ngày đầu dự báo khác lịch lookup.")
        return batch

    def _raw_hres(self, basin, day):
        if not self.accepts_hres:
            raise ValueError("Run này không nhận ECMWF HRES ngày đầu.")
        frame = self.load_forcing(self.cfg.data_dir, basin, self.cfg.dataset, temporal_resolution="1D")
        return frame.loc[pd.Timestamp(day), list(HRES)].to_dict()

    def _apply_hres(self, inputs, raw, hres, rain_multiplier, temperature_delta):
        values = dict(hres or {})
        if set(values) - set(HRES) or not all(np.isfinite(v) for v in values.values()):
            raise ValueError("ECMWF HRES thay thế cần đúng tên biến và giá trị hữu hạn.")
        if not np.isfinite(rain_multiplier) or rain_multiplier < 0 or not np.isfinite(temperature_delta):
            raise ValueError("Hệ số mưa cần hữu hạn và không âm. Mức cộng nhiệt cần hữu hạn.")
        for name, apply, change in (("ECMWF_tp", rain_multiplier != 1, lambda v: v * rain_multiplier),
                                    ("ECMWF_t2m", temperature_delta != 0, lambda v: v + temperature_delta)):
            if apply:
                old = values.get(name, raw[name])
                if not np.isfinite(old):
                    raise ValueError(f"Thiếu {name} ngày đầu.")
                values[name] = change(old)
        for name, value in values.items():
            scale = self.scales[name].item()
            if not np.isfinite(scale) or scale <= 0 or (name == "ECMWF_tp" and value < 0):
                raise ValueError(f"Giá trị ECMWF HRES ngoài miền: {name}")
            # LamaH chuyển giá trị về float32 trước khi dùng scaler.
            value = np.float32(value)
            normalized = (value - self.centers[name].to_numpy()) / self.scales[name].to_numpy()
            if not np.isfinite(normalized):
                raise ValueError(f"Giá trị ECMWF HRES ngoài miền: {name}")
            inputs[name] = inputs[name].clone()
            inputs[name][:, 0, :] = float(normalized)

    def predict(self, basin, origin_date, hres=None, *, rain_multiplier=1.0, temperature_delta=0.0):
        # hres là các giá trị ngày đầu trong đơn vị của dữ liệu phát hành.
        basin = str(basin)
        with self.lock:
            batch = self._window(basin, origin_date)
            if hres is not None or rain_multiplier != 1.0 or temperature_delta != 0.0:
                original = self._raw_hres(basin, origin_date)
                self._apply_hres(batch["x_d_forecast"], original, hres, rain_multiplier, temperature_delta)
            days, raw, _ = self._forward(batch)
        if not np.isfinite(raw).all():
            raise ValueError("Dự báo có giá trị không hữu hạn.")
        clipped = np.maximum(raw[0], 0)
        return [dict(basin_id=basin, origin_date=str(days[0, 0]), target_date=str(day), lead=k,
                     qmax_raw_mm_day=float(raw[0, k-1]), qmax_mm_day=float(clipped[k-1]),
                     qmax_m3_s=float(qmax_m3_s(clipped[k-1], self.areas[basin])))
                for k, day in enumerate(days[0], 1)]


def align_basin_lead(dates, raw, obs, reference_dates, lead):
    days = dates[:, lead-1]
    reference_dates = np.asarray(reference_dates, dtype="datetime64[D]")
    index = np.searchsorted(days, reference_dates)
    if np.any(index >= len(days)) or not np.array_equal(days[index], reference_dates):
        raise ValueError(f"Lịch dự báo hạn {lead} khác hợp đồng.")
    return raw[index, lead-1], obs[index, lead-1]
