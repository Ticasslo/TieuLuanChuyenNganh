from contextlib import redirect_stdout, suppress
from functools import lru_cache
from hashlib import file_digest
from importlib.metadata import distribution
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Lock, Thread
import traceback

FREEZE_H = "6f68f471bec255f67d446acd6c6556358ff9bfea"
# CPU H checkpoints are checked against the GPU inference exports in mm/day.
H_PRED_ATOL = 5e-4
H_CHECKPOINT_FILES = {
    "best_model.pt": "checkpoint_sha256",
    "config.yml": "config_sha256",
    "train_data/train_data_scaler.yml": "scaler_sha256",
}


class HWorkerError(RuntimeError):
    pass


class HWorker:
    """Run the Hybrid checkpoints under freeze-H in a separate Python process."""

    def __init__(self, python, bundle, raw, ids, areas, models):
        self.command = [str(Path(python).resolve()), "-u", str(Path(__file__).resolve()),
                        str(Path(bundle).resolve()), str(Path(raw).resolve())]
        self.startup = dict(ids=list(ids), areas=[float(area) for area in areas],
                            models={item["run"]: {key: item[key] for key in (*H_CHECKPOINT_FILES.values(), "n_params")}
                                    for item in models})
        self.process = None
        self.lock = Lock()

    @staticmethod
    def _read_replies(stream, replies):
        try:
            for line in stream:
                replies.put(line)
        finally:
            replies.put(None)

    def _exchange(self, request):
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.replies.get(timeout=60)
        if line is None:
            raise HWorkerError("Tiến trình suy luận H đã dừng.")
        reply = json.loads(line)
        if reply.get("kind") == "input":
            raise ValueError(reply["error"])
        if "error" in reply:
            raise HWorkerError(reply["error"])
        return reply["result"]

    def predict(self, run, basin, date, rain_multiplier=1.0, temperature_delta=0.0):
        with self.lock:
            try:
                if self.process is None:
                    self.process = subprocess.Popen(
                        self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8",
                        env={**os.environ, "PYTHONUTF8": "1"},
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    self.replies = Queue()
                    Thread(target=self._read_replies, args=(self.process.stdout, self.replies), daemon=True).start()
                    self._exchange(self.startup)
                return self._exchange(dict(run=run, basin=basin, date=date,
                                           rain_multiplier=rain_multiplier, temperature_delta=temperature_delta))
            except (OSError, Empty, json.JSONDecodeError, HWorkerError) as exc:
                self._close()
                if isinstance(exc, HWorkerError):
                    raise
                raise HWorkerError("Không nhận được kết quả từ tiến trình CPU của H.") from exc

    def _close(self):
        process, self.process = self.process, None
        if process is None:
            return
        with suppress(OSError):
            process.stdin.close()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()

    def close(self):
        with self.lock:
            self._close()


def serve(bundle, raw):
    output = sys.stdout

    def send(**reply):
        print(json.dumps(reply, allow_nan=False), file=output, flush=True)

    with redirect_stdout(sys.stderr):
        catalog = json.loads(sys.stdin.readline())
        install = json.loads(distribution("neuralhydrology").read_text("direct_url.json") or "{}")
        if install.get("vcs_info", {}).get("commit_id") != FREEZE_H:
            send(error=f"Môi trường suy luận H cần đúng commit {FREEZE_H}.")
            return
        import torch
        from forecast import DemoForecaster

        threads = max(1, int(os.environ.get("DEMO_THREADS", min(4, os.cpu_count() or 1))))
        torch.set_num_threads(threads)
        send(result=True)

        @lru_cache(maxsize=2)
        def predictor(run):
            metadata = catalog["models"][run]
            checkpoint = bundle / "checkpoints" / run
            for name, key in H_CHECKPOINT_FILES.items():
                with (checkpoint / name).open("rb") as f:
                    if file_digest(f, "sha256").hexdigest() != metadata[key]:
                        raise HWorkerError(f"Checkpoint H không khớp mã băm: {run}, {name}.")
            model = DemoForecaster(checkpoint, raw, catalog["ids"], catalog["areas"], None,
                                   device="cpu", n_params=metadata["n_params"])
            if model.mode != "first_day" or model.horizon != 7 or not model.accepts_hres:
                model.close()
                raise HWorkerError(f"Checkpoint H khác giao thức lai: {run}.")
            return model

        for line in sys.stdin:
            request = json.loads(line)
            try:
                model = predictor(request["run"])
                basin, date = request["basin"], request["date"]
                rain, temp = request["rain_multiplier"], request["temperature_delta"]
                baseline = model.predict(basin, date)
                changed = (baseline if (rain, temp) == (1, 0) else
                           model.predict(basin, date, rain_multiplier=rain, temperature_delta=temp))
                send(result=dict(baseline=baseline, changed=changed))
            except ValueError as exc:
                send(error=str(exc), kind="input")
            except HWorkerError as exc:
                send(error=str(exc))
            except Exception:
                traceback.print_exc()
                send(error="Tiến trình CPU của H gặp lỗi khi suy luận.")


if __name__ == "__main__":
    serve(Path(sys.argv[1]), Path(sys.argv[2]))
