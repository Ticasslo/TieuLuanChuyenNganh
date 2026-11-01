# Chuẩn bị bản sao kiểm tích hợp B2 + cores, không checkout/merge repo nguồn.
import io, json, pathlib, subprocess, tarfile, shutil, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT / 'NeuralHydrology'
B2_COMMIT = 'cac96c54379853af4f3f430fcd5ec0df55e5fb30'
CORE_COMMIT = 'fed53dca9815df963a5fa362558c448c769a6499'
(ROOT / '.cache').mkdir(exist_ok=True)
DEST = pathlib.Path(tempfile.mkdtemp(prefix='b2_core_check_', dir=ROOT / '.cache'))
def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])
with tarfile.open(fileobj=io.BytesIO(git('archive', B2_COMMIT))) as archive:
    archive.extractall(DEST, filter='data')
files = git('diff', '--name-only', 'f229e02..' + CORE_COMMIT).decode().splitlines()
for name in files:
    dest = DEST / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if name == 'neuralhydrology/utils/config.py':
        paths = []
        for ref in (B2_COMMIT, 'f229e02', CORE_COMMIT):
            path = DEST / (ref.replace('/', '_') + '_config.txt')
            path.write_bytes(git('show', ref + ':' + name))
            paths.append(str(path))
        merged = subprocess.run(['git', 'merge-file', '-p', *paths], capture_output=True, check=True)
        dest.write_bytes(merged.stdout)
    else:
        dest.write_bytes(git('show', CORE_COMMIT + ':' + name))
shutil.copy2(REPO / 'test/test_b3_masking.py', DEST / 'test/test_b3_masking.py')
(DEST / 'test/test_b2_core_integration.py').write_text('''import copy
import pytest
import torch
from neuralhydrology.modelzoo import get_model
from neuralhydrology.modelzoo.sequential_forecast import SequentialForecast
from neuralhydrology.datasetzoo.lamah import LamaH
from neuralhydrology.datautils.gpu_loader import GpuBatchLoader
from test.test_b2_source_inputs import b2_config

@pytest.mark.parametrize("core", ["lstm", "gru", "transformer", "s4d"])
def test_b2_grouped_missing_inputs_on_new_core(b2_config, core):
    cfg = b2_config(model="sequential_forecast", sf_core=core, sf_layers=1, sf_transformer_nheads=2,
                    nan_step_probability=.1, nan_sequence_probability=.12)
    ds = LamaH(cfg, is_train=True, period="train", scaler={})
    batch = next(iter(GpuBatchLoader(ds, batch_size=4, device="cpu", shuffle=False)))
    model = get_model(cfg)
    pred = model(batch)["y_hat"]
    assert pred.isfinite().all()
    pred.square().mean().backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    model.eval()
    batch = next(iter(GpuBatchLoader(ds, batch_size=4, device="cpu", shuffle=False)))
    if core == "lstm":
        old_cfg = copy.deepcopy(cfg)
        old_cfg.update_config(dict(model="sequential_forecast_lstm"))
        old = get_model(old_cfg).eval()
        model.load_state_dict({k.replace("lstm.", "core.rnn.", 1): v for k, v in old.state_dict().items()})
        torch.testing.assert_close(model(batch)["y_hat"], old(batch)["y_hat"], rtol=0, atol=0)
''', encoding='utf-8')
print(json.dumps({'copy': str(DEST), 'B2': git('rev-parse', B2_COMMIT).decode().strip(),
                  'cores': git('rev-parse', CORE_COMMIT).decode().strip(), 'files_from_cores': len(files)}))

print("Chạy từ thư mục copy bằng Python hiện hành: python -m pytest test/test_b2_source_inputs.py test/test_b3_masking.py test/test_b2_core_integration.py test/test_sequential_forecast.py -k 'not training_and_evaluation_run' --basetemp=.pytest_integration -p no:cacheprovider")
