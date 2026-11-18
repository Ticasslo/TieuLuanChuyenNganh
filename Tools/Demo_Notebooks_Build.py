import ast
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "Workspace/07_Demo"
EVALUATION = ROOT / "Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.py"
SOURCES = (
    "primary_event_tolerance_days", "basin_id", "Forecast", "check_same_context",
    "metrics", "unique_file", "sha256", "read_context", "read_contract", "digest_string", "contained_file",
    "verified_manifest", "c2_records", "c2_forecast",
)
CONSTANTS = (
    "METRICS", "LOG_FLOOR_MM_DAY", "TEST_START", "TEST_END", "FREEZE_C", "C2_NOTEBOOK", "C2_CONFIGS",
    "C2_SEEDS", "SUPPLEMENTARY_EVENT_TOLERANCE_DAYS", "D_COMPARE_NOTEBOOK", "D_CONFIGS",
)
NOTEBOOKS = {
    "d1": ("01_D1_LamaHCEExt_Inference", "T4 cu128, Internet On", "24 checkpoint C2, A3, C2 Compare và wheel Mamba."),
    "d2": ("02_D2_LamaHCEExt_CPU", "CPU, Internet On", "D1 đã Save, tám checkpoint đại diện C2, C2 Compare và A3."),
}


def extract_contract_reader():
    source = EVALUATION.read_text(encoding="utf-8-sig")
    lines = source.splitlines(keepends=True)
    nodes = ast.parse(source).body
    selected = []
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in SOURCES:
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            selected.append("".join(lines[start-1:node.end_lineno]))
        if isinstance(node, ast.Assign) and any(
                isinstance(x, ast.Name) and x.id in CONSTANTS
                for target in node.targets for x in ast.walk(target)):
            selected.append(ast.get_source_segment(source, node) + "\n")
    return "from dataclasses import dataclass, field\nimport itertools\nimport math\nimport re\n\n" + "\n".join(selected)


def as_notebook(source, filename):
    cells = []
    for index, part in enumerate(re.split(r"(?m)(?=^# %%)", source)):
        if not part.strip():
            continue
        ident = hashlib.sha256(f"{filename}:{index}".encode()).hexdigest()[:12]
        if part.startswith("# %% [markdown]"):
            text = [re.sub(r"^# ?", "", l) for l in part.splitlines(keepends=True)[1:]]
            cells.append(dict(cell_type="markdown", id=ident, metadata={}, source=text))
        else:
            ast.parse(part)
            cells.append(dict(cell_type="code", id=ident, metadata={}, source=part.splitlines(keepends=True),
                              execution_count=None, outputs=[]))
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    if ast.dump(ast.parse(code)) != ast.dump(ast.parse(source)):
        raise ValueError(f"Mã notebook khác tệp Python: {filename}")
    return dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                      "language_info": {"name": "python"}}, nbformat=4, nbformat_minor=5)


def build():
    runtime = DEMO / "DemoRuntime"
    common = "# %% Phần 1 - Nguồn, manifest và hợp đồng C2\n"
    common += (runtime/"audit.py").read_text(encoding="utf-8").rstrip() + "\n\n"
    common += "import numpy as np\nimport pandas as pd\n\n" + extract_contract_reader() + "\n\n"
    common += "# %% Phần 2 - Hàm dự báo dùng config, checkpoint, scaler đã lưu\n"
    common += (runtime/"forecast.py").read_text(encoding="utf-8").rstrip() + "\n\n"
    for key, (name, environment, inputs) in NOTEBOOKS.items():
        src = (f"# %% [markdown]\n# # {name}\n#\n# {environment}. {inputs}\n"
               f"# Hướng dẫn Input và nghiệm thu: {name}.md.\n\n" + common
               + "# %% Phần 3 - Đối chiếu và lưu kết quả\n"
               + (runtime/f"{key}.py").read_text(encoding="utf-8").rstrip() + "\n")
        folder = DEMO / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder/f"{name}.py").write_text(src, encoding="utf-8", newline="\n")
        nb = as_notebook(src,name)
        (folder/f"{name}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1)+"\n",encoding="utf-8",newline="\n")
        print(name, len(nb["cells"]))


if __name__ == "__main__":
    build()
