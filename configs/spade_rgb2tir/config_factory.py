from pathlib import Path

import yaml

from .registry import MODELS


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_refs(d, root):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                ref = v[2:-1]
                d[k] = root[ref]
            else:
                resolve_refs(v, root)

    elif isinstance(d, list):
        for i, v in enumerate(d):
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                ref = v[2:-1]
                d[i] = root[ref]
            else:
                resolve_refs(v, root)


def resolve_refs_and_cleanup(root):
    refs = set()

    def collect_refs(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    refs.add(v[2:-1])
                else:
                    collect_refs(v)
        elif isinstance(d, list):
            for v in d:
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    refs.add(v[2:-1])
                else:
                    collect_refs(v)

    collect_refs(root)

    # подставляем ссылки относительно всего root
    resolve_refs(root, root)

    # удаляем cfg-блоки, которые были использованы как ссылки
    for r in refs:
        if r in root:
            del root[r]


def build_config(model_name):
    base = Path(__file__).parent
    cfg = {}

    paths = MODELS[model_name]
    for section, rel_path in paths.items():
        section_dir = base / rel_path
        cfg.setdefault(section, {})

        for file in section_dir.iterdir():
            if file.is_file() and file.suffix == ".yaml":
                sub = load_cfg(file.as_posix())
                key = next(iter(sub))  # ← единственный верхний ключ
                cfg[section][key] = sub[key]

                resolve_refs_and_cleanup(cfg[section][key])
    return cfg
