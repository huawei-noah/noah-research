"""Install NLP assets declared in MindMemOS text processing config."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from mindmemos.config import DEFAULT_MINDMEMOS_CONFIG_ROOT, default_config_path, get_config, init_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if (value is None or value == "") and name.startswith("MINDMEMOS_"):
        suffix = name.removeprefix("MINDMEMOS_")
        value = os.getenv(f"MINDMEM_{suffix}") or os.getenv(f"MEMOS_{suffix}")
    return value if value not in (None, "") else default


def init_app_config() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    config_path = env_str("MINDMEMOS_CONFIG_PATH")
    if config_path:
        init_config(config_path=config_path)
        return

    config_name = env_str("MINDMEMOS_CONFIG_NAME", "dev")
    candidate = default_config_path(config_name)
    fallback = DEFAULT_MINDMEMOS_CONFIG_ROOT / "dev.example.yaml"
    init_config(config_name=config_name, config_path=candidate if candidate.exists() else fallback)


def model_is_available(model_name: str) -> bool:
    import spacy

    try:
        spacy.load(model_name)
    except OSError:
        return False
    return True


def install_model(model_name: str) -> None:
    if model_is_available(model_name):
        print(f"spaCy model already installed: {model_name}")
        return
    print(f"Installing spaCy model: {model_name}")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])


def main() -> None:
    init_app_config()
    cfg = get_config().algo_config.text_processing
    for model_name in (cfg.spacy_en_model, cfg.spacy_zh_model):
        install_model(model_name)


if __name__ == "__main__":
    main()
