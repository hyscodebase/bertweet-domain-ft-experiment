import json
from pathlib import Path

import yaml


def test_colab_notebook_references_gpu_workflow():
    notebook_path = Path("notebooks/bertweet_domain_ft_colab_gpu.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "nvidia-smi" in source
    assert "REPO_URL = 'https://github.com/hyscodebase/bertweet-domain-ft-experiment.git'" in source
    assert "pip install -r requirements.txt" in source
    assert "configs/experiment_colab.yaml" in source
    assert "python -m src.hf_data" in source
    assert "python -m src.run_baseline" in source


def test_colab_config_is_gpu_friendly():
    config = yaml.safe_load(Path("configs/experiment_colab.yaml").read_text(encoding="utf-8"))
    assert config["training"]["fp16"] is True
    assert config["training"]["batch_size"] <= 16
    assert config["hf_data"]["max_train_examples"] <= 1024
