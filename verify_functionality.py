#!/usr/bin/env python3
"""
Run verification steps from README "## Verify that everything works".
Output is written to verification.log (or path given as first argument).
Exit code 0 on success, non-zero on failure.
"""
import os
import sys
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "verification.log"
if len(sys.argv) > 1:
    LOG_PATH = Path(sys.argv[1])

ORCH = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ORCH))
from verify_subprocess_streaming import log_message, log_section, stream_run  # noqa: E402

TIMEOUT_DEFAULT = int(os.environ.get("VERIFY_TIMEOUT", "3600"))


def resolve_models_dir(repo_root: Path) -> Path:
    """Use shared ML_Production/models when repos are siblings; else ./models."""
    local_models = repo_root / "models"
    shared_candidates = [
        repo_root.parent / "ML_Production" / "models",
        repo_root.parent / "ML_Production__main" / "models",
    ]
    for shared in shared_candidates:
        if shared.is_dir():
            if not local_models.exists():
                local_models.symlink_to(os.path.relpath(shared, repo_root))
            return shared
    return local_models


def main():
    repo_root = Path(__file__).resolve().parent

    with open(LOG_PATH, "w", encoding="utf-8"):
        pass

    env_geotiff = {**os.environ, "GTIFF_SRS_SOURCE": "EPSG", "PYTHONUNBUFFERED": "1"}

    log_section(LOG_PATH, "=== CUDA check ===")
    ret = stream_run(
        f'{sys.executable} -c "import torch; print(\'CUDA available:\', torch.cuda.is_available()); print(\'Device:\', torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'N/A\')"',
        LOG_PATH,
        cwd=repo_root,
        env=env_geotiff,
        timeout=30,
    )
    log_message(LOG_PATH, f"Exit code: {ret}")
    if ret != 0:
        return ret
    try:
        import torch

        if not torch.cuda.is_available():
            log_message(LOG_PATH, "CUDA_UNAVAILABLE: CUDA required but not available")
            return 1
    except Exception as exc:
        log_message(LOG_PATH, f"CUDA_UNAVAILABLE: {type(exc).__name__}: {exc}")
        return 1

    models_dir = resolve_models_dir(repo_root)
    log_section(LOG_PATH, f"=== models directory: {models_dir} ===")

    token_file = repo_root / "my_hugging_face_token.txt"
    if token_file.exists():
        log_section(LOG_PATH, "=== download_upload_models_hf.py (download) ===")
        ret = stream_run(
            f"{sys.executable} src/ML_geo_production/download_upload_models_hf.py --download --token_file {token_file} --file_path {models_dir}/",
            LOG_PATH,
            cwd=repo_root,
            env=env_geotiff,
            timeout=TIMEOUT_DEFAULT,
        )
        log_message(LOG_PATH, f"Exit code: {ret}")
        if ret != 0:
            return ret
    else:
        log_section(LOG_PATH, "=== Skipping model download (my_hugging_face_token.txt not found) ===")

    log_section(LOG_PATH, "=== process_images.py (save_probs_preds_and_change_detection.json) ===")
    ret = stream_run(
        f"{sys.executable} src/ML_geo_production/process_images.py --json config_files/save_probs_preds_and_change_detection.json",
        LOG_PATH,
        cwd=repo_root,
        env=env_geotiff,
        timeout=TIMEOUT_DEFAULT,
    )
    log_message(LOG_PATH, f"Exit code: {ret}")
    return ret


if __name__ == "__main__":
    sys.exit(main())
