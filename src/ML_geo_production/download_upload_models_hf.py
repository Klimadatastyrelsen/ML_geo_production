#!/usr/bin/env python
"""Upload or download .pth models to/from a Hugging Face repo."""

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


def resolve_token(token_file: Path | None) -> str:
    if token_file is None:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent.parent
        token_file = repo_root.parent / "laz-superpoint_transformer" / "hftoken_write.txt"
    token_file = token_file.resolve()
    if not token_file.is_file():
        print(f"Error: token file not found: {token_file}", file=sys.stderr)
        sys.exit(1)
    token = token_file.read_text().strip()
    if not token:
        print("Error: token file is empty", file=sys.stderr)
        sys.exit(1)
    return token


def upload(api: HfApi, file_paths: list[str], repo_id: str, dry_run: bool):
    resolved = []
    for pattern in file_paths:
        resolved.extend(glob.glob(pattern))
    files = sorted({Path(f) for f in resolved if f.endswith(".pth") and Path(f).is_file()})
    if not files:
        print(f"Error: no .pth files matched: {file_paths}", file=sys.stderr)
        sys.exit(1)

    existing = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))

    failed = 0
    for f in files:
        name = f.name
        if name in existing:
            print(f"Skipped (already in repo): {name}")
            continue
        if dry_run:
            print(f"Would upload: {f} -> {repo_id}/{name}")
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Uploaded: {name}")
        except Exception as e:
            print(f"Failed to upload {name}: {e}", file=sys.stderr)
            failed += 1

    if failed:
        sys.exit(1)


def ensure_dest_dir(dest: Path) -> Path:
    """Return writable models directory; follow symlinks, create if missing."""
    if dest.is_symlink():
        dest = dest.resolve()
    elif dest.exists() and not dest.is_dir():
        raise FileExistsError(f"models path exists but is not a directory: {dest}")
    elif not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)
    return dest


def list_files_for_download(api: HfApi, repo_id: str) -> tuple[HfApi, list[str]]:
    """List repo files; if the token is rejected, retry anonymously (public repos)."""
    try:
        return api, api.list_repo_files(repo_id=repo_id, repo_type="model")
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", None)
        if not api.token or status != 401:
            raise
        print(
            "Warning: Hugging Face token was rejected (401, e.g. expired); retrying download anonymously",
            file=sys.stderr,
        )
        anonymous = HfApi(token=False)
        return anonymous, anonymous.list_repo_files(repo_id=repo_id, repo_type="model")


def load_config_models(config_path: Path) -> tuple[str | None, list[str]]:
    """Return (hf_repo_id, model file names) from a process_images-style JSON config."""
    parsed_json = json.loads(config_path.read_text())
    names = [Path(p).name for p in parsed_json.get("saved_models") or []]
    if not names:
        print(f"Error: no saved_models in {config_path}", file=sys.stderr)
        sys.exit(1)
    return parsed_json.get("hf_repo_id"), names


def download(api: HfApi, dest_dir: str, repo_id: str, dry_run: bool, only: list[str] | None = None):
    dest = ensure_dest_dir(Path(dest_dir))

    api, all_files = list_files_for_download(api, repo_id)
    pth_files = sorted(f for f in all_files if f.endswith(".pth"))
    if only is not None:
        missing = sorted(set(only) - set(pth_files))
        if missing:
            print(f"Error: not in repo {repo_id}: {missing}", file=sys.stderr)
            sys.exit(1)
        pth_files = sorted(only)
    if not pth_files:
        print(f"No .pth files found in repo {repo_id}")
        return

    failed = 0
    for name in pth_files:
        target = dest / name
        if target.exists():
            print(f"Skipped (already exists): {target}")
            continue
        if dry_run:
            print(f"Would download: {repo_id}/{name} -> {target}")
            continue
        try:
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=name,
                repo_type="model",
                token=api.token,
            )
            shutil.copy2(cached, target)
            print(f"Downloaded: {name} -> {target}")
        except Exception as e:
            print(f"Failed to download {name}: {e}", file=sys.stderr)
            failed += 1

    if failed:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload or download .pth models to/from Hugging Face.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--upload", action="store_true", help="Upload .pth file(s) to the repo")
    mode.add_argument("--download", action="store_true",
                      help="Download .pth files from the repo (all of them, or only those in --config)")

    parser.add_argument("--file_path", required=True, nargs="+",
                        help="Upload: path(s) or glob pattern for .pth files. Download: destination directory.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Download only the saved_models of this JSON config, from its hf_repo_id")
    parser.add_argument("--repo_id", default=None,
                        help="Hugging Face repo id (default: the config's hf_repo_id, else Danish-Agency-for-Cimate-Data/KDS_buildings)")
    parser.add_argument("--token_file", type=Path, default=None,
                        help="Path to HF token file (required for upload; optional for download of public repos)")
    parser.add_argument("--dry_run", action="store_true", help="Print actions without executing")

    args = parser.parse_args()

    only = None
    if args.config is not None:
        if not args.download:
            parser.error("--config is only used with --download")
        config_repo_id, only = load_config_models(args.config)
        args.repo_id = args.repo_id or config_repo_id
    args.repo_id = args.repo_id or "Danish-Agency-for-Cimate-Data/KDS_buildings"

    if args.dry_run:
        token = None
    elif args.download and args.token_file is None:
        token = False
    else:
        token = resolve_token(args.token_file)
    api = HfApi(token=token)

    if args.upload:
        upload(api, args.file_path, args.repo_id, args.dry_run)
    else:
        if len(args.file_path) != 1:
            print("Error: --download requires exactly one directory path", file=sys.stderr)
            sys.exit(1)
        download(api, args.file_path[0], args.repo_id, args.dry_run, only)


if __name__ == "__main__":
    main()
