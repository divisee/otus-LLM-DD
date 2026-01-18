#!/usr/bin/env python3
"""Сохранение / скачивание / удаление модели (как артефакта) в MLflow.

Это максимально совместимый способ "залить веса" в MLflow:
- мы логируем папку (snapshot) с файлами модели (HF weights/config/tokenizer) в artifacts.
- потом её можно скачать обратно по run_id + artifact_path.
- а "удаление" делаем через удаление run (для локального file-store/sqlite это реально чистит артефакты).

Почему не делаем прямое "delete artifact path":
- В MLflow нет стабильного публичного API для удаления отдельных артефактов внутри run.
- Надёжный путь: удалить весь run (или использовать файловое хранилище и rm -rf руками).

Команды:
  upload-hf   - скачать repo с Hugging Face и залогировать как артефакт
  upload-dir  - залогировать уже существующую локальную папку
  download    - скачать артефакт
  delete-run  - удалить run (и его артефакты)

Примеры:
  python mlflow_model_artifacts_manage.py upload-hf \
    --hf-repo Qwen/Qwen2.5-3B-Instruct \
    --artifact-path hf_models/qwen2.5-3b-instruct

  python mlflow_model_artifacts_manage.py download \
    --run-id <RUN_ID> \
    --artifact-path hf_models/qwen2.5-3b-instruct \
    --out-dir ./downloaded

  python mlflow_model_artifacts_manage.py delete-run --run-id <RUN_ID>

Переменные окружения:
- HF_TOKEN (опционально) для скачивания с Hugging Face
- MLFLOW_TRACKING_URI (опционально) переопределить tracking URI
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
from huggingface_hub import snapshot_download
from mlflow.tracking import MlflowClient

from config_loader import Config


@dataclass(frozen=True)
class UploadResult:
    run_id: str
    artifact_uri: str


def _resolve_tracking_uri(raw_uri: str) -> str:
    if raw_uri.startswith("sqlite:///"):
        path = raw_uri.replace("sqlite:///", "", 1)
        if not os.path.isabs(path):
            abs_path = Path(__file__).parent.joinpath(path).resolve()
            return f"sqlite:///{abs_path}"
    return raw_uri


def _set_mlflow_from_config(experiment_name: str | None) -> None:
    cfg = Config()
    tracking_uri = _resolve_tracking_uri(cfg.mlflow_tracking_uri)

    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    if experiment_name is None:
        experiment_name = cfg.mlflow_experiment_name
    mlflow.set_experiment(experiment_name)


def _assert_artifact_present(run_id: str, artifact_path: str) -> None:
    """Проверяет, что артефакт-папка существует в MLflow run.

    Важно: в MLflow UI артефакты могут не показываться из-за кеша/фильтров,
    но список через API — источник правды.
    """

    client = MlflowClient()

    # root listing (иногда полезно для отладки)
    _ = client.list_artifacts(run_id, path="")

    items = client.list_artifacts(run_id, path=artifact_path)
    if not items:
        raise RuntimeError(
            f"Артефакт не найден в run_id={run_id} по artifact_path='{artifact_path}'. "
            f"Проверьте tracking URI и права на artifact store."
        )

    # Печатаем первые элементы (файлы/директории)
    preview = [it.path for it in items[:20]]
    print(f"Artifact '{artifact_path}' exists. Preview (up to 20):")
    for p in preview:
        print(f"  - {p}")


def upload_local_dir_to_mlflow(
    local_dir: str,
    artifact_path: str,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> UploadResult:
    _set_mlflow_from_config(experiment_name)

    local_path = Path(local_dir)
    if not local_path.exists() or not local_path.is_dir():
        raise FileNotFoundError(f"local_dir не существует или не папка: {local_dir}")

    with mlflow.start_run(run_name=run_name or f"upload_dir:{local_path.name}") as run:
        mlflow.log_param("artifact_path", artifact_path)
        mlflow.log_param("local_dir", str(local_path))

        mlflow.log_artifacts(str(local_path), artifact_path=artifact_path)
        _assert_artifact_present(run.info.run_id, artifact_path)

        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        print(f"Uploaded dir to run_id={run.info.run_id}")
        print(f"Artifact URI: {artifact_uri}")
        return UploadResult(run_id=run.info.run_id, artifact_uri=artifact_uri)


def upload_hf_repo_to_mlflow(
    hf_repo: str,
    artifact_path: str,
    experiment_name: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> UploadResult:
    _set_mlflow_from_config(experiment_name)

    token = os.environ.get("HF_TOKEN")

    if cache_dir is None:
        cache_path = Path(".cache") / "hf_models" / hf_repo.replace("/", "__")
    else:
        cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=hf_repo,
        revision=revision,
        local_dir=str(cache_path),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    with mlflow.start_run(run_name=f"upload_hf_model:{hf_repo}") as run:
        mlflow.log_param("hf_repo", hf_repo)
        if revision:
            mlflow.log_param("hf_revision", revision)
        mlflow.log_param("artifact_path", artifact_path)
        mlflow.log_param("snapshot_path", snapshot_path)

        mlflow.log_artifacts(snapshot_path, artifact_path=artifact_path)
        _assert_artifact_present(run.info.run_id, artifact_path)

        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        print(f"Uploaded HF repo to run_id={run.info.run_id}")
        print(f"Artifact URI: {artifact_uri}")
        return UploadResult(run_id=run.info.run_id, artifact_uri=artifact_uri)


def download_artifact_from_mlflow(
    run_id: str,
    artifact_path: str,
    out_dir: str,
    experiment_name: str | None = None,
) -> str:
    _set_mlflow_from_config(experiment_name)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        dst_path=str(out_path),
    )
    print(f"Downloaded to: {local_path}")
    return local_path


def delete_run(run_id: str, experiment_name: str | None = None) -> None:
    _set_mlflow_from_config(experiment_name)

    client = MlflowClient()
    # В MLflow это помечает run как deleted. Для file backend обычно артефакты удаляются/скрываются.
    # Если нужно "жёстко" — можно дополнительно вызвать mlflow gc на backend или удалить папку artifacts.
    client.delete_run(run_id)
    print(f"Run marked as deleted: {run_id}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Manage model artifacts in MLflow (upload/download/delete-run)")
    sub = p.add_subparsers(dest="cmd", required=True)

    uphf = sub.add_parser("upload-hf", help="Download HF repo and log as MLflow artifact")
    uphf.add_argument("--hf-repo", required=True)
    uphf.add_argument("--artifact-path", required=True)
    uphf.add_argument("--experiment-name", default=None)
    uphf.add_argument("--revision", default=None)
    uphf.add_argument("--cache-dir", default=None)
    uphf.add_argument("--allow-pattern", action="append", default=None)
    uphf.add_argument("--ignore-pattern", action="append", default=None)

    updir = sub.add_parser("upload-dir", help="Log existing local directory as MLflow artifact")
    updir.add_argument("--local-dir", required=True)
    updir.add_argument("--artifact-path", required=True)
    updir.add_argument("--experiment-name", default=None)
    updir.add_argument("--run-name", default=None)

    dl = sub.add_parser("download", help="Download artifact folder from MLflow")
    dl.add_argument("--run-id", required=True)
    dl.add_argument("--artifact-path", required=True)
    dl.add_argument("--out-dir", required=True)
    dl.add_argument("--experiment-name", default=None)

    dr = sub.add_parser("delete-run", help="Delete MLflow run (artifacts are removed/hidden with the run)")
    dr.add_argument("--run-id", required=True)
    dr.add_argument("--experiment-name", default=None)

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.cmd == "upload-hf":
        upload_hf_repo_to_mlflow(
            hf_repo=args.hf_repo,
            artifact_path=args.artifact_path,
            experiment_name=args.experiment_name,
            revision=args.revision,
            cache_dir=args.cache_dir,
            allow_patterns=args.allow_pattern,
            ignore_patterns=args.ignore_pattern,
        )
        return

    if args.cmd == "upload-dir":
        upload_local_dir_to_mlflow(
            local_dir=args.local_dir,
            artifact_path=args.artifact_path,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
        )
        return

    if args.cmd == "download":
        download_artifact_from_mlflow(
            run_id=args.run_id,
            artifact_path=args.artifact_path,
            out_dir=args.out_dir,
            experiment_name=args.experiment_name,
        )
        return

    if args.cmd == "delete-run":
        delete_run(run_id=args.run_id, experiment_name=args.experiment_name)
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
