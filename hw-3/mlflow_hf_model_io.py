#!/usr/bin/env python3
"""MLflow ⇄ HuggingFace: загрузка весов в MLflow, скачивание и удаление run.

Сценарии:
1) upload: скачать модель с Hugging Face и залогировать папку модели в MLflow artifacts.
2) download: скачать артефакт модели из MLflow по run_id обратно в локальную папку.
3) delete-run: удалить run (самый надёжный способ удалить артефакты модели).

Пример:
  python mlflow_hf_model_io.py upload \
    --hf-repo Qwen/Qwen2.5-1.5B-Instruct \
    --artifact-path hf_models/qwen2.5-1.5b-instruct

  python mlflow_hf_model_io.py download \
    --run-id <RUN_ID> \
    --artifact-path hf_models/qwen2.5-1.5b-instruct \
    --out-dir ./downloaded_model

  python mlflow_hf_model_io.py delete-run --run-id <RUN_ID>

Примечания:
- Для приватных моделей/лимитов Hugging Face можно задать переменную окружения HF_TOKEN.
- MLflow tracking URI берётся из config.yaml (mlflow.tracking_uri) либо из MLFLOW_TRACKING_URI.
- В MLflow UI артефакты иногда не видны из-за кеша/не того backend; проверка после upload
  делает list_artifacts по run_id и подтверждает, что файлы реально записались.
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
    # Локальная копия логики, чтобы скрипт был самодостаточным.
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
    client = MlflowClient()
    items = client.list_artifacts(run_id, path=artifact_path)
    if not items:
        raise RuntimeError(
            f"Артефакт не найден в run_id={run_id} по artifact_path='{artifact_path}'. "
            f"Проверь tracking URI (MLFLOW_TRACKING_URI / config.yaml) и права на artifact store."
        )

    print(f"Artifact '{artifact_path}' exists. Preview (up to 20):")
    for it in items[:20]:
        print(f"  - {it.path}")


def upload_hf_model_to_mlflow(
    hf_repo: str,
    artifact_path: str,
    experiment_name: str | None = None,
    revision: str | None = None,
    local_dir: str | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> UploadResult:
    """Скачивает модель с HF и логирует папку в MLflow artifacts.

    artifact_path будет выглядеть как папка в артефактах, например:
      hf_models/qwen2.5-3b
    """

    _set_mlflow_from_config(experiment_name)

    # Куда будем скачивать снапшот HF
    if local_dir is None:
        local_dir_path = Path(".cache") / "hf_models" / hf_repo.replace("/", "__")
    else:
        local_dir_path = Path(local_dir)

    local_dir_path.mkdir(parents=True, exist_ok=True)

    # Используем HF_TOKEN, если он задан
    token = os.environ.get("HF_TOKEN")

    # snapshot_download кладёт полную структуру репозитория (config, tokenizer, веса и т.д.)
    snapshot_path = snapshot_download(
        repo_id=hf_repo,
        revision=revision,
        local_dir=str(local_dir_path),
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

        # Логируем всю папку модели как артефакт
        mlflow.log_artifacts(snapshot_path, artifact_path=artifact_path)
        _assert_artifact_present(run.info.run_id, artifact_path)

        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        print(f"Uploaded to run_id={run.info.run_id}")
        print(f"Artifact URI: {artifact_uri}")

        return UploadResult(run_id=run.info.run_id, artifact_uri=artifact_uri)


def download_model_from_mlflow(
    run_id: str,
    artifact_path: str,
    out_dir: str,
    experiment_name: str | None = None,
) -> str:
    """Скачивает модель-папку из MLflow artifacts в out_dir."""

    _set_mlflow_from_config(experiment_name)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # mlflow.artifacts.download_artifacts вернёт путь до скачанного артефакта
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        dst_path=str(out_path),
    )
    print(f"Downloaded to: {local_path}")
    return local_path


def delete_run(run_id: str, experiment_name: str | None = None) -> None:
    _set_mlflow_from_config(experiment_name)
    MlflowClient().delete_run(run_id)
    print(f"Run marked as deleted: {run_id}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Upload/download HuggingFace model weights to/from MLflow")
    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="Download HF model and log to MLflow artifacts")
    up.add_argument("--hf-repo", required=True, help="Hugging Face repo id, e.g. Qwen/Qwen2.5-3B-Instruct")
    up.add_argument("--artifact-path", required=True, help="MLflow artifact path, e.g. hf_models/qwen2.5-3b")
    up.add_argument("--experiment-name", default=None, help="Override MLflow experiment name")
    up.add_argument("--revision", default=None, help="HF revision (branch/tag/commit)")
    up.add_argument("--local-dir", default=None, help="Where to store HF snapshot before logging")
    up.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Pattern(s) to include when downloading (can be repeated).",
    )
    up.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="Pattern(s) to exclude when downloading (can be repeated).",
    )

    dl = sub.add_parser("download", help="Download model folder from MLflow artifacts")
    dl.add_argument("--run-id", required=True, help="MLflow run id")
    dl.add_argument("--artifact-path", required=True, help="Artifact path used during upload")
    dl.add_argument("--out-dir", required=True, help="Output directory")
    dl.add_argument("--experiment-name", default=None, help="Override MLflow experiment name")

    dr = sub.add_parser("delete-run", help="Delete MLflow run (removes/hides artifacts with the run)")
    dr.add_argument("--run-id", required=True, help="MLflow run id")
    dr.add_argument("--experiment-name", default=None, help="Override MLflow experiment name")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.cmd == "upload":
        upload_hf_model_to_mlflow(
            hf_repo=args.hf_repo,
            artifact_path=args.artifact_path,
            experiment_name=args.experiment_name,
            revision=args.revision,
            local_dir=args.local_dir,
            allow_patterns=args.allow_pattern,
            ignore_patterns=args.ignore_pattern,
        )
        return

    if args.cmd == "download":
        download_model_from_mlflow(
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
