"""Project state for Teich Studio: config.yaml and prompts.jsonl IO."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import yaml

from ..config import Config, PromptInput

PROMPT_FIELDS = ("prompt", "system", "github_repo", "follow_up_prompts")
JSONL_PROMPT_SUFFIXES = {".jsonl", ".ndjson"}
DIRECT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
CHAT_API_COMPATIBILITY_ERROR = (
    "Chat sessions require an OpenAI-compatible API. Use OpenRouter, OpenAI, "
    "or a compatible custom base URL instead of Anthropic's direct API."
)

DEFAULT_CONFIG_DATA: dict[str, Any] = {
    "agent": {"provider": "pi"},
    "model": {
        "model": "deepseek/deepseek-v4-flash",
        "approval_policy": "never",
        "sandbox": "danger-full-access",
        "reasoning_effort": "medium",
    },
    "api": {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": None,
        "wire_api": "responses",
    },
    "prompts_file": "prompts.jsonl",
    "prompts": [],
    "output": {
        "traces_dir": "./output",
        "sandbox_dir": "./sandbox",
        "failures_dir": "./failures",
        "pretty_name": "My Agent Traces",
    },
    "publish": {"repo_id": None, "hf_token": None, "private": False},
    "max_concurrency": 1,
    "timeout_seconds": 600,
    "developer_instructions": None,
}


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_base_url(value: str | None) -> str:
    return (value or "").strip().rstrip("/").lower()


def validate_chat_api_compatibility(config: Config) -> None:
    if config.get_agent_provider() != "chat":
        return
    api_provider = config.api.provider.strip().lower()
    if api_provider == "anthropic" or _normalize_base_url(config.get_base_url()) == DIRECT_ANTHROPIC_BASE_URL:
        raise ValueError(CHAT_API_COMPATIBILITY_ERROR)


class ProjectState:
    """Reads and writes the studio project's config.yaml and prompts file."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._lock = threading.RLock()

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def read_config_data(self) -> dict[str, Any]:
        with self._lock:
            if not self.config_path.exists():
                return dict(DEFAULT_CONFIG_DATA)
            with self.config_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise ValueError("config.yaml must contain a YAML mapping")
            return data

    def write_config_data(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Merge updates into the existing config and persist to config.yaml."""
        with self._lock:
            existing = self.read_config_data() if self.config_path.exists() else dict(DEFAULT_CONFIG_DATA)
            merged = _deep_merge(existing, updates)
            # Validate before writing so a bad save never corrupts the file.
            self._validate_config_data(merged)
            text = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True, default_flow_style=False)
            self.config_path.write_text(text, encoding="utf-8")
            return merged

    def _validate_config_data(self, data: dict[str, Any]) -> None:
        payload = dict(data)
        # prompts_file may not exist yet while the user is still setting up.
        prompts_file = payload.get("prompts_file")
        if isinstance(prompts_file, str) and prompts_file.strip():
            resolved = self.resolve_path(prompts_file)
            payload = {**payload, "prompts_file": str(resolved) if resolved.exists() else None}
        elif prompts_file is not None and not isinstance(prompts_file, str):
            payload = {**payload, "prompts_file": None}
        validate_chat_api_compatibility(Config(**payload))

    def load_config(self) -> Config:
        """Load a validated Config the same way the CLI does (env overrides included)."""
        with self._lock:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            cfg = Config.from_yaml(self.config_path)
            validate_chat_api_compatibility(cfg)
        cfg.output.traces_dir = self.resolve_path(cfg.output.traces_dir)
        cfg.output.sandbox_dir = self.resolve_path(cfg.output.sandbox_dir)
        cfg.output.failures_dir = self.resolve_path(cfg.output.failures_dir)
        return cfg

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.root / candidate).resolve()

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def prompts_path(self) -> Path:
        data = self.read_config_data()
        prompts_file = data.get("prompts_file")
        if isinstance(prompts_file, str) and prompts_file.strip():
            return self.resolve_path(prompts_file.strip())
        return self.root / "prompts.jsonl"

    def read_prompts(self) -> list[dict[str, Any]]:
        path = self.prompts_path()
        if not path.exists():
            return []
        return [self._row_from_prompt_input(item) for item in Config._load_prompt_inputs_from_file(path)]

    @staticmethod
    def _row_from_prompt_input(prompt_input: PromptInput) -> dict[str, Any]:
        row = prompt_input.model_dump(exclude_none=True)
        if not row.get("follow_up_prompts"):
            row.pop("follow_up_prompts", None)
        return ProjectState._normalize_prompt_row(row)

    @staticmethod
    def _normalize_prompt_row(row: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key in PROMPT_FIELDS:
            value = row.get(key)
            if key == "follow_up_prompts":
                if isinstance(value, list):
                    follow_ups = [str(item).strip() for item in value if str(item).strip()]
                    if follow_ups:
                        normalized[key] = follow_ups
            elif isinstance(value, str) and value.strip():
                normalized[key] = value
        return normalized

    def write_prompts(self, rows: list[dict[str, Any]]) -> Path:
        configured_path = self.prompts_path()
        path = self._writable_prompts_path(configured_path)
        normalized = []
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"Prompt entry {index} must be an object")
            clean = self._normalize_prompt_row(row)
            if not clean.get("prompt"):
                raise ValueError(f"Prompt entry {index} is missing prompt text")
            normalized.append(clean)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("w", encoding="utf-8") as handle:
                for row in normalized:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            # Make sure config points at the prompts file we just wrote.
            data = self.read_config_data()
            if self.config_path.exists() and (path != configured_path or not data.get("prompts_file")):
                self.write_config_data({"prompts_file": self._config_path_value(path)})
        return path

    def _writable_prompts_path(self, path: Path) -> Path:
        if path.suffix.lower() in JSONL_PROMPT_SUFFIXES:
            return path
        if path.suffix:
            return path.with_suffix(".jsonl")
        return path.with_name(f"{path.name}.jsonl")

    def _config_path_value(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def import_prompts_text(
        self, text: str, *, replace: bool, filename: str | None = None
    ) -> list[dict[str, Any]]:
        """Parse uploaded JSONL text and merge or replace the current prompts."""
        if filename:
            suffix = Path(filename).suffix.lower()
            if suffix not in JSONL_PROMPT_SUFFIXES:
                raise ValueError("Prompt uploads must be JSONL or NDJSON files")
        rows: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL on line {line_number}: {exc.msg}") from exc
            if isinstance(row, str):
                row = {"prompt": row}
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} must be a JSON object or string")
            clean = self._normalize_prompt_row(row)
            if not clean.get("prompt"):
                raise ValueError(f"Line {line_number} is missing prompt text")
            rows.append(clean)
        if not rows:
            raise ValueError("No prompts found in the uploaded file")
        merged = rows if replace else [*self.read_prompts(), *rows]
        self.write_prompts(merged)
        return merged

    # ------------------------------------------------------------------
    # Project scaffolding & traces
    # ------------------------------------------------------------------

    def ensure_initialized(self) -> None:
        """Create config.yaml (with defaults) and prompts.jsonl if missing."""
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            if not self.config_path.exists():
                from ..cli import CONFIG_TEMPLATE

                self.config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
            prompts = self.root / "prompts.jsonl"
            if not prompts.exists():
                prompts.write_text("", encoding="utf-8")

    def list_traces(self) -> list[dict[str, Any]]:
        data = self.read_config_data()
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        traces_dir = self.resolve_path(output.get("traces_dir") or "./output")
        failures_dir = self.resolve_path(output.get("failures_dir") or "./failures")
        if not traces_dir.exists():
            return []
        traces: list[dict[str, Any]] = []
        for path in sorted(traces_dir.rglob("*.jsonl")):
            if not path.is_file():
                continue
            relative_parts = path.relative_to(traces_dir).parts
            if any(part in {"partials", "failures"} for part in relative_parts):
                continue
            try:
                if path.resolve().is_relative_to(failures_dir.resolve()):
                    continue
            except (OSError, ValueError):
                pass
            stat = path.stat()
            traces.append(
                {
                    "name": path.relative_to(traces_dir).as_posix(),
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                }
            )
        traces.sort(key=lambda item: item["modified_at"], reverse=True)
        return traces

    def trace_file(self, name: str) -> Path:
        data = self.read_config_data()
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        traces_dir = self.resolve_path(output.get("traces_dir") or "./output")
        candidate = (traces_dir / name).resolve()
        if not candidate.is_relative_to(traces_dir.resolve()):
            raise ValueError("Invalid trace name")
        return candidate
