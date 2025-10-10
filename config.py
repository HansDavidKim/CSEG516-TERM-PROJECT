from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional


class EnvConfig(Mapping[str, str]):
    """Lightweight loader for values defined in a .env file with dict-style access."""

    def __init__(
        self,
        env_file: Path | str = None,
        *,
        encoding: str = "utf-8",
        override_with_os_environ: bool = True,
    ) -> None:
        self._path = self._resolve_env_path(env_file)
        self._encoding = encoding
        self._values: Dict[str, str] = {}

        if self._path.exists():
            self._values.update(self._read_env_file(self._path, encoding))

        if override_with_os_environ:
            # OS environment variables take precedence over .env definitions.
            self._values.update(os.environ)

    @staticmethod
    def _resolve_env_path(env_file: Optional[Path | str]) -> Path:
        if env_file is None:
            return Path(__file__).resolve().parent / ".env"
        env_path = Path(env_file)
        return env_path if env_path.is_absolute() else (Path(__file__).resolve().parent / env_path)

    @staticmethod
    def _read_env_file(path: Path, encoding: str) -> Dict[str, str]:
        values: Dict[str, str] = {}
        for raw_line in path.read_text(encoding=encoding).splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            value = EnvConfig._strip_quotes(raw_value.strip())
            values[key] = value
        return values

    @staticmethod
    def _strip_quotes(value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]
        return value

    def __getitem__(self, key: str) -> str:
        try:
            return self._values[key]
        except KeyError as exc:
            raise KeyError(f"Environment variable '{key}' is not configured.") from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._values.get(key, default)

    def as_dict(self) -> Dict[str, str]:
        return dict(self._values)


env_config = EnvConfig()

