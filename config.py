from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional


class Config(Mapping[str, Any]):
    """Generic mapping wrapper for configuration sections."""

    def __init__(
        self,
        data: Mapping[str, Any],
        *,
        section: str,
        source: Optional[Path] = None,
    ) -> None:
        self._data: Dict[str, Any] = dict(data)
        self._section = section
        self._source = source

    @property
    def section(self) -> str:
        return self._section

    @property
    def source(self) -> Optional[Path]:
        return self._source

    def __getitem__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:
            raise KeyError(f"'{key}' is not defined in [{self.section}] configuration.") from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    @classmethod
    def from_toml(
        cls,
        section: str,
        *,
        path: Path | str | None = None,
    ) -> "Config":
        resolved_path = _resolve_config_path(path)
        data = _load_toml(resolved_path)
        try:
            section_data = data[section]
        except KeyError as exc:
            raise KeyError(f"Section '{section}' not found in config file '{resolved_path}'.") from exc
        return cls(section_data, section=section, source=resolved_path)

    def __repr__(self) -> str:
        location = f" from {self._source}" if self._source else ""
        return f"{self.__class__.__name__}(section='{self.section}'{location})"


class ClassifierConfig(Config):
    """Configuration wrapper for classifier settings."""


class DatasetConfig(Config):
    """Configuration wrapper for dataset settings."""


class EnvConfig(Config):
    """Configuration loaded from a .env file with dict-style access."""

    def __init__(
        self,
        env_file: Path | str | None = None,
        *,
        encoding: str = "utf-8",
        override_with_os_environ: bool = True,
    ) -> None:
        self._path = self._resolve_env_path(env_file)
        self._encoding = encoding
        values: Dict[str, str] = {}

        if self._path.exists():
            values.update(self._read_env_file(self._path, encoding))

        if override_with_os_environ:
            values.update(os.environ)

        super().__init__(values, section="env", source=self._path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def encoding(self) -> str:
        return self._encoding

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

def _resolve_config_path(path: Path | str | None) -> Path:
    if path is None:
        return _DEFAULT_CONFIG_PATH
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parent / resolved
    return resolved.resolve()


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.toml"
_CONFIG_CACHE: Dict[Path, Dict[str, Any]] = {}


def _load_toml(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    if path in _CONFIG_CACHE:
        return _CONFIG_CACHE[path]
    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' does not exist.")
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    _CONFIG_CACHE[path] = data
    return data


env_config = EnvConfig()
classifier_config = ClassifierConfig.from_toml("classifier")
dataset_config = DatasetConfig.from_toml("dataset")
