from collections.abc import Mapping
from typing import TypeAlias, TypedDict

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | Mapping[str, "JsonValue"]


class WorkflowReport(TypedDict):
    schema_version: int
    selection: Mapping[str, JsonValue]
    evaluation: Mapping[str, JsonValue]
    calibration: Mapping[str, JsonValue]
    quantization: Mapping[str, JsonValue]
    artifact: Mapping[str, JsonValue]
    reload: Mapping[str, JsonValue]


class ArtifactReport(TypedDict):
    schema_version: int
    mode: str
    evaluation: Mapping[str, JsonValue]
    precision: Mapping[str, str]
    profile_hash: str
    profile_unchanged: bool
