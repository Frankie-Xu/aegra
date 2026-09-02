"""Shared search pagination cap for protocol search request models.

Read at validation time so MAX_SEARCH_LIMIT can change without reimporting models.
"""

from typing import Any

from pydantic_core import PydanticKnownError

from aegra_api.settings import settings

DEFAULT_SEARCH_LIMIT: int = 20


def enforce_search_limit(value: int | None) -> int | None:
    """Reject page sizes above the configured server cap."""
    if value is None:
        return value
    cap = settings.app.MAX_SEARCH_LIMIT
    if value > cap:
        # Same type/msg/ctx as Field(le=cap) so 422 payloads stay Pydantic-shaped.
        raise PydanticKnownError("less_than_equal", {"le": cap})
    return value


def search_limit_json_schema_extra(schema: dict[str, Any]) -> None:
    """Advertise the live cap on the integer branch of the OpenAPI schema."""
    cap = settings.app.MAX_SEARCH_LIMIT
    for option in schema.get("anyOf", []):
        if isinstance(option, dict) and option.get("type") == "integer":
            option["maximum"] = cap
            return
    schema["maximum"] = cap
