"""Validation tests for threads/store search `limit` (issue #471)."""

from typing import Any

import pytest
from pydantic import ValidationError

from aegra_api.models.search_limit import (
    DEFAULT_SEARCH_LIMIT,
    enforce_search_limit,
    search_limit_json_schema_extra,
)
from aegra_api.models.store import StoreSearchRequest
from aegra_api.models.threads import ThreadSearchRequest
from aegra_api.settings import settings

_DEFAULT_CAP: int = 1000
_INVALID_LIMITS: tuple[Any, ...] = ("abc", [], {}, 20.5)


@pytest.fixture(autouse=True)
def _pin_default_search_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", _DEFAULT_CAP)


def _integer_limit_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return the integer branch of an optional limit field schema."""
    for option in schema.get("anyOf", []):
        if isinstance(option, dict) and option.get("type") == "integer":
            return option
    return schema


def _limit_error(exc: ValidationError) -> dict[str, Any]:
    errors = [item for item in exc.errors() if item["loc"][-1] == "limit"]
    assert len(errors) == 1
    return errors[0]


def _store_request(**kwargs: Any) -> StoreSearchRequest:
    return StoreSearchRequest(namespace_prefix=["notes"], **kwargs)


class TestEnforceSearchLimit:
    """Shared helper used by both search request models."""

    def test_passes_none_through(self) -> None:
        assert enforce_search_limit(None) is None

    def test_passes_value_at_cap(self) -> None:
        assert enforce_search_limit(_DEFAULT_CAP) == _DEFAULT_CAP

    def test_schema_extra_sets_maximum_on_integer_anyof_branch(self) -> None:
        schema: dict[str, Any] = {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]}
        search_limit_json_schema_extra(schema)
        assert schema["anyOf"][0]["maximum"] == _DEFAULT_CAP
        assert "maximum" not in schema

    def test_schema_extra_sets_maximum_without_anyof(self) -> None:
        schema: dict[str, Any] = {"type": "integer"}
        search_limit_json_schema_extra(schema)
        assert schema["maximum"] == _DEFAULT_CAP

    def test_schema_extra_tracks_live_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 50)
        schema: dict[str, Any] = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
        search_limit_json_schema_extra(schema)
        assert schema["anyOf"][0]["maximum"] == 50


class TestThreadSearchRequestLimit:
    """ThreadSearchRequest.limit must accept LangGraph SDK page sizes."""

    @pytest.mark.parametrize("limit", [1, 20, 101, 500, 1000])
    def test_accepts_limits_within_default_cap(self, limit: int) -> None:
        request = ThreadSearchRequest(limit=limit)
        assert request.limit == limit

    def test_accepts_limit_equal_to_cap(self) -> None:
        request = ThreadSearchRequest(limit=settings.app.MAX_SEARCH_LIMIT)
        assert request.limit == settings.app.MAX_SEARCH_LIMIT

    def test_accepts_explicit_none_limit(self) -> None:
        request = ThreadSearchRequest(limit=None)
        assert request.limit is None

    @pytest.mark.parametrize("limit", [0, -1])
    def test_rejects_zero_and_negative_limits(self, limit: int) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ThreadSearchRequest(limit=limit)
        err = _limit_error(exc_info.value)
        assert err["type"] == "greater_than_equal"

    def test_rejects_limit_above_cap(self) -> None:
        cap = settings.app.MAX_SEARCH_LIMIT
        with pytest.raises(ValidationError) as exc_info:
            ThreadSearchRequest(limit=cap + 1)
        err = _limit_error(exc_info.value)
        assert err["type"] == "less_than_equal"
        assert err["ctx"] == {"le": cap}
        assert str(cap) in err["msg"]

    @pytest.mark.parametrize("limit", _INVALID_LIMITS)
    def test_rejects_non_integer_limits(self, limit: Any) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ThreadSearchRequest.model_validate({"limit": limit})
        err = _limit_error(exc_info.value)
        assert err["type"] in {"int_parsing", "int_type", "int_from_float"}

    def test_default_limit_is_unchanged(self) -> None:
        request = ThreadSearchRequest()
        assert request.limit == DEFAULT_SEARCH_LIMIT
        assert DEFAULT_SEARCH_LIMIT == 20
        assert DEFAULT_SEARCH_LIMIT != settings.app.MAX_SEARCH_LIMIT

    def test_honors_configured_max_search_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 50)
        request = ThreadSearchRequest(limit=50)
        assert request.limit == 50
        with pytest.raises(ValidationError) as exc_info:
            ThreadSearchRequest(limit=51)
        err = _limit_error(exc_info.value)
        assert err["type"] == "less_than_equal"
        assert err["ctx"] == {"le": 50}

    def test_openapi_schema_advertises_configured_cap(self) -> None:
        schema = ThreadSearchRequest.model_json_schema()["properties"]["limit"]
        integer_schema = _integer_limit_schema(schema)
        assert integer_schema["minimum"] == 1
        assert integer_schema["maximum"] == settings.app.MAX_SEARCH_LIMIT
        assert schema["default"] == DEFAULT_SEARCH_LIMIT

    def test_openapi_schema_tracks_live_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 50)
        schema = ThreadSearchRequest.model_json_schema()["properties"]["limit"]
        assert _integer_limit_schema(schema)["maximum"] == 50


class TestStoreSearchRequestLimit:
    """StoreSearchRequest.limit shares the same cap as thread search."""

    @pytest.mark.parametrize("limit", [1, 20, 101, 500, 1000])
    def test_accepts_limits_within_default_cap(self, limit: int) -> None:
        request = _store_request(limit=limit)
        assert request.limit == limit

    def test_accepts_limit_equal_to_cap(self) -> None:
        request = _store_request(limit=settings.app.MAX_SEARCH_LIMIT)
        assert request.limit == settings.app.MAX_SEARCH_LIMIT

    def test_accepts_explicit_none_limit(self) -> None:
        request = _store_request(limit=None)
        assert request.limit is None

    @pytest.mark.parametrize("limit", [0, -1])
    def test_rejects_zero_and_negative_limits(self, limit: int) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _store_request(limit=limit)
        err = _limit_error(exc_info.value)
        assert err["type"] == "greater_than_equal"

    def test_rejects_limit_above_cap(self) -> None:
        cap = settings.app.MAX_SEARCH_LIMIT
        with pytest.raises(ValidationError) as exc_info:
            _store_request(limit=cap + 1)
        err = _limit_error(exc_info.value)
        assert err["type"] == "less_than_equal"
        assert err["ctx"] == {"le": cap}
        assert str(cap) in err["msg"]

    @pytest.mark.parametrize("limit", _INVALID_LIMITS)
    def test_rejects_non_integer_limits(self, limit: Any) -> None:
        with pytest.raises(ValidationError) as exc_info:
            StoreSearchRequest.model_validate({"namespace_prefix": ["notes"], "limit": limit})
        err = _limit_error(exc_info.value)
        assert err["type"] in {"int_parsing", "int_type", "int_from_float"}

    def test_default_limit_is_unchanged(self) -> None:
        request = _store_request()
        assert request.limit == DEFAULT_SEARCH_LIMIT
        assert DEFAULT_SEARCH_LIMIT == 20
        assert DEFAULT_SEARCH_LIMIT != settings.app.MAX_SEARCH_LIMIT

    def test_honors_configured_max_search_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 50)
        request = _store_request(limit=50)
        assert request.limit == 50
        with pytest.raises(ValidationError) as exc_info:
            _store_request(limit=51)
        err = _limit_error(exc_info.value)
        assert err["type"] == "less_than_equal"
        assert err["ctx"] == {"le": 50}

    def test_openapi_schema_advertises_configured_cap(self) -> None:
        schema = StoreSearchRequest.model_json_schema()["properties"]["limit"]
        integer_schema = _integer_limit_schema(schema)
        assert integer_schema["minimum"] == 1
        assert integer_schema["maximum"] == settings.app.MAX_SEARCH_LIMIT
        assert schema["default"] == DEFAULT_SEARCH_LIMIT

    def test_openapi_schema_tracks_live_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 50)
        schema = StoreSearchRequest.model_json_schema()["properties"]["limit"]
        assert _integer_limit_schema(schema)["maximum"] == 50
