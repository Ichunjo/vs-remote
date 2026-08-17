from __future__ import annotations

import pytest

from vsremote import (
    RemoteAuthenticationError,
    RemoteCommandError,
    RemoteExecutionError,
    RemoteNotFoundError,
    RemotePayloadError,
    RemotePermissionError,
)
from vsremote.protocol import ResponseEnvelope, StatusCode


def test_status_code_exception_association() -> None:
    assert StatusCode.OK.exception_cls is None
    assert StatusCode.ERROR.exception_cls is RemoteExecutionError
    assert StatusCode.NOT_FOUND.exception_cls is RemoteNotFoundError
    assert StatusCode.INVALID_COMMAND.exception_cls is RemoteCommandError
    assert StatusCode.INVALID_PAYLOAD.exception_cls is RemotePayloadError
    assert StatusCode.UNAUTHORIZED.exception_cls is RemoteAuthenticationError
    assert StatusCode.PERMISSION_DENIED.exception_cls is RemotePermissionError


def test_status_code_raise_for_status() -> None:
    # Error status codes raise their corresponding type
    with pytest.raises(RemoteExecutionError) as exc_info:
        StatusCode.ERROR.raise_for_status("Execution failed", payload="err_details")
    assert exc_info.value.status == StatusCode.ERROR
    assert exc_info.value.payload == "err_details"
    assert "Execution failed" in str(exc_info.value)

    with pytest.raises(RemoteNotFoundError) as exc_info_nf:
        StatusCode.NOT_FOUND.raise_for_status("Output 5 not found")
    assert exc_info_nf.value.status == StatusCode.NOT_FOUND

    # Also catchable via KeyError
    with pytest.raises(KeyError):
        StatusCode.NOT_FOUND.raise_for_status("Output 5 not found")

    with pytest.raises(RemotePermissionError):
        StatusCode.PERMISSION_DENIED.raise_for_status("Denied")

    with pytest.raises(PermissionError):
        StatusCode.PERMISSION_DENIED.raise_for_status("Denied")

    with pytest.raises(RemoteAuthenticationError):
        StatusCode.UNAUTHORIZED.raise_for_status("Auth required")


def test_response_envelope_raise_for_status() -> None:
    ok_resp = ResponseEnvelope(status=StatusCode.OK, payload={"res": 1})
    ok_resp.raise_for_status("Should not raise")

    err_resp = ResponseEnvelope(status=StatusCode.ERROR, payload="SyntaxError on line 5")
    with pytest.raises(RemoteExecutionError) as exc_info:
        err_resp.raise_for_status("Failed to load script")
    assert "Failed to load script: SyntaxError on line 5" in str(exc_info.value)
    assert exc_info.value.status == StatusCode.ERROR
    assert exc_info.value.payload == "SyntaxError on line 5"
