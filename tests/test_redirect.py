from __future__ import annotations

import io
import logging
import sys

from vsremote.protocol import RemoteLogRecord, StreamEvent, StreamOutputEvent
from vsremote.server.redirect import StreamRedirector, capture_streams


def test_stream_redirector_protocol() -> None:
    events = list[StreamEvent]()
    redir = StreamRedirector("stdout", events.append)

    assert isinstance(redir, io.TextIOBase)
    assert redir.writable() is True
    assert redir.readable() is False
    assert redir.seekable() is False
    assert isinstance(redir.isatty(), bool)
    assert redir.closed is False

    # fileno delegates if supported or raises UnsupportedOperation
    try:
        fn = redir.fileno()
        assert isinstance(fn, int)
    except io.UnsupportedOperation:
        pass


def test_stream_redirector_context_manager_and_writes() -> None:
    events = list[StreamEvent]()
    original_stdout = sys.stdout

    with StreamRedirector("stdout", events.append) as redir:
        assert sys.stdout is redir
        redir.write("hello ")
        redir.writelines(["world\n", "second line\n"])
        redir.flush()

    assert sys.stdout is original_stdout
    outputs = [e for e in events if isinstance(e, StreamOutputEvent)]
    assert len(outputs) == 3
    assert outputs[0].stream == "stdout"
    assert outputs[0].text == "hello "
    assert outputs[1].text == "world\n"
    assert outputs[2].text == "second line\n"


def test_stream_redirection_suppressed_during_logging() -> None:
    events = list[StreamEvent]()
    test_logger = logging.getLogger("test_suppression")
    test_logger.propagate = False

    # Attach a standard StreamHandler writing to sys.stderr
    stream_handler = logging.StreamHandler(sys.stderr)
    test_logger.addHandler(stream_handler)

    try:
        with StreamRedirector("stderr", events.append):
            # Direct write to stderr should be captured
            sys.stderr.write("direct stderr output\n")

            # Logging call should be suppressed from StreamOutputEvent
            test_logger.warning("this is a logged warning")

            # Another direct write
            sys.stderr.write("second direct stderr\n")
    finally:
        test_logger.removeHandler(stream_handler)

    stderr_events = [e for e in events if isinstance(e, StreamOutputEvent) and e.stream == "stderr"]
    # Only direct writes should be captured, not the logger output
    texts = [e.text for e in stderr_events]
    assert "direct stderr output\n" in texts
    assert "second direct stderr\n" in texts
    assert not any("this is a logged warning" in t for t in texts)


def test_capture_streams_context_manager() -> None:
    events = list[StreamEvent]()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with capture_streams(events.append):
        assert sys.stdout is not original_stdout
        assert sys.stderr is not original_stderr

        print("captured print output")
        logging.getLogger().info("captured root log")

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr

    log_events = [e for e in events if isinstance(e, RemoteLogRecord)]
    stream_events = [e for e in events if isinstance(e, StreamOutputEvent)]

    assert any("captured root log" in e.msg for e in log_events)
    assert any("captured print output" in e.text for e in stream_events)
