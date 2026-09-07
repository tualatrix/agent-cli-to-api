from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from .openai_compat import normalize_message_content


@dataclass(frozen=True)
class StreamJsonResult:
    text: str
    usage: dict[str, int] | None


@dataclass(frozen=True)
class StreamDelta:
    content: str = ""
    reasoning: str = ""

    def __bool__(self) -> bool:
        return bool(self.content or self.reasoning)


class TextAssembler:
    """
    Some CLIs emit partial deltas and later emit a full final message.
    This helper turns mixed streams into clean deltas (and a final assembled text).
    """

    def __init__(self) -> None:
        self.text = ""

    def feed(self, incoming: str, *, incremental: bool = False) -> str:
        incoming = incoming or ""
        if not incoming:
            return ""
        if incoming == self.text:
            return ""
        if incremental:
            if incoming.startswith(self.text):
                delta = incoming[len(self.text) :]
                self.text = incoming
                return delta
            if self.text.startswith(incoming):
                return ""
            # Cursor --stream-partial-output often labels a rewritten snapshot
            # as subtype=delta. Concatenating that snapshot repeats the opening.
            incremental = False
        if incoming.startswith(self.text):
            delta = incoming[len(self.text) :]
            self.text = incoming
            return delta
        if self.text.startswith(incoming):
            # Older / shorter snapshot of text we already assembled.
            return ""
        if self.text and self.text in incoming:
            prefix, _, suffix = incoming.partition(self.text)
            self.text = incoming
            return f"{prefix}{suffix}"
        common = _common_prefix_len(self.text, incoming)
        # Later assistant/thinking events often resend a near-complete snapshot.
        # SSE clients cannot rewind, so only emit the unseen suffix. A full
        # replacement with no shared prefix is stored but not re-streamed.
        looks_like_snapshot = (
            "\n" in incoming
            or incoming.endswith(("。", "！", "？", ".", "!", "?", "\n"))
            or len(incoming) > 80
        )
        if self.text and (looks_like_snapshot or common >= min(len(self.text), 16)):
            self.text = incoming
            return incoming[common:] if common else ""
        self.text += incoming
        return incoming

    def unseen_since(self, already_sent: str) -> str:
        """Return text that was assembled but never yielded as an SSE delta."""
        latest = self.text or ""
        already_sent = already_sent or ""
        if not latest or latest == already_sent:
            return ""
        if not already_sent:
            return latest
        if latest.startswith(already_sent):
            return latest[len(already_sent) :]
        if already_sent.startswith(latest):
            return ""
        # A rewritten snapshot was stored without a shared prefix, or the
        # client already received an old opening plus a new suffix. Dumping
        # latest[common:] here repeats sentences in the session log.
        return ""


def _common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


async def iter_stream_json_events(
    *,
    cmd: list[str],
    env: dict[str, str] | None,
    timeout_seconds: int,
    stream_limit: int,
    event_callback: Callable[[dict], None] | None = None,
    stderr_callback: Callable[[str], None] | None = None,
) -> AsyncIterator[dict]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=stream_limit,
        env=env or os.environ.copy(),
    )

    stderr_buf: bytearray = bytearray()
    last_hint: str | None = None

    async def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        text_buf = ""
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                if stderr_callback and text_buf.strip():
                    for line in text_buf.splitlines():
                        line = line.strip()
                        if line:
                            stderr_callback(line)
                return
            stderr_buf.extend(chunk)
            if len(stderr_buf) > 64_000:
                del stderr_buf[:-64_000]
            if stderr_callback:
                text_buf += chunk.decode(errors="ignore")
                if "\n" in text_buf:
                    lines = text_buf.splitlines(keepends=False)
                    if not text_buf.endswith("\n"):
                        text_buf = lines.pop() if lines else ""
                    else:
                        text_buf = ""
                    for line in lines:
                        line = line.strip()
                        if line:
                            stderr_callback(line)

    drain_task = asyncio.create_task(_drain_stderr())
    try:
        if proc.stdout is None:
            raise RuntimeError("subprocess stdout not available")

        while True:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout_seconds)
            except (asyncio.TimeoutError, TimeoutError):
                proc.kill()
                await proc.wait()
                raise
            except ValueError as e:
                proc.kill()
                await proc.wait()
                msg = bytes(stderr_buf).decode(errors="ignore").strip()
                hint = (
                    f"subprocess output line exceeded asyncio stream limit ({stream_limit} bytes). "
                    "Increase CODEX_SUBPROCESS_STREAM_LIMIT."
                )
                raise RuntimeError(f"{hint}\n{msg}".strip()) from e

            if not line:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                evt = json.loads(raw.decode(errors="ignore"))
            except Exception:
                # Some CLIs print non-JSON lines even in stream-json mode.
                continue
            if evt.get("type") == "result" and isinstance(evt.get("result"), str) and evt.get("result"):
                last_hint = str(evt.get("result")).strip() or last_hint
            if evt.get("type") == "error" and isinstance(evt.get("message"), str) and evt.get("message"):
                last_hint = str(evt.get("message")).strip() or last_hint
            if event_callback:
                event_callback(evt)
            yield evt

        rc = await proc.wait()
        await drain_task
        if rc != 0:
            msg = bytes(stderr_buf).decode(errors="ignore").strip()
            raise RuntimeError(msg or last_hint or f"subprocess failed: {rc}")
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        if not drain_task.done():
            drain_task.cancel()


def extract_text_from_content(content: object) -> str:
    return extract_parts_from_content(content)[0]


def extract_parts_from_content(content: object) -> tuple[str, str]:
    if content is None:
        return "", ""
    if isinstance(content, str):
        return content, ""
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return normalize_message_content(content), ""

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "")
        if part_type in {"thinking", "reasoning", "thought"}:
            value = part.get("thinking") or part.get("reasoning") or part.get("text") or ""
            if isinstance(value, str) and value:
                reasoning_parts.append(value)
            continue
        if part_type == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
    return "".join(text_parts), "".join(reasoning_parts)


def _feed_final_result(assembler: TextAssembler, result: object) -> str:
    if not isinstance(result, str) or not result:
        return ""
    current = assembler.text or ""
    if current and not (
        result.startswith(current) or current in result or len(result) >= len(current)
    ):
        return ""
    return assembler.feed(result)


def extract_cursor_agent_delta(evt: dict, assembler: TextAssembler) -> str:
    return extract_cursor_agent_parts(evt, assembler).content


def extract_cursor_agent_parts(
    evt: dict,
    content_assembler: TextAssembler,
    reasoning_assembler: TextAssembler | None = None,
) -> StreamDelta:
    event_type = evt.get("type")
    if event_type == "thinking":
        if evt.get("subtype") == "completed":
            return StreamDelta()
        incoming = evt.get("text") if isinstance(evt.get("text"), str) else ""
        if reasoning_assembler is not None:
            incoming = reasoning_assembler.feed(incoming, incremental=True)
        return StreamDelta(reasoning=incoming)
    if event_type == "result":
        return StreamDelta(content=_feed_final_result(content_assembler, evt.get("result")))
    if event_type != "assistant":
        return StreamDelta()
    message = evt.get("message") or {}
    if not isinstance(message, dict):
        return StreamDelta()
    text, reasoning = extract_parts_from_content(message.get("content"))
    incremental = evt.get("subtype") == "delta"
    return StreamDelta(
        content=content_assembler.feed(text, incremental=incremental),
        reasoning=(
            reasoning_assembler.feed(reasoning, incremental=incremental)
            if reasoning_assembler is not None
            else reasoning
        ),
    )


def extract_claude_delta(evt: dict, assembler: TextAssembler) -> str:
    return extract_claude_parts(evt, assembler).content


def extract_claude_parts(
    evt: dict,
    content_assembler: TextAssembler,
    reasoning_assembler: TextAssembler | None = None,
) -> StreamDelta:
    event_type = evt.get("type")
    if event_type == "thinking":
        incoming = evt.get("text") if isinstance(evt.get("text"), str) else ""
        if evt.get("subtype") == "completed":
            return StreamDelta()
        if reasoning_assembler is not None:
            incoming = reasoning_assembler.feed(incoming, incremental=True)
        return StreamDelta(reasoning=incoming)
    if event_type == "result":
        return StreamDelta(content=_feed_final_result(content_assembler, evt.get("result")))
    if event_type != "assistant":
        return StreamDelta()
    message = evt.get("message") or {}
    if not isinstance(message, dict):
        return StreamDelta()
    text, reasoning = extract_parts_from_content(message.get("content"))
    return StreamDelta(
        content=content_assembler.feed(text),
        reasoning=(reasoning_assembler.feed(reasoning) if reasoning_assembler is not None else reasoning),
    )


def extract_gemini_delta(evt: dict, assembler: TextAssembler) -> str:
    return extract_gemini_parts(evt, assembler).content


def extract_gemini_parts(
    evt: dict,
    content_assembler: TextAssembler,
    reasoning_assembler: TextAssembler | None = None,
) -> StreamDelta:
    if evt.get("type") == "thinking":
        incoming = evt.get("text") if isinstance(evt.get("text"), str) else ""
        if reasoning_assembler is not None:
            incoming = reasoning_assembler.feed(incoming, incremental=True)
        return StreamDelta(reasoning=incoming)
    if evt.get("type") != "message":
        return StreamDelta()
    if evt.get("role") != "assistant":
        return StreamDelta()
    incoming = extract_text_from_content(evt.get("content"))
    reasoning = evt.get("reasoning") if isinstance(evt.get("reasoning"), str) else ""
    return StreamDelta(
        content=content_assembler.feed(incoming),
        reasoning=(reasoning_assembler.feed(reasoning) if reasoning_assembler is not None else reasoning),
    )


def extract_codex_cli_parts(
    evt: dict,
    content_assembler: TextAssembler,
    reasoning_assembler: TextAssembler | None = None,
) -> StreamDelta:
    if evt.get("type") != "item.completed":
        return StreamDelta()
    item = evt.get("item") or {}
    if not isinstance(item, dict):
        return StreamDelta()
    item_type = item.get("type")
    raw = item.get("text") if isinstance(item.get("text"), str) else ""
    if item_type == "reasoning":
        if reasoning_assembler is not None:
            raw = reasoning_assembler.feed(raw)
        return StreamDelta(reasoning=raw)
    if item_type == "agent_message":
        return StreamDelta(content=content_assembler.feed(raw))
    return StreamDelta()


def extract_codex_responses_parts(
    evt: dict,
    content_assembler: TextAssembler | None = None,
    reasoning_assembler: TextAssembler | None = None,
) -> StreamDelta:
    def _content(text: str, *, incremental: bool) -> str:
        if content_assembler is None:
            return text
        return content_assembler.feed(text, incremental=incremental)

    def _reasoning(text: str, *, incremental: bool) -> str:
        if reasoning_assembler is None:
            return text
        return reasoning_assembler.feed(text, incremental=incremental)

    event_type = evt.get("type")
    if event_type == "response.output_text.delta" and isinstance(evt.get("delta"), str):
        return StreamDelta(content=_content(evt["delta"], incremental=True))
    if event_type == "response.output_text.done" and isinstance(evt.get("text"), str):
        return StreamDelta(content=_content(evt["text"], incremental=False))
    if event_type in {
        "response.reasoning_summary_text.delta",
        "response.reasoning.delta",
        "response.reasoning_text.delta",
    } and isinstance(evt.get("delta"), str):
        return StreamDelta(reasoning=_reasoning(evt["delta"], incremental=True))
    if event_type in {
        "response.reasoning_summary_text.done",
        "response.reasoning.done",
    } and isinstance(evt.get("text"), str):
        return StreamDelta(reasoning=_reasoning(evt["text"], incremental=False))
    if event_type == "response.reasoning_summary_part.added":
        part = evt.get("part") or {}
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            return StreamDelta(reasoning=_reasoning(part["text"], incremental=False))
    if event_type in {"response.output_item.added", "response.output_item.done"}:
        item = evt.get("item") or {}
        if isinstance(item, dict) and item.get("type") == "reasoning":
            summary = item.get("summary")
            texts: list[str] = []
            if isinstance(item.get("text"), str) and item["text"]:
                texts.append(item["text"])
            if isinstance(summary, list):
                for part in summary:
                    if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]:
                        texts.append(part["text"])
            if texts:
                return StreamDelta(reasoning=_reasoning("".join(texts), incremental=False))
    return StreamDelta()


def extract_usage_from_claude_result(evt: dict) -> dict[str, int] | None:
    if evt.get("type") != "result":
        return None
    usage = evt.get("usage")
    if not isinstance(usage, dict):
        return None
    in_tokens = int(usage.get("input_tokens") or 0)
    out_tokens = int(usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": in_tokens,
        "completion_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
    }


def extract_usage_from_gemini_result(evt: dict) -> dict[str, int] | None:
    if evt.get("type") != "result":
        return None
    stats = evt.get("stats")
    if not isinstance(stats, dict):
        return None
    in_tokens = int(stats.get("input_tokens") or 0)
    out_tokens = int(stats.get("output_tokens") or 0)
    total = int(stats.get("total_tokens") or (in_tokens + out_tokens))
    return {
        "prompt_tokens": in_tokens,
        "completion_tokens": out_tokens,
        "total_tokens": total,
    }
