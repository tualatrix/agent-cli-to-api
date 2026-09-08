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

    ``journal=True`` is for thinking / status lines: each new event is a
    paragraph, not a rewrite of one document. Cursor concatenates those
    lines without newlines; we reinsert the breaks on emit.
    """

    def __init__(self, *, journal: bool = False) -> None:
        self.text = ""
        self.raw = ""
        self.journal = journal

    def feed(self, incoming: str, *, incremental: bool = False) -> str:
        incoming = incoming or ""
        if not incoming:
            return ""
        if incoming == self.raw or incoming == self.text:
            return ""
        if incremental:
            if incoming.startswith(self.raw):
                return self._extend_raw(incoming)
            if self.raw.startswith(incoming):
                return ""
            # Cursor --stream-partial-output often labels a rewritten snapshot
            # as subtype=delta. Concatenating that snapshot repeats the opening.
            incremental = False
        if incoming.startswith(self.raw):
            return self._extend_raw(incoming)
        if self.raw.startswith(incoming):
            # Older / shorter snapshot of text we already assembled.
            return ""
        if self.raw and self.raw in incoming:
            # Cursor's terminal `result` concatenates every assistant segment,
            # including pre-tool narration. Only take a trailing extension.
            suffix = incoming[incoming.index(self.raw) + len(self.raw) :]
            if not suffix:
                return ""
            return self._extend_raw(self.raw + suffix)
        if self.journal:
            if _is_token_crumb(incoming):
                return self._extend_raw(self.raw + incoming)
            return self._append_paragraph(incoming)
        if (
            self.raw
            and incoming.lstrip().startswith("当前会话没有绑定")
            and not self.raw.lstrip().startswith("当前会话没有绑定")
        ):
            return ""
        common = _common_prefix_len(self.raw, incoming)
        looks_like_snapshot = _looks_like_snapshot(incoming)
        if self.raw and (looks_like_snapshot or common >= min(len(self.raw), 16)):
            if len(incoming) < len(self.raw):
                # Stale earlier draft that shares an opening (often the
                # TutuStudio notice). Do not append that draft's suffix.
                return ""
            # Rewrite. SSE cannot rewind the old unique suffix, so emitting
            # incoming[common:] pastes the new tail onto the old draft.
            self.raw = incoming
            self.text = incoming
            return ""
        # Short crumb with no shared prefix: true token increment.
        if self.raw and not looks_like_snapshot and len(incoming) < 40:
            return self._extend_raw(self.raw + incoming)
        if self.raw:
            return ""
        self.raw = incoming
        self.text = incoming
        return incoming

    def _extend_raw(self, incoming: str) -> str:
        delta = incoming[len(self.raw) :]
        self.raw = incoming
        if not delta:
            return ""
        if _needs_paragraph_break(self.text, delta):
            delta = "\n\n" + delta
        self.text += delta
        return delta

    def _append_paragraph(self, incoming: str) -> str:
        self.raw = incoming
        delta = incoming if not self.text else "\n\n" + incoming
        self.text += delta
        return delta

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
        if latest in already_sent:
            return ""
        # A mid-stream rewrite was stored silently. Emit the last snapshot
        # once so the client is not left with a cut-off draft. Prefix a
        # paragraph break so the old tail and new draft are not jammed.
        return "\n\n" + latest


_NEW_THOUGHT_STARTS = (
    "正在",
    "准备",
    "接下来",
    "接着",
    "先看",
    "先在",
    "先对",
    "用户",
    "重点",
    "同时",
    "需要",
    "当前",
    "工作区",
    "现在",
    "确认",
    "发现",
    "评审",
    "验证",
    "重新",
    "根评",
    "未纳入",
    "未提交",
    "The ",
    "I ",
    "I'll ",
    "Reviewing",
    "Checking",
    "Looking",
    "Reading",
    "Let ",
    "Now ",
    "Got ",
    "No ",
    "Will ",
    "Starting",
    "**",
    "- ",
)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2CEAF
    )


def _looks_like_snapshot(text: str) -> bool:
    if not text:
        return False
    if "\n" in text or len(text) > 80:
        return True
    if text.endswith(("。", "！", "？", ".", "!", "?", "\n", "…")):
        return True
    # Cursor often cuts a rewrite before the trailing period, e.g.
    # "重新编译后再进那条 4 条回复的楼层看一眼。当前会话"
    return any(mark in text for mark in ("。", "！", "？"))


def _looks_like_new_sentence(text: str) -> bool:
    stripped = text.lstrip()
    return bool(stripped) and stripped.startswith(_NEW_THOUGHT_STARTS)


def _is_token_crumb(text: str) -> bool:
    if _looks_like_snapshot(text) or _looks_like_new_sentence(text):
        return False
    # A 20–40 char Chinese clause is a snapshot, not a token. English crumbs
    # like " world" stay below this CJK threshold.
    if sum(1 for char in text if _is_cjk(char)) >= 8:
        return False
    return 0 < len(text) < 40


def _needs_paragraph_break(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left[-1].isspace() or right[0].isspace():
        return False
    if right[0] in ".,;:!?。，、；：！？)]｝》'\"":
        return False
    if left.rstrip().endswith(("。", "！", "？", ".", "!", "?", "…", "：", ":", "`")):
        return True
    if _looks_like_new_sentence(right):
        return True
    if _is_cjk(left[-1]) != _is_cjk(right[0]):
        return True
    if left[-1].isalpha() and right[0].isupper() and right[0].isascii():
        return True
    return False


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
    return "".join(text_parts), "\n\n".join(reasoning_parts)


def _feed_final_result(assembler: TextAssembler, result: object) -> str:
    if not isinstance(result, str) or not result:
        return ""
    current = assembler.raw or assembler.text or ""
    if not current:
        state = getattr(assembler, "_cursor_stream", None)
        if isinstance(state, _CursorStreamState) and state.saw_tool:
            # Concatenated result still starts with pre-tool narration.
            return ""
        return assembler.feed(result)
    if result.startswith(current):
        return assembler.feed(result)
    # Last segment only: narration + answer, current is the answer (or a prefix).
    idx = result.rfind(current)
    if idx >= 0:
        return assembler.feed(result[idx:])
    return ""


def flush_cursor_held_content(assembler: TextAssembler) -> str:
    return _flush_held_cursor_complete(assembler)


# First assistant segment before a tool is usually "I'll open the file".
# Hold it so Cursor's concatenated `result` cannot paste it into content.
# Start streaming once the segment is clearly a real answer with no tool yet.
_CURSOR_PRE_TOOL_HOLD_CHARS = 320


@dataclass
class _CursorStreamState:
    saw_timestamp: bool = False
    held_complete: str = ""
    saw_tool: bool = False


def _cursor_stream_state(assembler: TextAssembler) -> _CursorStreamState:
    state = getattr(assembler, "_cursor_stream", None)
    if not isinstance(state, _CursorStreamState):
        state = _CursorStreamState()
        assembler._cursor_stream = state
    return state


def _cursor_assistant_is_duplicate(evt: dict, state: _CursorStreamState) -> bool:
    if evt.get("model_call_id"):
        return True
    if state.saw_timestamp and evt.get("timestamp_ms") is None and evt.get("subtype") != "delta":
        return True
    return False


def _flush_held_cursor_complete(assembler: TextAssembler) -> str:
    state = _cursor_stream_state(assembler)
    held = state.held_complete
    state.held_complete = ""
    if not held:
        return ""
    return assembler.feed(held)


def extract_cursor_agent_delta(evt: dict, assembler: TextAssembler) -> str:
    return extract_cursor_agent_parts(evt, assembler).content


def extract_cursor_agent_parts(
    evt: dict,
    content_assembler: TextAssembler,
    reasoning_assembler: TextAssembler | None = None,
) -> StreamDelta:
    event_type = evt.get("type")
    state = _cursor_stream_state(content_assembler)
    if event_type == "thinking":
        if evt.get("subtype") == "completed":
            return StreamDelta()
        incoming = evt.get("text") if isinstance(evt.get("text"), str) else ""
        if reasoning_assembler is not None:
            reasoning_assembler.journal = True
            incoming = reasoning_assembler.feed(incoming, incremental=True)
        return StreamDelta(reasoning=incoming)
    if event_type == "tool_call":
        held = state.held_complete
        state.held_complete = ""
        state.saw_tool = True
        if not held:
            held = content_assembler.text
            content_assembler.text = ""
            content_assembler.raw = ""
        if held:
            if reasoning_assembler is None:
                return StreamDelta(reasoning=held)
            if held in (reasoning_assembler.text or ""):
                return StreamDelta()
            reasoning_assembler.journal = True
            return StreamDelta(reasoning=reasoning_assembler.feed(held))
        return StreamDelta()
    if event_type == "result":
        content = _flush_held_cursor_complete(content_assembler)
        content += _feed_final_result(content_assembler, evt.get("result"))
        return StreamDelta(content=content)
    if event_type != "assistant":
        return StreamDelta()
    if _cursor_assistant_is_duplicate(evt, state):
        return StreamDelta()
    message = evt.get("message") or {}
    if not isinstance(message, dict):
        return StreamDelta()
    text, reasoning = extract_parts_from_content(message.get("content"))
    incremental = evt.get("subtype") == "delta" or evt.get("timestamp_ms") is not None
    if evt.get("timestamp_ms") is not None:
        state.saw_timestamp = True
    reasoning_delta = (
        reasoning_assembler.feed(reasoning, incremental=incremental)
        if reasoning_assembler is not None
        else reasoning
    )
    if incremental:
        if not state.saw_tool:
            if text and (
                not state.held_complete
                or text.startswith(state.held_complete)
                or len(text) >= len(state.held_complete)
            ):
                state.held_complete = text
            if len(state.held_complete) < _CURSOR_PRE_TOOL_HOLD_CHARS:
                return StreamDelta(reasoning=reasoning_delta)
        prior = _flush_held_cursor_complete(content_assembler)
        return StreamDelta(
            content=prior + content_assembler.feed(text, incremental=True),
            reasoning=reasoning_delta,
        )
    # Complete assistant message (one per tool-call gap). Hold it: a following
    # tool_call means this was narration, not the final answer.
    if (
        reasoning_assembler is not None
        and text
        and text in (reasoning_assembler.text or "")
    ):
        return StreamDelta(reasoning=reasoning_delta)
    if (
        state.held_complete
        and len(text) < len(state.held_complete)
        and _common_prefix_len(text, state.held_complete) >= 16
    ):
        return StreamDelta(reasoning=reasoning_delta)
    state.held_complete = text
    return StreamDelta(reasoning=reasoning_delta)


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
            reasoning_assembler.journal = True
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
