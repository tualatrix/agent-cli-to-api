from __future__ import annotations

import asyncio
import unittest

from codex_gateway.anthropic_compat import (
    openai_chat_completion_to_anthropic_message,
    openai_stream_to_anthropic_events,
)
from codex_gateway.gemini_cloudcode import _extract_parts_from_cloudcode_response
from codex_gateway.server import _chat_completion_to_responses
from codex_gateway.stream_json_cli import (
    TextAssembler,
    extract_claude_parts,
    extract_codex_cli_parts,
    extract_codex_responses_parts,
    extract_cursor_agent_parts,
    extract_gemini_parts,
    extract_parts_from_content,
)


class TextAssemblerTests(unittest.TestCase):
    def test_snapshot_extension_emits_suffix(self) -> None:
        assembler = TextAssembler()
        self.assertEqual(assembler.feed("先看工作区"), "先看工作区")
        self.assertEqual(assembler.feed("先看工作区\n再改代码"), "\n再改代码")
        self.assertEqual(assembler.text, "先看工作区\n再改代码")

    def test_repeated_snapshot_does_not_duplicate(self) -> None:
        assembler = TextAssembler()
        assembler.feed("工作区里有一张失败现场")
        self.assertEqual(assembler.feed("工作区里有一张失败现场"), "")
        self.assertEqual(assembler.text, "工作区里有一张失败现场")

    def test_replacement_snapshot_does_not_concatenate(self) -> None:
        assembler = TextAssembler()
        assembler.feed("工作区里有一张失败现场。这是旧描述。")
        delta = assembler.feed("应改 StreamDownKit，不是 PopAgent。")
        # SSE cannot rewind; keep the latest snapshot without re-streaming it.
        self.assertEqual(delta, "")
        self.assertEqual(assembler.text, "应改 StreamDownKit，不是 PopAgent。")

    def test_resent_opening_only_emits_suffix(self) -> None:
        assembler = TextAssembler()
        opening = (
            "当前会话没有绑定 TutuStudio Task，本轮不会记到 Work。"
            "我先在网关和各 provider 路径里查有没有 mid-request 注入（steer）的接口或转发。"
        )
        self.assertEqual(assembler.feed(opening), opening)
        self.assertEqual(assembler.feed(opening), "")
        self.assertEqual(assembler.feed(opening + "\n再看重复回复"), "\n再看重复回复")
        self.assertEqual(assembler.text, opening + "\n再看重复回复")

    def test_incremental_token_still_appends(self) -> None:
        assembler = TextAssembler()
        self.assertEqual(assembler.feed("Hello", incremental=True), "Hello")
        self.assertEqual(assembler.feed(" world", incremental=True), " world")
        self.assertEqual(assembler.text, "Hello world")

    def test_incremental_rewritten_snapshot_does_not_glue_old_suffix(self) -> None:
        assembler = TextAssembler()
        first = "先对照列表长按和「分享微博」是否都走同一条只带链接的分享实现。"
        later = "先对照列表长按和「分享微博」是否都走同一条只带链接是的。列表里长按微博后菜单里的「分享微博...」。"
        self.assertEqual(assembler.feed(first, incremental=True), first)
        # Mid-string rewrite must not emit later[common:] onto the old suffix.
        self.assertEqual(assembler.feed(later, incremental=True), "")
        self.assertEqual(assembler.text, later)
        self.assertEqual(assembler.unseen_since(first), "\n\n" + later)

    def test_incremental_punctuation_rewrite_does_not_glue_old_suffix(self) -> None:
        assembler = TextAssembler()
        first = "当前会话没有绑定 TutuStudio Task，本轮不会记录到 Work。"
        later = "当前会话没有绑定 TutuStudio Task，本轮不会记录到 Work工作区不在 NewLime"
        self.assertEqual(assembler.feed(first, incremental=True), first)
        self.assertEqual(assembler.feed(later, incremental=True), "")
        self.assertEqual(assembler.text, later)
        self.assertEqual(assembler.unseen_since(first), "\n\n" + later)

    def test_unseen_since_emits_rewritten_snapshot_at_end(self) -> None:
        assembler = TextAssembler()
        first = "工作区里有一张失败现场。这是旧描述。"
        self.assertEqual(assembler.feed(first), first)
        later = "应改 StreamDownKit，不是 PopAgent。补上被丢掉的后半段。"
        self.assertEqual(assembler.feed(later), "")
        self.assertEqual(assembler.text, later)
        self.assertEqual(assembler.unseen_since(first), "\n\n" + later)
        self.assertEqual(assembler.unseen_since(""), later)
        self.assertEqual(assembler.unseen_since(later), "")

    def test_late_short_notice_does_not_replace_long_answer(self) -> None:
        assembler = TextAssembler()
        answer = (
            "不完全是必须二选一，但启动、恢复、外部打开这三件事必须只有一个主人。"
            "现在出问题的正是这层混用，不是 Diagnostics 窗口本身用了 SwiftUI。"
        )
        notice = "当前会话没有绑定 TutuStudio Task，本轮不会记到 Work。要记录的话先发任务编号或 `@newtask` / `新"
        self.assertEqual(assembler.feed(answer), answer)
        self.assertEqual(assembler.feed(notice), "")
        self.assertEqual(assembler.text, answer)
        self.assertEqual(assembler.unseen_since(answer), "")

    def test_smashed_prefix_extension_inserts_paragraph_break(self) -> None:
        assembler = TextAssembler()
        first = "工作区里有 8 个改动文件"
        later = first + "`macOS/Localizable.xcstrings` 仍是无关抽取。"
        self.assertEqual(assembler.feed(first), first)
        self.assertEqual(assembler.feed(later), "\n\n`macOS/Localizable.xcstrings` 仍是无关抽取。")
        self.assertEqual(assembler.text, first + "\n\n`macOS/Localizable.xcstrings` 仍是无关抽取。")

    def test_rewritten_leftover_is_separated_from_old_draft(self) -> None:
        assembler = TextAssembler()
        first = "评审通过：诊断窗外壳与 Store/HUD 同一套 AppKit 所有权，剩余 Swift"
        later = (
            "评审没有阻塞问题，已提交到 `main`："
            "实现一致。`standardSessionWindowLeavesSwiftUIWindowGroup` 已通过。"
        )
        self.assertEqual(assembler.feed(first), first)
        self.assertEqual(assembler.feed(later), "")
        self.assertEqual(assembler.unseen_since(first), "\n\n" + later)

    def test_thinking_journal_keeps_status_lines_apart(self) -> None:
        content = TextAssembler()
        reasoning = TextAssembler()
        first = extract_cursor_agent_parts(
            {"type": "thinking", "subtype": "delta", "text": "准备改用 Shell"},
            content,
            reasoning,
        )
        second = extract_cursor_agent_parts(
            {"type": "thinking", "subtype": "delta", "text": "正在检查 macOS 诊断窗口的实现。"},
            content,
            reasoning,
        )
        self.assertEqual(first.reasoning, "准备改用 Shell")
        self.assertEqual(second.reasoning, "\n\n正在检查 macOS 诊断窗口的实现。")
        self.assertEqual(
            reasoning.text,
            "准备改用 Shell\n\n正在检查 macOS 诊断窗口的实现。",
        )

    def test_chinese_clause_is_not_glued_as_token_crumb(self) -> None:
        assembler = TextAssembler()
        first = (
            "根评论时间旁已能显示 `source`，回复页和楼层用的 `StatusComment` 没画出来。"
            "微博正文用的是 `region_name`，评论接口多半也有。接下来把该"
        )
        clause = "重新编译后再进那条 4 条回复的楼层看一眼。当前会话"
        fragment = (
            " `region_name`（和微博正文同一套，例如「发布于 华盛顿」）\n"
            "- 没有 `region_name` "
        )
        self.assertEqual(assembler.feed(first), first)
        self.assertEqual(assembler.feed(first + "评论接口里本来就有位置字段，回复页只画"), "评论接口里本来就有位置字段，回复页只画")
        # A cut-off later draft must not be treated as the next few tokens.
        self.assertEqual(assembler.feed(clause, incremental=True), "")
        self.assertEqual(assembler.feed(fragment, incremental=True), "")
        self.assertEqual(assembler.feed(clause, incremental=True), "")
        self.assertNotIn("重新编译后再进", assembler.text)
        self.assertNotIn("发布于 华盛顿", assembler.text)
        self.assertTrue(assembler.text.startswith("根评论时间旁已能显示"))


class ProviderReasoningSplitTests(unittest.TestCase):
    def test_cursor_thinking_events_are_not_content(self) -> None:
        content = TextAssembler()
        reasoning = TextAssembler()
        thinking = extract_cursor_agent_parts(
            {"type": "thinking", "subtype": "delta", "text": "先读图再回答"},
            content,
            reasoning,
        )
        extract_cursor_agent_parts(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "应改 StreamDownKit"}]},
            },
            content,
            reasoning,
        )
        done = extract_cursor_agent_parts(
            {"type": "result", "result": "应改 StreamDownKit"},
            content,
            reasoning,
        )
        self.assertEqual(thinking.content, "")
        self.assertEqual(thinking.reasoning, "先读图再回答")
        self.assertEqual(done.content, "应改 StreamDownKit")
        self.assertEqual(content.text, "应改 StreamDownKit")
        self.assertEqual(reasoning.text, "先读图再回答")

    def test_cursor_assistant_snapshot_resent_is_not_duplicated(self) -> None:
        content = TextAssembler()
        reasoning = TextAssembler()
        opening = "当前会话没有绑定 TutuStudio Task，本轮不会记到 Work。我先查 steer。"
        extract_cursor_agent_parts(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": opening}]},
            },
            content,
            reasoning,
        )
        extract_cursor_agent_parts(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": opening}]},
            },
            content,
            reasoning,
        )
        done = extract_cursor_agent_parts({"type": "result", "result": opening}, content, reasoning)
        self.assertEqual(done.content, opening)
        self.assertEqual(content.text, opening)

    def test_cursor_result_emits_unseen_suffix(self) -> None:
        content = TextAssembler()
        opening = "如果这里能一行行出 data: ...，就是 client 没消费 SSE（或打到了 /v1/responses）。"
        rest = "如果这里也是等很久再一大坨，就是当前 provider 模式本身不够细。"
        first = extract_cursor_agent_parts(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": opening}]},
            },
            content,
        )
        done = extract_cursor_agent_parts(
            {"type": "result", "result": opening + rest},
            content,
        )
        self.assertEqual(first.content, "")
        self.assertEqual(done.content, opening + "\n\n" + rest)
        self.assertEqual(content.text, opening + "\n\n" + rest)

    def test_cursor_pre_tool_narration_is_reasoning_not_content(self) -> None:
        content = TextAssembler()
        reasoning = TextAssembler()
        notice = (
            "当前会话没有绑定 TutuStudio Task，本轮不会记录到 Work。"
            "如需记录，请先发送任务编号（例如 `TSK-123`）。"
        )
        narration = notice + "\n\n我先打开这本 EPUB，从目录和前言判断它在讲什么。"
        answer = notice + "\n\n这是 O’Reilly 的早期预览电子书《Evals for AI Engineers》。"
        extract_cursor_agent_parts(
            {
                "type": "thinking",
                "subtype": "delta",
                "text": "正在查看用户提供的 EPUB 文件，准备分析其内容。",
            },
            content,
            reasoning,
        )
        extract_cursor_agent_parts(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": narration}]},
            },
            content,
            reasoning,
        )
        tool = extract_cursor_agent_parts(
            {"type": "tool_call", "subtype": "started", "call_id": "tool_1"},
            content,
            reasoning,
        )
        extract_cursor_agent_parts(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            },
            content,
            reasoning,
        )
        done = extract_cursor_agent_parts(
            {"type": "result", "result": narration + answer},
            content,
            reasoning,
        )
        self.assertTrue(tool.reasoning.endswith(narration))
        self.assertEqual(done.content, answer)
        self.assertNotIn("我先打开这本 EPUB", done.content)
        self.assertNotIn("我先打开这本 EPUB", content.text)
        self.assertIn("正在查看用户提供的 EPUB 文件", reasoning.text)
        self.assertIn("我先打开这本 EPUB", reasoning.text)

    def test_cursor_result_extends_last_segment_prefix(self) -> None:
        content = TextAssembler()
        reasoning = TextAssembler()
        narration = "我先打开这本 EPUB，从目录和前言判断它在讲什么。"
        prefix = "不完全是必须二选一，但启动、恢复、外部打开"
        rest = "这三件事必须只有一个主人。"
        extract_cursor_agent_parts(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": narration}]},
            },
            content,
            reasoning,
        )
        extract_cursor_agent_parts(
            {"type": "tool_call", "subtype": "started", "call_id": "tool_1"},
            content,
            reasoning,
        )
        extract_cursor_agent_parts(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": prefix}]},
            },
            content,
            reasoning,
        )
        done = extract_cursor_agent_parts(
            {"type": "result", "result": narration + prefix + rest},
            content,
            reasoning,
        )
        self.assertEqual(done.content, prefix + rest)
        self.assertEqual(content.text, prefix + rest)
        self.assertNotIn("我先打开这本 EPUB", content.text)

    def test_cursor_partial_pre_tool_narration_is_held_then_reasoning(self) -> None:
        content = TextAssembler()
        reasoning = TextAssembler()
        narration = "我先打开这本 EPUB，从目录和前言判断它在讲什么。"
        answer = "这是 O’Reilly 的早期预览电子书《Evals for AI Engineers》。"
        extract_cursor_agent_parts(
            {
                "type": "assistant",
                "subtype": "delta",
                "timestamp_ms": 1,
                "message": {"role": "assistant", "content": [{"type": "text", "text": narration}]},
            },
            content,
            reasoning,
        )
        extract_cursor_agent_parts(
            {"type": "tool_call", "subtype": "started", "call_id": "tool_1"},
            content,
            reasoning,
        )
        streamed = extract_cursor_agent_parts(
            {
                "type": "assistant",
                "subtype": "delta",
                "timestamp_ms": 2,
                "message": {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            },
            content,
            reasoning,
        )
        done = extract_cursor_agent_parts(
            {"type": "result", "result": narration + answer},
            content,
            reasoning,
        )
        self.assertEqual(streamed.content, answer)
        self.assertEqual(done.content, "")
        self.assertEqual(content.text, answer)
        self.assertIn(narration, reasoning.text)

    def test_cursor_result_does_not_prepend_earlier_segments(self) -> None:
        assembler = TextAssembler()
        assembler.feed("这是 O’Reilly 的早期预览电子书。")
        delta = assembler.feed("我先打开这本 EPUB。这是 O’Reilly 的早期预览电子书。")
        self.assertEqual(delta, "")
        self.assertEqual(assembler.text, "这是 O’Reilly 的早期预览电子书。")

    def test_stale_shorter_snapshot_does_not_append_old_draft(self) -> None:
        assembler = TextAssembler()
        notice = "当前会话没有绑定 TutuStudio Task，本轮不会记录到 Work。"
        answer = notice + "这是 O’Reilly 的早期预览电子书《Evals for AI Engineers》。"
        draft = notice + "我先打开这本 EPUB，从目录和前言判断它在讲什么。"
        self.assertEqual(assembler.feed(answer), answer)
        self.assertEqual(assembler.feed(draft), "")
        self.assertEqual(assembler.text, answer)

    def test_cursor_skips_buffered_assistant_flush(self) -> None:
        content = TextAssembler()
        answer = "这是 O’Reilly 的早期预览电子书。"
        extract_cursor_agent_parts(
            {
                "type": "assistant",
                "timestamp_ms": 1,
                "subtype": "delta",
                "message": {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            },
            content,
        )
        extract_cursor_agent_parts(
            {
                "type": "assistant",
                "model_call_id": "call_1",
                "timestamp_ms": 2,
                "message": {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            },
            content,
        )
        late = extract_cursor_agent_parts(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "我先打开这本 EPUB。"}]},
            },
            content,
        )
        done = extract_cursor_agent_parts({"type": "result", "result": answer}, content)
        self.assertEqual(late.content, "")
        self.assertEqual(done.content, answer)
        self.assertEqual(content.text, answer)

    def test_assistant_thinking_blocks_are_split(self) -> None:
        text, thinking = extract_parts_from_content(
            [
                {"type": "thinking", "thinking": "plan first"},
                {"type": "text", "text": "final answer"},
            ]
        )
        self.assertEqual(text, "final answer")
        self.assertEqual(thinking, "plan first")

    def test_claude_thinking_events_go_to_reasoning(self) -> None:
        content = TextAssembler()
        reasoning = TextAssembler()
        parts = extract_claude_parts(
            {"type": "thinking", "subtype": "delta", "text": "look at the stack"},
            content,
            reasoning,
        )
        self.assertEqual(parts.reasoning, "look at the stack")
        self.assertEqual(parts.content, "")

    def test_codex_cli_splits_reasoning_and_agent_message(self) -> None:
        content = TextAssembler()
        reasoning = TextAssembler()
        first = extract_codex_cli_parts(
            {"type": "item.completed", "item": {"type": "reasoning", "text": "need to inspect"}},
            content,
            reasoning,
        )
        second = extract_codex_cli_parts(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "fixed"}},
            content,
            reasoning,
        )
        self.assertEqual(first.reasoning, "need to inspect")
        self.assertEqual(second.content, "fixed")

    def test_codex_responses_reasoning_summary_delta(self) -> None:
        parts = extract_codex_responses_parts(
            {"type": "response.reasoning_summary_text.delta", "delta": "checking files"}
        )
        self.assertEqual(parts.reasoning, "checking files")
        self.assertEqual(parts.content, "")

    def test_codex_responses_done_does_not_repeat_delta(self) -> None:
        content = TextAssembler()
        reasoning = TextAssembler()
        first = extract_codex_responses_parts(
            {"type": "response.reasoning_summary_text.delta", "delta": "checking files"},
            content,
            reasoning,
        )
        second = extract_codex_responses_parts(
            {"type": "response.reasoning_summary_text.done", "text": "checking files"},
            content,
            reasoning,
        )
        third = extract_codex_responses_parts(
            {
                "type": "response.output_item.done",
                "item": {"type": "reasoning", "text": "checking files"},
            },
            content,
            reasoning,
        )
        self.assertEqual(first.reasoning, "checking files")
        self.assertEqual(second.reasoning, "")
        self.assertEqual(third.reasoning, "")
        self.assertEqual(reasoning.text, "checking files")

    def test_gemini_thinking_tokens_are_incremental(self) -> None:
        content = TextAssembler()
        reasoning = TextAssembler()
        first = extract_gemini_parts(
            {"type": "thinking", "text": "look"},
            content,
            reasoning,
        )
        second = extract_gemini_parts(
            {"type": "thinking", "text": " at stack"},
            content,
            reasoning,
        )
        self.assertEqual(first.reasoning, "look")
        self.assertEqual(second.reasoning, " at stack")
        self.assertEqual(reasoning.text, "look at stack")

    def test_chat_completion_to_responses_keeps_reasoning(self) -> None:
        converted = _chat_completion_to_responses(
            {
                "created": 1,
                "model": "gpt-5.5",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "应改 StreamDownKit",
                            "reasoning_content": "先查下划线画在哪一层",
                        }
                    }
                ],
            }
        )
        types = [item.get("type") for item in converted["output"]]
        self.assertEqual(types, ["reasoning", "message"])
        self.assertEqual(converted["output"][0]["summary"][0]["text"], "先查下划线画在哪一层")
        self.assertEqual(converted["output"][1]["content"][0]["text"], "应改 StreamDownKit")

    def test_gemini_thought_parts_are_split(self) -> None:
        text, reasoning = _extract_parts_from_cloudcode_response(
            {
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "I should inspect the renderer", "thought": True},
                                    {"text": "Change StreamDownKit."},
                                ]
                            }
                        }
                    ]
                }
            }
        )
        self.assertEqual(text, "Change StreamDownKit.")
        self.assertEqual(reasoning, "I should inspect the renderer")


class AnthropicReasoningCompatTests(unittest.TestCase):
    def test_non_stream_maps_reasoning_content_to_thinking_block(self) -> None:
        payload = openai_chat_completion_to_anthropic_message(
            {
                "id": "chatcmpl-1",
                "model": "gpt-5.5",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "应改 StreamDownKit",
                            "reasoning_content": "先查下划线画在哪一层",
                        }
                    }
                ],
            }
        )
        self.assertEqual(
            payload["content"],
            [
                {"type": "thinking", "thinking": "先查下划线画在哪一层"},
                {"type": "text", "text": "应改 StreamDownKit"},
            ],
        )

    def test_stream_emits_thinking_block_before_text(self) -> None:
        async def source():
            yield (
                'data: {"choices":[{"delta":{"reasoning_content":"先查一层"},"finish_reason":null}]}\n\n'
            )
            yield 'data: {"choices":[{"delta":{"content":"改 kit"},"finish_reason":null}]}\n\n'
            yield "data: [DONE]\n\n"

        async def collect():
            return [event async for event in openai_stream_to_anthropic_events(source(), model="claude")]

        events = asyncio.run(collect())
        joined = "".join(events)
        self.assertIn('"type": "thinking"', joined)
        self.assertIn('"thinking": "先查一层"', joined)
        self.assertIn('"text": "改 kit"', joined)
        thinking_at = joined.find('"type": "thinking"')
        text_at = joined.find('"type": "text"')
        self.assertGreater(text_at, thinking_at)


if __name__ == "__main__":
    unittest.main()
