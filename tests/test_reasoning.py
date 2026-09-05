from __future__ import annotations

import asyncio
import unittest

from codex_gateway.anthropic_compat import (
    openai_chat_completion_to_anthropic_message,
    openai_stream_to_anthropic_events,
)
from codex_gateway.gemini_cloudcode import _extract_parts_from_cloudcode_response
from codex_gateway.stream_json_cli import (
    TextAssembler,
    extract_claude_parts,
    extract_codex_cli_parts,
    extract_codex_responses_parts,
    extract_cursor_agent_parts,
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
        self.assertEqual(delta, "应改 StreamDownKit，不是 PopAgent。")
        self.assertEqual(assembler.text, "应改 StreamDownKit，不是 PopAgent。")

    def test_incremental_token_still_appends(self) -> None:
        assembler = TextAssembler()
        self.assertEqual(assembler.feed("Hello", incremental=True), "Hello")
        self.assertEqual(assembler.feed(" world", incremental=True), " world")
        self.assertEqual(assembler.text, "Hello world")


class ProviderReasoningSplitTests(unittest.TestCase):
    def test_cursor_thinking_events_are_not_content(self) -> None:
        content = TextAssembler()
        reasoning = TextAssembler()
        thinking = extract_cursor_agent_parts(
            {"type": "thinking", "subtype": "delta", "text": "先读图再回答"},
            content,
            reasoning,
        )
        answer = extract_cursor_agent_parts(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "应改 StreamDownKit"}]},
            },
            content,
            reasoning,
        )
        self.assertEqual(thinking.content, "")
        self.assertEqual(thinking.reasoning, "先读图再回答")
        self.assertEqual(answer.content, "应改 StreamDownKit")
        self.assertEqual(answer.reasoning, "")
        self.assertEqual(content.text, "应改 StreamDownKit")
        self.assertEqual(reasoning.text, "先读图再回答")

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
