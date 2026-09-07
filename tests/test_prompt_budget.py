import unittest

from codex_gateway.openai_compat import (
    ChatMessage,
    EARLIER_CONVERSATION_OMITTED,
    messages_to_prompt,
    trim_messages_to_prompt_budget,
)


class PromptBudgetTests(unittest.TestCase):
    def test_under_budget_is_unchanged(self) -> None:
        messages = [
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="hello"),
        ]
        self.assertIs(trim_messages_to_prompt_budget(messages, 10_000), messages)

    def test_drops_oldest_turns_and_keeps_latest_user(self) -> None:
        messages = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="A" * 400),
            ChatMessage(role="assistant", content="B" * 400),
            ChatMessage(role="user", content="latest question"),
        ]

        trimmed = trim_messages_to_prompt_budget(messages, 80)

        self.assertEqual(trimmed[0].content, "sys")
        self.assertEqual(trimmed[-1].content, "latest question")
        self.assertTrue(any(m.content == EARLIER_CONVERSATION_OMITTED for m in trimmed))
        self.assertLessEqual(len(messages_to_prompt(trimmed)), 80)

    def test_single_huge_message_is_truncated(self) -> None:
        messages = [ChatMessage(role="user", content="Z" * 10_000)]

        trimmed = trim_messages_to_prompt_budget(messages, 200)
        prompt = messages_to_prompt(trimmed)

        self.assertLessEqual(len(prompt), 200)
        self.assertIn("truncated", prompt)


if __name__ == "__main__":
    unittest.main()
