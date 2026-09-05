import unittest

from codex_gateway.claude_oauth import _content_to_anthropic_blocks
from codex_gateway.codex_responses import convert_chat_completions_to_codex_responses
from codex_gateway.openai_compat import (
    ChatCompletionRequest,
    ChatMessage,
    drop_stale_history_images,
    extract_image_urls,
    latest_user_message_has_images,
    prompt_with_attached_image_files,
)


OLD = "data:image/png;base64,AAAOLD"
NEW = "data:image/png;base64,BBBNEW"


def _user_image(url: str, text: str) -> ChatMessage:
    return ChatMessage(
        role="user",
        content=[
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    )


class ImageHistoryTests(unittest.TestCase):
    def test_drop_stale_history_images_keeps_only_latest_user_image(self) -> None:
        messages = [
            _user_image(OLD, "what is this?"),
            ChatMessage(role="assistant", content="an old poster"),
            _user_image(NEW, "now this one"),
        ]

        updated = drop_stale_history_images(messages)

        self.assertEqual(extract_image_urls(updated), [NEW])
        self.assertEqual(updated[0].content, [{"type": "text", "text": "what is this?"}])
        self.assertEqual(updated[2].content[1]["image_url"]["url"], NEW)

    def test_text_only_followup_keeps_earlier_image(self) -> None:
        messages = [
            _user_image(OLD, "what is this?"),
            ChatMessage(role="assistant", content="a cat"),
            ChatMessage(role="user", content="what color is it?"),
        ]

        updated = drop_stale_history_images(messages)
        self.assertEqual(extract_image_urls(updated), [OLD])
        self.assertTrue(latest_user_message_has_images(messages[:1]))
        self.assertFalse(latest_user_message_has_images(updated))

    def test_text_only_followup_is_not_a_new_image_turn(self) -> None:
        messages = [
            _user_image(OLD, "这个呢？"),
            ChatMessage(role="assistant", content="失败现场"),
            ChatMessage(role="user", content="你给我 review 一下"),
        ]
        updated = drop_stale_history_images(messages)
        self.assertTrue(extract_image_urls(updated))
        self.assertFalse(latest_user_message_has_images(updated))

    def test_codex_payload_does_not_forward_previous_user_image(self) -> None:
        req = ChatCompletionRequest(
            model="gpt-5.6-sol",
            messages=drop_stale_history_images(
                [
                    _user_image(OLD, "old"),
                    ChatMessage(role="assistant", content="old answer"),
                    _user_image(NEW, "new"),
                ]
            ),
        )
        payload = convert_chat_completions_to_codex_responses(
            req,
            model_name="gpt-5.6-sol",
            force_stream=False,
        )
        images = [
            part["image_url"]
            for item in payload["input"]
            if isinstance(item, dict)
            for part in item.get("content") or []
            if isinstance(part, dict) and part.get("type") == "input_image"
        ]
        self.assertEqual(images, [NEW])

    def test_extracts_input_image_and_anthropic_source(self) -> None:
        messages = [
            ChatMessage(
                role="user",
                content=[{"type": "input_image", "image_url": NEW}],
            )
        ]
        self.assertEqual(extract_image_urls(messages), [NEW])

        blocks = _content_to_anthropic_blocks(
            [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "BBBNEW"},
                }
            ]
        )
        self.assertEqual(blocks[0]["type"], "image")
        self.assertEqual(blocks[0]["source"]["data"], "BBBNEW")

    def test_extracts_popagent_mimetype_data(self) -> None:
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "这个呢？"},
                    {"type": "image", "mimeType": "image/png", "data": "BBBNEW"},
                ],
            )
        ]
        self.assertEqual(extract_image_urls(messages), [NEW])

        blocks = _content_to_anthropic_blocks(
            [{"type": "image", "mimeType": "image/png", "data": "BBBNEW"}]
        )
        self.assertEqual(blocks[0]["type"], "image")
        self.assertEqual(blocks[0]["source"]["media_type"], "image/png")
        self.assertEqual(blocks[0]["source"]["data"], "BBBNEW")

    def test_prompt_with_attached_image_files_points_at_this_turn(self) -> None:
        prompt = prompt_with_attached_image_files("USER: 这个呢？", ["/tmp/images/user-image-0.png"])
        self.assertIn("USER: 这个呢？", prompt)
        self.assertIn("/tmp/images/user-image-0.png", prompt)
        self.assertIn("Do not search the workspace for older screenshots", prompt)
