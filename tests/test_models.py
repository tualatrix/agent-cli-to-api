import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from codex_gateway import server
from codex_gateway.model_catalog import (
    DEFAULT_CLAUDE_MODELS,
    DEFAULT_CURSOR_MODELS,
    DEFAULT_GEMINI_MODELS,
    clear_model_cache,
    parse_cursor_model_list,
)
from codex_gateway.openai_compat import ChatCompletionRequest, ChatMessage


def _settings(**overrides):
    values = dict(
        bearer_token=None,
        provider="codex",
        default_model="gpt-5.6-sol",
        cursor_agent_model=None,
        claude_model=None,
        gemini_model=None,
        advertised_models=[],
        model_aliases={},
        allow_client_model_override=False,
        allow_client_provider_override=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class ModelListTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_model_cache()

    def test_codex_provider_advertises_current_codex_models_by_default(self) -> None:
        settings = _settings(allow_client_model_override=True)

        with mock.patch.object(server, "settings", settings):
            result = asyncio.run(server.list_models())

        model_ids = [item["id"] for item in result["data"]]
        self.assertEqual(
            model_ids,
            ["default", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"],
        )

    def test_explicit_advertised_models_override_codex_defaults(self) -> None:
        settings = _settings(
            advertised_models=["custom-model"],
            allow_client_model_override=True,
        )

        with mock.patch.object(server, "settings", settings):
            result = asyncio.run(server.list_models())

        self.assertEqual([item["id"] for item in result["data"]], ["custom-model"])

    def test_cursor_provider_fetches_live_models(self) -> None:
        settings = _settings(provider="cursor-agent", cursor_agent_model="auto")
        live = ["auto", "composer-2.5", "gpt-5.3-codex", "claude-opus-5-thinking-high"]

        with mock.patch.object(server, "settings", settings), mock.patch(
            "codex_gateway.model_catalog._fetch_cursor_models",
            new=mock.AsyncMock(return_value=live),
        ):
            result = asyncio.run(server.list_models())

        model_ids = [item["id"] for item in result["data"]]
        self.assertEqual(
            model_ids[:5],
            ["default", "auto", "composer-2.5", "gpt-5.3-codex", "claude-opus-5-thinking-high"],
        )
        for expected in live:
            self.assertIn(expected, model_ids)

    def test_cursor_provider_falls_back_when_live_fetch_fails(self) -> None:
        settings = _settings(provider="cursor-agent", cursor_agent_model="auto")

        with mock.patch.object(server, "settings", settings), mock.patch(
            "codex_gateway.model_catalog._fetch_cursor_models",
            new=mock.AsyncMock(side_effect=RuntimeError("not logged in")),
        ):
            result = asyncio.run(server.list_models())

        model_ids = [item["id"] for item in result["data"]]
        self.assertEqual(model_ids[0], "default")
        self.assertEqual(model_ids[1], "auto")
        for expected in DEFAULT_CURSOR_MODELS:
            self.assertIn(expected, model_ids)

    def test_claude_and_gemini_advertise_catalog_models(self) -> None:
        claude_settings = _settings(provider="claude", claude_model="sonnet")
        with mock.patch.object(server, "settings", claude_settings), mock.patch(
            "codex_gateway.model_catalog._fetch_claude_models",
            new=mock.AsyncMock(return_value=[]),
        ):
            claude_ids = [item["id"] for item in asyncio.run(server.list_models())["data"]]
        self.assertIn("claude-sonnet-4-6", claude_ids)
        for expected in DEFAULT_CLAUDE_MODELS:
            self.assertIn(expected, claude_ids)

        gemini_settings = _settings(provider="gemini", gemini_model="gemini-3-flash-preview")
        with mock.patch.object(server, "settings", gemini_settings), mock.patch(
            "codex_gateway.model_catalog._fetch_gemini_models",
            new=mock.AsyncMock(return_value=[]),
        ):
            gemini_ids = [item["id"] for item in asyncio.run(server.list_models())["data"]]
        self.assertIn("gemini-3-flash-preview", gemini_ids)
        for expected in DEFAULT_GEMINI_MODELS:
            self.assertIn(expected, gemini_ids)

    def test_parse_cursor_model_list(self) -> None:
        text = "\n".join(
            [
                "Available models",
                "",
                "auto - Auto (current, default)",
                "composer-2.5 - Composer 2.5",
                "gpt-5.3-codex-high - Codex 5.3 High",
            ]
        )
        self.assertEqual(
            parse_cursor_model_list(text),
            ["auto", "composer-2.5", "gpt-5.3-codex-high"],
        )


class ModelRoutingTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_model_cache()

    def test_cursor_honors_advertised_model_without_override_flag(self) -> None:
        settings = _settings(provider="cursor-agent", cursor_agent_model="auto")
        req = ChatCompletionRequest(
            model="composer-2.5",
            messages=[ChatMessage(role="user", content="hi")],
        )
        with mock.patch.object(server, "settings", settings), mock.patch(
            "codex_gateway.model_catalog._fetch_cursor_models",
            new=mock.AsyncMock(return_value=[]),
        ):
            provider, provider_model, requested_model = asyncio.run(server._resolve_request_provider(req))
        self.assertEqual(provider, "cursor-agent")
        self.assertEqual(provider_model, "composer-2.5")
        self.assertEqual(requested_model, "composer-2.5")

    def test_unknown_openai_default_falls_back_to_provider_default(self) -> None:
        settings = _settings(provider="cursor-agent", cursor_agent_model="auto")
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="hi")],
        )
        with mock.patch.object(server, "settings", settings), mock.patch(
            "codex_gateway.model_catalog._fetch_cursor_models",
            new=mock.AsyncMock(return_value=[]),
        ):
            provider, provider_model, requested_model = asyncio.run(server._resolve_request_provider(req))
        self.assertEqual(provider, "cursor-agent")
        self.assertIsNone(provider_model)
        self.assertEqual(requested_model, "auto")

    def test_codex_honors_advertised_model_when_override_disabled(self) -> None:
        settings = _settings(allow_client_model_override=False)
        req = ChatCompletionRequest(
            model="gpt-5.6-terra",
            messages=[ChatMessage(role="user", content="hi")],
        )
        with mock.patch.object(server, "settings", settings):
            provider, provider_model, requested_model = asyncio.run(server._resolve_request_provider(req))
        self.assertEqual(provider, "codex")
        self.assertEqual(provider_model, "gpt-5.6-terra")
        self.assertEqual(requested_model, "gpt-5.6-terra")

    def test_live_only_model_is_honored_after_list(self) -> None:
        settings = _settings(provider="cursor-agent", cursor_agent_model="auto")
        live = ["auto", "composer-2.6-nightly"]
        req = ChatCompletionRequest(
            model="composer-2.6-nightly",
            messages=[ChatMessage(role="user", content="hi")],
        )
        with mock.patch.object(server, "settings", settings), mock.patch(
            "codex_gateway.model_catalog._fetch_cursor_models",
            new=mock.AsyncMock(return_value=live),
        ):
            listed = asyncio.run(server.list_models())
            ids = [item["id"] for item in listed["data"]]
            self.assertIn("composer-2.6-nightly", ids)
            provider, provider_model, requested_model = asyncio.run(server._resolve_request_provider(req))
        self.assertEqual(provider, "cursor-agent")
        self.assertEqual(provider_model, "composer-2.6-nightly")
        self.assertEqual(requested_model, "composer-2.6-nightly")


if __name__ == "__main__":
    unittest.main()
