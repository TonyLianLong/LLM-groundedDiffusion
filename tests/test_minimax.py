"""Tests for MiniMax LLM provider support."""
import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMiniMaxModelNames(unittest.TestCase):
    """Unit tests for MiniMax model name handling."""

    def test_model_names_contains_minimax_m27(self):
        from utils.llm import model_names
        self.assertIn("MiniMax-M2.7", model_names)

    def test_model_names_contains_minimax_m27_highspeed(self):
        from utils.llm import model_names
        self.assertIn("MiniMax-M2.7-highspeed", model_names)

    def test_model_names_contains_minimax_m25(self):
        from utils.llm import model_names
        self.assertIn("MiniMax-M2.5", model_names)

    def test_model_names_contains_minimax_m25_highspeed(self):
        from utils.llm import model_names
        self.assertIn("MiniMax-M2.5-highspeed", model_names)

    def test_existing_models_still_present(self):
        from utils.llm import model_names
        for model in ["gpt-3.5-turbo", "gpt-4", "vicuna-13b"]:
            self.assertIn(model, model_names)


class TestGetFullModelName(unittest.TestCase):
    """Unit tests for get_full_model_name()."""

    def test_minimax_m27_unchanged(self):
        from utils.llm import get_full_model_name
        self.assertEqual(get_full_model_name("MiniMax-M2.7"), "MiniMax-M2.7")

    def test_minimax_m27_highspeed_unchanged(self):
        from utils.llm import get_full_model_name
        self.assertEqual(get_full_model_name("MiniMax-M2.7-highspeed"), "MiniMax-M2.7-highspeed")

    def test_minimax_m25_unchanged(self):
        from utils.llm import get_full_model_name
        self.assertEqual(get_full_model_name("MiniMax-M2.5"), "MiniMax-M2.5")

    def test_minimax_m25_highspeed_unchanged(self):
        from utils.llm import get_full_model_name
        self.assertEqual(get_full_model_name("MiniMax-M2.5-highspeed"), "MiniMax-M2.5-highspeed")

    def test_gpt35_shorthand_still_expands(self):
        from utils.llm import get_full_model_name
        self.assertEqual(get_full_model_name("gpt-3.5"), "gpt-3.5-turbo")


class TestMiniMaxApiKey(unittest.TestCase):
    """Unit tests for api_key.py MiniMax key handling."""

    def test_minimax_api_key_from_env(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key-123"}):
            import importlib
            import utils.api_key as api_key_module
            importlib.reload(api_key_module)
            self.assertEqual(api_key_module.minimax_api_key, "test-minimax-key-123")

    def test_minimax_api_key_default_without_env(self):
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import utils.api_key as api_key_module
            importlib.reload(api_key_module)
            self.assertEqual(api_key_module.minimax_api_key, "YOUR_MINIMAX_API_KEY")


class TestGetLlmKwargsMiniMax(unittest.TestCase):
    """Unit tests for get_llm_kwargs() with MiniMax models."""

    def _get_kwargs(self, model_name, minimax_key="fake-minimax-key"):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": minimax_key}):
            import importlib
            import utils.api_key as api_key_module
            importlib.reload(api_key_module)
            with patch("utils.llm.get_llm_kwargs.__globals__", {}):
                pass
            from utils.llm import get_llm_kwargs
            import utils.llm as llm_module
            # Patch the imported minimax_api_key inside the module
            with patch.object(api_key_module, "minimax_api_key", minimax_key):
                model, kwargs = get_llm_kwargs(model_name, "v0.1")
        return model, kwargs

    def test_minimax_m27_uses_minimax_api_base(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "fake-key"}):
            import importlib, utils.api_key
            importlib.reload(utils.api_key)
            from utils.llm import get_llm_kwargs
            model, kwargs = get_llm_kwargs("MiniMax-M2.7", "v0.1")
        self.assertEqual(kwargs.api_base, "https://api.minimax.io/v1")

    def test_minimax_m27_highspeed_uses_minimax_api_base(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "fake-key"}):
            import importlib, utils.api_key
            importlib.reload(utils.api_key)
            from utils.llm import get_llm_kwargs
            model, kwargs = get_llm_kwargs("MiniMax-M2.7-highspeed", "v0.1")
        self.assertEqual(kwargs.api_base, "https://api.minimax.io/v1")

    def test_minimax_m25_uses_minimax_api_base(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "fake-key"}):
            import importlib, utils.api_key
            importlib.reload(utils.api_key)
            from utils.llm import get_llm_kwargs
            model, kwargs = get_llm_kwargs("MiniMax-M2.5", "v0.1")
        self.assertEqual(kwargs.api_base, "https://api.minimax.io/v1")

    def test_minimax_uses_bearer_auth_header(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "my-minimax-key-xyz"}):
            import importlib, utils.api_key
            importlib.reload(utils.api_key)
            from utils.llm import get_llm_kwargs
            model, kwargs = get_llm_kwargs("MiniMax-M2.7", "v0.1")
        self.assertIn("Authorization", kwargs.headers)
        self.assertIn("my-minimax-key-xyz", kwargs.headers["Authorization"])

    def test_minimax_temperature_valid(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "fake-key"}):
            import importlib, utils.api_key
            importlib.reload(utils.api_key)
            from utils.llm import get_llm_kwargs
            model, kwargs = get_llm_kwargs("MiniMax-M2.7", "v0.1")
        self.assertGreater(kwargs.temperature, 0.0)
        self.assertLessEqual(kwargs.temperature, 1.0)

    def test_minimax_model_name_preserved(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "fake-key"}):
            import importlib, utils.api_key
            importlib.reload(utils.api_key)
            from utils.llm import get_llm_kwargs
            model, kwargs = get_llm_kwargs("MiniMax-M2.7", "v0.1")
        self.assertEqual(model, "MiniMax-M2.7")
        self.assertEqual(kwargs.model, "MiniMax-M2.7")

    def test_openai_still_uses_openai_base(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-openai-key"}):
            import importlib, utils.api_key
            importlib.reload(utils.api_key)
            from utils.llm import get_llm_kwargs
            model, kwargs = get_llm_kwargs("gpt-4", "v0.1")
        self.assertEqual(kwargs.api_base, "https://api.openai.com/v1")


class TestGetLayoutMiniMax(unittest.TestCase):
    """Unit tests for get_layout() using MiniMax (chat completions format)."""

    def _make_chat_response(self, content):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return mock_response

    def test_minimax_uses_chat_completions_endpoint(self):
        from utils.llm import get_layout
        from easydict import EasyDict

        mock_resp = self._make_chat_response("Objects: [('a cat', [50, 50, 100, 100])]\nBackground prompt: A room\nNegative prompt: ")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            kwargs = EasyDict(
                model="MiniMax-M2.7",
                template="{prompt}",
                api_base="https://api.minimax.io/v1",
                max_tokens=900,
                temperature=0.25,
                stop=None,
                headers={"Authorization": "Bearer fake-key"},
            )
            result = get_layout("a cat in a room", kwargs)

        call_url = mock_post.call_args[0][0]
        self.assertIn("chat/completions", call_url)

    def test_minimax_response_parsed_from_message_content(self):
        from utils.llm import get_layout
        from easydict import EasyDict

        expected_content = "Objects: [('a dog', [10, 10, 200, 200])]\nBackground prompt: park\nNegative prompt: "
        mock_resp = self._make_chat_response(expected_content)
        with patch("requests.post", return_value=mock_resp):
            kwargs = EasyDict(
                model="MiniMax-M2.7",
                template="{prompt}",
                api_base="https://api.minimax.io/v1",
                max_tokens=900,
                temperature=0.25,
                stop=None,
                headers={"Authorization": "Bearer fake-key"},
            )
            result = get_layout("a dog in a park", kwargs)

        self.assertEqual(result, expected_content)

    def test_minimax_highspeed_uses_chat_completions(self):
        from utils.llm import get_layout
        from easydict import EasyDict

        mock_resp = self._make_chat_response("Objects: []\nBackground prompt: sky\nNegative prompt: ")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            kwargs = EasyDict(
                model="MiniMax-M2.7-highspeed",
                template="{prompt}",
                api_base="https://api.minimax.io/v1",
                max_tokens=900,
                temperature=0.25,
                stop=None,
                headers={"Authorization": "Bearer fake-key"},
            )
            get_layout("a clear sky", kwargs)

        call_url = mock_post.call_args[0][0]
        self.assertIn("chat/completions", call_url)

    def test_minimax_request_body_contains_model(self):
        from utils.llm import get_layout
        from easydict import EasyDict

        mock_resp = self._make_chat_response("Objects: []\nBackground prompt: beach\nNegative prompt: ")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            kwargs = EasyDict(
                model="MiniMax-M2.5",
                template="{prompt}",
                api_base="https://api.minimax.io/v1",
                max_tokens=900,
                temperature=0.5,
                stop=None,
                headers={"Authorization": "Bearer fake-key"},
            )
            get_layout("a beach", kwargs)

        call_json = mock_post.call_args[1]["json"]
        self.assertEqual(call_json["model"], "MiniMax-M2.5")

    def test_minimax_request_body_contains_messages(self):
        from utils.llm import get_layout
        from easydict import EasyDict

        mock_resp = self._make_chat_response("Objects: []\nBackground prompt: forest\nNegative prompt: ")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            kwargs = EasyDict(
                model="MiniMax-M2.7",
                template="{prompt}",
                api_base="https://api.minimax.io/v1",
                max_tokens=900,
                temperature=0.25,
                stop=None,
                headers={"Authorization": "Bearer fake-key"},
            )
            get_layout("a forest", kwargs)

        call_json = mock_post.call_args[1]["json"]
        self.assertIn("messages", call_json)
        self.assertEqual(call_json["messages"][0]["role"], "user")

    def test_gpt_still_uses_chat_completions(self):
        from utils.llm import get_layout
        from easydict import EasyDict

        mock_resp = self._make_chat_response("Objects: []\nBackground prompt: city\nNegative prompt: ")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            kwargs = EasyDict(
                model="gpt-4",
                template="{prompt}",
                api_base="https://api.openai.com/v1",
                max_tokens=900,
                temperature=0.25,
                stop=None,
                headers={"Authorization": "Bearer fake-openai-key"},
            )
            get_layout("a city", kwargs)

        call_url = mock_post.call_args[0][0]
        self.assertIn("chat/completions", call_url)


class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests for MiniMax API (require MINIMAX_API_KEY env var)."""

    @unittest.skipUnless(os.environ.get("MINIMAX_API_KEY"), "MINIMAX_API_KEY not set")
    def test_minimax_m27_chat_completions_live(self):
        """Live test: verify MiniMax-M2.7 returns a valid chat completion."""
        import requests
        api_key = os.environ["MINIMAX_API_KEY"]
        resp = requests.post(
            "https://api.minimax.io/v1/chat/completions",
            json={
                "model": "MiniMax-M2.7",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 10,
                "temperature": 0.5,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("choices", data)
        self.assertIn("message", data["choices"][0])
        self.assertIn("content", data["choices"][0]["message"])

    @unittest.skipUnless(os.environ.get("MINIMAX_API_KEY"), "MINIMAX_API_KEY not set")
    def test_minimax_m27_highspeed_chat_completions_live(self):
        """Live test: verify MiniMax-M2.7-highspeed returns a valid chat completion."""
        import requests
        api_key = os.environ["MINIMAX_API_KEY"]
        resp = requests.post(
            "https://api.minimax.io/v1/chat/completions",
            json={
                "model": "MiniMax-M2.7-highspeed",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 10,
                "temperature": 0.5,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("choices", data)

    @unittest.skipUnless(os.environ.get("MINIMAX_API_KEY"), "MINIMAX_API_KEY not set")
    def test_get_llm_kwargs_minimax_live_request(self):
        """Integration test: get_llm_kwargs() + get_layout() with MiniMax."""
        import importlib, utils.api_key
        importlib.reload(utils.api_key)
        from utils.llm import get_llm_kwargs, get_layout

        model, kwargs = get_llm_kwargs("MiniMax-M2.7", "v0.1")
        self.assertEqual(kwargs.api_base, "https://api.minimax.io/v1")

        prompt = "A cat sitting on a table"
        response = get_layout(prompt, kwargs)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


if __name__ == "__main__":
    unittest.main()
