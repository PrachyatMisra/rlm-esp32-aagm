"""Focused tests for local retrieval and the Edge-RLM chat adapter."""

import json
import unittest
from unittest.mock import patch

try:
    from . import chat_agent
except ImportError:
    import chat_agent


class ChatAgentTests(unittest.TestCase):
    def test_retrieval_returns_project_sources(self):
        matches = chat_agent.retrieve("int8 quantization and memory footprint")
        self.assertTrue(matches)
        self.assertTrue(any("quant" in item["text"].lower() for item in matches))
        self.assertTrue(all(item["path"] and item["line"] > 0 for item in matches))

    def test_project_question_uses_local_retrieval_when_ollama_disabled(self):
        with patch.object(chat_agent, "CHAT_BACKEND", "local"):
            result = chat_agent.respond("How does recursive halting work?", mode="chat")
        self.assertEqual(result["kind"], "chat")
        self.assertTrue(result["sources"])
        self.assertEqual(result["sources"][0]["path"], "docs/PROJECT_REVIEW_REPORT.md")
        self.assertIn("project files", result["reply"].lower())

    def test_ollama_is_restricted_to_loopback(self):
        with patch.object(chat_agent, "OLLAMA_HOST", "http://example.com:11434"):
            with self.assertRaises(ValueError):
                chat_agent._ollama_reply("hello", [], [])

    def test_local_ollama_receives_conversation_history(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"message":{"content":"The local model answer."}}'

        history = [{"role": "user", "content": "Explain AAGM."}]
        with patch.object(chat_agent, "CHAT_BACKEND", "ollama"):
            with patch.object(chat_agent.urllib.request, "urlopen", return_value=FakeResponse()) as request:
                result = chat_agent.respond("And how does halting fit?", history=history, mode="chat")
        self.assertEqual(result["reply"], "The local model answer.")
        self.assertIn("Ollama", result["engine"])
        payload = json.loads(request.call_args.args[0].data)
        self.assertIn("Explain AAGM.", [message["content"] for message in payload["messages"]])

    def test_sentiment_mode_uses_inference_callback(self):
        calls = []

        def fake_inference(text, budget, profile, batt_mv):
            calls.append((text, budget, profile, batt_mv))
            return {
                "pred": 1,
                "logits": [-0.4, 1.2],
                "steps": 2,
                "mass": 0.98,
                "us": 1234,
                "cpu_mhz": 240,
            }

        result = chat_agent.respond(
            "A sharp, charming film.", mode="analyze", profile="PERF", inference=fake_inference
        )
        self.assertEqual(result["analysis"]["verdict"], "Positive")
        self.assertEqual(result["analysis"]["steps"], 2)
        self.assertEqual(calls, [("A sharp, charming film.", 8, "PERF", 4000.0)])

    def test_unknown_mode_falls_back_to_project_chat(self):
        with patch.object(chat_agent, "CHAT_BACKEND", "local"):
            result = chat_agent.respond("Tell me about the tokenizer.", mode="unexpected")
        self.assertEqual(result["kind"], "chat")
        self.assertTrue(result["sources"])


if __name__ == "__main__":
    unittest.main()
