import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class ProviderCleanupTests(unittest.TestCase):
    def test_python_runtime_has_no_google_ai_studio_route(self):
        forbidden = [
            "google_ai_studio",
            "google_api_key",
            "google_base_url",
            "_request_google",
            "_build_gemini_payload",
            "thinking_config",
            "model_safety_settings",
        ]
        matches: list[str] = []

        for path in PLUGIN_DIR.rglob("*.py"):
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            if path.relative_to(PLUGIN_DIR).parts[0] == "tests":
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    matches.append(f"{path.relative_to(PLUGIN_DIR)}: {token}")

        self.assertEqual([], matches)

    def test_readme_has_no_google_ai_studio_or_gemini_model_docs(self):
        readme = (PLUGIN_DIR / "README.md").read_text(encoding="utf-8")
        forbidden = [
            "Google AI Studio",
            "google_ai_studio",
            "gemini-3",
        ]

        matches = [token for token in forbidden if token in readme]

        self.assertEqual([], matches)


if __name__ == "__main__":
    unittest.main()
