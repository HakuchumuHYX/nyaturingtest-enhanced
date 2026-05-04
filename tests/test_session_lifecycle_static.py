import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class SessionLifecycleStaticTests(unittest.TestCase):
    def test_session_uses_injected_siliconflow_api_key(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("self._siliconflow_api_key = siliconflow_api_key", source)
        self.assertIn("api_key=self._siliconflow_api_key", source)
        self.assertNotIn('api_key=plugin_config.get("siliconflow_api_key", "")', source)

    def test_background_tasks_are_drained_before_close(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        state_source = (PLUGIN_DIR / "core" / "state_manager.py").read_text(encoding="utf-8")

        self.assertIn("async def drain_background_tasks", session_source)
        self.assertIn("asyncio.wait_for(asyncio.gather", session_source)
        self.assertIn("await state.session.drain_background_tasks", state_source)

        remove_group_state_index = state_source.index("async def remove_group_state")
        cleanup_index = state_source.index("async def cleanup_global_resources")
        remove_group_state_source = state_source[remove_group_state_index:cleanup_index]
        self.assertIn("await state.session.drain_background_tasks", remove_group_state_source)

    def test_append_self_message_accepts_bot_user_id(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("async def append_self_message(self, content: str, msg_id: str, bot_user_id: str)", session_source)
        self.assertIn("user_id=bot_user_id", session_source)
        self.assertNotIn("user_id=self.id # Bot", session_source)


if __name__ == "__main__":
    unittest.main()
