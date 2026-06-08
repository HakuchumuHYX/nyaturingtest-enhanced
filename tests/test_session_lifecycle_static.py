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

    def test_set_role_clears_preset_residue(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        set_role_start = session_source.index("async def set_role")
        role_method_start = session_source.index("    def role", set_role_start)
        set_role_source = session_source[set_role_start:role_method_start]

        self.assertIn("self.__aliases = []", set_role_source)
        self.assertIn('self.__examples_str = ""', set_role_source)

    def test_chatting_state_uses_idle_and_can_leave_active(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("IDLE = 0", session_source)
        self.assertNotIn("ILDE", session_source)
        self.assertIn('_ChattingState.ACTIVE\n            and self.willingness < runtime_settings["active_to_bubble_threshold"]', session_source)
        self.assertIn("self.__chatting_state = _ChattingState.BUBBLE", session_source)

    def test_long_term_memory_task_uses_safe_task_wrapper(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        save_memory_index = session_source.index("self.save_long_term_memory")
        task_block = session_source[save_memory_index - 120:save_memory_index + 160]
        self.assertIn("self._create_safe_task", task_block)

    def test_current_chunk_filter_does_not_use_dataclass_equality(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("chunk_message_ids", session_source)
        self.assertIn("m is chunk_msg", session_source)
        self.assertNotIn("if m not in messages_chunk", session_source)


if __name__ == "__main__":
    unittest.main()
