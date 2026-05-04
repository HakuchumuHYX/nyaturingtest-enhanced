import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class RepositorySplitTests(unittest.TestCase):
    def test_repository_is_split_by_domain_with_compatibility_facade(self):
        expected_modules = [
            "database/session_repository.py",
            "database/message_repository.py",
            "database/profile_repository.py",
            "database/token_repository.py",
            "database/enabled_group_repository.py",
        ]
        for relative in expected_modules:
            self.assertTrue((PLUGIN_DIR / relative).exists(), relative)

        facade = (PLUGIN_DIR / "database" / "repository.py").read_text(encoding="utf-8")
        for class_name in [
            "SessionStateRepository",
            "MessageRepository",
            "ProfileRepository",
            "TokenUsageRepository",
            "EnabledGroupRepository",
        ]:
            self.assertIn(class_name, facade)

    def test_enabled_group_orm_access_goes_through_repository(self):
        state_manager = (PLUGIN_DIR / "core" / "state_manager.py").read_text(encoding="utf-8")
        commands = (PLUGIN_DIR / "handlers" / "commands.py").read_text(encoding="utf-8")
        repository = (PLUGIN_DIR / "database" / "enabled_group_repository.py").read_text(encoding="utf-8")

        self.assertNotIn("EnabledGroupModel", state_manager)
        self.assertNotIn("EnabledGroupModel", commands)
        self.assertIn("EnabledGroupRepository.load_enabled_group_ids", state_manager)
        self.assertIn("EnabledGroupRepository.enable_group", commands)
        self.assertIn("EnabledGroupRepository.disable_group", commands)
        self.assertIn("get_or_create", repository)
        self.assertNotIn("EnabledGroupModel.create(group_id=group_id)", repository)

    def test_production_code_uses_narrow_repositories(self):
        offenders: list[str] = []
        for path in PLUGIN_DIR.rglob("*.py"):
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(PLUGIN_DIR)
            if relative.parts[0] in {"tests", "database"}:
                continue
            text = path.read_text(encoding="utf-8")
            if "database.repository import SessionRepository" in text:
                offenders.append(str(relative))

        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
