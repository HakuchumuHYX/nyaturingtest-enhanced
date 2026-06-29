import asyncio
import importlib.util
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PLUGIN_DIR.parents[1]
_MISSING = object()


def _restore_modules(saved):
    for name, module in saved.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_retention_module():
    module_name = "retention_under_test.database.retention"
    package_root = "retention_under_test"
    stub_names = [
        "nonebot",
        package_root,
        f"{package_root}.database",
        f"{package_root}.models",
        f"{package_root}.models.database",
        f"{package_root}.config",
        module_name,
    ]
    saved = {name: sys.modules.get(name, _MISSING) for name in stub_names}
    try:
        for name in stub_names:
            sys.modules.pop(name, None)

        nonebot = types.ModuleType("nonebot")
        nonebot.logger = types.SimpleNamespace(
            debug=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        )
        sys.modules["nonebot"] = nonebot

        for package in [
            package_root,
            f"{package_root}.database",
            f"{package_root}.models",
        ]:
            module = types.ModuleType(package)
            module.__path__ = []
            sys.modules[package] = module

        config = types.ModuleType(f"{package_root}.config")
        config.get_runtime_settings = lambda: {}
        sys.modules[f"{package_root}.config"] = config

        models = types.ModuleType(f"{package_root}.models.database")

        class FakeQuery:
            def __init__(self, model, filters):
                self.model = model
                self.filters = filters

            async def delete(self):
                self.model.deleted_filters.append(self.filters)
                return self.model.delete_count

        class FakeModel:
            deleted_filters = []
            delete_count = 0

            @classmethod
            def filter(cls, **filters):
                return FakeQuery(cls, filters)

        class FakeGlobalMessageModel(FakeModel):
            deleted_filters = []
            delete_count = 3

        class FakeInteractionLogModel(FakeModel):
            deleted_filters = []
            delete_count = 2

        class FakeTokenUsageModel(FakeModel):
            deleted_filters = []
            delete_count = 1

        models.GlobalMessageModel = FakeGlobalMessageModel
        models.InteractionLogModel = FakeInteractionLogModel
        models.TokenUsageModel = FakeTokenUsageModel
        sys.modules[f"{package_root}.models.database"] = models

        spec = importlib.util.spec_from_file_location(module_name, PLUGIN_DIR / "database" / "retention.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, models, saved
    except Exception:
        _restore_modules(saved)
        raise


class DatabaseRetentionTests(unittest.TestCase):
    def test_backup_retention_config_is_count_based_and_documented(self):
        config_source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        example_source = (PLUGIN_DIR / "config.example.json").read_text(encoding="utf-8")
        backup_source = (PLUGIN_DIR / "database" / "backup.py").read_text(encoding="utf-8")
        readme_source = (PLUGIN_DIR / "README.md").read_text(encoding="utf-8")

        self.assertIn('"backup_retention_count": 7', config_source)
        self.assertIn('"backup_retention_count": 7', example_source)
        self.assertIn("DEFAULT_BACKUP_RETENTION_COUNT", backup_source)
        self.assertIn("backup_retention_count", backup_source)
        self.assertNotIn("MAX_BACKUP_DAYS", backup_source)
        self.assertIn("backup_retention_count", readme_source)
        self.assertIn("按数量保留", readme_source)

    def test_backup_docs_call_out_sensitive_backup_contents(self):
        readme_source = (PLUGIN_DIR / "README.md").read_text(encoding="utf-8")

        for snippet in [
            "聊天内容",
            "用户 ID",
            "Token 使用量",
            "向量记忆",
            "外部存储",
            "自动、手动或重置前备份",
        ]:
            self.assertIn(snippet, readme_source)

    def test_raw_retention_config_defaults_to_disabled(self):
        config_source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        example_source = (PLUGIN_DIR / "config.example.json").read_text(encoding="utf-8")

        for snippet in [
            '"raw_message_retention_days": 0',
            '"raw_interaction_retention_days": 0',
            '"token_usage_retention_days": 0',
        ]:
            self.assertIn(snippet, config_source)
            self.assertIn(snippet, example_source)

    def test_raw_retention_deletes_only_enabled_raw_tables(self):
        module, models, saved = _load_retention_module()
        try:
            result = asyncio.run(module.cleanup_raw_data_retention({
                "raw_message_retention_days": 30,
                "raw_interaction_retention_days": 0,
                "token_usage_retention_days": 7,
            }))

            self.assertEqual(3, result["messages"])
            self.assertEqual(0, result["interactions"])
            self.assertEqual(1, result["token_usage"])
            self.assertEqual(1, len(models.GlobalMessageModel.deleted_filters))
            self.assertEqual(0, len(models.InteractionLogModel.deleted_filters))
            self.assertEqual(1, len(models.TokenUsageModel.deleted_filters))
            self.assertIn("time__lt", models.GlobalMessageModel.deleted_filters[0])
            self.assertIn("timestamp__lt", models.TokenUsageModel.deleted_filters[0])
        finally:
            _restore_modules(saved)

    def test_raw_retention_explicit_empty_settings_is_noop(self):
        module, models, saved = _load_retention_module()
        try:
            result = asyncio.run(module.cleanup_raw_data_retention({}))

            self.assertEqual({"messages": 0, "interactions": 0, "token_usage": 0}, result)
            self.assertEqual([], models.GlobalMessageModel.deleted_filters)
            self.assertEqual([], models.InteractionLogModel.deleted_filters)
            self.assertEqual([], models.TokenUsageModel.deleted_filters)
        finally:
            _restore_modules(saved)

    def test_raw_retention_module_does_not_touch_vector_memory(self):
        source = (PLUGIN_DIR / "database" / "retention.py").read_text(encoding="utf-8")

        self.assertNotIn("VectorMemory", source)
        self.assertNotIn("memory.vector", source)
        self.assertNotIn("vector_index", source)


if __name__ == "__main__":
    unittest.main()
