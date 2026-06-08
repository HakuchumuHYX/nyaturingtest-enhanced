from dataclasses import dataclass


@dataclass(frozen=True)
class CommandMeta:
    command: str
    description: str
    private_usage: str = ""


COMMANDS: tuple[CommandMeta, ...] = (
    CommandMeta("autochat <enable/disable>", "在本群启用或禁用 Autochat"),
    CommandMeta("status", "查看 Bot 状态、provider 错误和基础 metrics", "status <群号>"),
    CommandMeta("role", "查看当前角色", "role <群号>"),
    CommandMeta("set_role <角色名> <角色设定>", "设置角色，设定可包含空格", "set_role <群号> <角色名> <角色设定>"),
    CommandMeta("presets", "查看可用预设", "presets <群号>"),
    CommandMeta("set_preset <文件名>", "加载预设", "set_preset <群号> <文件名>"),
    CommandMeta("rag_debug <query>", "诊断 RAG 记忆检索"),
    CommandMeta("calm", "冷静并重置短期状态", "calm <群号>"),
    CommandMeta("reset_emotion", "仅重置 VAD 情绪", "reset_emotion <群号>"),
    CommandMeta("reset confirm", "先备份再完全重置本群", "reset <群号> confirm"),
    CommandMeta("token统计", "查看全部模型 Token 与 DeepSeek cache 统计"),
    CommandMeta("backup_data", "手动触发数据备份", "backup_data"),
    CommandMeta("help", "显示帮助", "help"),
)


def render_group_help() -> str:
    lines = ["可用命令:"]
    for item in COMMANDS:
        if item.command == "help":
            continue
        lines.append(f"- {item.command} - {item.description}")
    return "\n".join(lines)


def render_private_help() -> str:
    lines = ["可用命令(私聊需加群号):"]
    for item in COMMANDS:
        usage = item.private_usage or item.command
        lines.append(f"- {usage} - {item.description}")
    return "\n".join(lines)
