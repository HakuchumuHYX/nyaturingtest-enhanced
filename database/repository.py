from .enabled_group_repository import EnabledGroupRepository
from .message_repository import MessageRepository
from .profile_repository import ProfileRepository
from .session_repository import SessionStateRepository
from .token_repository import TokenUsageRepository


class SessionRepository(
    SessionStateRepository,
    MessageRepository,
    ProfileRepository,
    TokenUsageRepository,
    EnabledGroupRepository,
):
    """
    Backward-compatible facade for existing callers.

    New code should import the narrower repository class for its domain.
    """

    pass
