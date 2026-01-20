"""
Shared operations for Mind CLI and TUI.

This module provides the core logic for session operations,
separated from presentation concerns.
"""

from dataclasses import dataclass
from typing import Optional

from ..core.session_v2 import SessionV2
from ..core.draft_store import Draft
from ..core.message_store import Message
from ..core.users import User, UserStateEntry, PresenceState, parse_presence_shorthand


# ============================================================================
# Result types
# ============================================================================

@dataclass
class SendMessageResult:
    """Result of sending a message."""
    success: bool
    text: str
    error: Optional[str] = None


@dataclass
class AcceptDraftResult:
    """Result of accepting a draft."""
    success: bool
    draft: Optional[Draft] = None
    error: Optional[str] = None


@dataclass
class UserStateResult:
    """Result of updating user state."""
    success: bool
    user: Optional[User] = None
    state: Optional[UserStateEntry] = None
    error: Optional[str] = None


@dataclass
class SessionStatus:
    """Current session status."""
    session_dir: str
    iteration: int
    thoughts_count: int
    thoughts_active: int
    model: str
    is_drafting: bool
    # Drafting state
    awaiting_text: Optional[str] = None
    awaiting_iter: Optional[int] = None
    drafts_count: int = 0
    latest_draft: Optional[Draft] = None
    # Idle state
    history_count: int = 0
    # User state (active user)
    active_user_id: str = ""
    presence: str = "absent"
    status_text: str = ""
    users: Optional[list[User]] = None
    signal_state: Optional[dict] = None  # {consecutive_hard, threshold}


# ============================================================================
# History resolution (for CLI display)
# ============================================================================

def resolve_history_ref(ref: int, history: list[Message]) -> list[Message]:
    """
    Resolve a history reference to a list of messages.

    Args:
        ref: Either an iteration number (positive) or offset (negative).
             Positive: match single entry by iter number
             -1: latest single entry
             -N (N>1): last N entries
        history: List of messages (oldest first)

    Returns:
        List of matching messages (may be empty, single, or multiple)
    """
    if not history:
        return []

    if ref < 0:
        count = abs(ref)
        if count >= len(history):
            return list(history)
        return list(history[-count:])
    else:
        for entry in history:
            if entry.iter == ref:
                return [entry]
        return []


# ============================================================================
# Core operations
# ============================================================================

def send_message(session: SessionV2, text: str) -> SendMessageResult:
    """
    Send a message to the mind.

    Args:
        session: The session to send to
        text: Message text

    Returns:
        SendMessageResult with success status
    """
    text = text.strip()
    if not text:
        return SendMessageResult(success=False, text="", error="Empty message")

    if session.is_drafting:
        drafts = session.get_all_drafts()
        if drafts:
            return SendMessageResult(
                success=False,
                text=text,
                error=f"Cannot send message while awaiting response ({len(drafts)} draft(s) pending)"
            )
        else:
            return SendMessageResult(
                success=False,
                text=text,
                error="Cannot send message while awaiting response"
            )

    try:
        session.send_message(text)
        session.save()
        return SendMessageResult(success=True, text=text)
    except RuntimeError as e:
        return SendMessageResult(success=False, text=text, error=str(e))


def accept_draft(
    session: SessionV2,
    index: Optional[int] = None,
) -> AcceptDraftResult:
    """
    Accept a draft response by index.

    Args:
        session: The session
        index: Draft index (0-based). None = latest (highest index).

    Returns:
        AcceptDraftResult with the accepted draft
    """
    if not session.is_drafting:
        return AcceptDraftResult(success=False, error="No pending drafts to accept")

    drafts = session.get_all_drafts()
    if not drafts:
        return AcceptDraftResult(success=False, error="No drafts available")

    # Determine index to accept
    if index is None:
        index = drafts[-1].index  # Latest

    # Validate index exists
    valid_indices = [d.index for d in drafts]
    if index not in valid_indices:
        return AcceptDraftResult(
            success=False,
            error=f"Draft {index} not found. Valid: {valid_indices}"
        )

    try:
        accepted = session.accept_draft(index)
        session.save()
        return AcceptDraftResult(success=True, draft=accepted)
    except Exception as e:
        return AcceptDraftResult(success=False, error=str(e))


def mark_drafts_seen(
    session: SessionV2,
    indices: Optional[list[int]] = None,
) -> tuple[int, list[int]]:
    """
    Mark drafts as seen.

    Args:
        session: The session
        indices: List of draft indices (0-based). None = mark all.

    Returns:
        Tuple of (count marked, list of indices marked)
    """
    drafts = session.get_all_drafts()
    if not drafts:
        return 0, []

    if indices is None:
        # Mark all
        count = session.mark_drafts_seen(None)
        session.save()
        return count, [d.index for d in drafts]
    else:
        # Mark specific - validate indices first
        valid_indices = {d.index for d in drafts}
        to_mark = [i for i in indices if i in valid_indices]
        if to_mark:
            count = session.mark_drafts_seen(to_mark)
            session.save()
            return count, to_mark
        return 0, []


def set_user_state(
    session: SessionV2,
    user_id: str,
    presence: Optional[str] = None,
    status: Optional[str] = None,
    name: Optional[str] = None,
) -> UserStateResult:
    """
    Update user state (presence and/or status).

    Creates user if they don't exist.

    Args:
        session: The session
        user_id: User ID to update
        presence: New presence state (absent/reviewing/engaged or a/r/e)
        status: New status text
        name: Optional name (only used when creating user)

    Returns:
        UserStateResult with the updated user and state
    """
    # Parse presence shorthand
    parsed_presence: Optional[PresenceState] = None
    if presence:
        parsed_presence = parse_presence_shorthand(presence)
        if parsed_presence is None:
            return UserStateResult(
                success=False,
                error=f"Invalid presence: {presence}. Valid: absent (a), reviewing (r), engaged (e)"
            )

    # Get or create user
    user = session.user_registry.get_or_create(user_id, name or user_id)

    # Add state entry
    state = user.add_state(
        iter=session.iteration,
        presence=parsed_presence,
        status=status,
    )

    # Update last user and save
    session.user_registry.last_user_id = user_id
    session.save()

    return UserStateResult(success=True, user=user, state=state)


def create_user(
    session: SessionV2,
    user_id: str,
    name: str = "",
) -> UserStateResult:
    """
    Create a new user.

    Args:
        session: The session
        user_id: User ID (must be unique)
        name: Display name (defaults to user_id)

    Returns:
        UserStateResult with the new user
    """
    try:
        user = session.user_registry.create(user_id, name or user_id)
        session.user_registry.last_user_id = user_id
        session.save()
        return UserStateResult(success=True, user=user)
    except ValueError as e:
        return UserStateResult(success=False, error=str(e))


def get_session_status(session: SessionV2) -> SessionStatus:
    """
    Get current session status.

    Args:
        session: The session

    Returns:
        SessionStatus with current state
    """
    # Get users and active user state
    users = session.user_registry.list_users()
    active_user_id = session.user_registry.last_user_id
    presence = "absent"
    status_text = ""

    if active_user_id:
        user = session.user_registry.get(active_user_id)
        if user:
            latest = user.get_latest_state()
            if latest:
                presence = latest.presence
                status_text = latest.status

    status = SessionStatus(
        session_dir=str(session.session_dir),
        iteration=session.iteration,
        thoughts_count=session.thinking_pool.size(),
        thoughts_active=session.thinking_pool.active_size(),
        model=session.config.model,
        is_drafting=session.is_drafting,
        active_user_id=active_user_id,
        presence=presence,
        status_text=status_text,
        users=users,
        signal_state={
            'consecutive_hard': 0,  # Will be set by runner if available
            'threshold': session.config.hard_signal_threshold,
        },
    )

    if session.is_drafting:
        awaiting = session.get_awaiting_message()
        drafts = session.get_all_drafts()
        if awaiting:
            status.awaiting_text = awaiting.text
            status.awaiting_iter = awaiting.iter
        status.drafts_count = len(drafts)
        if drafts:
            status.latest_draft = drafts[-1]
    else:
        history = session.get_history()
        status.history_count = len(history)

    return status
