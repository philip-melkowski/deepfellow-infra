# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Extract the text the router classifies from a chat request."""

from server.models.api import ChatMessage, TextContentPart


def last_user_text(messages: list[ChatMessage]) -> str:
    """Text of the last user message (text parts joined by a newline); empty when there is none."""
    for message in reversed(messages):
        if message.role != "user":
            continue
        content = message.content
        if isinstance(content, str):
            return content
        return "\n".join(part.text for part in content if isinstance(part, TextContentPart))
    return ""
