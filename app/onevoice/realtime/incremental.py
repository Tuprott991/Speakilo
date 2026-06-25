"""Incremental ASR text stabilization and commit logic."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_TOKEN_RE = re.compile(r"\S+")


@dataclass(slots=True)
class CommitUpdate:
    """Stable source text that should now be translated."""

    text: str
    committed_text: str


@dataclass(slots=True)
class StableTextBuffer:
    """Two-layer text buffer for partial ASR and stable committed text.

    Partial ASR is allowed to change. Text is committed only when a prefix has
    survived repeated updates and the last few unstable words are held back.
    """

    stable_repeats: int = 2
    holdback_words: int = 2
    min_commit_words: int = 3
    committed_tokens: list[str] = field(default_factory=list)
    _last_uncommitted_tokens: list[str] = field(default_factory=list)
    _repeat_count: int = 0
    latest_partial_text: str = ""

    @property
    def committed_text(self) -> str:
        return " ".join(self.committed_tokens).strip()

    @property
    def has_commits(self) -> bool:
        return bool(self.committed_tokens)

    @property
    def unstable_text(self) -> str:
        return " ".join(self._last_uncommitted_tokens).strip()

    def update_partial(self, text: str) -> CommitUpdate | None:
        """Record one partial ASR update and return newly stable text if any."""
        normalized = _normalize_space(text)
        if not normalized:
            return None
        self.latest_partial_text = normalized

        tokens = _tokenize(normalized)
        uncommitted = self._remaining_after_committed(tokens)
        common = _common_prefix(self._last_uncommitted_tokens, uncommitted)
        if common:
            self._repeat_count += 1
        else:
            self._repeat_count = 1
        self._last_uncommitted_tokens = uncommitted

        if self._repeat_count < self.stable_repeats:
            return None

        commit_len = len(common) - self.holdback_words
        if commit_len < self.min_commit_words:
            return None

        commit_tokens = common[:commit_len]
        self.committed_tokens.extend(commit_tokens)
        self._last_uncommitted_tokens = uncommitted[commit_len:]
        self._repeat_count = 0
        return CommitUpdate(text=" ".join(commit_tokens).strip(), committed_text=self.committed_text)

    def finalize(self, final_text: str) -> CommitUpdate | None:
        """Commit all remaining final text when the utterance endpoint is reached."""
        normalized = _normalize_space(final_text)
        if not normalized:
            return None
        self.latest_partial_text = normalized
        tokens = _tokenize(normalized)
        remaining = self._remaining_after_committed(tokens)
        if not remaining:
            return None
        self.committed_tokens.extend(remaining)
        self._last_uncommitted_tokens = []
        self._repeat_count = 0
        return CommitUpdate(text=" ".join(remaining).strip(), committed_text=self.committed_text)

    def _remaining_after_committed(self, tokens: list[str]) -> list[str]:
        if not self.committed_tokens:
            return tokens
        prefix_len = len(self.committed_tokens)
        if _tokens_equal(tokens[:prefix_len], self.committed_tokens):
            return tokens[prefix_len:]

        committed_text = self.committed_text.lower()
        full_text = " ".join(tokens).lower()
        if committed_text and full_text.startswith(committed_text):
            return _tokenize(full_text[len(committed_text) :].strip())

        # ASR revised earlier words. Avoid destructive rollback; hold all text
        # as unstable until finalization resolves it.
        return tokens


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return [match.group(0) for match in _TOKEN_RE.finditer(_normalize_space(text))]


def _common_prefix(left: list[str], right: list[str]) -> list[str]:
    out = []
    for a, b in zip(left, right):
        if a.lower() != b.lower():
            break
        out.append(b)
    return out


def _tokens_equal(left: list[str], right: list[str]) -> bool:
    return len(left) == len(right) and all(a.lower() == b.lower() for a, b in zip(left, right))
