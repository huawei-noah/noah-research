from .base import MindMemOSError


class SkillError(MindMemOSError):
    """Base class for skill version-store errors."""


class SkillBundleError(SkillError):
    """Raised when a skill bundle is empty or contains no whitelisted file.

    The whitelist (see design §4.2) currently only admits ``SKILL.md``; a bundle
    that carries none of the whitelisted files cannot produce a stable
    ``content_hash`` and is rejected here.
    """


class SkillVersionNotFoundError(SkillError):
    """Requested skill version does not exist or is outside the caller's project."""


class SkillNotFoundError(SkillError):
    """Requested cloud skill does not exist or is outside the caller's project."""


class SkillContentNotFoundError(SkillError):
    """A skill version exists but its ``skill_blob`` content row is missing."""


class SkillEditError(SkillError):
    """Raised when a structured SKILL.md edit op cannot be parsed or applied.

    Covers malformed edit JSON, an unknown op, and anchors/targets that do not
    match the current document exactly once. In the evolve pipeline this is
    surfaced to the chat ``format_parser`` so the LLM gets a chance to retry the
    edit with text that actually matches (see ``components.skill.edit``).
    """
