"""Skill-related algorithm config (one module per skill capability).

Each skill capability keeps its tuning in its own module (e.g. ``evolve.py``);
add new ones here and re-export them.
"""

from .evolve import SkillEvolutionConfig

__all__ = ["SkillEvolutionConfig"]
