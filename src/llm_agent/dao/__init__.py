"""Data access objects."""

import os

from llm_agent.dao.utils import KG_TTL_PATHS

__all__ = []

kg_paths = _kg_paths.split(";") if (_kg_paths := os.getenv("KG_PATH")) else KG_TTL_PATHS
