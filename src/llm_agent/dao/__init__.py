"""Data access objects."""

import os

from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.dao.util import KG_TTL_PATH

__all__ = [
    "KnowledgeGraphDAO",
]

kg_path = os.getenv("KG_PATH", KG_TTL_PATH)
knowledge_graph_dao = KnowledgeGraphDAO(kg_path)
