# ruff: noqa: F403
"""Function execution implementations for the AlgorithmAgent."""

# Ensure that all function call handlers are registered
from llm_agent.agents.algorithm_agent.algorithm.exec.base import *  # noqa: I001
from llm_agent.agents.algorithm_agent.algorithm.exec.animation import *
from llm_agent.agents.algorithm_agent.algorithm.exec.contour import *
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import *
from llm_agent.agents.algorithm_agent.algorithm.exec.ensemble import *
from llm_agent.agents.algorithm_agent.algorithm.exec.plot import *
from llm_agent.agents.algorithm_agent.algorithm.exec.sst import *
from llm_agent.agents.algorithm_agent.algorithm.exec.wind import *
