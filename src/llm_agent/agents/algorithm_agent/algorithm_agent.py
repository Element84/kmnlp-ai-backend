import json
import logging
from string import Template
from typing import Generic, TypeVar

from e84_geoai_common.llm.core import (
    LLM,
    Agent,
    LLMInferenceConfig,
    LLMMessage,
)
from rich.pretty import pretty_repr

from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    ProblemSpec,
    SolutionSpec,
)

log = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = 10_000


ALGORITHM_AGENT_SYSTEM_PROMPT = Template("""\
You are an LLM specializing in designing algorithms to solve problems and \
describing them using the given specifications and contraints.

The problems to solve will be specified using this schema:
<problem_schema>
```json
${problem_schema}
```
</problem_schema>

You must produce a JSON response satisfying this schema:
<solution_schema>
```json
${solution_schema}
```
</solution_schema>
""")

ALGORITHM_AGENT_PROMPT_TEMPLATE = Template("""\
Design an algorithm to fulfill the request given below.

<problem_spec>
```json
${problem_spec}
```
</problem_spec>
""")

ProblemSpecType = TypeVar("ProblemSpecType", bound=ProblemSpec)
SolutionSpecType = TypeVar("SolutionSpecType", bound=SolutionSpec)


class AlgorithmAgent(Agent, Generic[ProblemSpecType, SolutionSpecType]):
    """Agent that designs algorithms to answer questions using data."""

    def __init__(
        self,
        llm: "LLM",
        *,
        problem_spec: type[ProblemSpecType] = AlgorithmProblemSpec,
        solution_spec: type[SolutionSpecType],
        inference_cfg: LLMInferenceConfig | None = None,
    ) -> None:
        """Constructor.

        Args:
            llm (LLM): An LLM instance.
            problem_spec (type[ProblemSpecType]): Pydantic model describing how
                problems will be specified to the agent. Deafults to
                AlgorithmProblemSpec.
            solution_spec (type[SolutionSpecType]): Pydantic model describing
                how solutions will be specified by the agent.
            inference_cfg (LLMInferenceConfig | None): Inference config.
                Defaults to None.
        """
        self.llm = llm
        self.tools = []
        self.problem_spec = problem_spec
        self.solution_spec = solution_spec
        self.system_prompt = ALGORITHM_AGENT_SYSTEM_PROMPT.safe_substitute(
            problem_schema=pretty_repr(
                self.problem_spec.model_json_schema(), indent_size=2
            ),
            solution_schema=pretty_repr(
                self.solution_spec.model_json_schema(), indent_size=2
            ),
        )
        self._prompt_template: Template = Template(
            ALGORITHM_AGENT_PROMPT_TEMPLATE.safe_substitute()
        )
        if inference_cfg is not None:
            if inference_cfg.system_prompt is not None:
                msg = (
                    "system_prompt should not be set in the provided "
                    "LLMInferenceConfig"
                )
                raise ValueError(msg)
            inference_cfg = inference_cfg.model_copy(
                update={
                    "system_prompt": self.system_prompt,
                    "json_mode": True,
                }
            )
        else:
            inference_cfg = LLMInferenceConfig(
                system_prompt=self.system_prompt,
                temperature=0,
                json_mode=True,
                max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            )
        self.inference_cfg = inference_cfg

    def run(self, problem: ProblemSpecType) -> SolutionSpecType:
        """Construct an algorithm to solve the given problem."""
        prompt = self._prompt_template.safe_substitute(
            problem_spec=pretty_repr(problem.model_dump(), indent_size=2)
        )
        response = self.llm.prompt(
            messages=[LLMMessage(role="user", content=prompt)],
            inference_cfg=self.inference_cfg,
        )
        text = response.to_text_only()
        try:
            solution_json = json.loads(text, strict=False)
        except json.JSONDecodeError:
            log.info(text)
            raise
        solution = self.solution_spec.model_validate(solution_json)
        return solution

    @property
    def prompt_template(self) -> str:
        """The prompt template used by the agent."""
        return self._prompt_template.template
