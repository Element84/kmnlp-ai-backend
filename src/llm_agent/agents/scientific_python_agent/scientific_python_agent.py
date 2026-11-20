import logging
from collections.abc import Callable
from string import Template
from typing import Any

from langfuse import observe  # type: ignore[reportUnknownVariableType]
from pydantic_ai import Agent, TextOutput
from pydantic_ai.models import Model
from pydantic_ai.models.bedrock import BedrockModelSettings

from llm_agent.agents.scientific_python_agent.utils import (
    ExecOutput,
    LLMCodeSolution,
)
from llm_agent.agents.scientific_python_agent.utils.agent_code_utils import (
    compute,
    filter_data_array,
    geocode_natural_language_description,
    geographic_coords_to_decimal_coords,
    mask_data_array_with_polygon,
    max_coords,
    min_coords,
    polygon_to_mask,
)
from llm_agent.agents.scientific_python_agent.utils.misc import (
    get_function_def,
    parse_code_solution,
)
from llm_agent.utils.utils import get_current_date

log = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = 2_000
DEFAULT_UTILS = [
    compute,
    geographic_coords_to_decimal_coords,
    filter_data_array,
    max_coords,
    min_coords,
    geocode_natural_language_description,
    polygon_to_mask,
    mask_data_array_with_polygon,
]

PYTHON_VERSION = 3.12

SCIENTIFIC_PYTHON_SYSTEM_PROMPT_TEMPLATE = Template("""\
You are an LLM specializing in writing scientific Python code to extract \
insights from N-dimensional data using libraries such as xarray.

The current date is: ${current_date}.

The code you write will be executed in a Python ${python_version} environment \
where the code below, inside `<setup-code></setup-code>` tags, has already \
been executed and its output is provided below inside \
`<setup-output></setup-output>` tags. Assume that your \
code will have access to everything defined or imported in the setup code.

Your output should comprise the following 3 parts:
1. Your understanding of the problem and plan of attack, enclosed inside \
`<thoughts></thoughts>` tags.
2. Your code for solving the problem, formatted like so:
<code>
```python
# your code here
```
</code>
Use the pre-defined utils listed inside `<utils></utils>` tags where possible.
3. Explanation for any parts of the code that are non-obvious, enclosed \
inside `<explanation></explanation>` tags. Don't be verbose; if the code is \
simple, keep this brief.

If the request is malformed or unrelated to the data or if the \
problem is not solvable, explain why in the `<thoughts></thoughts>` section \
and leave `<code></code>` and `<explanation></explanation>` empty. \
Similarly, if the request is a simple question that can be answered from the \
information in the setup code or setup output, without writing any code, \
answer it in the `explanation` section and leave the `code` section blank.

Important notes and tips:
- Remember to include import statements for any modules used in the code.
- Do NOT call `.compute()` directly on xarray objects. Use the \
provided `compute()` util function instead.
- Never call `.item()` unless you've already called `compute()` on the data \
array.
- Remember to call `compute()` before using a `DataArray` for plots.
- Always use `plt.show()` to display plots. Never save them to file.
- If printing out an answer in your code, prefix each line with "$$ANSWER$$". \
E.g. `print("$$ANSWER$$", "The max SST on Jan 01, 2023 was ...")`. This only \
applies to `<code></code>` blocks.

<setup-code>
```python
${setup_code}
```
</setup-code>

<setup-output>
${setup_output}
</setup-output>

<utils>
```python
${utils}
```
</utils>
""")

SCIENTIFIC_PYTHON_PROMPT_TEMPLATE = Template("""\
Write Python code to fulfill the request given below.

<request>
${query}
</request>
""")


class ScientificPythonAgent:
    """Writes scientific Python code to extract insights from data.

    Legacy agent, lightly migrated to wrap a Pydantic AI agent. The model emits a
    free-form text response containing ``<thoughts>``/``<code>``/``<explanation>``
    blocks, which ``parse_code_solution`` turns into an ``LLMCodeSolution`` via a
    ``TextOutput`` output function.
    """

    utils: list[Callable[..., Any]] = DEFAULT_UTILS

    def __init__(
        self,
        model: Model,
        *,
        setup_code: str = "",
        setup_output: str | ExecOutput = "",
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        """Constructor.

        Args:
            model: The Pydantic AI model to drive the agent. Build one with
                ``llm_agent.model_factory.make_bedrock_model``.
            setup_code: Code that sets up the Python environment where the LLM's
                code will be run. Defaults to "".
            setup_output: Output from running ``setup_code``. Defaults to "".
            max_output_tokens: Maximum number of output tokens.
        """
        self.setup_code = setup_code
        self.setup_output = str(setup_output)
        utils_str = "\n\n".join(get_function_def(func) for func in self.utils)
        self.system_prompt = SCIENTIFIC_PYTHON_SYSTEM_PROMPT_TEMPLATE.substitute(
            python_version=PYTHON_VERSION,
            setup_code=self.setup_code,
            setup_output=self.setup_output,
            utils=utils_str,
            current_date=get_current_date(),
        )
        self._prompt_template = SCIENTIFIC_PYTHON_PROMPT_TEMPLATE
        self._agent = Agent(
            model,
            output_type=TextOutput(parse_code_solution),
            instructions=self.system_prompt,
            model_settings=BedrockModelSettings(
                temperature=0,
                max_tokens=max_output_tokens,
                bedrock_cache_instructions=True,
            ),
        )

    @observe(as_type="agent", name=__name__)
    async def run(self, query: str) -> LLMCodeSolution:
        """Translate query into Python code."""
        prompt_str = self._prompt_template.substitute(query=query)
        result = await self._agent.run(prompt_str)
        return result.output

    @property
    def prompt_template(self) -> str:
        """The prompt template used by the agent."""
        return self._prompt_template.template
