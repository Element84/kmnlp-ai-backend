# AI Agent Instructions for KMNLP LLM Agent System

## Architecture Overview

This is a **geospatial data analysis** system using specialized LLM agents for scientific workflows. The system architecture follows an **agent orchestration pattern** with distinct specialized agents and tools:

- **OrchestrationAgent**: Top-level coordinator that breaks down complex user queries and uses tools to execute multi-step workflows
- **DatasetPickingAgent**: Identifies relevant climate/oceanographic datasets from an RDF knowledge graph
- **AlgorithmAgent**: Generates and executes data analysis algorithms using domain-specific function sets
- **ScientificPythonAgent**: Executes Python code for data analysis tasks

## Key Components & Data Flow

### Agent Orchestration Pattern
```
User Query → OrchestrationAgent → [DataSearchTool, DataAnalysisTool, TextResponseTool] → Response
                ↓
    DataSearchTool → DatasetPickingAgent → SPARQL queries → Dataset metadata
                ↓
    DataAnalysisTool → AlgorithmAgent → Algorithm generation → Execution engine
```

### Tool Integration
- **Agents extend `Agent[LLMAgentContext]`** for domain-specific tasks
- **Tools extend `LLMToolTyped`** for orchestration capabilities
- **Context flows through `LLMAgentContext`** containing place lookup and other shared state
- **Tools use `from_thread.run()` to call async agent methods from sync tool contexts**

### Critical Integration Points
- **`src/llm_agent/tools/data_analysis_tool.py`**: Bridges orchestration to algorithm generation via AlgorithmAgent
- **`src/llm_agent/agents/algorithm_agent/algorithm/exec.py`**: Executes generated algorithms using xarray/matplotlib
- **`src/llm_agent/dao/knowledge_graph_dao.py`**: SPARQL interface to RDF dataset knowledge graph
- **`src/llm_agent/data/dataset-kg-generated.ttl`**: Climate dataset metadata in RDF format

## Development Workflow

### Environment Setup
```bash
uv pip install -r pyproject.toml          # Install dependencies
uv pip install -r pyproject.toml --extra dev  # Install dev dependencies
pre-commit install                         # Install pre-commit hooks
```

### Code Quality Pipeline
- **Linting**: `scripts/lint.sh` runs Ruff, pyright, and shellcheck
- **Testing**: `scripts/test.sh` with pytest (unit tests) and integration tests requiring AWS
- **Evaluation**: `python scripts/run_evals.py` runs comprehensive agent evaluations

### Docker & CI/CD
- **Multi-stage Dockerfile** using UV for dependency management
- **GitLab CI** with build → test → integration → evals pipeline
- **ECR deployment** with role assumption for AWS resources

## Project-Specific Patterns

### Agent Implementation Pattern
```python
class MyAgent(Agent[LLMAgentContext]):
    def __init__(self, llm: LLM, inference_cfg: LLMInferenceConfig | None = None):
        # Set up tools and inference config with system prompt template
        self.inference_cfg = default_cfg.model_copy(update=inference_cfg.model_dump(...))

    async def run_single(self, messages: list[LLMMessage]) -> MyAgentMessage:
        # Single-turn agent execution pattern
```

### Tool Integration Pattern
```python
class MyTool(LLMToolTyped[InputT, UserOutputT, LLMOutputT]):
    def _execute_tool(self, context: LLMAgentContext, inp: InputT) -> ToolOutput | ErrorResponse:
        # Use from_thread.run() to call async agent methods
        out = from_thread.run(self._async_method, context, inp)
```

### Algorithm Function Sets
- **Located in `agents/algorithm_agent/algorithm/function_sets/`**
- **DataArrayFunctionSet**: xarray operations (slicing, filtering, reducing)
- **GeneralPlottingFunctionSet**: matplotlib visualization functions
- **Function sets are dynamically selected based on input variable types**

### Error Handling Convention
- **Return `ErrorResponse` objects rather than raising exceptions** in tool execution
- **Use `EnrichedValidationError` for Pydantic validation with context**
- **Tools wrap outputs in `EnrichedLLMToolResultContent` for debugging**

### Evaluation Framework
- **Located in `src/llm_agent/evaluation/`** with agent-specific evaluators
- **Uses `BaseEvaluator` pattern with typed metrics and aggregation**
- **Evaluation datasets in `evals/datasets/` as Python modules**
- **Integration with `scripts/run_evals.py` for automated evaluation runs**

## Key File Locations

- **Agent implementations**: `src/llm_agent/agents/{agent_name}/`
- **Tool implementations**: `src/llm_agent/tools/`
- **Evaluation code**: `src/llm_agent/evaluation/` and `evals/`
- **Algorithms & execution**: `src/llm_agent/agents/algorithm_agent/algorithm/`
- **Data models**: `src/llm_agent/model/`
- **Knowledge graph**: `src/llm_agent/data/dataset-kg-generated.ttl`

## Testing Notes

- **Unit tests**: Mock LLM responses using `e84_geoai_common.llm.tests.mock_bedrock_runtime`
- **Integration tests**: Require AWS credentials and real LLM calls (marked `@pytest.mark.integration`)
- **Environment variables required**: `GEOCODE_INDEX_HOST`, `GEOCODE_INDEX_PORT`, `GEOCODE_INDEX_REGION`
- **Use `CannedPlaceLookup` for test data in geospatial operations**
