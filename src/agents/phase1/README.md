# Phase 1: Code Interpreter Agent

## Overview

The Phase 1 Code Interpreter Agent is the "Brain" of the Data Scientist Agent. It operates on a Code Interpreter pattern where it:

1. **Reads an SOP** (Standard Operating Procedure) defining the workflow
2. **Generates Python code** to execute steps using local domain tools
3. **Runs code locally** in a privacy-preserving environment
4. **Updates state** based on execution results

## Architecture

### Privacy-Preserving Data Flow

**Critical Principle:** The LLM (Cloud) NEVER sees the full CSV content.

```
┌─────────────────────────────────────────────────────────────┐
│                         Cloud (LLM)                         │
│  - Sees: data_summary (metadata only)                       │
│  - Sees: columns, dtypes, n_rows, head(5)                   │
│  - Generates: Python code                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Local Execution                          │
│  - Executes: Generated Python code                          │
│  - Accesses: Full CSV data (locally)                        │
│  - Processes: Data using local tools                        │
│  - Returns: Metadata only (via Magic Output Protocol)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Metadata Synchronization                   │
│  - Inspects: Updated CSV locally                            │
│  - Generates: Fresh data_summary                            │
│  - Updates: Phase1State with new metadata                   │
└─────────────────────────────────────────────────────────────┘
```

### Workflow Graph

```
START
  │
  ▼
generate_code (LLM generates Python code)
  │
  ▼
execute_code (Run code locally, parse output)
  │
  ▼
check_execution (Check if execution succeeded)
  │
  ├─ success ──> advance_step (Move to next SOP step)
  │                  │
  │                  ▼
  │              should_continue? ──Yes──> generate_code
  │                  │
  │                  No
  │                  ▼
  │                 END
  │
  ├─ retry ──> reflect_and_fix (LLM analyzes error)
  │                  │
  │                  └──> generate_code (retry with fixed code)
  │
  └─ failed ──> END (max retries reached)
```

**Error Recovery:** The agent automatically retries failed executions up to 3 times (configurable). When code fails, the LLM analyzes the error and generates corrected code.

## Key Components

### 1. State (`state.py`)

**Phase1State** tracks:
- `sop_config`: The loaded SOP JSON
- `current_stage_id`: Current stage in workflow
- `current_step_index`: Current step within stage
- `execution_history`: Log of all executions
- `data_summary`: **CRITICAL** - Metadata only (columns, dtypes, shape, preview)
- `context_variables`: Dynamic variables (is_balanced, protected_columns, etc.)
- `current_data_path`: Local file path to current dataset

**DataSummary** contains:
```python
{
    "columns": ["Sample_ID", "Metabolite_1", ..., "Group"],
    "dtypes": {"Sample_ID": "object", "Metabolite_1": "float64", ...},
    "n_rows": 100,
    "n_cols": 22,
    "head_preview": "   Sample_ID  Metabolite_1  ...",
    "missing_values": {"Metabolite_5": 3, ...},
    "target_column": "Group"
}
```

### 2. Nodes (`nodes.py`)

#### `generate_code` - The Brain

Constructs a prompt for the LLM that includes:
- Current SOP step instruction
- Data summary (metadata only)
- Context variables
- Instructions for Magic Output Protocol

The LLM generates Python code that:
- Uses tools from `src.tools.analysis` and `src.tools.domain`
- Operates on file paths (not DataFrames)
- Prints state updates using Magic Output Protocol

#### `execute_code` - The Executor

1. Extracts generated code from LLM message
2. Saves code to temporary .py file
3. Executes using subprocess with log sanitization
4. Parses stdout for Magic Output Protocol
5. Updates context_variables and current_data_path
6. Refreshes data_summary if data path changed

#### `check_execution` - Execution Router

Routes workflow based on execution result:
- **success**: Execution succeeded → advance to next step
- **retry**: Execution failed but retries available → reflect and fix
- **failed**: Execution failed and max retries reached → end workflow

#### `reflect_and_fix` - Error Recovery

When code execution fails:
1. Analyzes the error message
2. Constructs reflection prompt for LLM
3. Asks LLM to understand error and generate fixed code
4. Updates retry count and error history
5. Loops back to generate_code with corrected approach

#### `get_data_summary` - Metadata Generator

Reads CSV locally and extracts only metadata:
- Column names and types
- Row/column counts
- First 5 rows preview
- Missing value counts

**Privacy Note:** This is the ONLY data information shared with the LLM.

### 3. Magic Output Protocol

Generated code can update state by printing JSON to stdout:

```python
import json

# After calculations...
state_updates = {
    "current_data_path": "/path/to/new_file.csv",
    "is_balanced": True,
    "feature_selection_method": "random_forest"
}

print(json.dumps({"__METABO_UPDATE__": state_updates}))
```

The `execute_code` node parses this and updates the state accordingly.

### 4. Metadata Synchronization

After every successful code execution:

1. Check if `current_data_path` changed
2. If yes, read the new CSV file locally
3. Generate fresh `data_summary`
4. Update state with new metadata

This ensures the LLM always knows:
- Current column names
- Current data shape
- Current data types

## Usage

### Basic Example

```python
from langchain_openai import ChatOpenAI
from src.agents.phase1.graph import create_phase1_agent

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Load SOP
sop = {
    "name": "Data Cleaning",
    "stages": [
        {
            "id": "1",
            "name": "Quality Check",
            "steps": [
                {
                    "id": "1.1",
                    "instruction": "Analyze dataset quality using analyze_dataset_quality tool"
                }
            ]
        }
    ]
}

# Create agent with error recovery (default: 3 retries per step)
graph, initial_state = create_phase1_agent(
    llm=llm,
    sop_config=sop,
    initial_data_path="data/sample.csv",
    target_column="Group",
    max_retries=3  # Optional: configure retry limit
)

# Run agent
result = graph.invoke(initial_state)

# Check results
print(f"Completed: {result['completed']}")
print(f"Steps executed: {len(result['execution_history'])}")
print(f"Final data: {result['current_data_path']}")

# Check if any retries occurred
if result.get('error_history'):
    print(f"Retry attempts: {len(result['error_history'])}")
```

### Error Recovery Example

```python
# The agent automatically handles errors and retries
# No additional code needed - just set max_retries

# Example: More aggressive retry strategy
graph, state = create_phase1_agent(
    llm=llm,
    sop_config=sop,
    initial_data_path="data/sample.csv",
    target_column="Group",
    max_retries=5  # Allow 5 retries per step
)

result = graph.invoke(state)

# Monitor retry behavior
for record in result["execution_history"]:
    if not record["success"]:
        print(f"Step {record['step_id']} failed initially")

for error in result.get("error_history", []):
    print(f"Retry {error['retry_attempt']} for step {error['step_id']}")
```

### Running the Example

```bash
# Run full example (requires OpenAI API key)
python -m src.agents.phase1.example

# Run minimal test (no LLM required)
python -m src.agents.phase1.example --minimal
```

## SOP Configuration

An SOP (Standard Operating Procedure) defines the workflow:

```json
{
    "name": "Data Cleaning Pipeline",
    "description": "Clean and prepare metabolomics data",
    "stages": [
        {
            "id": "1",
            "name": "Data Quality Assessment",
            "steps": [
                {
                    "id": "1.1",
                    "instruction": "Analyze dataset quality. Check for missing values and outliers."
                },
                {
                    "id": "1.2",
                    "instruction": "Remove columns with >50% missing values."
                }
            ]
        },
        {
            "id": "2",
            "name": "Data Cleaning",
            "steps": [
                {
                    "id": "2.1",
                    "instruction": "Impute missing values using median imputation."
                }
            ]
        }
    ]
}
```

## Available Tools

The agent can use tools from:

### Data Analysis Tools (`src.tools.analysis.data_analysis_tools`)
- `analyze_dataset_quality`: Analyze data quality
- `delete_columns_tool`: Remove columns
- `impute_missing_tool`: Impute missing values
- `cap_outliers_tool`: Cap outliers
- `pqn_normalization_tool`: PQN normalization
- `log2_transformation_tool`: Log2 transformation
- `auto_scaling_tool`: Auto scaling

### Feature Selection Tools (`src.tools.analysis.feature_selection_tools`)
- `run_random_forest_selector`: Random forest feature selection
- `run_lasso_selector`: LASSO feature selection
- `run_statistical_selector`: Statistical feature selection
- And 10+ more methods...

### Model Building Tools (`src.tools.analysis.model_building_tools`)
- `analyze_data_for_modeling`: Analyze data for modeling
- `apply_smote_nearmiss`: Apply SMOTE/NearMiss resampling
- `train_with_autogluon`: Train models with AutoGluon

### Domain Tools (`src.tools.domain`)
- `pubmed_retriever`: Search PubMed literature
- `pathway_scorer`: Score metabolic pathways
- `reaction_checker`: Check metabolic reactions

## Privacy Guarantees

1. **No Data Leakage to Cloud:**
   - LLM only sees metadata (columns, types, shape)
   - Full CSV data never leaves local environment

2. **File-Based Operations:**
   - All tools accept file paths, not DataFrames
   - Data processing happens locally

3. **Metadata Only Returns:**
   - Tools return JSON metadata, not raw data
   - Magic Output Protocol ensures controlled state updates

4. **Local Execution:**
   - All code runs in local subprocess
   - No external API calls for data processing

## Security Considerations

1. **Subprocess Isolation:**
   - Code runs in separate process
   - Timeout protection (default: 5 minutes)

2. **Working Directory Restriction:**
   - Code executes in specified working directory
   - File system access limited to data directory

3. **Error Handling:**
   - Execution errors captured and logged
   - Failed executions don't crash the agent

4. **Code Validation:**
   - Syntax validation before execution
   - Import restrictions (future enhancement)

## Future Enhancements

1. **Stronger Sandboxing:**
   - Docker container execution
   - Resource limits (CPU, memory)
   - Network isolation

2. **Import Restrictions:**
   - Whitelist allowed imports
   - Block dangerous modules (os.system, subprocess, etc.)

3. **Advanced Error Recovery:**
   - Error classification by type
   - Specialized reflection prompts per error category
   - Adaptive retry strategies

4. **Human-in-the-Loop:**
   - Optional approval before execution
   - Interactive debugging
   - Manual intervention after max retries

5. **Execution Caching:**
   - Cache execution results
   - Skip redundant computations

## Documentation

- **[Error Recovery Quick Start](../../docs/PHASE1_ERROR_RECOVERY_QUICKSTART.md)** - 5-minute guide to using error recovery
- **[Error Recovery Implementation](../../docs/PHASE1_ERROR_RECOVERY_IMPLEMENTATION.md)** - Full implementation details
- **[Error Recovery Design](../../docs/PHASE1_ERROR_RECOVERY_DESIGN.md)** - Design rationale and architecture
- **[Executor Usage](../../docs/PHASE1_EXECUTOR_USAGE.md)** - How to use the local executor
- **[Workflow Control](../../docs/PHASE1_WORKFLOW_CONTROL.md)** - Understanding workflow navigation

## Testing

```bash
# Run minimal test (no LLM)
python -m src.agents.phase1.example --minimal

# Run unit tests
pytest tests/agents/phase1/

# Run integration tests
pytest tests/agents/phase1/test_integration.py
```

## Troubleshooting

### Issue: "No module named 'sklearn'"

**Solution:** Install scikit-learn:
```bash
pip install scikit-learn
```

### Issue: "Execution timeout"

**Solution:** Increase timeout in executor:
```python
from src.agents.phase1.executor import LocalPythonExecutor
executor = LocalPythonExecutor(timeout=600)  # 10 minutes
```

### Issue: "Magic Output not parsed"

**Solution:** Ensure generated code prints JSON correctly:
```python
import json
print(json.dumps({"__METABO_UPDATE__": {"key": "value"}}))
```

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)
- [MetaboAgent Architecture](../../docs/REFACTORING_VERIFICATION_REPORT.md)
