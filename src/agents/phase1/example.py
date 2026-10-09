"""
Phase 1 Agent Example

This script demonstrates how to use the Phase 1 Code Interpreter Agent.

Usage:
    python -m src.agents.phase1.example

Requirements:
    - OpenAI API key in environment or .env file
    - Sample dataset at data/sample_data.csv
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def create_sample_data():
    """Create a sample dataset for testing."""
    import pandas as pd
    import numpy as np
    
    # Create sample metabolomics data
    np.random.seed(42)
    n_samples = 100
    n_metabolites = 20
    
    # Generate data
    data = {
        'Sample_ID': [f'S{i:03d}' for i in range(n_samples)],
        'Group': ['Control'] * 50 + ['Treatment'] * 50
    }
    
    # Add metabolite columns
    for i in range(n_metabolites):
        metabolite_name = f'Metabolite_{i+1}'
        # Control group: mean=10, std=2
        control_values = np.random.normal(10, 2, 50)
        # Treatment group: mean=12, std=2 (slightly higher)
        treatment_values = np.random.normal(12, 2, 50)
        data[metabolite_name] = np.concatenate([control_values, treatment_values])
    
    # Add some missing values (5%)
    df = pd.DataFrame(data)
    for col in df.columns:
        if col not in ['Sample_ID', 'Group']:
            mask = np.random.random(len(df)) < 0.05
            df.loc[mask, col] = np.nan
    
    # Save to CSV
    output_path = project_root / 'data' / 'sample_data.csv'
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Created sample data at {output_path}")
    print(f"  - {n_samples} samples")
    print(f"  - {n_metabolites} metabolites")
    print(f"  - 2 groups (Control, Treatment)")
    
    return str(output_path)


def create_simple_sop():
    """Create a simple SOP for testing."""
    return {
        "name": "Simple Data Cleaning",
        "description": "Basic data cleaning workflow",
        "stages": [
            {
                "id": "1",
                "name": "Quality Check",
                "steps": [
                    {
                        "id": "1.1",
                        "instruction": """Analyze the dataset quality using analyze_dataset_quality tool.
                        Print a summary of missing values and data distribution.
                        Store the analysis results."""
                    }
                ]
            },
            {
                "id": "2",
                "name": "Data Cleaning",
                "steps": [
                    {
                        "id": "2.1",
                        "instruction": """If there are missing values, impute them using median imputation.
                        Use the impute_missing_tool with method='median'.
                        Save the cleaned data to a new file with '_cleaned' suffix.
                        Update the current_data_path using the Magic Output Protocol."""
                    }
                ]
            }
        ]
    }


def run_example():
    """Run the Phase 1 agent example."""
    print("=" * 80)
    print("Phase 1 Code Interpreter Agent - Example")
    print("=" * 80)
    
    # Step 1: Create sample data
    print("\n[Step 1] Creating sample data...")
    data_path = create_sample_data()
    
    # Step 2: Create SOP
    print("\n[Step 2] Creating SOP...")
    sop = create_simple_sop()
    print(f"✓ SOP created: {sop['name']}")
    print(f"  - {len(sop['stages'])} stages")
    print(f"  - {sum(len(s['steps']) for s in sop['stages'])} total steps")
    
    # Step 3: Initialize agent
    print("\n[Step 3] Initializing agent...")
    
    try:
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("✗ Error: OPENAI_API_KEY not found in environment")
            print("  Please set it in .env file or environment variables")
            return
        
        # Create LLM
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        print("✓ LLM initialized (gpt-4)")
        
    except ImportError:
        print("✗ Error: langchain_openai not installed")
        print("  Install with: pip install langchain-openai")
        return
    
    # Step 4: Create agent
    print("\n[Step 4] Creating Phase 1 agent...")
    
    from src.agents.phase1.graph import create_phase1_agent
    
    graph, initial_state = create_phase1_agent(
        llm=llm,
        sop_config=sop,
        initial_data_path=data_path,
        target_column="Group"
    )
    
    print("✓ Agent created")
    print(f"  - Initial data: {initial_state['current_data_path']}")
    print(f"  - Rows: {initial_state['data_summary']['n_rows']}")
    print(f"  - Columns: {initial_state['data_summary']['n_cols']}")
    print(f"  - Target: {initial_state['data_summary']['target_column']}")
    
    # Step 5: Run agent
    print("\n[Step 5] Running agent...")
    print("-" * 80)
    
    try:
        result = graph.invoke(initial_state)
        
        print("-" * 80)
        print("\n✓ Agent execution completed!")
        
        # Print results
        print("\n[Results]")
        print(f"  - Completed: {result['completed']}")
        print(f"  - Error: {result.get('error', 'None')}")
        print(f"  - Steps executed: {len(result['execution_history'])}")
        print(f"  - Final data path: {result['current_data_path']}")
        
        # Print execution history
        print("\n[Execution History]")
        for i, record in enumerate(result['execution_history'], 1):
            print(f"\n  Step {record['step_id']}:")
            print(f"    - Success: {record['success']}")
            print(f"    - Execution time: {record['execution_time']:.2f}s")
            if record['state_updates']:
                print(f"    - State updates: {list(record['state_updates'].keys())}")
            if record['stderr']:
                print(f"    - Errors: {record['stderr'][:100]}...")
        
        # Print final data summary
        print("\n[Final Data Summary]")
        print(f"  - Rows: {result['data_summary']['n_rows']}")
        print(f"  - Columns: {result['data_summary']['n_cols']}")
        print(f"  - Column names: {', '.join(result['data_summary']['columns'][:5])}...")
        
    except Exception as e:
        print(f"\n✗ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)


def run_minimal_test():
    """Run a minimal test without LLM (for testing infrastructure)."""
    print("=" * 80)
    print("Phase 1 Agent - Minimal Test (No LLM)")
    print("=" * 80)
    
    # Create sample data
    print("\n[1] Creating sample data...")
    data_path = create_sample_data()
    
    # Test data summary generation
    print("\n[2] Testing data summary generation...")
    from src.agents.phase1.nodes import get_data_summary
    
    summary = get_data_summary(data_path, "Group")
    print("✓ Data summary generated:")
    print(f"  - Rows: {summary['n_rows']}")
    print(f"  - Columns: {summary['n_cols']}")
    print(f"  - Target: {summary['target_column']}")
    print(f"  - Column names: {', '.join(summary['columns'][:5])}...")
    
    # Test state creation
    print("\n[3] Testing state creation...")
    from src.agents.phase1.state import create_initial_state
    
    sop = create_simple_sop()
    state = create_initial_state(
        sop_config=sop,
        initial_data_path=data_path,
        initial_data_summary=summary,
        target_column="Group"
    )
    
    print("✓ State created:")
    print(f"  - Current stage: {state['current_stage_id']}")
    print(f"  - Current step: {state['current_step_index']}")
    print(f"  - Context variables: {list(state['context_variables'].keys())}")
    
    # Test code execution (simple example)
    print("\n[4] Testing code execution...")
    from src.agents.phase1.executor import LocalPythonExecutor
    
    executor = LocalPythonExecutor(timeout=10)
    
    test_code = """
import json

# Simple test
result = 2 + 2
print(f"Result: {result}")

# Test Magic Output Protocol
state_updates = {
    "test_value": result,
    "test_passed": True
}
print(json.dumps({"__METABO_UPDATE__": state_updates}))
"""
    
    result = executor.execute(test_code)
    print("✓ Code execution test:")
    print(f"  - Success: {result['success']}")
    print(f"  - Execution time: {result['execution_time']:.3f}s")
    print(f"  - Output: {result['stdout'].strip()}")
    
    # Test Magic Output parsing
    print("\n[5] Testing Magic Output Protocol parsing...")
    from src.agents.phase1.nodes import _parse_magic_output
    
    updates = _parse_magic_output(result['stdout'])
    print("✓ Magic Output parsed:")
    print(f"  - Updates: {updates}")
    
    print("\n" + "=" * 80)
    print("Minimal test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1 Agent Example")
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Run minimal test without LLM"
    )
    
    args = parser.parse_args()
    
    if args.minimal:
        run_minimal_test()
    else:
        run_example()
