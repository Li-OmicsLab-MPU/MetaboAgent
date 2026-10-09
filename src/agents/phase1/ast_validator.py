"""
AST-based Tool Call Validator

This module provides pre-execution validation of generated code using Abstract Syntax Tree (AST) analysis.
It catches common errors BEFORE execution, saving time and reducing retry cycles.

Key Features:
1. Syntax validation
2. Tool call verification (function exists, correct parameters)
3. Import statement validation
4. Type hint checking (basic)

This is P0-3 of the architecture improvements.
"""

import ast
import json
from typing import Dict, Any, List, Optional, Set, Tuple


class ToolCallValidator:
    """
    Validates generated code using AST analysis.
    
    This validator catches errors before execution:
    - Syntax errors
    - Unknown tool calls
    - Wrong parameter names
    - Missing imports
    """
    
    def __init__(self, tool_contracts: Dict[str, Any]):
        """
        Initialize validator with tool contracts.
        
        Args:
            tool_contracts: Dictionary of tool contracts from tool_contracts.py
        """
        self.tool_contracts = tool_contracts
        
        # Build quick lookup maps
        self.tool_names = set(tool_contracts.keys())
        self.tool_params = {}
        self.tool_class_names = {
            tool_name
            for tool_name in self.tool_names
            if any(name.startswith(f"{tool_name}.") for name in self.tool_names)
        }
        
        for tool_name, contract in tool_contracts.items():
            if "standard" in contract and isinstance(contract["standard"], dict):
                self.tool_params[tool_name] = set(contract["standard"]["params"].keys())
        
        # Column access validation configuration
        self.enable_column_validation = True  # Can be disabled if needed
        self.dataframe_var_names = {'df', 'data', 'result_df', 'diff_df', 'feature_df', 
                                     'train_data', 'test_data', 'X', 'y'}
    
    def validate(self, code: str, step_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate generated code.
        
        Args:
            code: Python code to validate
            step_id: Optional step ID for context-aware validation
        
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if validation passed
            - error_message: None if valid, error description if invalid
        """
        # Step 1: Syntax validation
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax Error at line {e.lineno}: {e.msg}\n{e.text}"

        security_error = self._validate_security_policy(tree)
        if security_error:
            return False, security_error
        
        # Step 2: Extract imports, local definitions and tool calls
        imports = self._extract_imports(tree)
        local_functions = self._extract_local_functions(tree)
        tool_calls = self._extract_tool_calls(tree, local_functions)
        
        # Step 3: Validate tool calls
        for tool_call in tool_calls:
            # Check if tool exists
            if tool_call["name"] not in self.tool_names:
                direct_function_fix = self._detect_direct_function_misuse(tool_call["name"])
                if direct_function_fix:
                    return False, direct_function_fix
                # Check if it's a known tool with typo
                similar = self._find_similar_tool(tool_call["name"])
                if similar:
                    return False, f"Tool '{tool_call['name']}' not found. Did you mean '{similar}'?"
                else:
                    # Not a registered tool, might be stdlib or sklearn - skip validation
                    continue
            
            # Check if tool is imported
            tool_module = self.tool_contracts[tool_call["name"]]["module"]
            
            # For method-level contracts (e.g., "ReactionCheckerTool.verify_pair"),
            # check if the class is imported instead of the method
            if "." in tool_call["name"]:
                # Extract class name from "ClassName.method_name"
                class_name = tool_call["name"].split(".")[0]
                if class_name in self.tool_names:
                    # Check if class is imported
                    if not self._is_tool_imported(class_name, tool_module, imports):
                        return False, f"Tool '{class_name}' is called but not imported. Add: from {tool_module} import {class_name}"
            else:
                # Direct function call
                if not self._is_tool_imported(tool_call["name"], tool_module, imports):
                    return False, f"Tool '{tool_call['name']}' is called but not imported. Add: from {tool_module} import {tool_call['name']}"
            
            # Check parameters (only for tools with known signatures)
            if tool_call["name"] in self.tool_params:
                expected_params = self.tool_params[tool_call["name"]]
                actual_params = set(tool_call["kwargs"])
                
                # Check for unexpected parameters
                unexpected = actual_params - expected_params
                if unexpected:
                    return False, f"Tool '{tool_call['name']}' got unexpected parameter(s): {', '.join(unexpected)}. Expected: {', '.join(expected_params)}"
        
        # Step 4: Check for unsafe DataFrame column access (NEW)
        if self.enable_column_validation:
            unsafe_accesses = self._detect_unsafe_column_access(tree, code)
            if unsafe_accesses:
                return False, self._format_column_access_error(unsafe_accesses)
        
        # Step 5: Check for temporary feature file output format (NEW)
        # Pass step_id for context-aware validation
        temp_file_warnings = self._validate_temp_feature_files(code, step_id)
        if temp_file_warnings:
            return False, temp_file_warnings
        
        # All validations passed
        return True, None

    def _validate_security_policy(self, tree: ast.AST) -> Optional[str]:
        """Reject network, process-spawning and dynamic-code primitives."""
        denied_modules = {
            "aiohttp", "asyncio", "ctypes", "ftplib", "http", "importlib",
            "paramiko", "requests", "smtplib", "socket", "subprocess",
            "telnetlib", "urllib", "webbrowser",
        }
        denied_builtins = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}
        denied_calls = {
            "os.system", "os.popen", "os.spawnl", "os.spawnle", "os.spawnlp",
            "os.spawnlpe", "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
            "os.execl", "os.execle", "os.execlp", "os.execlpe", "os.execv",
            "os.execve", "os.execvp", "os.execvpe", "os.chdir", "shutil.rmtree",
        }

        def _qualified_name(node: ast.AST) -> str:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                parent = _qualified_name(node.value)
                return f"{parent}.{node.attr}" if parent else node.attr
            return ""

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in denied_modules:
                        return f"Security policy: import '{alias.name}' is not allowed in generated code"
            elif isinstance(node, ast.ImportFrom):
                root = str(node.module or "").split(".", 1)[0]
                if root in denied_modules:
                    return f"Security policy: import from '{node.module}' is not allowed in generated code"
            elif isinstance(node, ast.Call):
                call_name = _qualified_name(node.func)
                if call_name in denied_builtins or call_name in denied_calls:
                    return f"Security policy: call '{call_name}(...)' is not allowed"
                root = call_name.split(".", 1)[0]
                if root in denied_modules:
                    return f"Security policy: external/network call '{call_name}(...)' is not allowed"

        return None

    def _detect_direct_function_misuse(self, tool_name: str) -> Optional[str]:
        """
        Detect a common LLM mistake where a function tool is treated like an
        object and called as `tool.get(...)`.
        """
        direct_function_tools = {
            "generate_reaction_ratios",
            "generate_taxonomy_sums",
            "generate_pathway_scores",
        }

        if not tool_name.endswith(".get"):
            return None

        base_name = tool_name[:-4]
        if base_name not in direct_function_tools:
            return None

        return (
            f"Tool '{tool_name}' not found because '{base_name}' is a DIRECT FUNCTION, "
            f"not an object or dict. Call it exactly as `{base_name}(...)`, then inspect "
            f"the returned dict via `result['status']` or `result.get('status')`. "
            f"Do NOT write `{base_name}.get(...)`."
        )
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Extract all import statements from AST.
        
        Returns:
            List of import info dicts: {type: 'import'|'from', module: str, names: List[str]}
        """
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "names": [alias.asname if alias.asname else alias.name]
                    })
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                names = [alias.name for alias in node.names]
                imports.append({
                    "type": "from",
                    "module": module,
                    "names": names
                })
        
        return imports
    
    def _extract_tool_calls(self, tree: ast.AST, local_functions: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract all function calls that might be tool calls.
        
        Returns:
            List of call info dicts: {name: str, args: List, kwargs: List[str]}
        """
        # First, extract tool instances (e.g., checker = ReactionCheckerTool())
        tool_instances = self._extract_tool_instances(tree)
        local_functions = local_functions or set()
        
        calls = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Get function name
                func_name = None
                
                if isinstance(node.func, ast.Name):
                    # Direct function call: delete_columns_tool(...)
                    func_name = node.func.id
                    if func_name in local_functions:
                        continue
                elif isinstance(node.func, ast.Attribute):
                    # Method call: obj.method(...)
                    if isinstance(node.func.value, ast.Name):
                        var_name = node.func.value.id
                        method_name = node.func.attr
                        
                        # Check if this is a known tool instance
                        if var_name in tool_instances:
                            # This is a tool instance method call
                            # Example: checker.verify_pair(...) where checker = ReactionCheckerTool()
                            tool_class = tool_instances[var_name]
                            func_name = f"{tool_class}.{method_name}"
                        elif var_name in {
                            "generate_reaction_ratios",
                            "generate_taxonomy_sums",
                            "generate_pathway_scores",
                        }:
                            # Some LLM outputs incorrectly treat direct function tools
                            # as objects, e.g. generate_reaction_ratios.get(...).
                            func_name = f"{var_name}.{method_name}"
                        elif var_name.islower():
                            # Not a tool instance, skip validation
                            # Example: df.merge(...), scaler.fit_transform(...)
                            continue
                        else:
                            # Class method call (uppercase names)
                            # Example: MyClass.static_method()
                            func_name = f"{var_name}.{method_name}"
                    else:
                        func_name = node.func.attr
                
                if func_name:
                    # Extract keyword argument names
                    kwargs = [kw.arg for kw in node.keywords if kw.arg]
                    
                    calls.append({
                        "name": func_name,
                        "args": len(node.args),
                        "kwargs": kwargs
                    })
        
        return calls

    def _extract_local_functions(self, tree: ast.AST) -> Set[str]:
        """
        Extract names of locally defined functions so the validator does not
        mistake helper functions for registered tools.
        """
        local_functions: Set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local_functions.add(node.name)

        return local_functions
    
    def _extract_tool_instances(self, tree: ast.AST) -> Dict[str, str]:
        """
        Extract tool instance variable names to class names mapping.
        
        Example: checker = ReactionCheckerTool() -> {"checker": "ReactionCheckerTool"}
        
        Returns:
            Dictionary mapping variable names to tool class names
        """
        instances = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Check assignment statements
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    
                    # Check if right side is a tool class instantiation
                    if isinstance(node.value, ast.Call):
                        if isinstance(node.value.func, ast.Name):
                            class_name = node.value.func.id
                            # Only treat real tool classes as instances.
                            # Direct function tools such as generate_reaction_ratios(...)
                            # return plain values and must not be tracked as objects.
                            if class_name in self.tool_class_names:
                                instances[var_name] = class_name
        
        return instances
    
    def _is_tool_imported(self, tool_name: str, tool_module: str, imports: List[Dict]) -> bool:
        """
        Check if a tool is properly imported.
        
        Args:
            tool_name: Name of the tool
            tool_module: Expected module path
            imports: List of import statements
        
        Returns:
            True if tool is imported correctly
        """
        for imp in imports:
            if imp["type"] == "from":
                # Check if module matches and tool name is in imported names
                if tool_module in imp["module"] and tool_name in imp["names"]:
                    return True
                # Also check for wildcard imports
                if tool_module in imp["module"] and "*" in imp["names"]:
                    return True
        
        return False
    
    def _find_similar_tool(self, tool_name: str) -> Optional[str]:
        """
        Find similar tool name (for typo suggestions).
        
        Uses simple string similarity (Levenshtein distance).
        """
        from difflib import get_close_matches
        
        matches = get_close_matches(tool_name, self.tool_names, n=1, cutoff=0.6)
        return matches[0] if matches else None
    
    def _detect_unsafe_column_access(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """
        Detect unsafe DataFrame column access patterns.
        
        Detects patterns like:
        1. df['column_name'] - Direct subscript with hardcoded string
        2. df[col_var] - Subscript with variable (without prior check)
        
        Safe patterns (not flagged):
        1. if 'column' in df.columns: df['column']
        2. df.get('column', default)
        3. df.iloc[:, 0] - Position-based access
        4. df[df.columns[0]] - Dynamic column name
        
        Args:
            tree: AST tree
            code: Original code string (for context)
        
        Returns:
            List of unsafe access info: {line: int, column: str, pattern: str}
        """
        unsafe_accesses = []
        code_lines = code.split('\n')
        
        # Check if code has column existence checks
        has_column_check = self._has_column_existence_check(code)
        
        for node in ast.walk(tree):
            # Detect pattern: df['column_name']
            if isinstance(node, ast.Subscript):
                # Check if it's a DataFrame variable
                if isinstance(node.value, ast.Name):
                    var_name = node.value.id
                    
                    # Check if it's a common DataFrame variable name
                    if var_name in self.dataframe_var_names:
                        # Check if it's a string literal (hardcoded column name)
                        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                            column_name = node.slice.value
                            
                            # Skip if it's a safe pattern
                            if self._is_safe_column_access(node, column_name, code_lines, has_column_check):
                                continue
                            
                            unsafe_accesses.append({
                                'line': node.lineno,
                                'variable': var_name,
                                'column': column_name,
                                'pattern': f"{var_name}['{column_name}']"
                            })
        
        return unsafe_accesses
    
    def _has_column_existence_check(self, code: str) -> bool:
        """
        Check if code contains column existence checks.
        
        Looks for patterns like:
        - 'column' in df.columns
        - if 'column' in df.columns:
        """
        check_patterns = [
            ' in df.columns',
            ' in data.columns',
            ' in diff_df.columns',
            '.get(',
        ]
        
        return any(pattern in code for pattern in check_patterns)
    
    def _is_safe_column_access(self, node: ast.Subscript, column_name: str, 
                                code_lines: List[str], has_column_check: bool) -> bool:
        """
        Check if a column access is in a safe context.
        
        Safe contexts:
        1. Protected columns (Sample_ID, Group) - guaranteed to exist
        2. After a column existence check
        3. Inside a try-except block
        4. Using .get() method
        
        Args:
            node: AST Subscript node
            column_name: Column name being accessed
            code_lines: Lines of code
            has_column_check: Whether code has any column checks
        
        Returns:
            True if access is safe
        """
        # SAFE: Access to protected columns
        # These columns are guaranteed to exist throughout the pipeline
        # and don't need existence checks
        PROTECTED_COLUMNS = [
            'Sample_ID',      # Sample identifier column
            'Group',          # Target/group column
            'target',         # Common target column name
            'sample_id_column',  # Alternative sample ID name
            'target_column',  # Alternative target column name
        ]
        
        if column_name in PROTECTED_COLUMNS:
            return True
        
        # If code has column checks, assume it's safe (conservative)
        # This avoids false positives when checks are present
        if has_column_check:
            # Check if this specific column is checked
            line_num = node.lineno - 1
            
            # Look at previous lines for column check
            for i in range(max(0, line_num - 5), line_num):
                if i < len(code_lines):
                    line = code_lines[i]
                    # Check if this column is specifically checked
                    if f"'{column_name}' in" in line or f'"{column_name}" in' in line:
                        return True
        
        return False
    
    def _format_column_access_error(self, unsafe_accesses: List[Dict]) -> str:
        """
        Format column access error message with helpful suggestions.
        
        Args:
            unsafe_accesses: List of unsafe access info
        
        Returns:
            Formatted error message
        """
        error_lines = []
        error_lines.append("⚠️  Unsafe DataFrame column access detected!")
        error_lines.append("")
        error_lines.append("Hardcoded column names can cause KeyError if the column doesn't exist.")
        error_lines.append("")
        
        # Show first 3 unsafe accesses
        for access in unsafe_accesses[:3]:
            error_lines.append(f"  Line {access['line']}: {access['pattern']}")
        
        if len(unsafe_accesses) > 3:
            error_lines.append(f"  ... and {len(unsafe_accesses) - 3} more")
        
        error_lines.append("")
        error_lines.append("📝 Recommended fixes:")
        error_lines.append("")
        error_lines.append("  Option 1: Check column existence first")
        error_lines.append("  ----------------------------------------")
        error_lines.append("  if 'column_name' in df.columns:")
        error_lines.append("      value = df['column_name']")
        error_lines.append("")
        error_lines.append("  Option 2: Try multiple possible column names")
        error_lines.append("  ---------------------------------------------")
        error_lines.append("  for col in ['Feature', 'metabolite_name', 'Metabolite']:")
        error_lines.append("      if col in df.columns:")
        error_lines.append("          value = df[col]")
        error_lines.append("          break")
        error_lines.append("")
        error_lines.append("  Option 3: Use first column as fallback")
        error_lines.append("  ---------------------------------------")
        error_lines.append("  if 'Feature' in df.columns:")
        error_lines.append("      value = df['Feature']")
        error_lines.append("  elif len(df.columns) > 0:")
        error_lines.append("      value = df.iloc[:, 0]  # Use first column")
        
        return "\n".join(error_lines)
    
    def _validate_temp_feature_files(self, code: str, step_id: Optional[str] = None) -> Optional[str]:
        """
        Validate that temporary feature files don't include Group column.
        
        This prevents merge conflicts when integrating features in step 1.5.4.
        
        Args:
            code: Python code to validate
            step_id: Optional step ID for context-aware validation
            
        Returns:
            Error message if validation fails, None otherwise
        """
        # CRITICAL: Step 1.5.4 is the FINAL INTEGRATION step
        # It MUST include the Group column from the base dataset
        # Only temporary feature generation steps (1.5.1, 1.5.2, 1.5.3) should exclude Group
        if step_id == "1.5.4":
            # This is the integration step - allow Group column
            return None
        
        # Check if this is a temporary feature file generation step
        temp_files = [
            'temp_ratios.csv',
            'temp_sums.csv',
            'temp_pathway_scores.csv',
            'temp_pathways.csv',  # legacy compatibility
        ]
        
        # Check if code is saving to any temp file
        is_temp_file_step = any(temp_file in code for temp_file in temp_files)
        
        if not is_temp_file_step:
            return None
        
        # Pattern 1: Check for explicit Group column inclusion in dataframe creation
        # e.g., output_df = data[['Sample_ID', 'Group']].copy()
        if "[['Sample_ID', 'Group']]" in code or '["Sample_ID", "Group"]' in code:
            return (
                "VALIDATION ERROR: Temporary feature files should NOT include 'Group' column.\n\n"
                "Problem: Code includes 'Group' in the output dataframe:\n"
                "  output_df = data[['Sample_ID', 'Group']].copy()  # ❌ WRONG\n\n"
                "This will cause merge conflicts in step 1.5.4 when integrating features.\n\n"
                "Solution: Only include Sample_ID and feature columns:\n"
                "  output_df = data[['Sample_ID']].copy()  # ✓ CORRECT\n"
                "  # Then add feature columns\n"
                "  for col_name, values in features.items():\n"
                "      output_df[col_name] = values\n\n"
                "The Group column will be merged from the base dataset in step 1.5.4."
            )
        
        # Pattern 2: Check for Group column in to_csv operations
        # This is a softer check - warn if Group might be included
        if 'to_csv' in code and 'Group' in code:
            # Check if there's a pattern like: df['Group'] = ... before to_csv
            lines = code.split('\n')
            has_group_assignment = False
            has_to_csv = False
            
            for i, line in enumerate(lines):
                if "['Group']" in line or '["Group"]' in line:
                    # Check if it's an assignment (not just a check)
                    if '=' in line and 'if' not in line and '==' not in line:
                        has_group_assignment = True
                if 'to_csv' in line and any(temp in line for temp in temp_files):
                    has_to_csv = True
                    # Check if this to_csv is after the Group assignment
                    if has_group_assignment:
                        return (
                            "VALIDATION ERROR: Temporary feature files should NOT include 'Group' column.\n\n"
                            "Problem: Code assigns Group column before saving to temporary file.\n\n"
                            "Solution: Remove Group column assignment for temporary feature files:\n"
                            "  # ❌ WRONG:\n"
                            "  output_df['Group'] = data['Group']\n"
                            "  output_df.to_csv('data/temp_ratios.csv', index=False)\n\n"
                            "  # ✓ CORRECT:\n"
                            "  # Don't include Group column in temporary files\n"
                            "  output_df.to_csv('data/temp_ratios.csv', index=False)\n\n"
                            "The Group column will be merged from the base dataset in step 1.5.4."
                        )
        
        return None


def validate_code_with_contracts(code: str, tool_contracts: Dict[str, Any], step_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Convenience function for validating code.
    
    Args:
        code: Python code to validate
        tool_contracts: Tool contracts dictionary
        step_id: Optional step ID for context-aware validation
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = ToolCallValidator(tool_contracts)
    return validator.validate(code, step_id)


def validate_generated_code_security(code: str) -> Tuple[bool, Optional[str]]:
    """Run only the syntax and security-policy layer for executor entrypoints."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax Error at line {exc.lineno}: {exc.msg}"
    error = ToolCallValidator({})._validate_security_policy(tree)
    return error is None, error


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test cases
    from tool_contracts import TOOL_CONTRACTS
    
    print("="*80)
    print("AST Validator Test Suite")
    print("="*80)
    
    # Test 1: Valid code
    print("\n[Test 1] Valid Code")
    code1 = """
from src.tools.analysis.data_analysis_tools import delete_columns_tool
import json

result_json = delete_columns_tool(columns=['bad_col'], data_path='data.csv')
result = json.loads(result_json)
"""
    validator = ToolCallValidator(TOOL_CONTRACTS)
    is_valid, error = validator.validate(code1)
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if error:
        print(f"  Error: {error}")
    
    # Test 2: Syntax error
    print("\n[Test 2] Syntax Error")
    code2 = """
from src.tools.analysis.data_analysis_tools import delete_columns_tool
result = delete_columns_tool(columns=['bad_col', data_path='data.csv')  # Missing ]
"""
    is_valid, error = validator.validate(code2)
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if error:
        print(f"  Error: {error}")
    
    # Test 3: Missing import
    print("\n[Test 3] Missing Import")
    code3 = """
import json

result_json = delete_columns_tool(columns=['bad_col'], data_path='data.csv')
result = json.loads(result_json)
"""
    is_valid, error = validator.validate(code3)
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if error:
        print(f"  Error: {error}")
    
    # Test 4: Wrong parameter name
    print("\n[Test 4] Wrong Parameter Name")
    code4 = """
from src.tools.analysis.data_analysis_tools import delete_columns_tool
import json

result_json = delete_columns_tool(cols=['bad_col'], file_path='data.csv')  # Wrong param names
result = json.loads(result_json)
"""
    is_valid, error = validator.validate(code4)
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if error:
        print(f"  Error: {error}")
    
    # Test 5: Typo in tool name
    print("\n[Test 5] Typo in Tool Name")
    code5 = """
from src.tools.analysis.data_analysis_tools import delete_column_tool  # Missing 's'
import json

result_json = delete_column_tool(columns=['bad_col'], data_path='data.csv')
"""
    is_valid, error = validator.validate(code5)
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if error:
        print(f"  Error: {error}")
    
    print("\n" + "="*80)
    print("Test Suite Complete")
    print("="*80)
