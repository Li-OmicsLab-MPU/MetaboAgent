"""
Local Python Executor

This module provides a secure local execution environment for generated Python code.

Security Features:
- Subprocess isolation
- Timeout protection
- Working directory restriction
- Error handling and logging
- Log sanitization (Privacy Guard)

Privacy Features:
- All execution happens locally
- No data sent to external services
- File-based operations only
- Sensitive information masked in logs
"""

import os
import json
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional


_SENSITIVE_MAGIC_KEYS = {
    "head_preview", "data_preview", "raw_preview", "raw_rows", "row_data",
    "records", "observations", "raw_data", "dataframe", "sample_values",
    "patient_values", "subject_values", "sample_ids", "patient_ids", "subject_ids",
}


def _sanitize_magic_value(value: Any, key: str = "") -> Any:
    """Remove row- or subject-level payloads while preserving workflow metadata."""
    normalized_key = str(key or "").strip().lower()
    if normalized_key in _SENSITIVE_MAGIC_KEYS or normalized_key.endswith("_preview"):
        return "[REDACTED_ROW_LEVEL_DATA]"
    if isinstance(value, dict):
        return {str(k): _sanitize_magic_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_magic_value(item, normalized_key) for item in value]
    if isinstance(value, str):
        sanitized = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            '[MASKED_EMAIL]',
            value,
        )
        sanitized = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '[MASKED_IP]', sanitized)
        return sanitized
    return value


def _sanitize_magic_line(line: str) -> str:
    """Parse and sanitize Magic Output; malformed payloads fail closed."""
    try:
        payload = json.loads(line)
        if not isinstance(payload, dict) or "__METABO_UPDATE__" not in payload:
            raise ValueError("missing __METABO_UPDATE__ object")
        payload["__METABO_UPDATE__"] = _sanitize_magic_value(
            payload["__METABO_UPDATE__"],
            "__METABO_UPDATE__",
        )
        sanitized = json.dumps(payload, ensure_ascii=False, default=str)
        max_chars = int(os.getenv("METABO_MAGIC_OUTPUT_MAX_CHARS", "262144"))
        if len(sanitized) > max_chars:
            return json.dumps({
                "__METABO_UPDATE__": {
                    "error": "Magic Output rejected: payload exceeds privacy size limit",
                    "error_type": "privacy_guard",
                }
            })
        return sanitized
    except Exception:
        return json.dumps({
            "__METABO_UPDATE__": {
                "error": "Magic Output rejected: invalid or unsafe payload",
                "error_type": "privacy_guard",
            }
        })


def sanitize_log(log_text: str, max_length: int = 2000) -> str:
    """
    Sanitize log output to prevent data leakage.
    
    Privacy Guard: This function ensures that logs sent to the LLM
    do not contain sensitive information or excessive data.
    
    Args:
        log_text: Raw log text from execution
        max_length: Maximum length of sanitized log (default: 2000 chars)
    
    Returns:
        Sanitized log text with Magic Output preserved
    
    Sanitization steps:
        1. Parse and privacy-filter Magic Output lines (__METABO_UPDATE__)
        2. Truncate other content to max_length
        3. Mask patterns that look like sensitive IDs (ONLY in non-Magic lines)
        4. Remove potential data dumps
    """
    if not log_text:
        return log_text
    
    # Magic Output remains machine-readable, but is no longer exempt from privacy filtering.
    magic_output_lines = []
    other_lines = []
    
    for line in log_text.split('\n'):
        if '__METABO_UPDATE__' in line:
            magic_output_lines.append(_sanitize_magic_line(line))
        else:
            other_lines.append(line)
    
    # Reconstruct: Magic Output first (UNTOUCHED), then sanitized other content
    preserved_magic = '\n'.join(magic_output_lines)
    other_content = '\n'.join(other_lines)
    
    # Calculate remaining space for other content
    magic_length = len(preserved_magic)
    remaining_space = max_length - magic_length - 50  # Reserve 50 chars for truncation message
    
    if remaining_space < 0:
        return preserved_magic
    
    # Step 1: Truncate other content if needed
    if len(other_content) > remaining_space:
        truncated = other_content[-remaining_space:]
        other_content = f"[... truncated {len(other_content) - remaining_space} chars ...]\n{truncated}"
    
    # Step 2: Mask patterns that look like sensitive IDs or data (ONLY in other_content)
    # CRITICAL FIX: Use a pattern that doesn't match decimal parts of floats
    # Pattern 1: Long hex strings (potential IDs) - must contain at least one letter a-f
    # This prevents matching pure numeric sequences like decimal parts of floats
    other_content = re.sub(r'\b(?=.*[a-fA-F])[0-9a-fA-F]{16,}\b', '[MASKED_ID]', other_content)
    
    # Pattern 2: Email-like patterns
    other_content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[MASKED_EMAIL]', other_content)
    
    # Pattern 3: File paths with potential sensitive names (keep structure)
    other_content = re.sub(r'/home/[^/\s]+', '/home/[USER]', other_content)
    other_content = re.sub(r'C:\\Users\\[^\\]+', r'C:\\Users\\[USER]', other_content)
    
    # Pattern 4: IP addresses
    other_content = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[MASKED_IP]', other_content)
    
    # Combine sanitized Magic Output and ordinary logs.
    if preserved_magic and other_content:
        return preserved_magic + '\n' + other_content
    elif preserved_magic:
        return preserved_magic
    else:
        return other_content
    
# Tool Enforcement Rules for Step 1.5
# These rules ensure LLM uses provided tool classes instead of manual implementation
TOOL_ENFORCEMENT_RULES = {
    '1.5.1': {
        'name': 'Reaction Ratio Generation',
        'required': [
            'from src.tools.domain.feature_generator import generate_reaction_ratios',
            'generate_reaction_ratios(',
        ],
        'forbidden': [
            'ReactionCheckerTool',
            'itertools.combinations',
            '.find_connected_pairs(',
        ],
        'error_message': (
            'Step 1.5.1 MUST use the centralized feature generator entrypoint '
            '`generate_reaction_ratios(...)` from `src.tools.domain.feature_generator`.\n'
            'Import exactly: "from src.tools.domain.feature_generator import generate_reaction_ratios".\n'
            'Use `pre_engineering_data_path` when available, write to the temporary engineered '
            'feature file, and set `include_label=False`.\n'
            'DO NOT manually enumerate metabolite pairs, DO NOT call ReactionCheckerTool directly '
            'in SOP-generated code, and DO NOT compute raw division ratios by hand.'
        )
    },
    '1.5.2': {
        'name': 'Taxonomy Sum Generation',
        'required': [
            'from src.tools.domain.feature_generator import generate_taxonomy_sums',
            'generate_taxonomy_sums(',
        ],
        'forbidden': [
            'TaxonomyGroupTool',
            '.group_metabolites(',
        ],
        'error_message': (
            'Step 1.5.2 MUST use the centralized feature generator entrypoint '
            '`generate_taxonomy_sums(...)` from `src.tools.domain.feature_generator`.\n'
            'Import exactly: "from src.tools.domain.feature_generator import generate_taxonomy_sums".\n'
            'Use `pre_engineering_data_path` when available, write to the temporary engineered '
            'feature file, and set `include_label=False`.\n'
            'DO NOT manually group and sum all taxonomy classes inside SOP-generated code; the '
            'generator already applies the direction-consistency and member-count constraints.'
        )
    },
    '1.5.3': {
        'name': 'Pathway Score Calculation',
        'required': [
            'from src.tools.domain.feature_generator import generate_pathway_scores',
            'generate_pathway_scores(',
        ],
        'forbidden': [
            'calculate_pathway_scores(',
        ],
        'error_message': (
            'Step 1.5.3 MUST use the centralized feature generator entrypoint '
            '`generate_pathway_scores(...)` from `src.tools.domain.feature_generator`.\n'
            'Import exactly: "from src.tools.domain.feature_generator import generate_pathway_scores".\n'
            'Use `pre_engineering_data_path` when available, pass `phase0_output` when available, '
            'write to the temporary engineered feature file, and set `include_label=False`.\n'
            'DO NOT call the low-level `calculate_pathway_scores()` entrypoint directly in SOP-generated '
            'code, and DO NOT implement manual ssGSEA/pathway scoring logic.'
        )
    },
    '1.5.4': {
        'name': 'Engineered Feature Merge',
        'required': [
            'from src.tools.domain.feature_generator import merge_engineered_features',
            'merge_engineered_features(',
        ],
        'forbidden': [
            "pd.read_csv(ratios_path)",
            "pd.read_csv(sums_path)",
            "pd.read_csv(pathway_path)",
            "ratios_df.shape[1]",
            "sums_df.shape[1]",
            "pathway_df.shape[1]",
        ],
        'error_message': (
            'Step 1.5.4 MUST use the centralized merge entrypoint '
            '`merge_engineered_features(...)` from `src.tools.domain.feature_generator`.\n'
            'Import exactly: "from src.tools.domain.feature_generator import merge_engineered_features".\n'
            'Use `pre_engineering_data_path` when available as base data, pass the temp engineered '
            'feature file paths, and read counts from `result[\"feature_breakdown\"]` instead of '
            'manually inferring counts from CSV column totals.\n'
            'DO NOT manually load each temp feature file and count columns with DataFrame.shape[1], '
            'because that miscounts join keys / placeholder columns.'
        )
    },
    '1.5.5': {
        'name': 'Auto Scaling For Engineered Matrix',
        'required': [
            'from src.tools.analysis.data_analysis_tools import auto_scaling_tool',
            'auto_scaling_tool(',
        ],
        'forbidden': [
            'def has_engineered_markers(',
            'def resolve_input_path(',
            'def choose_engineered_source(',
            'csv.reader(',
            'engineered_features_merged =',
            'n_ratio_features =',
            'n_sum_features =',
            'n_pathway_scores =',
            'prerequisite_not_met',
            'Auto-scaling aborted:',
        ],
        'error_message': (
            'Step 1.5.5 MUST directly call `auto_scaling_tool(...)` from '
            '`src.tools.analysis.data_analysis_tools`.\n'
            'Import exactly: "from src.tools.analysis.data_analysis_tools import auto_scaling_tool".\n'
            'Use the already prepared `current_data_path` as the scaling input; '
            '`_maybe_prepare_stage15_scaling_input()` has already switched it to the merged '
            'engineered matrix when that artifact exists.\n'
            'DO NOT define helper functions such as `has_engineered_markers()` or '
            '`resolve_input_path()` / `choose_engineered_source()` to rediscover the engineered '
            'file; that duplicates runtime logic and has repeatedly triggered invalid code generation.\n'
            'DO NOT add prerequisite guards based on `engineered_features_merged`, '
            '`n_ratio_features`, `n_sum_features`, or `n_pathway_scores`; if the merged matrix '
            'exists, scale it directly.'
        )
    },
    '5.2.1': {
        'name': 'Unified Stability Selection',
        'required': [
            'run_train_only_stability_selection',
            'build_stable_panel_from_logs',
        ],
        'forbidden': [],
        'error_message': (
            '❌ Step 5.2.1 MANDATORY Function Call Validation Failed!\n\n'
            '═══════════════════════════════════════════════════════════\n'
            'This step must use the unified Phase 1 stability-selection pipeline\n'
            '═══════════════════════════════════════════════════════════\n\n'
            'MANDATORY FUNCTIONS (BOTH REQUIRED):\n'
            '  1️⃣  run_train_only_stability_selection() - MUST be called FIRST\n'
            '  2️⃣  build_stable_panel_from_logs() - MUST be called SECOND\n\n'
            'EXECUTION SEQUENCE:\n'
            '  ```python\n'
            '  from src.tools.analysis.feature_selection_tools import (\n'
            '      run_train_only_stability_selection,\n'
            '      build_stable_panel_from_logs\n'
            '  )\n'
            '  \n'
            '  # Step 1: MANDATORY\n'
            '  stab_result = run_train_only_stability_selection(...)\n'
            '  \n'
            '  # Step 2: MANDATORY\n'
            '  panel_result = json.loads(build_stable_panel_from_logs(...))\n'
            '  ```\n\n'
            'CRITICAL RULES:\n'
            '  ❌ DO NOT skip either function call\n'
            '  ❌ DO NOT change the execution order\n'
            '  ❌ DO NOT fall back to legacy threshold schedule or adaptive voting\n'
            '  ✅ ALWAYS check panel_result["success"]\n'
            '  ✅ ALWAYS raise an error if any step fails\n'
        )
    },
    '5.2.2': {
        'name': 'Stability Audit And Stable Core Review',
        'required': [],  # Allow flexible implementation - voting logic is straightforward
        'forbidden': [],
        'error_message': ''
    },
    '5.2.3': {
        'name': 'Stable Panel Handoff Validation',
        'required': [],
        'forbidden': [],
        'error_message': ''
    }
}


class LocalPythonExecutor:
    """
    Executes Python code in a local subprocess with security controls.
    
    This executor provides:
    - Subprocess isolation
    - Timeout protection
    - Capture of stdout/stderr
    - Working directory management
    - Tool enforcement for critical steps
    
    Example:
        >>> executor = LocalPythonExecutor(timeout=1200)
        >>> result = executor.execute(code, working_dir="/data", step_id="1.5.1")
        >>> print(result["stdout"])
    """
    
    def __init__(self, timeout: int = 1200, python_executable: Optional[str] = None):
        """
        Initialize the executor.
        
        Args:
            timeout: Maximum execution time in seconds (default: 1200 = 20 minutes)
            python_executable: Path to Python executable (default: current interpreter)
        """
        self.timeout = timeout
        self.python_executable = python_executable or sys.executable

    @staticmethod
    def _filtered_environment(extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Pass runtime settings but strip credentials from generated-code processes."""
        sensitive_fragments = (
            "KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL",
            "OPENAI", "ANTHROPIC", "AZURE", "AWS_", "GCP_", "GOOGLE_APPLICATION",
            "SSH_", "DATABASE_URL", "DSN",
        )
        safe_env = {
            key: value
            for key, value in os.environ.items()
            if not any(fragment in key.upper() for fragment in sensitive_fragments)
        }
        for key, value in dict(extra_env or {}).items():
            if not any(fragment in str(key).upper() for fragment in sensitive_fragments):
                safe_env[str(key)] = str(value)
        safe_env["PYTHONDONTWRITEBYTECODE"] = "1"
        safe_env["NO_PROXY"] = "*"
        safe_env["no_proxy"] = "*"
        # Native numerical libraries inspect the host CPU count independently
        # from application-level n_jobs settings. On large shared servers this
        # can exhaust the sandbox's thread budget after several fitted models.
        # A single-thread default is deterministic and callers may explicitly
        # raise it with METABO_EXEC_NUM_THREADS when the deployment allows it.
        thread_limit = str(max(1, int(os.getenv("METABO_EXEC_NUM_THREADS", "1"))))
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            safe_env[variable] = thread_limit
        return safe_env

    def _resource_limiter(self):
        """Return a pre-exec hook whose limits are inherited by child processes."""
        timeout = int(self.timeout)
        max_memory_mb = int(os.getenv("METABO_EXEC_MAX_MEMORY_MB", "65536"))
        max_file_mb = int(os.getenv("METABO_EXEC_MAX_FILE_MB", "8192"))
        # RLIMIT_NPROC is per Unix user, not per subprocess tree. This shared
        # research server already runs many user processes, so the default must
        # remain above the current account total while still imposing a ceiling.
        max_processes = int(os.getenv("METABO_EXEC_MAX_PROCESSES", "4096"))

        def _apply_limits() -> None:
            os.setsid()
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_CPU, (timeout + 30, timeout + 30))
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (max_file_mb * 1024 * 1024, max_file_mb * 1024 * 1024),
            )
            resource.setrlimit(
                resource.RLIMIT_AS,
                (max_memory_mb * 1024 * 1024, max_memory_mb * 1024 * 1024),
            )
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (max_processes, max_processes))
            except (ValueError, OSError):
                pass

        return _apply_limits

    def _sandbox_command(
        self,
        temp_file: str,
        project_root: str,
        working_dir: str,
    ) -> tuple[list[str], str]:
        """Build a bubblewrap namespace with no network and project-scoped writes."""
        bwrap = shutil.which("bwrap")
        require_bwrap = os.getenv("METABO_EXEC_REQUIRE_BWRAP", "1") != "0"
        if not bwrap:
            if require_bwrap:
                raise RuntimeError("bubblewrap is required but not installed")
            return [self.python_executable, temp_file], "resource_limited_subprocess"

        command = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-ipc",
            "--unshare-uts",
        ]
        # Some managed/NFS servers allow user namespaces but deny the
        # CAP_NET_ADMIN operation bubblewrap needs to bring up loopback in a
        # network namespace. Keep the filesystem/IPC/UTS sandbox and make the
        # network namespace optional for that deployment shape.
        if os.getenv("METABO_EXEC_UNSHARE_NET", "1") != "0":
            command.insert(3, "--unshare-net")
        for system_path in (
            "/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt",
            "/var/lib", "/var/cache/fontconfig", "/sys",
        ):
            if os.path.exists(system_path):
                command.extend(["--ro-bind", system_path, system_path])

        # Preserve an equivalent logical project path (for example
        # /home/.../MetaboAgent when project_root resolves to
        # /nas/.../MetaboAgent). Runtime configuration may legitimately retain
        # that logical spelling even though both paths address the same tree.
        project_aliases = []
        logical_cwd = str(os.environ.get("PWD", "") or "").strip()
        if logical_cwd and logical_cwd != project_root and os.path.isdir(logical_cwd):
            try:
                if os.path.samefile(logical_cwd, project_root):
                    project_aliases.append(logical_cwd)
            except OSError:
                pass

        # Create only the parent directories required for the Python environment
        # and this project; no other user home is visible inside the namespace.
        visible_paths = [
            Path(sys.prefix).resolve(),
            Path(project_root).resolve(),
            *(Path(alias) for alias in project_aliases),
        ]
        parents = set()
        for visible_path in visible_paths:
            for parent in visible_path.parents:
                if str(parent) not in {"/", "/usr", "/opt"}:
                    parents.add(str(parent))
        for parent in sorted(parents, key=lambda value: (value.count(os.sep), value)):
            command.extend(["--dir", parent])

        python_prefix = str(Path(sys.prefix).resolve())
        command.extend(["--ro-bind", python_prefix, python_prefix])
        command.extend(["--bind", project_root, project_root])
        for alias in project_aliases:
            command.extend(["--bind", project_root, alias])
        command.extend([
            "--proc", "/proc",
            "--dev-bind", "/dev", "/dev",
            "--tmpfs", "/tmp",
            "--chdir", working_dir,
            "--",
            self.python_executable,
            temp_file,
        ])
        return command, "bubblewrap"

    @staticmethod
    def _canonicalize_logical_paths(text: str, project_root: str) -> str:
        """Resolve shell-visible symlink paths before entering the namespace."""
        normalized_text = str(text or "")
        logical_cwd = str(os.environ.get("PWD", "") or "").strip()
        candidate_roots = set()
        if logical_cwd:
            candidate_roots.add(logical_cwd)

        # A launcher may start Python from the physical /nas path while the
        # active runtime configuration still contains the equivalent /home
        # path. Discover project-root spellings present in the generated code
        # and canonicalize only aliases that resolve to this same directory.
        project_name = re.escape(Path(project_root).name)
        alias_pattern = re.compile(
            rf"(?<![A-Za-z0-9_.-])(/[A-Za-z0-9_./-]+/{project_name})(?=/|['\"]|\s)"
        )
        candidate_roots.update(alias_pattern.findall(normalized_text))

        for candidate in sorted(candidate_roots, key=len, reverse=True):
            if not candidate or candidate == project_root or not os.path.isdir(candidate):
                continue
            try:
                if os.path.samefile(candidate, project_root):
                    normalized_text = normalized_text.replace(candidate, project_root)
            except OSError:
                continue
        return normalized_text
    
    def _validate_code(self, code: str, step_id: Optional[str] = None) -> Optional[str]:
        """
        Validate code against tool enforcement rules.
        
        This method ensures that critical steps (like Step 1.5) use the provided
        tools instead of manual implementation. This prevents:
        - Token explosions from generating too many features
        - Data loss from incorrect manual implementations
        - Pipeline failures from buggy code
        
        Args:
            code: Python code to validate
            step_id: Step identifier (e.g., "1.5.1")
        
        Returns:
            Error message string if validation fails, None if validation passes
        
        Example:
            >>> error = executor._validate_code(code, step_id="1.5.1")
            >>> if error:
            >>>     print(f"Validation failed: {error}")
        """
        from .ast_validator import validate_generated_code_security

        security_valid, security_error = validate_generated_code_security(code)
        if not security_valid:
            return security_error

        # If no step_id or step not in enforcement rules, skip tool-specific validation
        if not step_id or step_id not in TOOL_ENFORCEMENT_RULES:
            return None
        
        rules = TOOL_ENFORCEMENT_RULES[step_id]
        
        # Check required patterns
        for required_pattern in rules['required']:
            if required_pattern not in code:
                error_msg = (
                    f"Tool Enforcement Failed for {rules['name']} (Step {step_id}):\n"
                    f"Required function '{required_pattern}' not found in code.\n\n"
                    f"{rules['error_message']}\n\n"
                    f"Your code must import and call this function. "
                    f"Do NOT implement the logic manually."
                )
                return error_msg
        
        # Check forbidden patterns with smart detection
        for forbidden_pattern in rules['forbidden']:
            # Special handling for 'for ' - only detect actual for-loops, not in comments/strings
            if forbidden_pattern == 'for ':
                # Use AST to detect actual for-loops
                try:
                    import ast
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.For):
                            error_msg = (
                                f"Tool Enforcement Failed for {rules['name']} (Step {step_id}):\n"
                                f"Forbidden pattern 'for-loop' detected in code.\n\n"
                                f"{rules['error_message']}\n\n"
                                f"You MUST use the provided tool function. "
                                f"Remove all manual for-loops and call the tool directly."
                            )
                            return error_msg
                except SyntaxError:
                    # If code has syntax errors, let it fail during execution
                    pass
            else:
                # For other patterns, use simple string matching
                if forbidden_pattern in code:
                    error_msg = (
                        f"Tool Enforcement Failed for {rules['name']} (Step {step_id}):\n"
                        f"Forbidden pattern '{forbidden_pattern}' detected in code.\n\n"
                        f"{rules['error_message']}\n\n"
                        f"You MUST use the provided tool function. "
                        f"Remove all manual implementation code and call the tool directly."
                    )
                    return error_msg
        
        # Validation passed
        return None
    
    def execute(
        self,
        code: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        step_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code in a subprocess.
        
        Args:
            code: Python code to execute
            working_dir: Working directory for execution (default: current dir)
            env: Environment variables (default: inherit from parent)
            step_id: Step identifier for tool enforcement (e.g., "1.5.1")
        
        Returns:
            Dictionary with execution results:
            - success: bool - Whether execution succeeded
            - stdout: str - Standard output
            - stderr: str - Standard error
            - execution_time: float - Time taken in seconds
            - return_code: int - Process return code
        
        Example:
            >>> code = "print('Hello, World!')"
            >>> result = executor.execute(code, step_id="1.5.1")
            >>> assert result["success"]
            >>> assert "Hello, World!" in result["stdout"]
        """
        # Validate code against tool enforcement rules
        validation_error = self._validate_code(code, step_id)
        if validation_error:
            return {
                "success": False,
                "stdout": "",
                "stderr": validation_error,
                "execution_time": 0.0,
                "return_code": -1
            }
        
        # === P0-1: 自动注入环境设置 ===
        # 在代码开头自动注入 sys.path 和项目根目录
        project_root = str(Path(os.getcwd()).resolve())
        env_setup = f'''
# === AUTO-INJECTED: Environment Setup ===
import sys
import os
import json

# 自动注入项目根目录到 sys.path（确保能导入 src.tools）
_project_root = r"{project_root}"
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 设置工作目录为项目根目录（确保相对路径正确）
os.chdir(_project_root)

# Make json.dumps resilient to numpy / pandas scalar outputs commonly produced
# by generated analysis code when emitting the Magic Output Protocol payload.
_metabo_original_json_dumps = json.dumps

def _metabo_json_default(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, set):
        return list(value)
    return str(value)

def _metabo_json_dumps(obj, *args, **kwargs):
    user_default = kwargs.get("default")
    if user_default is None:
        kwargs["default"] = _metabo_json_default
    else:
        def _combined_default(value):
            try:
                return user_default(value)
            except TypeError:
                return _metabo_json_default(value)
        kwargs["default"] = _combined_default
    return _metabo_original_json_dumps(obj, *args, **kwargs)

json.dumps = _metabo_json_dumps
# === END AUTO-INJECTION ===

'''
        # 将环境设置代码前置到用户代码之前
        full_code = env_setup + code
        full_code = self._canonicalize_logical_paths(full_code, project_root)
        
        start_time = time.time()
        
        try:
            requested_path = str(working_dir or project_root)
            if os.path.isfile(requested_path):
                requested_path = os.path.dirname(requested_path)
            requested_working_dir = str(Path(requested_path).resolve())
            if os.path.commonpath([project_root, requested_working_dir]) != project_root:
                raise PermissionError(
                    f"working_dir must remain inside project root: {project_root}"
                )
            os.makedirs(requested_working_dir, exist_ok=True)

            exec_temp_dir = Path(project_root) / ".metabo_exec"
            exec_temp_dir.mkdir(parents=True, exist_ok=True)

            # Create temporary file for code (使用注入后的完整代码)
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8',
                dir=str(exec_temp_dir),
            ) as f:
                f.write(full_code)  # 写入注入后的代码
                temp_file = f.name

            # Prepare a credential-free environment.
            exec_env = self._filtered_environment(env)
            
            # Add project root to PYTHONPATH so generated code can import src.tools
            if 'PYTHONPATH' in exec_env:
                exec_env['PYTHONPATH'] = f"{project_root}{os.pathsep}{exec_env['PYTHONPATH']}"
            else:
                exec_env['PYTHONPATH'] = project_root

            command, sandbox_backend = self._sandbox_command(
                temp_file=temp_file,
                project_root=project_root,
                working_dir=requested_working_dir,
            )

            # Execute code
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=project_root,
                env=exec_env,
                preexec_fn=self._resource_limiter(),
            )
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except Exception:
                pass  # Ignore cleanup errors
            
            execution_time = time.time() - start_time
            
            # Privacy Guard: Sanitize logs before returning
            sanitized_stdout = sanitize_log(result.stdout, max_length=2000)
            sanitized_stderr = sanitize_log(result.stderr, max_length=2000)
            
            return {
                "success": result.returncode == 0,
                "stdout": sanitized_stdout,
                "stderr": sanitized_stderr,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "sandbox_backend": sandbox_backend,
            }
        
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            
            # Clean up temp file
            try:
                if 'temp_file' in locals():
                    os.unlink(temp_file)
            except Exception:
                pass
            
            # Privacy Guard: Sanitize timeout message
            error_msg = f"Execution timeout ({self.timeout} seconds)"
            
            return {
                "success": False,
                "stdout": "",
                "stderr": sanitize_log(error_msg, max_length=2000),
                "execution_time": execution_time,
                "return_code": -1
            }
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Clean up temp file
            try:
                if 'temp_file' in locals():
                    os.unlink(temp_file)
            except Exception:
                pass
            
            # Privacy Guard: Sanitize exception message
            error_msg = f"Execution error: {str(e)}"
            
            return {
                "success": False,
                "stdout": "",
                "stderr": sanitize_log(error_msg, max_length=2000),
                "execution_time": execution_time,
                "return_code": -1
            }
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate Python code syntax without executing it.
        
        Args:
            code: Python code to validate
        
        Returns:
            Dictionary with validation results:
            - valid: bool - Whether code is syntactically valid
            - error: Optional[str] - Error message if invalid
        
        Example:
            >>> result = executor.validate_code("print('hello')")
            >>> assert result["valid"]
        """
        try:
            compile(code, '<string>', 'exec')
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Syntax error at line {e.lineno}: {e.msg}"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}"
            }


class ExecutionSandbox:
    """
    A sandboxed execution environment with additional safety features.
    
    This class extends LocalPythonExecutor with:
    - Resource limits
    - Restricted imports
    - File system access controls
    
    The underlying executor uses a bubblewrap namespace on Linux, strips
    credentials, disables networking and applies resource limits.
    """
    
    def __init__(
        self,
        timeout: int = 1200,
        max_memory_mb: Optional[int] = None,
        allowed_imports: Optional[list] = None
    ):
        """
        Initialize the sandbox.
        
        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Optional memory limit override in MB
            allowed_imports: Reserved for an application-specific allowlist
        """
        self.executor = LocalPythonExecutor(timeout=timeout)
        self.max_memory_mb = max_memory_mb
        self.allowed_imports = allowed_imports
        if max_memory_mb is not None:
            os.environ.setdefault("METABO_EXEC_MAX_MEMORY_MB", str(int(max_memory_mb)))
    
    def execute(
        self,
        code: str,
        working_dir: Optional[str] = None,
        step_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute code in the sandbox.
        
        Args:
            code: Python code to execute
            working_dir: Working directory for execution
            step_id: Step identifier for tool enforcement
        
        Returns:
            Execution results dictionary
        """
        return self.executor.execute(code, working_dir, step_id=step_id)
