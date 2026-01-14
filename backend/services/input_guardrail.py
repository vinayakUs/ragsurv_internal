"""
Input Guardrail Service

Comprehensive input validation to protect against malicious queries including:
- SQL injection
- Python code injection  
- Hacking/exploit keywords
- XSS patterns
- Path traversal
- Command injection

Usage:
    from backend.services.input_guardrail import validate_query
    
    is_safe, error_message = validate_query(user_query)
    if not is_safe:
        raise HTTPException(status_code=400, detail=error_message)
"""

import re
import logging
from typing import Tuple, List, Pattern

logger = logging.getLogger(__name__)


# =============================================================================
# SQL Injection Patterns
# =============================================================================
SQL_KEYWORDS: List[str] = [
    # DML/DDL statements
    r"\bSELECT\b", r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b",
    r"\bDROP\b", r"\bCREATE\b", r"\bALTER\b", r"\bTRUNCATE\b",
    r"\bUNION\b", r"\bJOIN\b", r"\bFROM\b", r"\bWHERE\b",
    r"\bGRANT\b", r"\bREVOKE\b",
    # SQL functions and commands
    r"\bEXEC\b", r"\bEXECUTE\b", r"\bCAST\b", r"\bCONVERT\b",
    r"\bDECLARE\b", r"\bFETCH\b", r"\bCURSOR\b",
    r"\bxp_cmdshell\b", r"\bsp_executesql\b",
    # Information schema
    r"\bINFORMATION_SCHEMA\b", r"\bSYSOBJECTS\b", r"\bSYSTABLES\b",
]

SQL_INJECTION_PATTERNS: List[str] = [
    # Comment-based injection
    r"--",                          # SQL single-line comment
    r"/\*",                         # SQL block comment start
    r"\*/",                         # SQL block comment end
    # Quote-based injection
    r"'\s*OR\s+'",                  # ' OR '
    r"'\s*OR\s+\d+\s*=\s*\d+",      # ' OR 1=1
    r"'\s*;\s*",                    # '; (statement terminator)
    r"''\s*",                       # Empty string injection
    r"\b1\s*=\s*1\b",               # 1=1 always true
    r"\b0\s*=\s*0\b",               # 0=0 always true
    r"'\s*AND\s+'",                 # ' AND '
    # Hex/encoded injection
    r"0x[0-9a-fA-F]+",              # Hex literals
    r"\\x[0-9a-fA-F]{2}",           # Hex escape sequences
    # Stacked queries
    r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)",
]


# =============================================================================
# Python Code Injection Patterns
# =============================================================================
PYTHON_INJECTION_PATTERNS: List[str] = [
    # Import statements
    r"\bimport\s+\w+",
    r"\bfrom\s+\w+\s+import\b",
    r"__import__\s*\(",
    # Dangerous functions
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"\bcompile\s*\(",
    r"\bgetattr\s*\(",
    r"\bsetattr\s*\(",
    r"\bdelattr\s*\(",
    r"\bhasattr\s*\(",
    # OS and subprocess
    r"\bos\.system\s*\(",
    r"\bos\.popen\s*\(",
    r"\bos\.exec\w*\s*\(",
    r"\bos\.spawn\w*\s*\(",
    r"\bsubprocess\.",
    r"\bcommands\.",
    # File operations
    r"\bopen\s*\([^)]*['\"][rwab]",    # open with mode
    r"\bfile\s*\(",
    r"\bcodecs\.open\s*\(",
    # Magic methods and attributes
    r"__builtins__",
    r"__class__",
    r"__bases__",
    r"__subclasses__",
    r"__globals__",
    r"__code__",
    r"__reduce__",
    r"__getattribute__",
    # Pickle/deserialization
    r"\bpickle\.",
    r"\bcPickle\.",
    r"\byaml\.load\s*\(",
    r"\bjson\.loads?\s*\([^)]*cls\s*=",
    # Lambda abuse for code execution
    r"lambda\s*.*:\s*(exec|eval|__import__)",
]


# =============================================================================
# Hacking/Exploit Keywords
# =============================================================================
HACKING_KEYWORDS: List[str] = [
    # Direct hacking terms
    r"\bhow\s+to\s+hack\b",
    r"\bhack\s+(into|this|the)\b",
    r"\bhacking\s+(tutorial|guide|method)\b",
    r"\bexploit\s+(vulnerability|database|system)\b",
    r"\bbypass\s+(security|authentication|login|firewall)\b",
    r"\bpentesting\b", r"\bpenetration\s+test\b",
    # Malware terms
    r"\breverse\s+shell\b",
    r"\bbackdoor\b",
    r"\brootkit\b",
    r"\bkeylogger\b",
    r"\btrojan\b",
    r"\bmalware\b",
    r"\bransomware\b",
    r"\bphishing\b",
    # Network attacks
    r"\bDDoS\b", r"\bDoS\s+attack\b",
    r"\bman[\s-]in[\s-]the[\s-]middle\b",
    r"\barp\s+spoofing\b",
    r"\bdns\s+poisoning\b",
    # Password attacks
    r"\bbrute\s*force\s*(attack|password)\b",
    r"\bpassword\s+crack\b",
    r"\bdictionary\s+attack\b",
    r"\bcredential\s+(stuff|dump)\b",
    r"\bsteal\s+(password|credentials|data)\b",
    # Privilege escalation
    r"\bprivilege\s+escalation\b",
    r"\broot\s+access\b",
    r"\badmin\s+access\b",
]


# =============================================================================
# XSS (Cross-Site Scripting) Patterns  
# =============================================================================
XSS_PATTERNS: List[str] = [
    # Script tags
    r"<\s*script",
    r"</\s*script\s*>",
    # JavaScript protocol
    r"javascript\s*:",
    r"vbscript\s*:",
    r"data\s*:\s*text/html",
    # Event handlers
    r"\bon\w+\s*=",                 # onclick=, onerror=, onload=, etc.
    # HTML injection
    r"<\s*iframe",
    r"<\s*embed",
    r"<\s*object",
    r"<\s*img\s+[^>]*onerror",
    r"<\s*svg\s+[^>]*onload",
    # Expression injection
    r"expression\s*\(",
    r"url\s*\(\s*['\"]?javascript",
]


# =============================================================================
# Path Traversal Patterns
# =============================================================================
PATH_TRAVERSAL_PATTERNS: List[str] = [
    r"\.\./",                       # Unix path traversal
    r"\.\.\\",                      # Windows path traversal
    r"/etc/passwd",                 # Common Unix target
    r"/etc/shadow",
    r"/etc/hosts",
    r"C:\\Windows",                 # Windows system paths
    r"C:\\Users",
    r"C:\\Program Files",
    r"%systemroot%",
    r"%windir%",
]


# =============================================================================
# Command Injection Patterns
# =============================================================================
COMMAND_INJECTION_PATTERNS: List[str] = [
    # Shell operators
    r"\|\s*\w+",                    # Pipe to command
    r";\s*\w+",                     # Command chaining
    r"&&\s*\w+",                    # AND chaining
    r"\|\|\s*\w+",                  # OR chaining
    r"\$\(",                        # Command substitution
    r"`[^`]+`",                     # Backtick command substitution
    # Common shell commands
    r"\bcat\s+[/\\]",
    r"\bls\s+[/\\-]",
    r"\brm\s+[/\\-]",
    r"\bwget\s+",
    r"\bcurl\s+",
    r"\bchmod\s+",
    r"\bchown\s+",
    r"\bkill\s+",
    r"\bsudo\s+",
    r"\bsh\s+-c\b",
    r"\bbash\s+-c\b",
    r"\bpowershell\b",
    r"\bcmd\.exe\b",
    r"\bcmd\s+/c\b",
]


# =============================================================================
# Compile all patterns for performance
# =============================================================================
def _compile_patterns(patterns: List[str], flags: int = re.IGNORECASE) -> List[Pattern]:
    """Compile regex patterns for efficient matching."""
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, flags))
        except re.error as e:
            logger.warning(f"Failed to compile pattern '{pattern}': {e}")
    return compiled


# Pre-compiled pattern groups
_SQL_KEYWORD_PATTERNS = _compile_patterns(SQL_KEYWORDS)
_SQL_INJECTION_COMPILED = _compile_patterns(SQL_INJECTION_PATTERNS)
_PYTHON_INJECTION_COMPILED = _compile_patterns(PYTHON_INJECTION_PATTERNS)
_HACKING_KEYWORDS_COMPILED = _compile_patterns(HACKING_KEYWORDS)
_XSS_PATTERNS_COMPILED = _compile_patterns(XSS_PATTERNS)
_PATH_TRAVERSAL_COMPILED = _compile_patterns(PATH_TRAVERSAL_PATTERNS)
_COMMAND_INJECTION_COMPILED = _compile_patterns(COMMAND_INJECTION_PATTERNS)


# =============================================================================
# Validation Functions
# =============================================================================
def _check_patterns(text: str, patterns: List[Pattern], description: str) -> Tuple[bool, str]:
    """
    Check text against a list of compiled patterns.
    
    Returns:
        (is_safe, error_message): Tuple where is_safe is False if blocked
    """
    for pattern in patterns:
        if pattern.search(text):
            match = pattern.search(text)
            matched_text = match.group()[:30] if match else "unknown"
            logger.warning(f"Blocked query - {description}: '{matched_text}...'")
            return False, f"Query blocked: {description} detected. This type of query is not allowed."
    return True, ""


def check_sql_injection(text: str) -> Tuple[bool, str]:
    """Check for SQL injection patterns."""
    # Check keywords
    is_safe, msg = _check_patterns(text, _SQL_KEYWORD_PATTERNS, "SQL keyword")
    if not is_safe:
        return is_safe, msg
    
    # Check injection patterns
    return _check_patterns(text, _SQL_INJECTION_COMPILED, "SQL injection pattern")


def check_python_injection(text: str) -> Tuple[bool, str]:
    """Check for Python code injection patterns."""
    return _check_patterns(text, _PYTHON_INJECTION_COMPILED, "Python code injection")


def check_hacking_keywords(text: str) -> Tuple[bool, str]:
    """Check for hacking/exploit related keywords."""
    return _check_patterns(text, _HACKING_KEYWORDS_COMPILED, "Potentially malicious request")


def check_xss(text: str) -> Tuple[bool, str]:
    """Check for XSS (Cross-Site Scripting) patterns."""
    return _check_patterns(text, _XSS_PATTERNS_COMPILED, "XSS pattern")


def check_path_traversal(text: str) -> Tuple[bool, str]:
    """Check for path traversal patterns."""
    return _check_patterns(text, _PATH_TRAVERSAL_COMPILED, "Path traversal attempt")


def check_command_injection(text: str) -> Tuple[bool, str]:
    """Check for command injection patterns."""
    return _check_patterns(text, _COMMAND_INJECTION_COMPILED, "Command injection")


def check_query_length(text: str, max_length: int = 5000) -> Tuple[bool, str]:
    """Check for excessively long queries (potential DoS)."""
    if len(text) > max_length:
        logger.warning(f"Blocked query - exceeds max length: {len(text)} chars")
        return False, f"Query blocked: Query too long. Maximum {max_length} characters allowed."
    return True, ""


def check_null_bytes(text: str) -> Tuple[bool, str]:
    """Check for null byte injection."""
    if '\x00' in text or '%00' in text.lower():
        logger.warning("Blocked query - null byte injection")
        return False, "Query blocked: Invalid characters detected."
    return True, ""


# =============================================================================
# Main Validation Function
# =============================================================================
def validate_query(query: str) -> Tuple[bool, str]:
    """
    Validate a user query against all security patterns.
    
    Args:
        query: The user's input query string
        
    Returns:
        Tuple of (is_safe, error_message)
        - is_safe: True if query passes all checks, False otherwise
        - error_message: Empty string if safe, otherwise description of issue
        
    Example:
        is_safe, error = validate_query("SELECT * FROM users")
        if not is_safe:
            raise HTTPException(status_code=400, detail=error)
    """
    if not query or not isinstance(query, str):
        return True, ""  # Empty queries handled elsewhere
    
    # Run all validation checks in order
    validators = [
        check_query_length,
        check_null_bytes,
        check_sql_injection,
        check_python_injection,
        check_hacking_keywords,
        check_xss,
        check_path_traversal,
        check_command_injection,
    ]
    
    for validator in validators:
        is_safe, error = validator(query)
        if not is_safe:
            return False, error
    
    return True, ""


def validate_query_batch(queries: List[str]) -> Tuple[bool, str, int]:
    """
    Validate a batch of queries.
    
    Returns:
        Tuple of (is_safe, error_message, failed_index)
        - is_safe: True if all queries pass
        - error_message: Error message if any query fails
        - failed_index: Index of first failed query (-1 if all pass)
    """
    for i, query in enumerate(queries):
        is_safe, error = validate_query(query)
        if not is_safe:
            return False, f"Query at index {i}: {error}", i
    
    return True, "", -1


# =============================================================================
# CLI Testing
# =============================================================================
if __name__ == "__main__":
    # Quick test
    test_cases = [
        ("SELECT * FROM users", False, "SQL"),
        ("import os; os.system('rm -rf /')", False, "Python"),
        ("how to hack into the database", False, "Hacking"),
        ("<script>alert('xss')</script>", False, "XSS"),
        ("../../../etc/passwd", False, "Path Traversal"),
        ("What are the meeting agenda items?", True, "Legitimate"),
        ("Tell me about the annual report", True, "Legitimate"),
    ]
    
    print("=" * 60)
    print("Input Guardrail Quick Test")
    print("=" * 60)
    
    passed = 0
    for query, should_pass, category in test_cases:
        is_safe, error = validate_query(query)
        result_ok = is_safe == should_pass
        passed += 1 if result_ok else 0
        status = "? PASS" if result_ok else "? FAIL"
        print(f"{status} [{category}] '{query[:40]}...' -> {'ALLOWED' if is_safe else 'BLOCKED'}")
    
    print("=" * 60)
    print(f"Results: {passed}/{len(test_cases)} tests passed")
