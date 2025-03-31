"""
Validation utilities for alpha expressions and related data.
Provides functions to check expression syntax and data integrity.
"""

import re
from typing import Dict, List, Tuple, Optional, Set

class ValidationError(Exception):
    """Raised when validation fails."""
    pass

def validate_alpha_expression(expression: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an alpha expression for syntax errors.
    
    Args:
        expression: Alpha expression to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not expression.strip():
        return False, "Expression is empty"
    
    # Check for balanced parentheses
    if expression.count('(') != expression.count(')'):
        return False, "Unbalanced parentheses"
    
    # Check for overly simple expressions
    if re.match(r'^\d+\.?$|^[a-zA-Z_]+$', expression):
        return False, "Expression too simple (just a number or variable name)"
    
    # Check for function calls
    if not re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(', expression):
        return False, "No function calls found in expression"
    
    # Check for common syntax errors
    if expression.count(',') > 0 and expression.count('(') == 0:
        return False, "Commas without function calls"
    
    if re.search(r'\(\s*\)', expression):
        return False, "Empty function calls"
    
    # Check for missing operators between terms
    if re.search(r'\)\s*\(', expression):
        return False, "Missing operator between terms"
    
    return True, None

def extract_symbols_from_expression(expression: str) -> Set[str]:
    """
    Extract symbol names from an alpha expression.
    
    Args:
        expression: Alpha expression to analyze
        
    Returns:
        Set of symbol names
    """
    # Extract all potential symbol names (alphanumeric words)
    symbols = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expression))
    
    # Filter out common operators and functions
    common_operators = {
        # Arithmetic operators
        'add', 'subtract', 'multiply', 'divide', 'power',
        
        # Comparison operators
        'greater', 'less', 'equal', 'not_equal',
        
        # Logical operators
        'and', 'or', 'not', 'if', 'where', 'if_else',
        
        # Time series operators
        'ts_mean', 'ts_std_dev', 'ts_min', 'ts_max', 'ts_sum',
        'ts_product', 'ts_rank', 'ts_delta', 'ts_returns', 'ts_delay',
        'ts_correlation', 'ts_covariance', 'ts_skewness', 'ts_kurtosis',
        
        # Cross-sectional operators
        'rank', 'zscore', 'winsorize', 'sigmoid', 'scale',
        
        # Group operators
        'group_rank', 'group_zscore', 'group_mean', 'group_sum',
        'group_min', 'group_max', 'group_std_dev',
        
        # Vector operators
        'vec_sum', 'vec_mean', 'vec_std_dev', 'vec_min', 'vec_max',
        
        # Miscellaneous operators
        'log', 'sqrt', 'abs', 'sign', 'exp', 'round', 'floor', 'ceiling',
        
        # Flow control
        'return'
    }
    
    # Return symbols that aren't common operators
    return {s for s in symbols if s.lower() not in common_operators}

def validate_simulation_settings(settings: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate simulation settings.
    
    Args:
        settings: Dictionary of simulation settings
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Required fields
    required_fields = [
        'instrumentType', 'region', 'universe', 'delay',
        'neutralization', 'truncation', 'pasteurization'
    ]
    
    for field in required_fields:
        if field not in settings:
            return False, f"Missing required field: {field}"
    
    # Valid instrument types
    valid_instrument_types = ['EQUITY', 'FUTURES', 'CRYPTO', 'FOREX']
    if settings.get('instrumentType') not in valid_instrument_types:
        return False, f"Invalid instrumentType: {settings.get('instrumentType')}"
    
    # Valid regions
    valid_regions = ['USA', 'CHN', 'JPN', 'EUR', 'ASIA', 'KOR', 'TWN', 'GBR', 'HKG', 'GLOBAL']
    if settings.get('region') not in valid_regions:
        return False, f"Invalid region: {settings.get('region')}"
    
    # Valid universes
    valid_universes = ['TOP3000', 'TOP1000', 'TOP500', 'TOP100', 'ALL']
    if settings.get('universe') not in valid_universes:
        return False, f"Invalid universe: {settings.get('universe')}"
    
    # Valid neutralization
    valid_neutralization = ['INDUSTRY', 'SECTOR', 'MARKET', 'NONE']
    if settings.get('neutralization') not in valid_neutralization:
        return False, f"Invalid neutralization: {settings.get('neutralization')}"
    
    # Numeric fields
    if not isinstance(settings.get('delay', 1), int) or settings.get('delay', 1) < 0:
        return False, "Delay must be a non-negative integer"
    
    if not isinstance(settings.get('decay', 0), int) or settings.get('decay', 0) < 0:
        return False, "Decay must be a non-negative integer"
    
    # Truncation between 0 and 1
    truncation = settings.get('truncation', 0.08)
    if not isinstance(truncation, (int, float)) or truncation < 0 or truncation > 1:
        return False, "Truncation must be between 0 and 1"
    
    return True, None

def extract_parameters_from_expression(expression: str) -> List[Tuple[int, int, int]]:
    """
    Extract numeric parameters and their positions from an expression.
    
    Args:
        expression: Alpha expression to analyze
        
    Returns:
        List of tuples (value, start_position, end_position)
    """
    # Find numeric parameters that aren't part of variable names
    pattern = r'(?<=[,()\s])\d+(?![a-zA-Z])'
    parameters = []
    
    for match in re.finditer(pattern, expression):
        value = int(match.group())
        start_pos = match.start()
        end_pos = match.end()
        parameters.append((value, start_pos, end_pos))
    
    return parameters

def create_expression_variant(base_expression: str, positions: List[Tuple[int, int]], params: List[int]) -> str:
    """
    Create a new expression with substituted parameters.
    
    Args:
        base_expression: Original expression
        positions: List of (start, end) positions to replace
        params: New parameter values
        
    Returns:
        New expression with substituted parameters
    """
    result = base_expression
    offset = 0  # Account for length changes
    
    # Sort positions in reverse order to avoid changing positions
    # of subsequent replacements
    sorted_positions = sorted(positions, reverse=True)
    
    for (start, end), new_value in zip(sorted_positions, params):
        new_str = str(new_value)
        adjusted_start = start
        adjusted_end = end
        
        result = result[:adjusted_start] + new_str + result[adjusted_end:]
    
    return result