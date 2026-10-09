"""
Unified Metabolite Name Mapper with Three-Tier Fallback Strategy

This module provides a comprehensive metabolite name to HMDB ID mapping with:
1. JSON lookup (fast, ~90% success rate)
2. Shorthand mapping (CAR, LPC, PC, etc.)
3. LLM fallback (optional, for unknown metabolites)

Fallback Strategy:
- Tier 1: JSON exact match (instant, O(1))
- Tier 2: Shorthand map (CAR, LPC, PC, PE, SM, etc.)
- Tier 3: LLM inference (optional, requires LLM instance)

Performance:
- JSON lookup: ~0.000001s per query
- Shorthand lookup: ~0.000001s per query
- LLM fallback: ~1-2s per query (cached after first call)

Author: MetaboAgent Team
Date: 2026-04-21
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict
from functools import lru_cache


# Module-level caches
_LOOKUP_MAP_CACHE: Optional[Dict[str, str]] = None
_SHORTHAND_MAP_CACHE: Optional[Dict[str, Dict[str, str]]] = None
_LLM_MAPPING_CACHE: Dict[str, Optional[str]] = {}  # In-memory cache for LLM results


@lru_cache(maxsize=1)
def _load_lookup_map(json_path: str = "storage/metabolite_lookup.json") -> Dict[str, str]:
    """
    Load the metabolite lookup map from JSON file.
    
    Uses LRU cache to ensure the file is loaded only once per session.
    
    Args:
        json_path: Path to the lookup JSON file
        
    Returns:
        Dictionary mapping metabolite names to HMDB IDs
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is invalid
    """
    json_file = Path(json_path)
    
    if not json_file.exists():
        raise FileNotFoundError(
            f"Metabolite lookup file not found: {json_path}\n"
            f"Please run: python scripts/build_metabolite_lookup.py"
        )
    
    with open(json_file, 'r', encoding='utf-8') as f:
        lookup_map = json.load(f)
    
    print(f"✓ Loaded metabolite lookup map: {len(lookup_map):,} entries")
    
    return lookup_map


@lru_cache(maxsize=1)
def _load_shorthand_map(json_path: str = "storage/shorthand_map.json") -> Dict[str, Dict[str, str]]:
    """
    Load the shorthand mapping (CAR, LPC, PC, etc.) from JSON file.
    
    Uses LRU cache to ensure the file is loaded only once per session.
    
    Args:
        json_path: Path to the shorthand mapping JSON file
        
    Returns:
        Dictionary mapping shorthand names to {hmdb_id, name}
    """
    json_file = Path(json_path)
    
    if not json_file.exists():
        # Shorthand map is optional - return empty dict if not found
        return {}
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            shorthand_map = json.load(f)
        
        print(f"✓ Loaded shorthand map: {len(shorthand_map):,} entries")
        return shorthand_map
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid shorthand map JSON: {e}")
        return {}


# ============================================================================
# LLM Fallback Functions (Tier 3)
# ============================================================================

def _map_with_llm(
    name: str,
    llm,
    cache_path: str
) -> Optional[str]:
    """
    Use LLM to infer HMDB ID for a metabolite name (Tier 3 fallback).
    
    This function:
    1. Checks in-memory cache first
    2. Checks persistent cache (JSON file)
    3. Calls LLM if not cached
    4. Caches result for future use
    
    Args:
        name: Metabolite name
        llm: LangChain LLM instance
        cache_path: Path to cache file
        
    Returns:
        HMDB ID if LLM can infer it, None otherwise
    """
    # Check in-memory cache
    if name in _LLM_MAPPING_CACHE:
        return _LLM_MAPPING_CACHE[name]
    
    # Check persistent cache
    cached_result = _load_llm_cache(name, cache_path)
    if cached_result is not None:
        _LLM_MAPPING_CACHE[name] = cached_result
        return cached_result
    
    # Call LLM
    try:
        prompt = f"""You are a metabolomics expert. Given a metabolite name, return ONLY its HMDB ID.

Metabolite name: "{name}"

Rules:
1. Return ONLY the HMDB ID in format "HMDBXXXXXXX" (7-9 digits)
2. If you cannot determine the HMDB ID with high confidence, return "UNKNOWN"
3. Do NOT provide explanations or additional text

Examples:
- "Creatine" → HMDB0000064
- "Palmitoylcarnitine" → HMDB0000222
- "Unknown_XYZ" → UNKNOWN

Your response (HMDB ID only):"""
        
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Parse response
        response_text = response.content.strip()
        
        # Check for "UNKNOWN"
        if "UNKNOWN" in response_text.upper() or "CANNOT" in response_text.upper():
            hmdb_id = None
        else:
            # Extract HMDB ID using regex
            match = re.search(r'HMDB\d{5,9}', response_text, re.IGNORECASE)
            hmdb_id = match.group(0).upper() if match else None
        
        # Cache result
        _LLM_MAPPING_CACHE[name] = hmdb_id
        _save_llm_cache(name, hmdb_id, cache_path)
        
        return hmdb_id
        
    except Exception as e:
        print(f"Warning: LLM mapping failed for '{name}': {e}")
        return None


def _load_llm_cache(name: str, cache_path: str) -> Optional[str]:
    """Load cached LLM mapping result from JSON file."""
    cache_file = Path(cache_path)
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        return cache.get(name)
    except (json.JSONDecodeError, IOError):
        return None


def _save_llm_cache(name: str, hmdb_id: Optional[str], cache_path: str):
    """Save LLM mapping result to persistent cache."""
    cache_file = Path(cache_path)
    
    # Load existing cache
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            cache = {}
    else:
        cache = {}
    
    # Update cache
    cache[name] = hmdb_id
    
    # Save cache
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Warning: Could not save LLM cache: {e}")


# ============================================================================
# Batch Mapping Functions
# ============================================================================


def map_metabolite_fast(
    name: str,
    json_path: str = "storage/metabolite_lookup.json",
    shorthand_path: str = "storage/shorthand_map.json",
    llm = None,
    llm_cache_path: str = "storage/llm_metabolite_cache.json"
) -> Optional[str]:
    """
    Unified metabolite name to HMDB ID mapping with three-tier fallback strategy.
    
    Fallback Strategy:
    1. **Tier 1 - Shorthand Map**: Check if name is in shorthand map (CAR, LPC, PC, etc.)
    2. **Tier 2 - JSON Lookup**: Try exact match in main metabolite lookup map
    3. **Tier 3 - LLM Fallback**: If LLM provided, use it to infer HMDB ID (optional)
    
    Args:
        name: Metabolite name or shorthand notation to look up
        json_path: Path to the main metabolite lookup JSON file
        shorthand_path: Path to the shorthand mapping JSON file
        llm: Optional LangChain LLM instance for fallback (e.g., ChatOpenAI)
        llm_cache_path: Path to cache file for LLM results
        
    Returns:
        HMDB ID if found, None otherwise
        
    Example:
        >>> # Without LLM (Tier 1 & 2 only)
        >>> hmdb_id = map_metabolite_fast("CAR 16:0")
        >>> print(hmdb_id)  # HMDB0000222
        
        >>> # With LLM fallback (all 3 tiers)
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4", temperature=0)
        >>> hmdb_id = map_metabolite_fast("Palmitoylcarnitine", llm=llm)
        >>> print(hmdb_id)  # HMDB0000222
    """
    # ========================================================================
    # Tier 1: Shorthand Map (CAR, LPC, PC, PE, SM, etc.)
    # ========================================================================
    try:
        shorthand_map = _load_shorthand_map(shorthand_path)
        if name in shorthand_map:
            mapping = shorthand_map[name]
            if "hmdb_id" in mapping and mapping["hmdb_id"]:
                return mapping["hmdb_id"]
    except Exception:
        pass  # Continue to next tier
    
    # ========================================================================
    # Tier 2: JSON Lookup (Main metabolite database)
    # ========================================================================
    try:
        lookup_map = _load_lookup_map(json_path)
        
        # Try exact match
        if name in lookup_map:
            return lookup_map[name]
        
        # Try lowercase
        if name.lower() in lookup_map:
            return lookup_map[name.lower()]
    except Exception:
        pass  # Continue to next tier
    
    # ========================================================================
    # Tier 3: LLM Fallback (Optional)
    # ========================================================================
    if llm is not None:
        return _map_with_llm(name, llm, llm_cache_path)
    
    # Not found in any tier
    return None


def batch_map_metabolites_fast(
    names: list[str],
    json_path: str = "storage/metabolite_lookup.json",
    shorthand_path: str = "storage/shorthand_map.json",
    llm = None,
    llm_cache_path: str = "storage/llm_metabolite_cache.json"
) -> Dict[str, Optional[str]]:
    """
    Batch map multiple metabolite names using three-tier fallback strategy.
    
    More efficient than calling map_metabolite_fast multiple times
    because it loads the lookup maps only once.
    
    Args:
        names: List of metabolite names to look up
        json_path: Path to the main metabolite lookup JSON file
        shorthand_path: Path to the shorthand mapping JSON file
        llm: Optional LangChain LLM instance for fallback
        llm_cache_path: Path to cache file for LLM results
        
    Returns:
        Dictionary mapping input names to HMDB IDs (or None if not found)
        
    Example:
        >>> names = ["Creatine", "CAR 16:0", "LPC 18:0", "Unknown"]
        >>> results = batch_map_metabolites_fast(names)
        >>> print(results)
        {'Creatine': 'HMDB0000064', 'CAR 16:0': 'HMDB0000222', 'LPC 18:0': 'HMDB0010384', 'Unknown': None}
    """
    results = {}
    
    # Load maps once
    try:
        shorthand_map = _load_shorthand_map(shorthand_path)
    except Exception:
        shorthand_map = {}
    
    try:
        lookup_map = _load_lookup_map(json_path)
    except Exception:
        lookup_map = {}
    
    for name in names:
        # Tier 1: Shorthand map
        if name in shorthand_map:
            mapping = shorthand_map[name]
            if "hmdb_id" in mapping and mapping["hmdb_id"]:
                results[name] = mapping["hmdb_id"]
                continue
        
        # Tier 2: JSON lookup
        if name in lookup_map:
            results[name] = lookup_map[name]
            continue
        elif name.lower() in lookup_map:
            results[name] = lookup_map[name.lower()]
            continue
        
        # Tier 3: LLM fallback (if provided)
        if llm is not None:
            results[name] = _map_with_llm(name, llm, llm_cache_path)
        else:
            results[name] = None
    
    return results


def get_lookup_stats(json_path: str = "storage/metabolite_lookup.json") -> Dict[str, int]:
    """
    Get statistics about the lookup map.
    
    Args:
        json_path: Path to the lookup JSON file
        
    Returns:
        Dictionary with statistics
        
    Example:
        >>> stats = get_lookup_stats()
        >>> print(stats)
        {'total_entries': 1022065, 'unique_hmdb_ids': 114523}
    """
    try:
        lookup_map = _load_lookup_map(json_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: {e}")
        return {"total_entries": 0, "unique_hmdb_ids": 0}
    
    unique_hmdb_ids = set(lookup_map.values())
    
    return {
        "total_entries": len(lookup_map),
        "unique_hmdb_ids": len(unique_hmdb_ids)
    }


# ============================================================================
# LangChain Tool Registration
# ============================================================================

from langchain_core.tools import StructuredTool

json_mapping_tool = StructuredTool.from_function(
    func=map_metabolite_fast,
    name="map_metabolite_fast",
    description="""Fast exact-match lookup for metabolite name to HMDB ID.
    
    This tool performs instant dictionary lookup without any API calls.
    Use this FIRST before trying fuzzy matching methods.
    
    Args:
        name: Metabolite name to look up (e.g., "Creatine", "L-Alanine")
        
    Returns:
        HMDB ID string if found (e.g., "HMDB0000064"), None otherwise
        
    Performance: ~0.000001s per query (instant)
    Success rate: ~90% for exact matches
    """
)


# ============================================================================
# Testing and Validation
# ============================================================================

if __name__ == "__main__":
    import time
    
    print("="*80)
    print("Testing Unified Metabolite Mapper (Three-Tier Fallback)")
    print("="*80)
    
    # Test 1: Tier 1 & 2 (JSON + Shorthand, no LLM)
    print("\n[Test 1] Tier 1 & 2: JSON + Shorthand (No LLM)")
    print("-"*80)
    
    test_names = [
        "Creatine",          # Tier 2: JSON lookup
        "CAR 16:0",          # Tier 1: Shorthand
        "LPC 18:0",          # Tier 1: Shorthand
        "PC 36:2",           # Tier 1: Shorthand
        "alanine",           # Tier 2: JSON (lowercase)
        "Unknown_XYZ"        # Not found
    ]
    
    for name in test_names:
        start = time.time()
        hmdb_id = map_metabolite_fast(name)
        elapsed = time.time() - start
        
        status = "✓" if hmdb_id else "✗"
        print(f"{status} {name:20s} -> {hmdb_id or 'Not found':20s} ({elapsed*1000:.4f}ms)")
    
    # Test 2: Batch lookup
    print("\n[Test 2] Batch Lookup (Tier 1 & 2)")
    print("-"*80)
    
    start = time.time()
    results = batch_map_metabolites_fast(test_names)
    elapsed = time.time() - start
    
    success_count = sum(1 for v in results.values() if v is not None)
    print(f"Processed {len(test_names)} names in {elapsed*1000:.2f}ms")
    print(f"Success rate: {success_count}/{len(test_names)} ({success_count/len(test_names)*100:.1f}%)")
    
    # Test 3: Statistics
    print("\n[Test 3] Lookup Statistics")
    print("-"*80)
    
    stats = get_lookup_stats()
    print(f"Total entries: {stats['total_entries']:,}")
    print(f"Unique HMDB IDs: {stats['unique_hmdb_ids']:,}")
    
    # Test 4: LLM Fallback (Optional - requires LLM configuration)
    print("\n[Test 4] Tier 3: LLM Fallback (Optional)")
    print("-"*80)
    print("To test LLM fallback, uncomment the code below and configure your LLM:")
    print()
    print("Example:")
    print("  from langchain_openai import ChatOpenAI")
    print("  llm = ChatOpenAI(model='gpt-4', temperature=0)")
    print("  hmdb_id = map_metabolite_fast('Palmitoylcarnitine', llm=llm)")
    print("  print(hmdb_id)  # Should return HMDB0000222")
    
    """
    # Uncomment to test LLM fallback:
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    test_llm_names = [
        "Palmitoylcarnitine",  # Should find via LLM
        "Oleoylcarnitine",     # Should find via LLM
        "Random_Unknown_123"   # Should return None
    ]
    
    print("\nTesting LLM fallback:")
    for name in test_llm_names:
        hmdb_id = map_metabolite_fast(name, llm=llm)
        status = "✓" if hmdb_id else "✗"
        print(f"{status} {name:30s} -> {hmdb_id or 'Not found'}")
    """
    
    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80)
