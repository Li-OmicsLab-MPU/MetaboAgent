"""
Enhanced Metabolite Name Mapper with Bidirectional Mapping

Supports both forward (name -> HMDB ID) and reverse (HMDB ID -> name) mapping.
This is essential for matching Phase 0 prior biomarkers with dataset features.

Author: MetaboAgent Team
Date: 2026-04-15
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.feature_name_standardizer import standardize_feature_name


_HMDB_PATTERN = re.compile(r"(HMDB\d{7})", re.IGNORECASE)
_STEREO_PREFIX_PATTERN = re.compile(r"^(?:D|L|DL|D_L|L_D|R|S|RS|R_S|S_R)_+")


def _extract_hmdb_token(value: str) -> Optional[str]:
    """Extract a normalized HMDB token from free text or a feature name."""
    text = str(value or "").strip().upper()
    if not text:
        return None
    match = _HMDB_PATTERN.search(text)
    return match.group(1).upper() if match else None


def _canonical_variants(value: str) -> Set[str]:
    """Build conservative canonical variants for matching biomarker aliases."""
    standardized = standardize_feature_name(str(value or ""))
    if not standardized:
        return set()

    variants = {standardized}
    stripped = standardized
    while True:
        updated = _STEREO_PREFIX_PATTERN.sub("", stripped)
        if not updated or updated == stripped:
            break
        variants.add(updated)
        stripped = updated
    return {variant for variant in variants if variant}


def _build_column_index(column_names: List[str]) -> Dict[str, Any]:
    """Pre-compute lookup tables for dataset feature columns."""
    by_upper: Dict[str, str] = {}
    by_hmdb: Dict[str, List[str]] = defaultdict(list)
    by_variant: Dict[str, List[str]] = defaultdict(list)
    records: List[Dict[str, Any]] = []

    for column in column_names:
        column_name = str(column or "").strip()
        if not column_name:
            continue
        upper_name = column_name.upper()
        hmdb_token = _extract_hmdb_token(column_name)
        variants = _canonical_variants(column_name)

        by_upper.setdefault(upper_name, column_name)
        if hmdb_token:
            by_hmdb[hmdb_token].append(column_name)
        for variant in variants:
            by_variant[variant].append(column_name)

        records.append(
            {
                "column": column_name,
                "upper": upper_name,
                "hmdb": hmdb_token,
                "variants": variants,
            }
        )

    return {
        "by_upper": by_upper,
        "by_hmdb": dict(by_hmdb),
        "by_variant": dict(by_variant),
        "records": records,
    }


class MetaboliteNameMapper:
    """
    Bidirectional metabolite name mapper.
    
    Supports:
    - Forward mapping: metabolite name -> HMDB ID
    - Reverse mapping: HMDB ID -> metabolite name(s)
    - Fuzzy matching: find best match for a name in dataset columns
    """
    
    def __init__(
        self, 
        json_path: str = "storage/metabolite_lookup.json",
        shorthand_path: str = "storage/shorthand_map.json"
    ):
        """
        Initialize the mapper.
        
        Args:
            json_path: Path to the metabolite lookup JSON file
            shorthand_path: Path to the shorthand mapping JSON file (CAR, LPC, PC, etc.)
        """
        self.json_path = json_path
        self.shorthand_path = shorthand_path
        self._forward_map: Optional[Dict[str, str]] = None
        self._reverse_map: Optional[Dict[str, List[str]]] = None
        self._shorthand_map: Optional[Dict[str, Dict[str, str]]] = None
        self._load_maps()
    
    def _load_maps(self):
        """Load forward, reverse, and shorthand mapping dictionaries."""
        # Load main metabolite lookup
        json_file = Path(self.json_path)
        
        if not json_file.exists():
            raise FileNotFoundError(
                f"Metabolite lookup file not found: {self.json_path}\n"
                f"Please run: python scripts/build_metabolite_lookup.py"
            )
        
        with open(json_file, 'r', encoding='utf-8') as f:
            self._forward_map = json.load(f)
        
        # Build reverse map: HMDB ID -> list of names
        self._reverse_map = {}
        for name, hmdb_id in self._forward_map.items():
            if hmdb_id not in self._reverse_map:
                self._reverse_map[hmdb_id] = []
            self._reverse_map[hmdb_id].append(name)
        
        # Load shorthand map (CAR, LPC, PC, etc.)
        shorthand_file = Path(self.shorthand_path)
        if shorthand_file.exists():
            with open(shorthand_file, 'r', encoding='utf-8') as f:
                self._shorthand_map = json.load(f)
            print(f"✓ Loaded metabolite mapper:")
            print(f"  - Forward entries: {len(self._forward_map):,}")
            print(f"  - Unique HMDB IDs: {len(self._reverse_map):,}")
            print(f"  - Shorthand entries: {len(self._shorthand_map):,}")
        else:
            self._shorthand_map = {}
            print(f"✓ Loaded metabolite mapper:")
            print(f"  - Forward entries: {len(self._forward_map):,}")
            print(f"  - Unique HMDB IDs: {len(self._reverse_map):,}")
            print(f"  - Shorthand map not found: {self.shorthand_path}")
    
    def normalize_shorthand(self, feature_name: str) -> str:
        """
        Normalize shorthand feature names (CAR, LPC, PC, PE, SM, etc.) to HMDB IDs or metabolite names.
        
        This method converts lipidomics shorthand notation to standardized identifiers:
        - "CAR 16:0" -> "HMDB0000222" (if HMDB ID available)
        - "CAR 16:0" -> "Palmitoylcarnitine" (if only name available)
        - "Unknown Feature" -> "Unknown Feature" (if not in shorthand map)
        
        Args:
            feature_name: Original feature name (e.g., "CAR 16:0", "LPC 18:0")
            
        Returns:
            Normalized feature name (HMDB ID preferred, then metabolite name, then original)
        """
        if not self._shorthand_map or feature_name not in self._shorthand_map:
            return feature_name
        
        mapping = self._shorthand_map[feature_name]
        
        # Priority 1: Return HMDB ID if available
        if "hmdb_id" in mapping and mapping["hmdb_id"]:
            return mapping["hmdb_id"]
        
        # Priority 2: Return metabolite name if available
        if "name" in mapping and mapping["name"]:
            return mapping["name"]
        
        # Priority 3: Return original if no mapping found
        return feature_name
    
    def name_to_hmdb(self, name: str) -> Optional[str]:
        """
        Map metabolite name to HMDB ID.
        
        This method tries multiple strategies:
        1. Normalize shorthand notation (CAR, LPC, PC, etc.) first
        2. Try exact match in forward map
        3. Try lowercase match in forward map
        
        Args:
            name: Metabolite name or shorthand notation
            
        Returns:
            HMDB ID if found, None otherwise
        """
        # Strategy 1: Try shorthand normalization first
        normalized = self.normalize_shorthand(name)
        
        # If normalized to HMDB ID, return it directly
        if normalized.startswith("HMDB"):
            return normalized
        
        # If normalized to a metabolite name, use it for lookup
        lookup_name = normalized
        
        # Strategy 2: Try exact match
        if lookup_name in self._forward_map:
            return self._forward_map[lookup_name]
        
        # Strategy 3: Try lowercase
        if lookup_name.lower() in self._forward_map:
            return self._forward_map[lookup_name.lower()]
        
        return None
    
    def hmdb_to_names(self, hmdb_id: str) -> List[str]:
        """
        Map HMDB ID to all known metabolite names.
        
        Args:
            hmdb_id: HMDB ID (e.g., "HMDB0000064")
            
        Returns:
            List of metabolite names (empty if not found)
        """
        return self._reverse_map.get(hmdb_id, [])
    
    def match_in_columns(
        self, 
        metabolite_name: str, 
        column_names: List[str],
        match_threshold: float = 0.8
    ) -> Optional[str]:
        """
        Find the best matching column for a metabolite name.
        
        This function tries multiple strategies:
        1. Exact HMDB ID match
        2. Partial name match in column
        3. Column name in metabolite name
        
        Args:
            metabolite_name: Metabolite name to match
            column_names: List of column names in dataset
            match_threshold: Minimum similarity threshold (not used yet)
            
        Returns:
            Best matching column name, or None if no match found
        """
        # Strategy 1: Try to get HMDB ID and match exactly
        hmdb_id = self.name_to_hmdb(metabolite_name)
        if hmdb_id and hmdb_id in column_names:
            return hmdb_id
        
        # Strategy 2: Partial match - metabolite name in column
        metabolite_lower = metabolite_name.lower()
        for col in column_names:
            col_lower = col.lower()
            if metabolite_lower in col_lower or col_lower in metabolite_lower:
                return col
        
        # Strategy 3: If column is HMDB ID, check if it maps to this metabolite
        for col in column_names:
            if col.startswith('HMDB'):
                col_names = self.hmdb_to_names(col)
                for col_name in col_names:
                    if col_name.lower() == metabolite_lower:
                        return col
        
        return None
    
    def batch_match_in_columns(
        self,
        metabolite_names: List[str],
        column_names: List[str]
    ) -> Dict[str, Optional[str]]:
        """
        Batch match multiple metabolite names to dataset columns.
        
        Args:
            metabolite_names: List of metabolite names to match
            column_names: List of column names in dataset
            
        Returns:
            Dictionary mapping metabolite names to matched column names
        """
        results = {}
        for name in metabolite_names:
            matched_col = self.match_in_columns(name, column_names)
            results[name] = matched_col
        
        return results

    def resolve_biomarker_in_columns(
        self,
        biomarker_term: str,
        column_names: List[str],
    ) -> Dict[str, Any]:
        """
        Resolve one Phase 0 biomarker term onto a dataset feature column.

        Matching priority is intentionally conservative:
        1. Exact column match
        2. HMDB-based match, including aliases that map to the same HMDB ID
        3. Canonical name match after standardization / stereo-prefix stripping
        4. Canonical containment match as a last resort
        """
        biomarker = str(biomarker_term or "").strip()
        if not biomarker:
            return {
                "matched_column": None,
                "match_type": "empty_term",
                "hmdb_id": None,
                "candidate_aliases": [],
                "candidate_variants": [],
            }

        column_index = _build_column_index(column_names)
        by_upper = column_index["by_upper"]
        by_hmdb = column_index["by_hmdb"]
        by_variant = column_index["by_variant"]
        column_records = column_index["records"]

        biomarker_upper = biomarker.upper()
        if biomarker_upper in by_upper:
            return {
                "matched_column": by_upper[biomarker_upper],
                "match_type": "exact_column",
                "hmdb_id": _extract_hmdb_token(biomarker),
                "candidate_aliases": [biomarker],
                "candidate_variants": sorted(_canonical_variants(biomarker)),
            }

        hmdb_ids: List[str] = []
        direct_hmdb = _extract_hmdb_token(biomarker)
        if direct_hmdb:
            hmdb_ids.append(direct_hmdb)
        mapped_hmdb = self.name_to_hmdb(biomarker)
        if mapped_hmdb:
            hmdb_ids.append(mapped_hmdb.upper())
        hmdb_ids = list(dict.fromkeys([hmdb for hmdb in hmdb_ids if hmdb]))

        alias_names: List[str] = [biomarker]
        for hmdb_id in hmdb_ids:
            alias_names.extend(self.hmdb_to_names(hmdb_id))
        alias_names = list(dict.fromkeys([str(name or "").strip() for name in alias_names if str(name or "").strip()]))

        variant_to_origin: Dict[str, str] = {}
        for alias_name in alias_names:
            for variant in _canonical_variants(alias_name):
                variant_to_origin.setdefault(variant, alias_name)

        for hmdb_id in hmdb_ids:
            hmdb_matches = list(by_hmdb.get(hmdb_id, []) or [])
            if hmdb_matches:
                return {
                    "matched_column": hmdb_matches[0],
                    "match_type": "hmdb_id",
                    "hmdb_id": hmdb_id,
                    "candidate_aliases": alias_names,
                    "candidate_variants": sorted(variant_to_origin),
                }

        for variant, origin in variant_to_origin.items():
            exact_matches = list(by_variant.get(variant, []) or [])
            if exact_matches:
                match_type = "canonical_name" if origin == biomarker else "hmdb_alias"
                return {
                    "matched_column": exact_matches[0],
                    "match_type": match_type,
                    "hmdb_id": hmdb_ids[0] if hmdb_ids else None,
                    "candidate_aliases": alias_names,
                    "candidate_variants": sorted(variant_to_origin),
                }

        best_partial_match: Optional[Tuple[int, int, str, str]] = None
        for variant, origin in variant_to_origin.items():
            if len(variant) < 6:
                continue
            for record in column_records:
                for column_variant in record["variants"]:
                    if variant in column_variant or column_variant in variant:
                        score = (
                            abs(len(column_variant) - len(variant)),
                            len(column_variant),
                            record["column"],
                            origin,
                        )
                        if best_partial_match is None or score < best_partial_match:
                            best_partial_match = score

        if best_partial_match is not None:
            return {
                "matched_column": best_partial_match[2],
                "match_type": "canonical_partial",
                "hmdb_id": hmdb_ids[0] if hmdb_ids else None,
                "candidate_aliases": alias_names,
                "candidate_variants": sorted(variant_to_origin),
            }

        return {
            "matched_column": None,
            "match_type": "unresolved",
            "hmdb_id": hmdb_ids[0] if hmdb_ids else None,
            "candidate_aliases": alias_names,
            "candidate_variants": sorted(variant_to_origin),
        }
    
    def get_mapping_report(
        self,
        metabolite_names: List[str],
        column_names: List[str]
    ) -> Dict:
        """
        Generate a detailed mapping report.
        
        Args:
            metabolite_names: List of metabolite names to match
            column_names: List of column names in dataset
            
        Returns:
            Dictionary with mapping statistics and details
        """
        matches = self.batch_match_in_columns(metabolite_names, column_names)
        
        matched = {k: v for k, v in matches.items() if v is not None}
        unmatched = {k: v for k, v in matches.items() if v is None}
        
        return {
            "total_metabolites": len(metabolite_names),
            "matched_count": len(matched),
            "unmatched_count": len(unmatched),
            "match_rate": len(matched) / len(metabolite_names) if metabolite_names else 0,
            "matched_pairs": matched,
            "unmatched_metabolites": list(unmatched.keys())
        }


# ============================================================================
# Convenience Functions
# ============================================================================

_GLOBAL_MAPPER: Optional[MetaboliteNameMapper] = None


def extract_phase0_biomarker_terms(phase0_output: Optional[Dict[str, Any]]) -> List[str]:
    """Return canonical identifier/name terms from all supported Phase 0 schemas."""
    if not isinstance(phase0_output, dict):
        return []
    feature_definitions = phase0_output.get("feature_definitions", {}) or {}
    sources = [
        phase0_output.get("confirmed_biomarkers", []),
        phase0_output.get("final_priors", []),
        phase0_output.get("prior_metabolites", []),
        phase0_output.get("ranked_biomarkers", []),
        feature_definitions.get("target_metabolites", []),
    ]
    terms: List[str] = []
    for source in sources:
        if not isinstance(source, list):
            continue
        for biomarker in source:
            if isinstance(biomarker, dict):
                for key in (
                    "id", "hmdb_id", "metabolite_id", "name",
                    "metabolite_name", "canonical_name",
                ):
                    term = str(biomarker.get(key, "") or "").strip()
                    if term:
                        terms.append(term)
            else:
                term = str(biomarker or "").strip()
                if term:
                    terms.append(term)
    return list(dict.fromkeys(terms))


def get_mapper(
    json_path: str = "storage/metabolite_lookup.json",
    shorthand_path: str = "storage/shorthand_map.json"
) -> MetaboliteNameMapper:
    """
    Get or create the global mapper instance.
    
    Args:
        json_path: Path to the metabolite lookup JSON file
        shorthand_path: Path to the shorthand mapping JSON file
        
    Returns:
        MetaboliteNameMapper instance
    """
    global _GLOBAL_MAPPER
    if _GLOBAL_MAPPER is None:
        _GLOBAL_MAPPER = MetaboliteNameMapper(json_path, shorthand_path)
    return _GLOBAL_MAPPER


def map_prior_biomarkers_to_columns(
    prior_biomarkers: List[str],
    dataset_columns: List[str],
    json_path: str = "storage/metabolite_lookup.json",
    shorthand_path: str = "storage/shorthand_map.json"
) -> Tuple[List[str], Dict]:
    """
    Map Phase 0 prior biomarkers to dataset column names.
    
    This is the main function to use for matching prior biomarkers
    with dataset features.
    
    Args:
        prior_biomarkers: List of metabolite names from Phase 0
        dataset_columns: List of column names in the dataset
        json_path: Path to the metabolite lookup JSON file
        shorthand_path: Path to the shorthand mapping JSON file
        
    Returns:
        Tuple of (matched_columns, mapping_report)
        - matched_columns: List of dataset column names that match prior biomarkers
        - mapping_report: Detailed report with statistics
        
    Example:
        >>> prior_biomarkers = ["Creatine", "L-Alanine", "Glucose"]
        >>> dataset_columns = ["HMDB0000064", "HMDB0001310", "Sample_ID", "Group"]
        >>> matched, report = map_prior_biomarkers_to_columns(prior_biomarkers, dataset_columns)
        >>> print(matched)
        ['HMDB0000064', 'HMDB0001310']
        >>> print(report['match_rate'])
        0.67
    """
    resolution = resolve_prior_biomarkers_to_columns(
        prior_biomarkers,
        dataset_columns,
        json_path=json_path,
        shorthand_path=shorthand_path,
    )
    return list(resolution.get("matched_columns", []) or []), dict(resolution.get("resolution_report", {}) or {})


def resolve_prior_biomarkers_to_columns(
    prior_biomarkers: List[str],
    dataset_columns: List[str],
    json_path: str = "storage/metabolite_lookup.json",
    shorthand_path: str = "storage/shorthand_map.json",
) -> Dict[str, Any]:
    """Resolve Phase 0 prior biomarkers onto dataset columns with detailed tracing."""
    mapper = get_mapper(json_path, shorthand_path)
    unique_biomarkers = list(dict.fromkeys([str(term or "").strip() for term in prior_biomarkers if str(term or "").strip()]))

    matched_columns: List[str] = []
    matched_pairs: Dict[str, str] = {}
    matched_details: Dict[str, Dict[str, Any]] = {}
    unmatched_biomarkers: List[str] = []
    match_type_counts: Dict[str, int] = {}

    for biomarker in unique_biomarkers:
        result = mapper.resolve_biomarker_in_columns(biomarker, dataset_columns)
        matched_column = str(result.get("matched_column") or "").strip()
        match_type = str(result.get("match_type") or "unresolved")
        match_type_counts[match_type] = match_type_counts.get(match_type, 0) + 1

        if matched_column:
            matched_columns.append(matched_column)
            matched_pairs[biomarker] = matched_column
            matched_details[biomarker] = {
                "matched_column": matched_column,
                "match_type": match_type,
                "hmdb_id": result.get("hmdb_id"),
                "candidate_aliases": list(result.get("candidate_aliases", []) or []),
                "candidate_variants": list(result.get("candidate_variants", []) or []),
            }
        else:
            unmatched_biomarkers.append(biomarker)

    unique_matched_columns = list(dict.fromkeys(matched_columns))
    resolution_report = {
        "total_metabolites": len(unique_biomarkers),
        "matched_count": len(matched_pairs),
        "unmatched_count": len(unmatched_biomarkers),
        "match_rate": len(matched_pairs) / len(unique_biomarkers) if unique_biomarkers else 0.0,
        "matched_pairs": matched_pairs,
        "matched_columns": unique_matched_columns,
        "matched_details": matched_details,
        "unmatched_metabolites": unmatched_biomarkers,
        "match_type_counts": match_type_counts,
    }
    return {
        "matched_columns": unique_matched_columns,
        "protected_features": unique_matched_columns,
        "available_features": unique_matched_columns,
        "resolution_report": resolution_report,
    }


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Testing Enhanced Metabolite Name Mapper")
    print("="*80)
    
    # Test 1: Shorthand normalization
    print("\n[Test 1] Shorthand Normalization (CAR features)")
    print("-"*80)
    
    mapper = MetaboliteNameMapper()
    
    test_shorthands = ["CAR 14:0", "CAR 16:0", "CAR 18:0", "CAR 18:1", "CAR 18:2", "CAR 20:0", "Unknown Feature"]
    for shorthand in test_shorthands:
        normalized = mapper.normalize_shorthand(shorthand)
        hmdb_id = mapper.name_to_hmdb(shorthand)
        print(f"  {shorthand:20s} -> normalized: {normalized:30s} -> HMDB: {hmdb_id}")
    
    # Test 2: Basic forward and reverse mapping
    print("\n[Test 2] Forward and Reverse Mapping")
    print("-"*80)
    
    # Forward mapping
    test_names = ["Creatine", "L-Alanine", "Glucose"]
    for name in test_names:
        hmdb_id = mapper.name_to_hmdb(name)
        print(f"  {name} -> {hmdb_id}")
        
        if hmdb_id:
            # Reverse mapping
            names = mapper.hmdb_to_names(hmdb_id)
            print(f"    {hmdb_id} -> {names[:3]}...")  # Show first 3 names
    
    # Test 3: Match in columns
    print("\n[Test 3] Match Prior Biomarkers to Dataset Columns")
    print("-"*80)
    
    prior_biomarkers = [
        "Creatine",
        "L-Alanine", 
        "Glucose",
        "3-Hydroxybutyric acid",
        "CAR 16:0",  # Test shorthand
        "CAR 18:1",  # Test shorthand
        "Unknown Metabolite"
    ]
    
    dataset_columns = [
        "Sample_ID",
        "Group",
        "HMDB0000064",  # Creatine
        "HMDB0001310",  # L-Alanine
        "Glucose",
        "CAR 16:0",
        "HMDB0000222",  # Palmitoylcarnitine (CAR 16:0)
        "HMDB0000688",  # Oleoylcarnitine (CAR 18:1)
        "L-Urobilin"
    ]
    
    matched_columns, report = map_prior_biomarkers_to_columns(
        prior_biomarkers, 
        dataset_columns
    )
    
    print(f"  Total prior biomarkers: {report['total_metabolites']}")
    print(f"  Matched: {report['matched_count']}")
    print(f"  Unmatched: {report['unmatched_count']}")
    print(f"  Match rate: {report['match_rate']:.1%}")
    print(f"\n  Matched pairs:")
    for biomarker, column in report['matched_pairs'].items():
        print(f"    '{biomarker}' -> '{column}'")
    
    if report['unmatched_metabolites']:
        print(f"\n  Unmatched metabolites:")
        for metabolite in report['unmatched_metabolites']:
            print(f"    - {metabolite}")
    
    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80)
