#!/usr/bin/env python3
"""
Pathway Map Builder
Extracts metabolite-pathway mappings from HMDB cleaned JSON data.

Input: data/hmdb_metabolites_serum_cleaned.json
Output: storage/metabolite_pathway_map.json

Structure:
{
  "HMDB0000001": ["Pathway 1", "Pathway 2", ...],
  "HMDB0000002": ["Pathway 3", ...],
  ...
}
"""

import json
from pathlib import Path
from typing import Dict, List


def extract_pathway_names(pathways_data: List) -> List[str]:
    """
    Extract pathway names from the pathways field.
    
    Handles both formats:
    - List of dicts: [{'name': '...', 'smpdb_id': '...', 'kegg_map_id': '...'}, ...]
    - List of strings: ['Pathway 1', 'Pathway 2', ...]
    
    Args:
        pathways_data: List of pathway information (dicts or strings)
        
    Returns:
        List of pathway names (strings)
    """
    pathway_names = []
    
    if not pathways_data:
        return pathway_names
    
    for pathway in pathways_data:
        if isinstance(pathway, dict):
            # New format: extract 'name' field
            name = pathway.get('name', '')
            if name:
                pathway_names.append(name)
        elif isinstance(pathway, str):
            # Old format: already a string
            if pathway:
                pathway_names.append(pathway)
    
    return pathway_names


def build_pathway_map(
    input_json: str = "data/hmdb_metabolites_serum_cleaned.json",
    output_json: str = "storage/metabolite_pathway_map.json"
) -> Dict[str, List[str]]:
    """
    Build metabolite-pathway mapping from HMDB cleaned JSON.
    
    Args:
        input_json: Path to input JSON file
        output_json: Path to output JSON file
        
    Returns:
        Dictionary mapping HMDB IDs to pathway names
    """
    print("="*70)
    print("  Pathway Map Builder")
    print("="*70)
    
    # Load input data
    print(f"\nLoading data from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        metabolites = json.load(f)
    
    print(f"Loaded {len(metabolites):,} metabolites")
    
    # Build pathway map
    print("\nBuilding pathway map...")
    pathway_map = {}
    
    metabolites_with_pathways = 0
    total_pathway_entries = 0
    
    for metabolite in metabolites:
        hmdb_id = metabolite.get('hmdb_id')
        pathways_data = metabolite.get('pathways', [])
        
        if not hmdb_id:
            continue
        
        # Extract pathway names
        pathway_names = extract_pathway_names(pathways_data)
        
        if pathway_names:
            pathway_map[hmdb_id] = pathway_names
            metabolites_with_pathways += 1
            total_pathway_entries += len(pathway_names)
    
    # Statistics
    print("\n" + "-"*70)
    print("Statistics:")
    print("-"*70)
    print(f"Total metabolites: {len(metabolites):,}")
    print(f"Metabolites with pathways: {metabolites_with_pathways:,}")
    print(f"Metabolites without pathways: {len(metabolites) - metabolites_with_pathways:,}")
    print(f"Total pathway entries: {total_pathway_entries:,}")
    print(f"Average pathways per metabolite: {total_pathway_entries / metabolites_with_pathways:.1f}" 
          if metabolites_with_pathways > 0 else "N/A")
    print(f"Coverage: {metabolites_with_pathways / len(metabolites) * 100:.1f}%")
    
    # Show sample entries
    print("\n" + "-"*70)
    print("Sample entries:")
    print("-"*70)
    
    sample_count = 0
    for hmdb_id, pathways in pathway_map.items():
        if sample_count >= 5:
            break
        print(f"\n{hmdb_id}:")
        for i, pathway in enumerate(pathways[:3], 1):
            print(f"  {i}. {pathway}")
        if len(pathways) > 3:
            print(f"  ... and {len(pathways) - 3} more")
        sample_count += 1
    
    # Save output
    print("\n" + "-"*70)
    print(f"Saving to {output_json}...")
    
    # Ensure output directory exists
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(pathway_map, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(pathway_map):,} metabolite-pathway mappings")
    
    # File size
    file_size = output_path.stat().st_size
    print(f"✓ File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    
    print("\n" + "="*70)
    print("✓ Pathway map built successfully!")
    print("="*70)
    
    return pathway_map


def query_pathways(hmdb_id: str, pathway_map_file: str = "storage/metabolite_pathway_map.json") -> List[str]:
    """
    Query pathways for a specific metabolite.
    
    Args:
        hmdb_id: HMDB ID to query
        pathway_map_file: Path to pathway map JSON file
        
    Returns:
        List of pathway names for the metabolite
    """
    with open(pathway_map_file, 'r', encoding='utf-8') as f:
        pathway_map = json.load(f)
    
    return pathway_map.get(hmdb_id, [])


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_json = sys.argv[1]
        output_json = sys.argv[2] if len(sys.argv) > 2 else "storage/metabolite_pathway_map.json"
    else:
        input_json = "data/hmdb_metabolites_serum_cleaned.json"
        output_json = "storage/metabolite_pathway_map.json"
    
    # Build pathway map
    pathway_map = build_pathway_map(input_json, output_json)
    
    # Test query
    print("\n" + "="*70)
    print("Test Query:")
    print("="*70)
    
    test_id = "HMDB0000001"
    pathways = query_pathways(test_id, output_json)
    
    print(f"\nQuery: {test_id}")
    if pathways:
        print(f"Found {len(pathways)} pathways:")
        for i, pathway in enumerate(pathways, 1):
            print(f"  {i}. {pathway}")
    else:
        print("No pathways found")
    
    print("\n" + "="*70)
