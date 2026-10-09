#!/usr/bin/env python3
"""
Metabolite-Pathway Map Builder (from PathBank)
Inverts the pathway_member_map.json to create a metabolite-centric view.

Strategy: Use high-quality PathBank data instead of messy HMDB XML parsing.

Input: storage/pathway_member_map.json
  Format: {"Pathway Name": ["HMDB0000001", "HMDB0000002", ...]}

Output: storage/pathbank_pathway_map.json
  Format: {"HMDB0000001": ["Pathway A", "Pathway B", ...]}

Logic:
  1. Load pathway_member_map.json
  2. Invert the dictionary structure
  3. For each pathway and its member list:
     - For each member HMDB ID:
       - Append the pathway name to that metabolite's pathway list
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def invert_pathway_map(
    input_json: str = "storage/pathway_member_map.json",
    output_json: str = "storage/pathbank_pathway_map.json"
) -> Dict[str, List[str]]:
    """
    Invert pathway-to-metabolites map to metabolite-to-pathways map.
    
    Args:
        input_json: Path to pathway_member_map.json
        output_json: Path to output pathbank_pathway_map.json
        
    Returns:
        Dictionary mapping HMDB IDs to pathway names
    """
    print("="*70)
    print("  Metabolite-Pathway Map Builder (from PathBank)")
    print("="*70)
    
    # Load input data
    print(f"\nLoading PathBank data from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        pathway_member_map = json.load(f)
    
    print(f"Loaded {len(pathway_member_map):,} pathways")
    
    # Invert the map
    print("\nInverting pathway map...")
    metabolite_pathway_map = defaultdict(list)
    
    total_members = 0
    pathways_processed = 0
    
    for pathway_name, member_ids in pathway_member_map.items():
        pathways_processed += 1
        
        if not member_ids:
            continue
        
        for hmdb_id in member_ids:
            if hmdb_id:  # Skip empty strings
                metabolite_pathway_map[hmdb_id].append(pathway_name)
                total_members += 1
        
        # Progress indicator
        if pathways_processed % 100 == 0:
            print(f"  Processed {pathways_processed:,} pathways...", end='\r')
    
    print(f"  Processed {pathways_processed:,} pathways... Done!")
    
    # Convert defaultdict to regular dict
    metabolite_pathway_map = dict(metabolite_pathway_map)
    
    # Statistics
    print("\n" + "-"*70)
    print("Statistics:")
    print("-"*70)
    print(f"Input pathways: {len(pathway_member_map):,}")
    print(f"Output metabolites: {len(metabolite_pathway_map):,}")
    print(f"Total pathway-metabolite relationships: {total_members:,}")
    
    if metabolite_pathway_map:
        pathway_counts = [len(pathways) for pathways in metabolite_pathway_map.values()]
        avg_pathways = sum(pathway_counts) / len(pathway_counts)
        max_pathways = max(pathway_counts)
        min_pathways = min(pathway_counts)
        
        print(f"Average pathways per metabolite: {avg_pathways:.1f}")
        print(f"Max pathways for a metabolite: {max_pathways}")
        print(f"Min pathways for a metabolite: {min_pathways}")
    
    # Show sample entries
    print("\n" + "-"*70)
    print("Sample entries:")
    print("-"*70)
    
    sample_count = 0
    for hmdb_id, pathways in sorted(metabolite_pathway_map.items())[:5]:
        print(f"\n{hmdb_id}:")
        for i, pathway in enumerate(pathways[:3], 1):
            print(f"  {i}. {pathway}")
        if len(pathways) > 3:
            print(f"  ... and {len(pathways) - 3} more")
        sample_count += 1
    
    # Pathway distribution
    print("\n" + "-"*70)
    print("Pathway count distribution:")
    print("-"*70)
    
    bins = [0, 1, 5, 10, 20, 50, 100, float('inf')]
    bin_labels = ['0', '1-5', '6-10', '11-20', '21-50', '51-100', '100+']
    distribution = {label: 0 for label in bin_labels}
    
    for count in pathway_counts:
        for i in range(len(bins) - 1):
            if bins[i] <= count < bins[i+1]:
                distribution[bin_labels[i]] += 1
                break
    
    for label, count in distribution.items():
        if count > 0:
            percentage = count / len(metabolite_pathway_map) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {label:>6} pathways: {count:>6,} ({percentage:>5.1f}%) {bar}")
    
    # Save output
    print("\n" + "-"*70)
    print(f"Saving to {output_json}...")
    
    # Ensure output directory exists
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metabolite_pathway_map, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(metabolite_pathway_map):,} metabolite-pathway mappings")
    
    # File size
    file_size = output_path.stat().st_size
    print(f"✓ File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    
    print("\n" + "="*70)
    print("✓ Metabolite-pathway map built successfully!")
    print("="*70)
    
    return metabolite_pathway_map


def query_pathways(
    hmdb_id: str,
    pathway_map_file: str = "storage/pathbank_pathway_map.json"
) -> List[str]:
    """
    Query pathways for a specific metabolite.
    
    Args:
        hmdb_id: HMDB ID to query
        pathway_map_file: Path to pathbank_pathway_map.json
        
    Returns:
        List of pathway names for the metabolite
    """
    with open(pathway_map_file, 'r', encoding='utf-8') as f:
        pathway_map = json.load(f)
    
    return pathway_map.get(hmdb_id, [])


def compare_with_hmdb_map(
    pathbank_map_file: str = "storage/pathbank_pathway_map.json",
    hmdb_map_file: str = "storage/metabolite_pathway_map.json"
) -> None:
    """
    Compare PathBank-derived map with HMDB-derived map.
    
    Args:
        pathbank_map_file: Path to PathBank-derived map
        hmdb_map_file: Path to HMDB-derived map
    """
    print("\n" + "="*70)
    print("Comparison: PathBank vs HMDB Maps")
    print("="*70)
    
    # Load both maps
    with open(pathbank_map_file, 'r') as f:
        pathbank_map = json.load(f)
    
    try:
        with open(hmdb_map_file, 'r') as f:
            hmdb_map = json.load(f)
    except FileNotFoundError:
        print(f"\n⚠️  HMDB map not found: {hmdb_map_file}")
        print("   Run build_pathway_map.py to generate it for comparison.")
        return
    
    print(f"\nPathBank map: {len(pathbank_map):,} metabolites")
    print(f"HMDB map: {len(hmdb_map):,} metabolites")
    
    # Find common and unique metabolites
    pathbank_ids = set(pathbank_map.keys())
    hmdb_ids = set(hmdb_map.keys())
    
    common_ids = pathbank_ids & hmdb_ids
    only_pathbank = pathbank_ids - hmdb_ids
    only_hmdb = hmdb_ids - pathbank_ids
    
    print(f"\nCommon metabolites: {len(common_ids):,}")
    print(f"Only in PathBank: {len(only_pathbank):,}")
    print(f"Only in HMDB: {len(only_hmdb):,}")
    
    # Compare pathway counts for common metabolites
    if common_ids:
        print("\nPathway count comparison (for common metabolites):")
        
        sample_ids = sorted(common_ids)[:5]
        for hmdb_id in sample_ids:
            pathbank_count = len(pathbank_map[hmdb_id])
            hmdb_count = len(hmdb_map[hmdb_id])
            print(f"\n{hmdb_id}:")
            print(f"  PathBank: {pathbank_count} pathways")
            print(f"  HMDB: {hmdb_count} pathways")
            
            # Show overlap
            pathbank_set = set(pathbank_map[hmdb_id])
            hmdb_set = set(hmdb_map[hmdb_id])
            overlap = pathbank_set & hmdb_set
            print(f"  Overlap: {len(overlap)} pathways")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_json = sys.argv[1]
        output_json = sys.argv[2] if len(sys.argv) > 2 else "storage/pathbank_pathway_map.json"
    else:
        input_json = "storage/pathway_member_map.json"
        output_json = "storage/pathbank_pathway_map.json"
    
    # Build metabolite-pathway map
    metabolite_pathway_map = invert_pathway_map(input_json, output_json)
    
    # Test query
    print("\n" + "="*70)
    print("Test Query:")
    print("="*70)
    
    # Test with a few known metabolites
    test_ids = ["HMDB0000001", "HMDB0000122", "HMDB0000148"]
    
    for test_id in test_ids:
        pathways = query_pathways(test_id, output_json)
        
        print(f"\n{test_id}:")
        if pathways:
            print(f"  Found {len(pathways)} pathways:")
            for i, pathway in enumerate(pathways[:3], 1):
                print(f"    {i}. {pathway}")
            if len(pathways) > 3:
                print(f"    ... and {len(pathways) - 3} more")
        else:
            print("  No pathways found in PathBank")
    
    # Compare with HMDB map if it exists
    compare_with_hmdb_map(output_json, "storage/metabolite_pathway_map.json")
    
    print("\n" + "="*70)
