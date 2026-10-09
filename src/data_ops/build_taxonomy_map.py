"""
Build taxonomy map from HMDB metabolites data.

This script processes the cleaned HMDB metabolites JSON file and creates
a taxonomy mapping that associates each metabolite HMDB ID with its
taxonomic classification (class and sub_class).
"""

import json
from pathlib import Path


def build_taxonomy_map(
    input_path: str = "data/hmdb_metabolites_cleaned.json",
    output_path: str = "storage/taxonomy_map.json"
) -> dict:
    """
    Build taxonomy map from HMDB metabolites data.
    
    Args:
        input_path: Path to the cleaned HMDB metabolites JSON file
        output_path: Path to save the taxonomy map
        
    Returns:
        Dictionary mapping HMDB IDs to taxonomy information
    """
    print(f"Loading metabolites from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        metabolites = json.load(f)
    
    print(f"Processing {len(metabolites)} metabolites...")
    taxonomy_map = {}
    
    for item in metabolites:
        hmdb_id = item.get("hmdb_id")
        if not hmdb_id:
            continue
            
        # Extract taxonomy information with fallback to "Unclassified"
        class_value = item.get("class")
        sub_class_value = item.get("sub_class")
        
        taxonomy_map[hmdb_id] = {
            "class": class_value if class_value else "Unclassified",
            "sub_class": sub_class_value if sub_class_value else "Unclassified"
        }
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save taxonomy map
    print(f"Saving taxonomy map to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(taxonomy_map, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    total = len(taxonomy_map)
    unclassified_class = sum(1 for v in taxonomy_map.values() if v["class"] == "Unclassified")
    unclassified_subclass = sum(1 for v in taxonomy_map.values() if v["sub_class"] == "Unclassified")
    
    print(f"\nTaxonomy map built successfully!")
    print(f"Total metabolites: {total}")
    print(f"Unclassified class: {unclassified_class} ({unclassified_class/total*100:.2f}%)")
    print(f"Unclassified sub_class: {unclassified_subclass} ({unclassified_subclass/total*100:.2f}%)")
    
    return taxonomy_map


if __name__ == "__main__":
    build_taxonomy_map()
