"""
Build disease map from HMDB metabolites data.

This script processes the cleaned HMDB metabolites JSON file and creates
an inverted index mapping disease names to lists of metabolite dictionaries.
Each metabolite entry includes: id, name, and top 3 synonyms.
Disease names are normalized (lowercase, stripped whitespace) for better matching.
"""

import json
from pathlib import Path
from typing import Dict, List, TypedDict


class MetaboliteInfo(TypedDict):
    """Metabolite information stored in disease map."""
    id: str
    name: str
    synonyms: List[str]


def normalize_disease_name(disease: str) -> str:
    """
    Normalize disease name for consistent matching.
    
    Args:
        disease: Raw disease name
        
    Returns:
        Normalized disease name (lowercase, stripped whitespace)
    """
    return disease.strip().lower()


def build_disease_map(
    input_path: str = "data/hmdb_metabolites_cleaned.json",
    output_path: str = "storage/disease_map.json"
) -> Dict[str, List[MetaboliteInfo]]:
    """
    Build disease map from HMDB metabolites data.
    
    Creates an inverted index mapping disease names to lists of metabolite dictionaries.
    Each metabolite entry includes: id, name, and top 3 synonyms.
    
    Args:
        input_path: Path to the cleaned HMDB metabolites JSON file
        output_path: Path to save the disease map
        
    Returns:
        Dictionary mapping normalized disease names to lists of metabolite info dicts
    """
    print(f"Loading metabolites from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        metabolites = json.load(f)
    
    print(f"Processing {len(metabolites)} metabolites...")
    disease_map: Dict[str, List[MetaboliteInfo]] = {}
    metabolites_with_diseases = 0
    total_disease_associations = 0
    
    for item in metabolites:
        hmdb_id = item.get("hmdb_id")
        if not hmdb_id:
            continue
        
        # Extract metabolite information
        name = item.get("name", "")
        synonyms = item.get("synonyms", [])
        # Take top 3 synonyms to save space
        top_synonyms = synonyms[:3] if isinstance(synonyms, list) else []
        
        diseases = item.get("diseases", [])
        if diseases:
            metabolites_with_diseases += 1
            
        for disease in diseases:
            if disease:  # Skip empty strings
                normalized_disease = normalize_disease_name(disease)
                
                if normalized_disease not in disease_map:
                    disease_map[normalized_disease] = []
                
                # Create metabolite info dictionary
                metabolite_info: MetaboliteInfo = {
                    "id": hmdb_id,
                    "name": name,
                    "synonyms": top_synonyms
                }
                
                disease_map[normalized_disease].append(metabolite_info)
                total_disease_associations += 1
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save disease map
    print(f"Saving disease map to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(disease_map, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    unique_diseases = len(disease_map)
    avg_metabolites_per_disease = (
        sum(len(metabolite_list) for metabolite_list in disease_map.values()) / unique_diseases
        if unique_diseases > 0 else 0
    )
    
    print(f"\nDisease map built successfully!")
    print(f"Total metabolites: {len(metabolites)}")
    print(f"Metabolites with diseases: {metabolites_with_diseases} ({metabolites_with_diseases/len(metabolites)*100:.2f}%)")
    print(f"Unique diseases: {unique_diseases}")
    print(f"Total disease associations: {total_disease_associations}")
    print(f"Average metabolites per disease: {avg_metabolites_per_disease:.2f}")
    
    # Show top 10 diseases by metabolite count
    if disease_map:
        print("\nTop 10 diseases by metabolite count:")
        sorted_diseases = sorted(disease_map.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (disease, metabolite_list) in enumerate(sorted_diseases[:10], 1):
            print(f"  {i}. {disease}: {len(metabolite_list)} metabolites")
    
    return disease_map


if __name__ == "__main__":
    build_disease_map()
