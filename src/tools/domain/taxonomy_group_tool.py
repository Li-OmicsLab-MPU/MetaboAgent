"""
Taxonomy Group Tool for metabolite classification and grouping.

This tool groups metabolites by their taxonomic classification (class or sub_class)
to identify patterns and relationships in metabolite sets.
"""

import json
from pathlib import Path
from typing import Dict, List, Literal


class TaxonomyGroupTool:
    """
    Tool for grouping metabolites by taxonomic classification.
    
    This tool loads a pre-built taxonomy map and provides methods to group
    metabolites by their class or sub_class taxonomy levels.
    """
    
    def __init__(self, taxonomy_map_path: str = "storage/taxonomy_map.json"):
        """
        Initialize the TaxonomyGroupTool.
        
        Args:
            taxonomy_map_path: Path to the taxonomy map JSON file
            
        Raises:
            FileNotFoundError: If taxonomy map file doesn't exist
            json.JSONDecodeError: If taxonomy map file is invalid JSON
        """
        self.taxonomy_map_path = taxonomy_map_path
        self.taxonomy_map = self._load_taxonomy_map()
        
    def _load_taxonomy_map(self) -> Dict[str, Dict[str, str]]:
        """
        Load taxonomy map from JSON file.
        
        Returns:
            Dictionary mapping HMDB IDs to taxonomy information
        """
        map_file = Path(self.taxonomy_map_path)
        if not map_file.exists():
            raise FileNotFoundError(
                f"Taxonomy map not found at {self.taxonomy_map_path}. "
                "Please run src/data_ops/build_taxonomy_map.py first."
            )
        
        with open(map_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def group_metabolites(
        self,
        metabolite_ids: List[str],
        level: Literal["class", "sub_class"] = "sub_class"
    ) -> Dict[str, List[str]]:
        """
        Group metabolites by their taxonomic classification.
        
        Args:
            metabolite_ids: List of HMDB IDs to group
            level: Taxonomy level to group by ("class" or "sub_class").
                   Defaults to "sub_class" as it's more biologically relevant.
                   
        Returns:
            Dictionary mapping taxonomy names to lists of HMDB IDs.
            Only includes groups with 2 or more members.
            
        Example:
            >>> tool = TaxonomyGroupTool()
            >>> ids = ["HMDB0000001", "HMDB0000002", "HMDB0000005"]
            >>> groups = tool.group_metabolites(ids, level="sub_class")
            >>> print(groups)
            {
                "Amino acids, peptides, and analogues": ["HMDB0000001"],
                "Amines": ["HMDB0000002"]
            }
        """
        if level not in ["class", "sub_class"]:
            raise ValueError(f"Invalid level '{level}'. Must be 'class' or 'sub_class'.")
        
        # Group metabolites by taxonomy
        groups: Dict[str, List[str]] = {}
        unknown_ids = []
        
        for metabolite_id in metabolite_ids:
            if metabolite_id not in self.taxonomy_map:
                unknown_ids.append(metabolite_id)
                continue
            
            taxonomy_value = self.taxonomy_map[metabolite_id][level]
            
            if taxonomy_value not in groups:
                groups[taxonomy_value] = []
            groups[taxonomy_value].append(metabolite_id)
        
        # Filter to only groups with >= 2 members
        filtered_groups = {
            taxonomy: ids
            for taxonomy, ids in groups.items()
            if len(ids) >= 2
        }
        
        # Log unknown IDs if any
        if unknown_ids:
            print(f"Warning: {len(unknown_ids)} metabolite IDs not found in taxonomy map: {unknown_ids[:5]}{'...' if len(unknown_ids) > 5 else ''}")
        
        return filtered_groups
    
    def get_taxonomy_info(self, metabolite_id: str) -> Dict[str, str]:
        """
        Get taxonomy information for a single metabolite.
        
        Args:
            metabolite_id: HMDB ID of the metabolite
            
        Returns:
            Dictionary with "class" and "sub_class" keys
            
        Raises:
            KeyError: If metabolite ID not found in taxonomy map
        """
        if metabolite_id not in self.taxonomy_map:
            raise KeyError(f"Metabolite ID '{metabolite_id}' not found in taxonomy map")
        
        return self.taxonomy_map[metabolite_id]
    
    def get_statistics(self, metabolite_ids: List[str]) -> Dict[str, any]:
        """
        Get taxonomy statistics for a list of metabolites.
        
        Args:
            metabolite_ids: List of HMDB IDs
            
        Returns:
            Dictionary with statistics about taxonomy distribution
        """
        class_groups = self.group_metabolites(metabolite_ids, level="class")
        subclass_groups = self.group_metabolites(metabolite_ids, level="sub_class")
        
        # Count all metabolites (including singletons)
        all_class_counts = {}
        all_subclass_counts = {}
        
        for metabolite_id in metabolite_ids:
            if metabolite_id in self.taxonomy_map:
                class_val = self.taxonomy_map[metabolite_id]["class"]
                subclass_val = self.taxonomy_map[metabolite_id]["sub_class"]
                
                all_class_counts[class_val] = all_class_counts.get(class_val, 0) + 1
                all_subclass_counts[subclass_val] = all_subclass_counts.get(subclass_val, 0) + 1
        
        return {
            "total_metabolites": len(metabolite_ids),
            "unique_classes": len(all_class_counts),
            "unique_sub_classes": len(all_subclass_counts),
            "class_groups_with_2plus": len(class_groups),
            "sub_class_groups_with_2plus": len(subclass_groups),
            "top_classes": sorted(all_class_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "top_sub_classes": sorted(all_subclass_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
