#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reaction Checker Tool for MetaboAgent

This module provides tools for querying biochemical reaction networks
and verifying metabolite relationships.

Author: MetaboAgent Team
Date: 2026-01-03
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class ReactionCheckerTool:
    """
    Tool for querying biochemical reaction networks and verifying
    metabolite relationships.
    
    This tool loads pre-built reaction graphs and reaction class information,
    providing methods to verify connections between metabolites and retrieve
    reaction mechanism descriptions.
    
    Attributes:
        graph: Dict mapping HMDB IDs to lists of connection objects
        rc_info: Dict mapping RC IDs to mechanism descriptions
        storage_path: Path to the storage directory
    
    Example:
        >>> tool = ReactionCheckerTool()
        >>> result = tool.verify_pair("HMDB0000902", "HMDB0001487")
        >>> print(result)
        {
            "is_connected": True,
            "reaction_type": "Aldehyde to Secondary alcohol",
            "rc_id": "RC00001",
            "confidence": "High (KEGG Verified)",
            "direction": "bidirectional"
        }
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the reaction checker tool.
        
        Args:
            storage_path: Path to storage directory. If None, uses default
                         relative path from project root.
                         
        Raises:
            FileNotFoundError: If data files cannot be found
            ValueError: If data files have invalid structure
        """
        # Determine storage path
        if storage_path is None:
            # Use relative path from project root
            # __file__ = .../MetaboAgent/src/tools/domain/reaction_checker.py
            # Need 3 parents to get to project root: domain -> tools -> src -> MetaboAgent
            base_dir = Path(__file__).parent.parent.parent.parent
            self.storage_path = base_dir / "storage"
        else:
            self.storage_path = Path(storage_path)
        
        # Initialize attributes
        self.graph: Dict[str, List[Dict[str, str]]] = {}
        self.rc_info: Dict[str, str] = {}
        
        # Load data files
        self._load_data_files()
        
        # Validate loaded data
        self._validate_graph_structure(self.graph)
    
    def _load_data_files(self) -> None:
        """
        Load reaction graph and RC info from JSON files.
        
        Raises:
            FileNotFoundError: If files don't exist
            ValueError: If JSON structure is invalid
        """
        graph_path = self.storage_path / "hmdb_reaction_graph.json"
        rc_info_path = self.storage_path / "reaction_class_info.json"
        
        # Load reaction graph
        try:
            with open(graph_path, 'r', encoding='utf-8') as f:
                self.graph = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Reaction graph file not found: {graph_path}. "
                f"Please ensure the file exists or run build_reaction_map.py first."
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in reaction graph file {graph_path}: {str(e)}"
            )
        
        # Load RC info
        try:
            with open(rc_info_path, 'r', encoding='utf-8') as f:
                self.rc_info = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"RC info file not found: {rc_info_path}. "
                f"Please ensure the file exists or run build_reaction_map.py first."
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in RC info file {rc_info_path}: {str(e)}"
            )
    
    def _validate_graph_structure(self, graph: Dict) -> None:
        """
        Validate that graph has correct structure.
        
        Args:
            graph: The graph to validate
            
        Raises:
            ValueError: If graph structure is invalid
        """
        if not isinstance(graph, dict):
            raise ValueError(
                f"Invalid graph structure: graph must be a dict, got {type(graph)}"
            )
        
        for hmdb_id, connections in graph.items():
            if not isinstance(connections, list):
                raise ValueError(
                    f"Invalid graph structure: connections for {hmdb_id} "
                    f"must be a list, got {type(connections)}"
                )
            
            for conn in connections:
                if not isinstance(conn, dict):
                    raise ValueError(
                        f"Invalid connection object for {hmdb_id}: "
                        f"must be dict, got {type(conn)}"
                    )
                if "target" not in conn or "rc_id" not in conn:
                    raise ValueError(
                        f"Invalid connection object for {hmdb_id}: "
                        f"missing required fields 'target' or 'rc_id'. Got: {conn}"
                    )
                if not isinstance(conn["target"], str) or not isinstance(conn["rc_id"], str):
                    raise ValueError(
                        f"Invalid connection object for {hmdb_id}: "
                        f"'target' and 'rc_id' must be strings. Got: {conn}"
                    )
    
    def verify_pair(self, id_a: str, id_b: str) -> Dict:
        """
        Verify if two metabolites have a direct reaction relationship.
        
        This method checks bidirectionally and automatically looks up
        the reaction mechanism description using the RC ID.
        
        Args:
            id_a: First HMDB ID (e.g., "HMDB0000902")
            id_b: Second HMDB ID (e.g., "HMDB0001487")
            
        Returns:
            Dictionary with connection information:
            {
                "is_connected": bool,
                "reaction_type": str,  # Mechanism description
                "rc_id": str,          # Reaction class ID
                "confidence": str,     # Verification level
                "direction": str       # "A->B", "B->A", or "bidirectional"
            }
            
            If not connected, returns {"is_connected": False}
            
        Raises:
            ValueError: If either ID is None or empty
            
        Example:
            >>> tool = ReactionCheckerTool()
            >>> result = tool.verify_pair("HMDB0000902", "HMDB0001487")
            >>> print(result["is_connected"])
            True
        """
        # Input validation
        if not id_a or id_a is None:
            raise ValueError("id_a cannot be None or empty")
        if not id_b or id_b is None:
            raise ValueError("id_b cannot be None or empty")
        
        # Check if IDs exist in graph
        if id_a not in self.graph and id_b not in self.graph:
            return {"is_connected": False}
        
        # Find connection
        conn_a_to_b = self._find_connection(id_a, id_b)
        conn_b_to_a = self._find_connection(id_b, id_a)
        
        # Determine connection status
        if conn_a_to_b and conn_b_to_a:
            # Bidirectional connection
            rc_id = conn_a_to_b["rc_id"]
            direction = "bidirectional"
        elif conn_a_to_b:
            # A -> B
            rc_id = conn_a_to_b["rc_id"]
            direction = "A->B"
        elif conn_b_to_a:
            # B -> A
            rc_id = conn_b_to_a["rc_id"]
            direction = "B->A"
        else:
            # Not connected
            return {"is_connected": False}
        
        # Look up reaction mechanism
        reaction_type = self.get_reaction_mechanism(rc_id)
        
        return {
            "is_connected": True,
            "reaction_type": reaction_type,
            "rc_id": rc_id,
            "confidence": "High (KEGG Verified)",
            "direction": direction
        }
    
    def _find_connection(self, id_a: str, id_b: str) -> Optional[Dict]:
        """
        Find connection object between two metabolites.
        
        Args:
            id_a: First HMDB ID
            id_b: Second HMDB ID
            
        Returns:
            Connection object with rc_id, or None if not connected
        """
        if id_a not in self.graph:
            return None
        
        connections = self.graph[id_a]
        for conn in connections:
            if conn["target"] == id_b:
                return conn
        
        return None
    
    def get_reaction_mechanism(self, rc_id: str) -> str:
        """
        Get human-readable description of a reaction mechanism.
        
        Args:
            rc_id: Reaction class ID (e.g., "RC00371")
            
        Returns:
            Description string or "RC ID not found" message
            
        Raises:
            ValueError: If rc_id is None or empty
            
        Example:
            >>> tool = ReactionCheckerTool()
            >>> desc = tool.get_reaction_mechanism("RC00371")
            >>> print(desc)
            "Aldehyde to Secondary alcohol"
        """
        # Input validation
        if not rc_id or rc_id is None:
            raise ValueError("rc_id cannot be None or empty")
        
        # Look up description
        if rc_id in self.rc_info:
            return self.rc_info[rc_id]
        else:
            return f"RC ID not found: {rc_id}"
    
    def get_neighbors(self, hmdb_id: str) -> List[str]:
        """
        Get all neighbor metabolites (directly connected via reactions) for a given metabolite.
        
        This method is essential for Phase 2 mechanistic mining strategy, which generates
        ratio features based on biochemical reaction relationships.
        
        Args:
            hmdb_id: HMDB ID of the metabolite (e.g., "HMDB0000054")
            
        Returns:
            List of neighbor HMDB IDs that have direct reaction relationships.
            Returns empty list if the metabolite is not in the graph.
            
        Raises:
            ValueError: If hmdb_id is None or empty
            
        Example:
            >>> tool = ReactionCheckerTool()
            >>> neighbors = tool.get_neighbors("HMDB0000054")
            >>> print(f"Found {len(neighbors)} neighbors")
            >>> for neighbor in neighbors[:5]:
            ...     print(f"  - {neighbor}")
            Found 12 neighbors
              - HMDB0000123
              - HMDB0000456
              - HMDB0000789
              ...
        
        Notes:
            - This method only returns direct neighbors (1-hop connections)
            - The graph is based on KEGG reaction data
            - Used by Phase 2 PToT engine for mechanistic mining strategy
        """
        # Input validation
        if not hmdb_id or hmdb_id is None:
            raise ValueError("hmdb_id cannot be None or empty")
        
        # Check if the metabolite exists in the graph
        if hmdb_id not in self.graph:
            return []
        
        # Extract all neighbor target IDs
        neighbors = []
        for conn in self.graph[hmdb_id]:
            neighbors.append(conn["target"])
        
        return neighbors
    
    def find_connected_pairs(self, hmdb_ids: List[str]) -> List[Dict]:
        """
        Batch process a list of HMDB IDs and find ALL valid biochemical 
        reaction pairs among them based on the internal KEGG graph.
        
        This method uses graph network reverse lookup with O(N) complexity,
        utilizing set operations for fast filtering. It eliminates the need
        for manual loops or combinatorial pair generation.
        
        Args:
            hmdb_ids: List of HMDB IDs present in the current dataset
                     
        Returns:
            List of dictionaries containing valid connected pairs:
            [
                {
                    "id_a": "HMDB0000064",
                    "id_b": "HMDB0000161",
                    "rc_id": "RC00001",
                    "reaction_type": "Mechanism description"
                }, ...
            ]
            
        Example:
            >>> tool = ReactionCheckerTool()
            >>> hmdb_list = ["HMDB0000064", "HMDB0000562", "HMDB0003339"]
            >>> pairs = tool.find_connected_pairs(hmdb_list)
            >>> print(f"Found {len(pairs)} connected pairs")
            >>> for pair in pairs:
            ...     print(f"{pair['id_a']} <-> {pair['id_b']}: {pair['reaction_type']}")
        """
        valid_pairs = []
        hmdb_set = set(hmdb_ids)  # Convert to set for O(1) lookups
        
        # Iterate only through the IDs present in our data
        for id_a in hmdb_set:
            if id_a in self.graph:
                for conn in self.graph[id_a]:
                    id_b = conn["target"]
                    # Check if target is also in our data
                    # Use id_a < id_b to prevent duplicate pairs (A-B and B-A)
                    if id_b in hmdb_set and id_a < id_b:
                        rc_id = conn["rc_id"]
                        reaction_type = self.get_reaction_mechanism(rc_id)
                        valid_pairs.append({
                            "id_a": id_a,
                            "id_b": id_b,
                            "rc_id": rc_id,
                            "reaction_type": reaction_type
                        })
        
        return valid_pairs


if __name__ == "__main__":
    """
    Example usage and basic testing
    """
    print("=" * 60)
    print("Reaction Checker Tool - Example Usage")
    print("=" * 60)
    
    # Initialize tool
    print("\n1. Initializing ReactionCheckerTool...")
    tool = ReactionCheckerTool()
    print(f"   ✓ Loaded {len(tool.graph)} metabolites")
    print(f"   ✓ Loaded {len(tool.rc_info)} reaction class descriptions")
    
    # Test verify_pair with known connected pair
    print("\n2. Testing verify_pair with connected metabolites...")
    result = tool.verify_pair("HMDB0000902", "HMDB0001487")
    print(f"   Query: HMDB0000902 <-> HMDB0001487")
    print(f"   Connected: {result['is_connected']}")
    if result['is_connected']:
        print(f"   Reaction Type: {result['reaction_type']}")
        print(f"   RC ID: {result['rc_id']}")
        print(f"   Direction: {result['direction']}")
    
    # Test verify_pair with unconnected pair
    print("\n3. Testing verify_pair with unconnected metabolites...")
    result = tool.verify_pair("HMDB9999999", "HMDB9999998")
    print(f"   Query: HMDB9999999 <-> HMDB9999998")
    print(f"   Connected: {result['is_connected']}")
    
    # Test get_reaction_mechanism
    print("\n4. Testing get_reaction_mechanism...")
    desc = tool.get_reaction_mechanism("RC00371")
    print(f"   RC00371: {desc}")
    
    # Test with invalid RC ID
    desc = tool.get_reaction_mechanism("RC99999")
    print(f"   RC99999: {desc}")
    
    # Test find_connected_pairs (NEW)
    print("\n5. Testing find_connected_pairs (Graph Network Reverse Lookup)...")
    test_hmdb_list = [
        "HMDB0000064", "HMDB0000562", "HMDB0003339", 
        "HMDB0003423", "HMDB0000157", "HMDB0000195"
    ]
    print(f"   Input: {len(test_hmdb_list)} HMDB IDs")
    pairs = tool.find_connected_pairs(test_hmdb_list)
    print(f"   ✓ Found {len(pairs)} connected pairs:")
    for pair in pairs:
        print(f"      - {pair['id_a']} <-> {pair['id_b']}")
        print(f"        Type: {pair['reaction_type']}")
        print(f"        RC ID: {pair['rc_id']}")
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)



# LangChain Tool Integration

try:
    from langchain.tools import BaseTool
    from pydantic import BaseModel, Field
    
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object
    BaseModel = object
    
    def Field(*args, **kwargs):
        return None


class VerifyPairInput(BaseModel):
    """Input schema for verify_pair tool."""
    id_a: str = Field(description="First HMDB ID (e.g., 'HMDB0000902')")
    id_b: str = Field(description="Second HMDB ID (e.g., 'HMDB0001487')")


class FindConnectedPairsInput(BaseModel):
    """Input schema for find_connected_pairs tool."""
    hmdb_ids: List[str] = Field(
        description="List of HMDB IDs from the current dataset (e.g., ['HMDB0000064', 'HMDB0000562'])"
    )


class VerifyPairTool(BaseTool):
    """
    LangChain tool wrapper for metabolite pair verification.
    
    Use this tool to verify if two metabolites have a direct 
    substrate-product relationship in biochemical reactions. 
    Useful for validating ratio features in metabolomics analysis.
    
    This tool provides evidence from KEGG reaction databases about
    whether two metabolites are directly connected through biochemical
    reactions, including the specific reaction mechanism.
    
    Example:
        >>> tool = VerifyPairTool()
        >>> result = tool._run("HMDB0000902", "HMDB0001487")
        >>> print(result)
        {
          "is_connected": true,
          "reaction_type": "Cyclic amine to Aryl imine",
          "rc_id": "RC00001",
          "confidence": "High (KEGG Verified)",
          "direction": "bidirectional"
        }
    """
    
    name: str = "verify_metabolite_pair"
    description: str = (
        "Verify if two metabolites have a direct biochemical reaction "
        "relationship. Input should be two HMDB IDs (e.g., 'HMDB0000902' and 'HMDB0001487'). "
        "Returns connection status, reaction type (mechanism description), "
        "RC ID (reaction class identifier), confidence level, and direction. "
        "This tool is useful for validating ratio features in metabolomics analysis "
        "by confirming substrate-product relationships."
    )
    args_schema: type[BaseModel] = VerifyPairInput
    
    # Declare checker as a class variable
    checker: ReactionCheckerTool = None
    
    def __init__(self, **kwargs):
        """Initialize the tool with a ReactionCheckerTool instance."""
        super().__init__(**kwargs)
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, 'checker', ReactionCheckerTool())
    
    def _run(self, id_a: str, id_b: str) -> str:
        """
        Execute the tool.
        
        Args:
            id_a: First HMDB ID
            id_b: Second HMDB ID
            
        Returns:
            JSON-formatted string with verification result
        """
        try:
            result = self.checker.verify_pair(id_a, id_b)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "is_connected": False
            }, indent=2)
    
    async def _arun(self, id_a: str, id_b: str) -> str:
        """
        Async execution (not implemented).
        
        Args:
            id_a: First HMDB ID
            id_b: Second HMDB ID
            
        Raises:
            NotImplementedError: Async execution not supported
        """
        raise NotImplementedError("Async execution not supported for this tool")


class FindConnectedPairsTool(BaseTool):
    """
    LangChain tool wrapper for batch metabolite pair discovery.
    
    Use this tool to find ALL biochemically connected metabolite pairs
    in a dataset using graph network reverse lookup. This eliminates the
    need for manual loops or combinatorial pair generation.
    
    This tool performs O(N) graph intersection to identify all valid
    substrate-product relationships among the provided HMDB IDs.
    
    Example:
        >>> tool = FindConnectedPairsTool()
        >>> hmdb_list = ["HMDB0000064", "HMDB0000562", "HMDB0003339"]
        >>> result = tool._run(hmdb_list)
        >>> print(result)
        [
          {
            "id_a": "HMDB0000064",
            "id_b": "HMDB0000562",
            "rc_id": "RC00615",
            "reaction_type": "Lactam to Carboxylate"
          }
        ]
    """
    
    name: str = "find_connected_metabolite_pairs"
    description: str = (
        "Find ALL biochemically connected metabolite pairs in a dataset "
        "using graph network reverse lookup. Input should be a list of HMDB IDs "
        "(e.g., ['HMDB0000064', 'HMDB0000562', 'HMDB0003339']). "
        "Returns a list of all valid pairs with their reaction types, RC IDs, "
        "and mechanism descriptions. This tool uses O(N) complexity and eliminates "
        "the need for manual loops or combinatorial pair generation. "
        "Essential for generating biologically valid ratio features."
    )
    args_schema: type[BaseModel] = FindConnectedPairsInput
    
    # Declare checker as a class variable
    checker: ReactionCheckerTool = None
    
    def __init__(self, **kwargs):
        """Initialize the tool with a ReactionCheckerTool instance."""
        super().__init__(**kwargs)
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, 'checker', ReactionCheckerTool())
    
    def _run(self, hmdb_ids: List[str]) -> str:
        """
        Execute the tool.
        
        Args:
            hmdb_ids: List of HMDB IDs from the dataset
            
        Returns:
            JSON-formatted string with list of connected pairs
        """
        try:
            result = self.checker.find_connected_pairs(hmdb_ids)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "valid_pairs": []
            }, indent=2)
    
    async def _arun(self, hmdb_ids: List[str]) -> str:
        """
        Async execution (not implemented).
        
        Args:
            hmdb_ids: List of HMDB IDs
            
        Raises:
            NotImplementedError: Async execution not supported
        """
        raise NotImplementedError("Async execution not supported for this tool")
