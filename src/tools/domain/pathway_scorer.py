"""
Pathway Scoring Tool

This tool calculates ssGSEA (Single Sample Gene Set Enrichment Analysis) scores
for metabolic pathways based on metabolite expression data.

The tool uses a pathway member map to identify which metabolites belong to each pathway,
then calculates enrichment scores that represent pathway activity in each sample.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional


def _detect_id_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("Sample_ID", "sample_id", "SampleID", "sampleid", "ROW_ID", "row_id", "__row_id__"):
        if col in df.columns:
            return col
    return None


def _detect_label_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("Group", "group", "target", "Target", "Class", "class", "Label", "label"):
        if col in df.columns:
            return col
    return None


class PathwayScorer:
    """
    Calculate pathway activity scores using ssGSEA algorithm.
    """
    
    def __init__(self, pathway_map_path: str = "storage/pathbank_pathway_map_reverse.json"):
        """
        Initialize PathwayScorer.
        
        Args:
            pathway_map_path: Path to the pathway member map JSON file
        """
        self.pathway_map_path = pathway_map_path
        self.pathway_map = self._load_pathway_map()
    
    def _load_pathway_map(self) -> Dict[str, List[str]]:
        """Load pathway member map from JSON file."""
        path = Path(self.pathway_map_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Pathway map not found at {self.pathway_map_path}. "
                "Please run src/rag/build_pathway_map.py first."
            )
        
        with open(path, 'r') as f:
            pathway_map = json.load(f)
        
        print(f"Loaded pathway map with {len(pathway_map)} pathways")
        return pathway_map
    
    def _validate_inputs(
        self, 
        data_matrix: pd.DataFrame, 
        pathway_names: List[str]
    ) -> tuple[List[str], List[str]]:
        """
        Validate inputs and return valid pathways and warnings.
        
        Args:
            data_matrix: Sample x HMDB_ID expression matrix
            pathway_names: List of pathway names to score
            
        Returns:
            Tuple of (valid_pathway_names, warnings)
        """
        warnings = []
        valid_pathways = []
        
        # Check pathway names
        for pathway_name in pathway_names:
            if pathway_name not in self.pathway_map:
                warnings.append(f"Pathway '{pathway_name}' not found in pathway map")
            else:
                valid_pathways.append(pathway_name)
        
        if not valid_pathways:
            raise ValueError("No valid pathways found in the provided pathway_names")
        
        # Check data matrix columns (HMDB IDs)
        data_hmdb_ids = set(data_matrix.columns)
        all_pathway_hmdb_ids = set()
        for pathway in valid_pathways:
            all_pathway_hmdb_ids.update(self.pathway_map[pathway])
        
        overlap = data_hmdb_ids.intersection(all_pathway_hmdb_ids)
        if len(overlap) == 0:
            warnings.append(
                "No HMDB IDs in data matrix match any pathway members. "
                "Check that column names are HMDB IDs (e.g., 'HMDB0000001')"
            )
        else:
            coverage = len(overlap) / len(all_pathway_hmdb_ids) * 100
            print(f"Data matrix covers {len(overlap)}/{len(all_pathway_hmdb_ids)} "
                  f"({coverage:.1f}%) of pathway metabolites")
        
        return valid_pathways, warnings
    
    def _calculate_ssgsea_score(
        self, 
        expression_values: np.ndarray, 
        gene_set_indices: List[int]
    ) -> float:
        """
        Calculate ssGSEA score for a single sample and gene set.
        
        This is a simplified implementation of ssGSEA that:
        1. Ranks genes by expression
        2. Calculates enrichment score based on rank positions of gene set members
        
        Args:
            expression_values: Expression values for all genes (1D array)
            gene_set_indices: Indices of genes in the gene set
            
        Returns:
            ssGSEA enrichment score
        """
        n_genes = len(expression_values)
        
        if len(gene_set_indices) == 0:
            return 0.0
        
        # Rank genes by expression (higher expression = higher rank)
        ranks = np.argsort(np.argsort(expression_values)) + 1
        
        # Get ranks of gene set members
        gene_set_ranks = ranks[gene_set_indices]
        
        # Calculate enrichment score
        # This is a simplified version: mean rank normalized by total genes
        mean_rank = np.mean(gene_set_ranks)
        expected_rank = (n_genes + 1) / 2
        
        # Normalize to [-1, 1] range
        score = (mean_rank - expected_rank) / (n_genes / 2)
        
        return score
    
    def calculate_scores(
        self, 
        data_matrix: pd.DataFrame, 
        pathway_names: List[str]
    ) -> pd.DataFrame:
        """
        Calculate ssGSEA pathway activity scores.
        
        Args:
            data_matrix: DataFrame with samples as rows and HMDB IDs as columns
            pathway_names: List of pathway names to calculate scores for
            
        Returns:
            DataFrame with samples as rows and pathway scores as columns
            
        Example:
            >>> scorer = PathwayScorer()
            >>> data = pd.DataFrame({
            ...     'HMDB0000001': [1.2, 2.3, 0.5],
            ...     'HMDB0000002': [0.8, 1.5, 2.1]
            ... }, index=['Sample1', 'Sample2', 'Sample3'])
            >>> scores = scorer.calculate_scores(data, ['Urea Cycle'])
        """
        # Validate inputs
        valid_pathways, warnings = self._validate_inputs(data_matrix, pathway_names)
        
        # Print warnings
        for warning in warnings:
            print(f"Warning: {warning}")
        
        # Initialize results DataFrame
        results = pd.DataFrame(
            index=data_matrix.index,
            columns=valid_pathways
        )
        
        # Get list of all HMDB IDs in data
        data_hmdb_ids = list(data_matrix.columns)
        
        # Calculate scores for each pathway
        for pathway_name in valid_pathways:
            pathway_members = self.pathway_map[pathway_name]
            
            # Find indices of pathway members in data matrix
            member_indices = [
                i for i, hmdb_id in enumerate(data_hmdb_ids)
                if hmdb_id in pathway_members
            ]
            
            if len(member_indices) == 0:
                print(f"Warning: No members of '{pathway_name}' found in data matrix")
                results[pathway_name] = 0.0
                continue
            
            # Calculate ssGSEA score for each sample
            for sample_idx in range(len(data_matrix)):
                expression_values = data_matrix.iloc[sample_idx].values
                score = self._calculate_ssgsea_score(expression_values, member_indices)
                results.loc[results.index[sample_idx], pathway_name] = score
        
        # Convert to float type
        results = results.astype(float)
        
        print(f"\nCalculated scores for {len(valid_pathways)} pathways "
              f"across {len(data_matrix)} samples")
        
        return results
    
    def get_available_pathways(self) -> List[str]:
        """Get list of all available pathway names."""
        return sorted(self.pathway_map.keys())
    
    def get_pathway_members(self, pathway_name: str) -> Optional[List[str]]:
        """Get list of HMDB IDs for a specific pathway."""
        return self.pathway_map.get(pathway_name)


# LangChain Tool Wrapper
def calculate_pathway_scores(
    data_matrix_path: str,
    pathway_names: List[str],
    output_path: Optional[str] = None,
    include_group: bool = True
) -> str:
    """
    Use this tool to calculate ssGSEA pathway activity scores for specific pathways.
    Useful for generating high-level pathway features from metabolite data.
    
    Args:
        data_matrix_path: Path to CSV file with metabolite expression data 
                         (rows=samples, columns=HMDB_IDs)
        pathway_names: List of pathway names to calculate scores for
                      (e.g., ["Urea Cycle", "Arginine Metabolism"])
        output_path: Optional path to save results CSV file
        include_group: Whether to include Group column in output (default: True)
                      Set to False when generating temporary feature files for merging
        
    Returns:
        String describing the results and statistics
        
    Example:
        >>> result = calculate_pathway_scores(
        ...     data_matrix_path="data/metabolite_expression.csv",
        ...     pathway_names=["Urea Cycle", "Arginine Metabolism"],
        ...     include_group=False  # For temporary feature files
        ... )
    """
    try:
        # Load the full table first so the target column is never silently consumed
        # as row index when datasets do not contain a sample-id column.
        full_df = pd.read_csv(data_matrix_path)
        id_col = _detect_id_column(full_df)
        label_col = _detect_label_column(full_df)
        
        # CRITICAL FIX: Filter to only feature columns (HMDB IDs or numeric)
        # Exclude common metadata columns that cause type errors
        # BUT preserve id/label information for optional output stitching
        metadata_cols = ['Batch', 'batch', 'Class', 'class', 
                        'Label', 'label', 'Target', 'target']
        if id_col:
            metadata_cols.append(id_col)
        if label_col and label_col not in metadata_cols:
            metadata_cols.append(label_col)
        
        # Keep only columns that are:
        # 1. Not in metadata list, AND
        # 2. Either start with 'HMDB' OR are numeric
        feature_cols = []
        for col in full_df.columns:
            if col in metadata_cols:
                continue
            if col.startswith('HMDB') or pd.api.types.is_numeric_dtype(full_df[col]):
                feature_cols.append(col)
        
        # Filter data to only feature columns
        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns found in data matrix")
        
        print(f"Filtered to {len(feature_cols)} feature columns (excluded {len(full_df.columns) - len(feature_cols)} metadata columns)")
        feature_data = full_df[feature_cols].copy()
        if id_col:
            feature_data.index = full_df[id_col]
        
        # Initialize scorer
        scorer = PathwayScorer()
        
        # Calculate scores
        scores = scorer.calculate_scores(feature_data, pathway_names)
        
        export_df = scores.reset_index(drop=True)
        if id_col:
            export_df.insert(0, id_col, full_df[id_col].values)
        if label_col and include_group:
            insert_at = 1 if id_col else 0
            export_df.insert(insert_at, label_col, full_df[label_col].values)
        
        # Save if output path provided
        if output_path:
            export_df.to_csv(output_path, index=False)
            save_msg = f"\nResults saved to {output_path}"
        else:
            save_msg = ""
        
        # Generate summary statistics (only for numeric pathway columns, exclude Group)
        numeric_cols = list(scores.columns)
        numeric_scores = scores[numeric_cols]
        
        summary = f"""
Pathway Scoring Results:
- Samples: {len(scores)}
- Pathways: {len(numeric_cols)}
- Score range: [{numeric_scores.min().min():.3f}, {numeric_scores.max().max():.3f}]

Pathway Statistics:
"""
        for pathway in numeric_cols:
            mean_score = scores[pathway].mean()
            std_score = scores[pathway].std()
            summary += f"  {pathway}: mean={mean_score:.3f}, std={std_score:.3f}\n"
        
        summary += save_msg
        
        # Return JSON format for better parsing
        import json
        result = {
            "success": True,
            "output_path": output_path if output_path else None,
            "n_samples": len(scores),
            "n_pathways": len(scores.columns),
            "n_pathway_scores": len(scores.columns),
            "pathways": list(scores.columns),
            "summary": summary
        }
        return json.dumps(result)
        
    except Exception as e:
        import json
        return json.dumps({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        })


# ============================================================================
# LangChain Tool Wrapper
# ============================================================================
from langchain_core.tools import StructuredTool

calculate_pathway_scores_tool = StructuredTool.from_function(
    func=calculate_pathway_scores,
    name="calculate_pathway_scores",
    description=calculate_pathway_scores.__doc__
)


# Convenience function for direct use
def score_pathways(
    data_matrix: pd.DataFrame,
    pathway_names: List[str]
) -> pd.DataFrame:
    """
    Convenience function to calculate pathway scores directly.
    
    Args:
        data_matrix: DataFrame with samples as rows and HMDB IDs as columns
        pathway_names: List of pathway names to score
        
    Returns:
        DataFrame with pathway scores
    """
    scorer = PathwayScorer()
    return scorer.calculate_scores(data_matrix, pathway_names)


if __name__ == "__main__":
    # Example usage
    print("PathwayScorer Tool")
    print("=" * 50)
    
    scorer = PathwayScorer()
    print(f"\nAvailable pathways: {len(scorer.get_available_pathways())}")
    print("\nExample pathways:")
    for pathway in scorer.get_available_pathways()[:5]:
        members = scorer.get_pathway_members(pathway)
        print(f"  - {pathway}: {len(members)} members")
