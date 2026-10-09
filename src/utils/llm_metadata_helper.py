"""
LLM Metadata Helper - Semantic Label Encoding

This module provides LLM-driven semantic reasoning for metadata interpretation,
specifically for determining the positive class in binary classification tasks.

Core Design:
- Single Source of Truth: Encode labels at Phase 1 Stage 0 using LLM
- Semantic Understanding: LLM interprets clinical context to identify disease/case class
- Eliminates downstream encoding inconsistencies and positive/negative class inversion

Author: MetaboAgent Team
Date: 2026-04-27
"""

import json
import logging
import os
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataReasoner:
    """
    LLM-driven metadata semantic reasoner for clinical data.
    
    This class uses large language models to interpret clinical scenarios
    and determine which class should be considered the "positive class"
    (typically disease, mutation, treatment response, or event occurrence).
    
    Design Philosophy:
    - Semantic over Alphabetical: Use medical logic, not string sorting
    - Single Source of Truth: Encode once at data ingestion, use everywhere
    - Fail-Safe: Graceful degradation to heuristic rules if LLM fails
    
    Example:
        >>> reasoner = MetadataReasoner()
        >>> positive_class = reasoner.infer_positive_class(
        ...     scenario="Crohn's Disease vs Healthy Control",
        ...     labels=["Control", "CD"]
        ... )
        >>> print(positive_class)  # "CD"
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the metadata reasoner.
        
        Args:
            llm_client: Optional LLM client instance. If None, will create one
                       using environment variables (OPENAI_API_KEY, OPENAI_API_BASE)
        """
        self.llm = llm_client
        
        # Lazy initialization: only create LLM client when needed
        if self.llm is None:
            self._initialize_llm_client()
    
    def _initialize_llm_client(self):
        """
        Initialize LLM client using environment variables.
        
        Uses OpenAI-compatible API with configuration from .env:
        - OPENAI_API_KEY: API key
        - OPENAI_API_BASE: API base URL
        """
        try:
            from langchain_openai import ChatOpenAI
            
            # Get API configuration from environment
            api_key = os.getenv('OPENAI_API_KEY')
            api_base = os.getenv('OPENAI_API_BASE')
            
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment. LLM reasoning will be disabled.")
                self.llm = None
                return
            
            # Initialize ChatOpenAI client
            self.llm = ChatOpenAI(
                model="gpt-4o",  # Use GPT-4 for better reasoning
                temperature=0,  # Deterministic output
                openai_api_key=api_key,
                openai_api_base=api_base if api_base else None
            )
            
            logger.info("LLM client initialized successfully")
            
        except ImportError:
            logger.error("langchain_openai not installed. Please install: pip install langchain-openai")
            self.llm = None
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm = None
    
    def infer_positive_class(
        self,
        scenario: str,
        labels: List[str],
        return_reasoning: bool = False
    ) -> str:
        """
        Use LLM to infer which class should be the positive class.
        
        The LLM analyzes the clinical scenario and label names to determine
        which class represents the "case" group (disease, mutation, event, etc.)
        
        Args:
            scenario: Clinical research scenario description
            labels: List of class labels (e.g., ["Control", "CD"])
            return_reasoning: If True, return (positive_class, reasoning) tuple
        
        Returns:
            str: The label that should be encoded as positive class (1)
            or Tuple[str, str]: (positive_class, reasoning) if return_reasoning=True
        
        Fallback Strategy:
        1. Try LLM reasoning (primary method)
        2. If LLM fails, use keyword heuristics
        3. If heuristics fail, use minority class detection
        4. If all fail, default to last label in list
        
        Examples:
            >>> reasoner = MetadataReasoner()
            >>> positive = reasoner.infer_positive_class(
            ...     scenario="Crohn's Disease biomarker study",
            ...     labels=["Control", "CD"]
            ... )
            >>> print(positive)  # "CD"
            
            >>> positive, reasoning = reasoner.infer_positive_class(
            ...     scenario="Cancer vs Normal",
            ...     labels=["Normal", "Cancer"],
            ...     return_reasoning=True
            ... )
            >>> print(positive)  # "Cancer"
            >>> print(reasoning)  # "Cancer represents the disease state..."
        """
        # Validate inputs
        if not labels or len(labels) == 0:
            logger.error("Empty labels list provided")
            return labels[-1] if labels else ""
        
        if len(labels) == 1:
            logger.warning("Only one label provided, returning it as positive class")
            return labels[0]
        
        # Try LLM reasoning first
        if self.llm is not None:
            try:
                positive_class, reasoning = self._llm_infer(scenario, labels)
                
                # Validate that returned class is in the labels list
                if positive_class in labels:
                    logger.info(f"LLM identified positive class: {positive_class}")
                    if return_reasoning:
                        return positive_class, reasoning
                    return positive_class
                else:
                    logger.warning(f"LLM returned invalid class '{positive_class}', falling back to heuristics")
            
            except Exception as e:
                logger.error(f"LLM inference failed: {e}. Falling back to heuristics.")
        
        # Fallback to heuristic methods
        positive_class = self._heuristic_infer(scenario, labels)
        reasoning = "Determined using keyword heuristics (LLM unavailable)"
        
        if return_reasoning:
            return positive_class, reasoning
        return positive_class
    
    def _llm_infer(self, scenario: str, labels: List[str]) -> Tuple[str, str]:
        """
        Use LLM to infer positive class.
        
        Args:
            scenario: Clinical scenario description
            labels: List of class labels
        
        Returns:
            Tuple[str, str]: (positive_class, reasoning)
        
        Raises:
            Exception: If LLM call fails or returns invalid JSON
        """
        # Construct prompt
        prompt = f"""[Role]
You are a senior clinical data scientist specializing in metabolomics and biomarker discovery.

[Task]
Based on the provided clinical research scenario and class labels, determine which class should be defined as the "Positive Class" (typically the disease, mutation, treatment response, or event occurrence group).

[Context]
- Research Scenario: {scenario}
- Class Labels: {labels}

[Output Format]
Return ONLY a valid JSON object (no markdown, no code blocks):
{{
    "positive_class": "exact_class_name",
    "reasoning": "brief medical logic explanation"
}}

[Rules]
1. The positive_class MUST be one of the provided labels (exact match)
2. In medical studies, the positive class is typically:
   - Disease vs Control → Disease
   - Case vs Control → Case
   - Mutant vs Wild-type → Mutant
   - Responder vs Non-responder → Responder
   - Event vs No-event → Event
3. Provide clear, concise reasoning based on medical conventions
4. Output ONLY the JSON object, no additional text"""

        # Call LLM
        response = self.llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Clean up response (remove markdown if present)
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON
        try:
            result = json.loads(response_text)
            positive_class = result.get('positive_class', '')
            reasoning = result.get('reasoning', 'No reasoning provided')
            
            if not positive_class:
                raise ValueError("LLM response missing 'positive_class' field")
            
            return positive_class, reasoning
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            raise
    
    def _heuristic_infer(self, scenario: str, labels: List[str]) -> str:
        """
        Use keyword heuristics to infer positive class.
        
        This is a fallback method when LLM is unavailable or fails.
        
        Three-tier heuristic:
        1. Keyword detection (disease-related terms)
        2. Minority class detection (if class distribution is known)
        3. Default rule (last label in list)
        
        Args:
            scenario: Clinical scenario description
            labels: List of class labels
        
        Returns:
            str: Inferred positive class
        """
        # Convert all labels to strings for consistent processing
        labels = [str(label) for label in labels]
        
        # Tier 1: Keyword detection
        disease_keywords = [
            'disease', 'patient', 'case', 'cd', 'uc', 'ibd',
            'cancer', 'tumor', 'mutant', 'mutation',
            'responder', 'response', 'event', 'positive',
            'affected', 'sick', 'ill', 'disorder'
        ]
        
        control_keywords = [
            'control', 'healthy', 'normal', 'wild', 'wildtype',
            'non-responder', 'no-event', 'negative', 'unaffected'
        ]
        
        # Check each label against keywords
        for label in labels:
            label_lower = label.lower()
            
            # Check if label contains disease keywords
            if any(keyword in label_lower for keyword in disease_keywords):
                logger.info(f"Keyword heuristic: '{label}' identified as positive class")
                return label
        
        # Check if any label is explicitly a control
        non_control_labels = []
        for label in labels:
            label_lower = label.lower()
            if not any(keyword in label_lower for keyword in control_keywords):
                non_control_labels.append(label)
        
        if len(non_control_labels) == 1:
            logger.info(f"Control exclusion heuristic: '{non_control_labels[0]}' identified as positive class")
            return non_control_labels[0]
        
        # Tier 2: Minority class detection (not implemented here, requires data)
        # This would be implemented in the calling code where data is available
        
        # Tier 3: Default rule - use last label
        logger.warning(f"Using default rule: last label '{labels[-1]}' as positive class")
        return labels[-1]
    
    def encode_labels(
        self,
        labels: List[str],
        positive_class: str
    ) -> Dict[str, int]:
        """
        Create label encoding mapping with positive class as 1.
        
        Args:
            labels: List of unique class labels
            positive_class: The label to encode as 1 (positive class)
        
        Returns:
            Dict[str, int]: Mapping from label to encoded value
                           {positive_class: 1, other_class: 0}
        
        Example:
            >>> reasoner = MetadataReasoner()
            >>> mapping = reasoner.encode_labels(["Control", "CD"], "CD")
            >>> print(mapping)  # {"CD": 1, "Control": 0}
        """
        if positive_class not in labels:
            raise ValueError(f"Positive class '{positive_class}' not in labels list: {labels}")
        
        # Create mapping: positive class = 1, others = 0
        mapping = {}
        for label in labels:
            if label == positive_class:
                mapping[label] = 1
            else:
                mapping[label] = 0
        
        return mapping


# ============================================================================
# Convenience Functions
# ============================================================================

def infer_and_encode_labels(
    scenario: str,
    labels: List[str],
    llm_client=None
) -> Tuple[Dict[str, int], str, str]:
    """
    Convenience function to infer positive class and create encoding mapping.
    
    Args:
        scenario: Clinical research scenario
        labels: List of class labels
        llm_client: Optional LLM client instance
    
    Returns:
        Tuple containing:
        - mapping: Dict[str, int] - Label encoding mapping
        - positive_class: str - Identified positive class
        - reasoning: str - Explanation of the decision
    
    Example:
        >>> mapping, positive, reasoning = infer_and_encode_labels(
        ...     scenario="Crohn's Disease study",
        ...     labels=["Control", "CD"]
        ... )
        >>> print(mapping)  # {"CD": 1, "Control": 0}
        >>> print(positive)  # "CD"
    """
    reasoner = MetadataReasoner(llm_client=llm_client)
    
    # Infer positive class with reasoning
    positive_class, reasoning = reasoner.infer_positive_class(
        scenario=scenario,
        labels=labels,
        return_reasoning=True
    )
    
    # Create encoding mapping
    mapping = reasoner.encode_labels(labels, positive_class)
    
    return mapping, positive_class, reasoning


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("LLM Metadata Helper - Test Suite")
    print("="*80)
    
    # Test 1: Crohn's Disease scenario
    print("\n[Test 1] Crohn's Disease vs Control")
    print("-"*80)
    
    reasoner = MetadataReasoner()
    positive, reasoning = reasoner.infer_positive_class(
        scenario="Crohn's Disease biomarker discovery study",
        labels=["Control", "CD"],
        return_reasoning=True
    )
    
    print(f"Positive Class: {positive}")
    print(f"Reasoning: {reasoning}")
    
    mapping = reasoner.encode_labels(["Control", "CD"], positive)
    print(f"Encoding Mapping: {mapping}")
    
    # Test 2: Cancer scenario
    print("\n[Test 2] Cancer vs Normal")
    print("-"*80)
    
    positive, reasoning = reasoner.infer_positive_class(
        scenario="Lung cancer early detection study",
        labels=["Normal", "Cancer"],
        return_reasoning=True
    )
    
    print(f"Positive Class: {positive}")
    print(f"Reasoning: {reasoning}")
    
    mapping = reasoner.encode_labels(["Normal", "Cancer"], positive)
    print(f"Encoding Mapping: {mapping}")
    
    # Test 3: Convenience function
    print("\n[Test 3] Using convenience function")
    print("-"*80)
    
    mapping, positive, reasoning = infer_and_encode_labels(
        scenario="Type 2 Diabetes metabolomics study",
        labels=["Healthy", "T2D"]
    )
    
    print(f"Positive Class: {positive}")
    print(f"Reasoning: {reasoning}")
    print(f"Encoding Mapping: {mapping}")
    
    print("\n" + "="*80)
    print("✅ All tests completed!")
    print("="*80)
