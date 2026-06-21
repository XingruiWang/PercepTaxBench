"""
Domain-specific reasoning generators for Chain-of-Thought outputs.

Each module exposes functions that accept a CoTReasoningGenerator instance and
return structured reasoning steps for a particular class of question types.
"""

from .affordance import generate_affordance_reasoning  # noqa: F401
from .compositional import generate_compositional_reasoning  # noqa: F401
from .description import generate_description_reasoning  # noqa: F401
from .material import generate_material_reasoning  # noqa: F401
from .physical import generate_physical_property_reasoning  # noqa: F401
from .repurposing import generate_repurposing_reasoning  # noqa: F401
from .spatial import (
    build_spatial_relation_clause,  # noqa: F401
    generate_spatial_reasoning,  # noqa: F401
)
from .taxonomy import generate_taxonomy_aware_reasoning  # noqa: F401
from .function import generate_function_reasoning  # noqa: F401

__all__ = [
    "generate_affordance_reasoning",
    "generate_compositional_reasoning",
    "generate_description_reasoning",
    "generate_material_reasoning",
    "generate_physical_property_reasoning",
    "generate_repurposing_reasoning",
    "generate_spatial_reasoning",
    "build_spatial_relation_clause",
    "generate_taxonomy_aware_reasoning",
    "generate_function_reasoning",
]

