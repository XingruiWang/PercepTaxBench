import re
from typing import List, Optional

__all__ = [
    "clean_description_punctuation",
    "clean_text",
    "normalize_cluster_key",
    "clean_cluster_name",
    "cluster_label",
    "naturalize_cluster_phrase",
    "summarize_feature_list",
    "format_object_list",
    "render_domain_reference",
    "compose_entry_summary",
    "render_filter_trait",
    "cluster_feature_override",
]

# Shared descriptor phrases used across reasoning templates
_CLUSTER_DESCRIPTORS = {
    "material": ("material group", "material groups"),
    "affordance": ("affordance", "affordances"),
    "function": ("function", "functions"),
    "physical_property": ("physical property group", "physical property groups"),
    "physical": ("physical property group", "physical property groups"),
    "default": ("taxonomy group", "taxonomy groups"),
}

_CLUSTER_PHRASE_OVERRIDES = {
    "affordancehouseholdfacilityoperations": "household or facility operations tasks",
}

_CLUSTER_FEATURE_OVERRIDES = {
    "affordancehouseholdfacilityoperations": "integrated storage, shelving, and utility surfaces used to support household or facility chores",
    "rigid": "sturdy framing and inflexible structure",
    "movable": "lightweight construction and reachable grasp points",
    "contain": "hollow volume with openings for storage",
    "hollow": "internal cavity or void space for placement",
}

_SUFFIX_MAP = {
    "material": "materials",
    "affordance": "affordances",
    "function": "functions",
    "physical_property": "traits",
    "physical": "traits",
}


def clean_description_punctuation(text: str) -> str:
    """Remove specific punctuation from description text for better readability."""
    if not text:
        return ""
    cleaned = text.replace('&', 'and').replace('/', ', ').replace('‑', '-')
    cleaned = re.sub(r'\([^)]*\)', '', cleaned)
    cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
    cleaned = re.sub(r'[^\w\s\.,!?:-]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', cleaned)
    return cleaned


def clean_text(text: str) -> str:
    """General purpose text cleaner used across reasoning outputs."""
    if not text:
        return ""
    cleaned = clean_description_punctuation(text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'[,;:]+$', '', cleaned)
    cleaned = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', cleaned)
    return cleaned.strip()


def normalize_cluster_key(text: Optional[str]) -> str:
    if not text:
        return ""
    normalized = str(text).lower().replace('&', 'and')
    return re.sub(r'[^a-z0-9]+', '', normalized)


def clean_cluster_name(cluster_name: str) -> str:
    cleaned = cluster_name.replace(' & ', ' and ').replace(' / ', ', ').replace('‑', '-')
    cleaned = cleaned.replace('(', '').replace(')', '').replace('  ', ' ')
    return cleaned.strip()


def cluster_label(domain: Optional[str], plural: bool = False) -> str:
    descriptor = _CLUSTER_DESCRIPTORS.get((domain or '').lower(), _CLUSTER_DESCRIPTORS["default"])
    return descriptor[1 if plural else 0]


def naturalize_cluster_phrase(cluster_name: Optional[str], domain: Optional[str] = None) -> Optional[str]:
    if not cluster_name:
        return None
    normalized = normalize_cluster_key(cluster_name)
    if normalized in _CLUSTER_PHRASE_OVERRIDES:
        return _CLUSTER_PHRASE_OVERRIDES[normalized]
    cleaned = clean_cluster_name(cluster_name)
    if not cleaned:
        return None
    parts = [part.strip() for part in cleaned.split(',') if part.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        phrase = parts[0]
    elif len(parts) == 2:
        phrase = f"{parts[0]} or {parts[1]}"
    else:
        phrase = ', '.join(parts[:-1]) + f", or {parts[-1]}"
    return phrase.replace('_', ' ').lower()


def summarize_feature_list(clauses: List[str]) -> Optional[str]:
    cleaned_clauses: List[str] = []
    for clause in clauses:
        cleaned = clean_text(clause)
        if cleaned and cleaned not in cleaned_clauses:
            cleaned_clauses.append(cleaned)
    if not cleaned_clauses:
        return None
    if len(cleaned_clauses) == 1:
        return cleaned_clauses[0]
    limited = cleaned_clauses[:3]
    if len(limited) == 2:
        return f"{limited[0]} and {limited[1]}"
    return ', '.join(limited[:-1]) + f", and {limited[-1]}"


def format_object_list(items: List[str]) -> str:
    names = [item for item in items if item]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ', '.join(names[:-1]) + f", and {names[-1]}"


def render_domain_reference(domain: Optional[str], label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    label_clean = clean_text(label)
    suffix = _SUFFIX_MAP.get((domain or '').lower())
    if suffix:
        return f"{label_clean} {suffix}"
    return label_clean


def compose_entry_summary(entry: dict) -> Optional[str]:
    if not entry:
        return None
    label_phrase = render_domain_reference(entry.get("domain"), entry.get("label"))
    features = clean_text(entry.get("features") or "") if entry.get("features") else ""
    if label_phrase and features:
        return f"{label_phrase} ({features})"
    if label_phrase:
        return label_phrase
    if features:
        return features
    return None


def render_filter_trait(filter_type: Optional[str], filter_value: Optional[str], feature_summary: Optional[str]) -> str:
    if feature_summary:
        return feature_summary
    domain_key = (filter_type or '').lower()
    natural_value = naturalize_cluster_phrase(filter_value, domain_key) or clean_cluster_name(filter_value or '')
    if natural_value:
        suffix = _SUFFIX_MAP.get(domain_key)
        if suffix:
            return f"{natural_value} {suffix}"
        return natural_value
    return "the required traits"


def cluster_feature_override(key: str) -> Optional[str]:
    return _CLUSTER_FEATURE_OVERRIDES.get(normalize_cluster_key(key))
