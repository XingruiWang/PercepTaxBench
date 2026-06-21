# Question Generation Process Summary (Research Paper Version)

Our question generation pipeline systematically converts structured object annotations into large-scale, semantically grounded question–answer pairs for a vision–language QA benchmark. Starting from taxonomy-based mappings across material, affordance, and function domains, we construct parameterized templates that capture diverse semantic intents. These templates are expanded through LLM-assisted rephrasing to enhance linguistic variety while preserving answer consistency. 

To ensure question quality, we implement cluster-based filtering that excludes inappropriate candidate objects. For example, when generating material property questions, objects assigned to abstract or unclassified material clusters (e.g., "depicted scenes" or "occupations") are filtered out, as these do not represent concrete material properties. Similarly, for physical property questions, objects in the "no physical properties" cluster are excluded to maintain semantic validity.

Additionally, we enforce answer space uniqueness by suppressing question generation when multiple objects in the scene could validly answer the same question. For instance, if a scene contains multiple wooden objects, a question like "Which object is made of wood?" would not be generated, as it lacks a unique answer. This ensures each question has exactly one correct answer, preventing ambiguity in the benchmark.

Human verification at multiple stages—taxonomy validation, template review, and final QA inspection—ensures grammatical correctness, semantic clarity, and visual answerability. Combined with automated quality checks on scene presence, taxonomy validity, and answer uniqueness, this process enables scalable yet high-quality generation of diverse, interpretable QA data grounded in real and simulated visual contexts.

