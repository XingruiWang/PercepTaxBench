# Question Generation Process Description

## From Taxonomy Annotations to Question-Answer Pairs

Our question generation pipeline follows a systematic, multi-stage process that transforms structured object annotations into diverse, high-quality question-answer pairs. The process consists of four main stages: (1) annotation extraction and taxonomy mapping, (2) template construction, (3) LLM-assisted rephrasing for diversity, and (4) human verification at each stage.

### Stage 1: Annotation Extraction and Taxonomy Mapping

Objects in each scene are first extracted from annotation files (either from simulation scene metadata or real image detection outputs). Each detected object is mapped to our hierarchical taxonomy system, which organizes objects across seven domains: material, affordance, function, physical properties, shape, texture, and environment. Objects are assigned to one or more taxonomy clusters within each domain based on their semantic properties. For example, a "wooden chair" would be mapped to the "Wood & Plant-Based Solids" cluster in the material domain, the "Furniture" cluster in the affordance domain, and the "Functional Seating" cluster in the function domain. This taxonomy mapping ensures that questions are grounded in semantically meaningful object categories rather than raw class names.

### Stage 2: Template Construction

For each question type (e.g., material property, spatial relation, affordance), we carefully construct multiple template variants that capture the same semantic intent while varying in phrasing, formality, and contextual framing. Templates are designed to be both linguistically natural and precise in their semantic requirements. For instance, repurposing questions include five distinct variants ranging from direct queries ("What object in the scene can someone grab and repurpose as a shield?") to scenario-based formulations ("During an unexpected emergency, which object could be quickly repurposed as a makeshift shield?"). Each template is parameterized to accept object names, material types, physical properties, or other domain-specific attributes extracted from the taxonomy annotations. This template-based approach ensures consistency while allowing for controlled variation.

### Stage 3: LLM-Assisted Rephrasing for Diversity

To further enhance linguistic diversity and naturalness, we employ large language models (LLMs) to generate additional rephrasings of our template-based questions. For each template variant, we provide the LLM with the original question, the target object(s), and relevant context (e.g., scene type, object properties), and request alternative phrasings that preserve the semantic intent while varying sentence structure, vocabulary, and stylistic register. The LLM-generated variants are then filtered to ensure they maintain the same answer space and semantic constraints as the original templates. This process significantly expands the linguistic diversity of our dataset while maintaining semantic consistency.

### Stage 4: Human Verification

Human verification is integrated at multiple stages of the pipeline to ensure quality and correctness. First, taxonomy cluster assignments are manually reviewed to ensure objects are correctly categorized. Second, template variants are validated by human annotators to confirm they are grammatically correct, semantically clear, and appropriate for the target question type. Third, LLM-generated rephrasings are reviewed to filter out any that introduce ambiguity, change the answer space, or contain unnatural phrasing. Finally, a sample of generated question-answer pairs is manually verified to ensure the questions are answerable from the image and that the ground-truth answers are correct. This multi-stage human verification process ensures that the final dataset maintains high quality standards while benefiting from the scalability of automated generation.

### Quality Assurance

Throughout the pipeline, we implement several automated quality checks: (1) objects must be present in the scene annotations, (2) taxonomy mappings must exist for all objects used in questions, (3) template parameters must be correctly filled (e.g., material names must be non-empty and valid), (4) questions must have exactly one correct answer, and (5) answer objects must be visually distinct and identifiable in the image. These automated checks, combined with human verification, ensure that the final dataset is both large-scale and high-quality.

