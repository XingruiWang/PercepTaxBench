# Property Format Templates

This document lists all property format templates that are appended to questions based on question type.

## General Format

Properties are appended to the question text in this format:
```
[Original Question] Choice properties: [Choice1] ([property1]; [property2]); [Choice2] ([property1]); [Choice3] ([property1]; [property2]; [property3]).
```

Each choice's properties are formatted as: `([property1]; [property2]; ...)`

---

## Question Type Templates

### 1. Material Questions (`material_*`)

**Template:**
```
(made of [material]; material cluster: [cluster1], [cluster2])
```

**Example:**
```
Red box (made of Metals & Alloys; material cluster: Metals & Alloys)
Green box (made of Stone, Concrete & Mineral; material cluster: Stone, Concrete & Mineral)
```

**Properties included:**
- `made of [material]` - Single material value
- `material cluster: [cluster1], [cluster2]` - Non-void material cluster names (comma-separated)

---

### 2. Affordance Questions (`affordance_*`)

**Template:**
```
(affordance: [cluster1], [cluster2])
```

**Example:**
```
Red box (affordance: Sit / Ride / Attend)
Green box (affordance: Contain / Carry / Package)
```

**Properties included:**
- `affordance: [cluster1], [cluster2]` - Non-void affordance cluster names (comma-separated)

---

### 3. Function Questions (`function_*`)

**Template:**
```
(function: [function]; function cluster: [cluster1], [cluster2])
```

**Example:**
```
Red box (function: sitting; function cluster: Seating)
Green box (function: storage; function cluster: Storage)
```

**Properties included:**
- `function: [function]` - Single function value
- `function cluster: [cluster1], [cluster2]` - Non-void function cluster names (comma-separated)

---

### 4. Physical Property Questions (`physical_*`)

**Template:**
```
(physical properties: [prop1], [prop2], [prop3]; physical property cluster: [cluster1], [cluster2])
```

**Example:**
```
Red box (physical properties: heavy, rigid, movable, stable; physical property cluster: Rigid, Movable)
Green box (physical properties: light, flexible; physical property cluster: Light, Flexible)
```

**Properties included:**
- `physical properties: [prop1], [prop2], ...` - Physical property values (comma-separated)
- `physical property cluster: [cluster1], [cluster2]` - Non-void physical property cluster names (comma-separated)

---

### 5. Repurposing Questions (`repurposing_*`)

**Template:**
```
(affordance: [cluster1], [cluster2]; made of [material]; physical properties: [prop1], [prop2])
```

**Example:**
```
Red box (affordance: Sit / Ride / Attend; made of Metals & Alloys; physical properties: heavy, rigid, movable)
Green box (affordance: Contain / Carry / Package; made of Wood and Plant-based Solids; physical properties: light, hollow)
```

**Properties included:**
- `affordance: [cluster1], [cluster2]` - Non-void affordance cluster names
- `made of [material]` - Material value
- `physical properties: [prop1], [prop2]` - Physical property values (comma-separated)

---

### 6. Compositional Questions (`compositional_*`)

**Template:**
```
(made of [material]; affordance: [cluster1], [cluster2]; physical properties: [prop1], [prop2]; function: [function])
```

**Example:**
```
Red box (made of Metals & Alloys; affordance: Sit / Ride / Attend; physical properties: heavy, rigid, movable; function: seating)
Green box (made of Wood and Plant-based Solids; affordance: Contain / Carry / Package; physical properties: light, hollow; function: storage)
```

**Properties included:**
- `made of [material]` - Material value
- `affordance: [cluster1], [cluster2]` - Non-void affordance cluster names
- `physical properties: [prop1], [prop2]` - Physical property values (comma-separated)
- `function: [function]` - Function value

---

### 7. Counterfactual Questions (`counterfactual_*`)

**Template:**
```
(made of [material]; physical properties: [prop1], [prop2])
```

**Example:**
```
Red box (made of Metals & Alloys; physical properties: heavy, rigid, heat-resistant)
Green box (made of Wood and Plant-based Solids; physical properties: light, flammable)
```

**Properties included:**
- `made of [material]` - Material value
- `physical properties: [prop1], [prop2]` - Physical property values (comma-separated)

---

### 8. Functional Questions (`functional_*`)

**Template:**
```
(function: [function]; affordance: [cluster1], [cluster2])
```

**Example:**
```
Red box (function: sitting; affordance: Sit / Ride / Attend)
Green box (function: storage; affordance: Contain / Carry / Package)
```

**Properties included:**
- `function: [function]` - Function value
- `affordance: [cluster1], [cluster2]` - Non-void affordance cluster names

---

### 9. Description Matching Questions (`description_matching`)

**Template:**
```
(description: [description]; made of [material]; function: [function])
```

**Example:**
```
Red box (description: A red, soft, rectangular seating object; made of Textiles, Fibers & Leather; function: seating)
Green box (description: A wooden storage container; made of Wood and Plant-based Solids; function: storage)
```

**Properties included:**
- `description: [description]` - Object description
- `made of [material]` - Material value
- `function: [function]` - Function value

---

## Complete Example

**Original Question:**
```
Which object is made of metals and alloys? Objects to choose from: Red box, Green box, Blue box
```

**After Property Addition:**
```
Which object is made of metals and alloys? Objects to choose from: Red box, Green box, Blue box Choice properties: Red box (made of Metals & Alloys; material cluster: Metals & Alloys); Green box (made of Stone, Concrete & Mineral; material cluster: Stone, Concrete & Mineral); Blue box (made of Wood and Plant-based Solids; material cluster: Wood and Plant-based Solids).
```

---

## Notes

1. **Void Cluster Filtering**: All cluster names are filtered to exclude void clusters (e.g., "No Clear Function", "Abstract / Depictions / Scenes/ Occupations", etc.)

2. **Empty Properties**: If an object has no properties for a given question type, no property text is appended for that choice.

3. **Property Separator**: Multiple properties within a choice are separated by `; ` (semicolon + space)

4. **Cluster Separator**: Multiple cluster names are separated by `, ` (comma + space)

5. **Choice Separator**: Multiple choices in the appended text are separated by `; ` (semicolon + space)

6. **Format Consistency**: All properties follow the pattern: `[property_type]: [value(s)]`

