# Counting illustrated object dataset

30 textured tabletop GLBs (0.11 × 0.11 × 0.0061 m), 200 two-tile base tasks, and 800 instructions. English instructions match the existing evaluation environment. Images are original programmatically drawn object illustrations, not photographs or free-standing object meshes. GLBs embed their PNG textures.

- `pairs.json`: existing evaluation format, one base task per row.
- `questions.json` / `questions.csv`: four conditions saved separately per task; individual condition JSON files contain 200 records each.
- `assets.json`: artwork metadata and paths; `model_db.json`: simulator metadata.
- `shapes/<id>/textured.glb` and `collision.obj`: visible tile and collision box.
- `images/` and `contact_sheet.png`: source artwork and preview.

Knowledge tasks require interpreting quantities in everyday situations or testing vertical reflection of the entire colored design. Object conditions explicitly identify the target; detailed conditions add visible appearance. Symmetry negatives vary across one-sided marks, vertical displacement, color mismatch, compressed width, and missing bands. These controlled edits can provide shortcuts; this is an illustrated reasoning benchmark, not a natural-image benchmark. Each category reuses its 30 assets; 200 tasks are not 200 independent visual scenes. Some semantic templates recur across objects and quantities. Left/right answers are balanced.

The four conditions share a question_id and scene: evaluate them as paired observations, not independent samples. No train/test split is imposed. Split by object and template family to reduce leakage; reuse of images across splits can still leak visual identity. Canonical stored spatial labels assume no scene swap. The environment dynamically recalculates spatial instructions when swapping. Use instruction_condition=explicit_object_detailed (or its visual_description alias) for the detailed condition; knowledge, explicit_object, and explicit_spatial already work directly.

Regenerate from repository root: `python scripts/dataset_generation/generate_counting_symmetry.py` (Pillow, numpy, trimesh). Validate: `python scripts/dataset_generation/validate_counting_symmetry.py`.
