"""
Model 1 - the model decides presentation, not just extraction.

Eight targets per turn: intent, a multi-label variable set, aggregation, location and time
spans, and the three presentation decisions (detail, chart, insights) earlier models left to
Python. It commits to a reading rather than asking.

This is the only model served; see backend/registry.py, MODEL_RULES.md and ARCHITECTURE.md.
"""
