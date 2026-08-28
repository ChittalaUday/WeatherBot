"""
v2 contracts and dataset generator - library code, not a model.

The v2 and v3 models are both deleted. What survives is what the dataset chain still needs: the
slot enums in schema.py (Intent, Variable, Aggregation, Slots) and the conversation generator in
dataset.py, whose output data/v3_dataset.csv is now the multi-turn fixture behind
tests/test_conversations.py.

See ARCHITECTURE.md for why.
"""
