"""
The conversation store: turns, feedback and the training rows they become.
Run: python tests/test_store_units.py

Moved out of `backend/store.py`, which keeps its `main()` - that one is a real CLI for reading
the database, not a self-check.
"""

from _root import ROOT  # noqa: F401 - puts the repo root on sys.path


def check_store():
    """Self-check: a turn plus a human label must come back out as a training row."""
    import tempfile

    from backend.store import (
        Path,
        connect,
        feedback_for,
        json,
        record_feedback,
        record_turn,
        stats,
        training_rows,
    )

    with tempfile.TemporaryDirectory() as tmp:
        connection = connect(Path(tmp) / "check.db")
        turn = record_turn(connection, "s1", "angara vs hyderbad", intent="TEMPERATURE",
                           action="COMPARE", confidence=0.25,
                           location=["angara", "hyderbad"], time_raw=[], time_norm=[],
                           outcome="clarified", detail="low confidence")
        record_feedback(connection, turn, "down", model="v1")
        record_feedback(connection, turn, "choice", intent="RAIN", action="COMPARE",
                        error_type="intent_confusion")

        row = connection.execute(
            "SELECT COUNT(*) n FROM feedback WHERE turn_id = ?", (turn,)).fetchone()
        assert row is not None
        rows_for_turn = row["n"]
        assert rows_for_turn == 1, rows_for_turn
        current = feedback_for(connection, turn)
        assert current is not None
        assert current["kind"] == "choice" and current["revisions"] == 1, current
        assert current["model"] == "v1", current      # untouched fields survive the update

        rows = training_rows(connection)
        assert len(rows) == 1, rows
        assert rows[0]["weather_intent"] == "RAIN", rows          # the human label wins
        assert json.loads(rows[0]["location"]) == ["angara", "hyderbad"], rows

        # an approved guess is not evidence unless the model was already confident
        low = record_turn(connection, "s1", "wind", intent="WIND_SPEED", action="GET",
                          confidence=0.30, outcome="answered")
        record_feedback(connection, low, "up", intent="WIND_SPEED", action="GET")
        assert len(training_rows(connection, include_approved=True)) == 1, "low-confidence up leaked"
        assert stats(connection)["turns"] == 2
        print("store demo OK:", rows[0])

def main():
    """Every check in this file, in order. Any assertion failure stops it."""
    for check in (check_store,):
        print(f"{check.__name__}:")
        check()
    print("\n1 check(s) passed")


if __name__ == "__main__":
    main()
