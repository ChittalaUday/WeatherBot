"""
Text handling: normalization, span tagging, and v4 entity extraction.
Run: python tests/test_text_units.py

Moved out of `src/normalize.py`, `src/tagger.py` and `src/v4/entities.py`. The entity module
keeps its command-line branch, which extracts from a sentence given as an argument.
"""

from _root import ROOT  # noqa: F401 - puts the repo root on sys.path


def check_normalize():
    """Self-check: shorthand folds, place names and casing survive."""
    from src.normalize import normalize
    result = normalize("What’s da wthr in KKD tmrw??")
    assert result.normalized == "what is the weather in KKD tomorrow??", result.normalized
    assert ["da", "the"] in result.replacements and ["tmrw", "tomorrow"] in result.replacements
    assert result.original == "What’s da wthr in KKD tmrw??"        # audit trail intact

    # place names are never rewritten here - that is the resolver's job (Rule 4.1)
    assert "KKD" in normalize("weather in KKD").normalized
    assert normalize("rain in Nokha").normalized == "rain in Nokha"

    # repeated characters, shouting, and unicode punctuation
    assert normalize("soooo hot").normalized == "soo hot"
    assert normalize("TEMP IN BZA").normalized == "TEMPERATURE IN BZA"
    assert "degrees" in normalize("is it 40° in Guntur").normalized

    # a clean sentence must come out unchanged apart from spacing
    assert (normalize("will it rain in Guntur tomorrow?").normalized
            == "will it rain in Guntur tomorrow?")
    print("normalize demo OK:", normalize("What’s da wthr in KKD tmrw??").model_dump())

def check_tagger():
    """Self-check: the tagger must generalise past the names it was trained on."""
    from src.tagger import SpanTagger, normalize_time
    train = [
        ("what is the rain in Guntur tomorrow?", {"location": ["Guntur"], "time": ["tomorrow"]}),
        ("temperature in Nokha at 6 PM", {"location": ["Nokha"], "time": ["6 PM"]}),
        ("compare rain in Guntur and Nokha today", {"location": ["Guntur", "Nokha"], "time": ["today"]}),
        ("humidity in Ausa, Latur tonight", {"location": ["Ausa, Latur"], "time": ["tonight"]}),
        ("wind in Nokha", {"location": ["Nokha"], "time": []}),
        ("set an alert for rain in Ausa tomorrow", {"location": ["Ausa"], "time": ["tomorrow"]}),
        ("rain today", {"location": [], "time": ["today"]}),
    ] * 12
    tagger = SpanTagger(metric_nouns=["rain", "temperature", "humidity", "wind"], min_word_freq=6)
    tagger.fit([t for t, _ in train], [s for _, s in train])

    unseen = tagger.predict("what is the rain in Kakinada tomorrow?")
    assert unseen["location"] == ["Kakinada"], unseen        # a name never trained on
    assert unseen["time"] == ["tomorrow"], unseen

    clock = tagger.predict("temperature in Nokha at 4:15 pm")
    assert "4:15 pm" in clock["time"], clock                 # regex pass covers unseen clocks

    # one canonical shape per expression, whatever the user typed
    assert normalize_time("tommorrow") == "tomorrow"
    assert normalize_time("TMRW") == "tomorrow"
    assert normalize_time("6:45 pm") == "18:45"
    assert normalize_time("at 8 AM") == "08:00"
    assert normalize_time("from 6 pm to 9 pm") == "18:00-21:00"
    assert normalize_time("in 45 minutes") == "next 45 minutes"
    assert normalize_time("on Friday") == "friday"
    assert normalize_time("sowing week") == "sowing week"    # unknown passes through, never dropped
    print("tagger demo OK:", unseen, clock)

def check_entities():
    """Self-check: longest match wins, spans stay verbatim, unknown words stay unknown."""
    from src.v4.entities import extract, vocabulary_report
    assert extract("should i spray pesticide on my cotton field tomorrow") == {
        "material": ["pesticide"], "crop": ["cotton"]}
    assert extract("can we play cricket in Guntur tomorrow") == {"sport": ["cricket"]}
    assert extract("can i wash my white clothes today")["clothing"] == ["white clothes"], \
        "longest match must beat the word inside it"
    assert extract("bengal gram sowing")["crop"] == ["bengal gram"]
    assert extract("Should I take the Motorcycle") == {"transport": ["Motorcycle"]}, \
        "spans come back cased as typed"
    assert extract("will it rain tomorrow") == {}, "no entity is a normal answer"
    assert extract("scattered showers") == {}, "substring must not match inside a word"

    report = vocabulary_report()
    assert not report["duplicates across types"], report["duplicates across types"]
    print(f"entities demo OK: {report['terms']} terms over {report['types']} types "
          f"{report['per type']}")

def main():
    """Every check in this file, in order. Any assertion failure stops it."""
    for check in (check_normalize, check_tagger, check_entities,):
        print(f"{check.__name__}:")
        check()
    print("\n3 check(s) passed")


if __name__ == "__main__":
    main()
