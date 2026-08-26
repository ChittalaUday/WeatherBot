"""
Builds data/locations.csv - the LOCATION vocabulary src/build_dataset.py generates prompts
from. Three sources, in this order:

  1. names    the `shapes` schema over psql - real Indian villages / blocks / districts /
              states, sampled read-only
  2. world    GeoNames cities15000 - international cities, so the vocabulary is not India-only
  3. codes    GeoNames alternate names - the shortcuts people actually type: HYD, VTZ, BLR,
              LON, TYO, DXB, and state codes like AP / TS / HP
  4. typos    3-8 generated misspellings per name, so the span tagger sees "hyderbad" and
              "vishakapatnam" during training instead of meeting them for the first time in
              production

READ-ONLY BY CONSTRUCTION (step 1): every statement is a SELECT against shapes.* and the
connection is opened with default_transaction_read_only=on, so the server rejects any write
even if this file is edited carelessly. Only `name` and the shape centroid are read;
identifiers (`unq`, `*_id`) are never exported.

Mix is village-heavy (weather queries are asked about villages, and villages carry their own
centroid column). test_dataset.py asserts >=1000 names, village-led, >=80% inside India, so
the international quota is computed from the India count rather than fixed - see WORLD_SHARE.
A fifth of the names is reserved for the eval split so the held-out entity vocabulary stays
held out; a name's misspellings inherit its split, which is why they are a column and not
extra rows.

Usage:
    python src/fetch_locations.py                  # all four steps
    python src/fetch_locations.py --no-db          # reuse the names already in the CSV
    python src/fetch_locations.py --skip-codes     # offline: no GeoNames download
"""

import argparse
import csv
import math
import os
import random
import re
import subprocess
import sys
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.build_dataset import _misspell  # one realistic keyboard slip, already tuned

# lat/lng envelope for India (minx, miny, maxx, maxy)
INDIA_BBOX = "st_makeenvelope(68.0, 6.0, 97.5, 37.6, 4326)"

# level -> (rows inside India, rows outside India). Village-heavy on purpose. shapes.village,
# .block and .district hold India only, so the non-India remainder comes from foreign states
# and countries, topped up by the GeoNames world cities below.
QUOTAS = {
    "village": (800, 0), "block": (130, 0), "district": (70, 0),
    "state": (32, 20), "country": (0, 45),
}
CENTROID_LEVELS = {"village", "country"}  # tables carrying a ready-made centroid column

# Junk and unusable rows: placeholders, numeric codes, one-word noise, ampersand mouthfuls.
NAME_FILTER = """{col} is not null and length({col}) between 4 and 28
    and {col} !~* 'unknown|unnamed|^n[./ ]?a$' and {col} !~ '[0-9&/()]'
    and {col} ~ '^[A-Za-z][A-Za-z .-]+$'"""


# Ancestors of each level, so a village can be written out as "Angara, East Godavari" or
# "Angara, Rajahmundry, East Godavari, Andhra Pradesh" by the dataset builder.
ANCESTRY = {
    "village": [("block", "b", "v.block_id = b.unq"),
                ("district", "d", "b.dist_id = d.unq"),
                ("state", "s", "d.state_id = s.unq")],
    "block": [("district", "d", "v.dist_id = d.unq"), ("state", "s", "d.state_id = s.unq")],
    "district": [("state", "s", "v.state_id = s.unq")],
    "state": [("country", "c", "v.country_id = c.unq")],
    "country": [],
}

# --- GeoNames ---------------------------------------------------------------
# Only three files, all small. The global allCountries / alternateNamesV2 dumps are hundreds
# of megabytes and hold nothing these three do not.
#   cities15000.zip        3.3 MB  34k cities, 244 countries, codes inline in column 4
#   IN.zip                15.7 MB  every Indian feature - needed to turn a village name into
#                                  a geonameid, which is the only key the codes file has
#   alternatenames/IN.zip   1.4 MB  typed codes for those ids (iata / abbr / unlc)
GEONAMES = "https://download.geonames.org/export/dump"
CACHE = ROOT / "data" / ".geonames_cache"

# Which alternate-name rows are a shortcut a human would type. `link` is a URL, `post` a
# postcode, `wkdt` a Wikidata id - none of those are things anyone types into a chat box.
CODE_LANGS = {"iata", "abbr", "unlc"}
CODE_MIN, CODE_MAX = 2, 6

# International share of the vocabulary. test_dataset.py asserts >=80% India, so this stays
# under that with room to spare rather than sitting on the line.
WORLD_SHARE = 0.14
PER_COUNTRY = 4          # so the list is not 90% United States and China
MIN_CITY_POP = 15000     # what cities15000 already holds; stated here to be explicit

TYPO_RANGE = (3, 8)      # misspellings generated per name
TYPO_MIN_LEN = 4         # _misspell indexes word[i+1] and word[i+2]; shorter words are unsafe

# How far a GeoNames point may sit from our centroid before it is a different place, and
# which GeoNames feature class that level should be. A big state's geometric centre is
# hundreds of km from the town GeoNames marks it by, while two villages 30 km apart sharing a
# name are genuinely two villages. Class matters as much as distance: without it Delhi the
# state matches Delhi the city and comes back with an airport code.
#   A = country / state / region / ...   P = city / village / ...
LEVEL_MATCH = {
    "village": (25.0, "P"), "city": (25.0, "P"), "block": (60.0, "A"),
    "district": (120.0, "A"), "state": (500.0, "A"), "country": (900.0, "A"),
}


def query(level: str, limit: int, inside: bool) -> str:
    """SELECT name, ancestor names and centroid for one admin level, in/outside the India bbox."""
    point = "v.centroid" if level in CENTROID_LEVELS else "st_pointonsurface(v.geometry)"
    shape = "v.centroid" if level in CENTROID_LEVELS else "v.geometry"
    # TABLESAMPLE keeps the 624k-row village scan cheap; REPEATABLE makes it reproducible.
    sample = " tablesample system (1.2) repeatable (7)" if level == "village" else ""
    where = f"{shape} && {INDIA_BBOX}" if inside else f"not ({shape} && {INDIA_BBOX})"
    joins = "".join(f" left join shapes.{tbl} {alias} on {on}" for tbl, alias, on in ANCESTRY[level])
    # Fixed 4 ancestor slots keep every row the same width regardless of level.
    parents = [f"{alias}.name" for _, alias, _ in ANCESTRY[level]]
    parents += ["null"] * (3 - len(parents))
    # md5 shuffle in the outer query: DISTINCT ON forces an alphabetical inner sort, and
    # taking the LIMIT off that would hand back nothing but A-names.
    return f"""
        select name, lat, lng, p1, p2, p3 from (
            select distinct on (lower(v.name)) v.name as name,
                   round(st_y({point})::numeric, 4) as lat,
                   round(st_x({point})::numeric, 4) as lng,
                   {parents[0]} as p1, {parents[1]} as p2, {parents[2]} as p3
            from shapes.{level} v{sample}{joins}
            where {where} and {NAME_FILTER.format(col="v.name")}
            order by lower(v.name), 1
        ) picked
        order by md5(name)
        limit {limit};
    """


def run_psql(sql: str, env: dict) -> list:
    """One read-only psql call; returns rows as lists of strings."""
    proc = subprocess.run(
        ["psql", "-h", env["DB_HOST"], "-p", env["DB_PORT"], "-U", env["DB_USER"],
         "-d", env["DB_NAME"], "-At", "-F", "\t", "-v", "ON_ERROR_STOP=1",
         "-c", "set default_transaction_read_only = on", "-c", sql],
        capture_output=True, text=True,
        env={**os.environ, "PGPASSWORD": env["DB_PASSWORD"], "PGCONNECT_TIMEOUT": "10"},
    )
    if proc.returncode:
        sys.exit(f"psql failed: {proc.stderr.strip()}")
    return [line.split("\t") for line in proc.stdout.splitlines() if "\t" in line]


def read_env(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"{path} not found - DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME required")
    env = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    missing = {"DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"} - env.keys()
    if missing:
        sys.exit(f"{path} missing keys: {sorted(missing)}")
    return env


def from_db(env: dict) -> list[dict]:
    """Step 1: real Indian place names, sampled read-only from the shapes schema."""
    rows, seen = [], set()
    for level, (n_in, n_out) in QUOTAS.items():
        for inside, limit in ((True, n_in), (False, n_out)):
            if not limit:
                continue
            got = 0
            for name, lat, lng, *parents in run_psql(query(level, limit, inside), env):
                if name.lower() in seen:
                    continue
                seen.add(name.lower())
                # parents run child -> parent; blank out junk so the builder can skip them
                clean = [p for p in parents if p and p.upper() != p.lower() and "UNKNOWN" not in p.upper()]
                rows.append({"name": name, "level": level, "lat": lat, "lng": lng,
                             "in_india": int(inside), "parents": " | ".join(clean)})
                got += 1
            print(f"  {level:9s} {'inside' if inside else 'outside':7s} India: {got}/{limit}")
    return rows


# --- GeoNames downloads -----------------------------------------------------

def fetch(path: str) -> Path:
    """Download one GeoNames file into data/.geonames_cache and unzip it. Cached across runs.

    The cache key keeps the directory: `IN.zip` and `alternatenames/IN.zip` are different
    files that both unzip to a file called `IN.txt`, so a flat cache silently serves one for
    the other.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    key = path.replace("/", "_")
    archive = CACHE / key
    if not archive.exists():
        print(f"  downloading {path} ...", flush=True)
        urllib.request.urlretrieve(f"{GEONAMES}/{path}", archive)
    if not key.endswith(".zip"):
        return archive
    target = CACHE / key[:-4]                      # a directory per archive, never shared
    if not target.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    return next(p for p in sorted(target.iterdir()) if p.suffix == ".txt" and p.name != "readme.txt")


def fold(text: str) -> str:
    """'Visākhapatnam' -> 'visakhapatnam'. GeoNames writes diacritics, the shapes DB does not."""
    stripped = unicodedata.normalize("NFKD", text)
    return "".join(c for c in stripped if not unicodedata.combining(c)).lower().strip()


def _clean_codes(raw) -> list[str]:
    """Keep the ones a person would type; drop the ones only a machine uses."""
    out = set()
    for code in raw:
        code = code.strip().upper()
        if not (code.isascii() and code.isalnum() and CODE_MIN <= len(code) <= CODE_MAX):
            continue
        # UN/LOCODEs are the country code plus three letters - INHYD, INVIZ, INGTR. The tail
        # is the part people actually type.
        if len(code) == 5 and code.startswith("IN"):
            out.add(code[2:])
        else:
            out.add(code)
    return sorted(out)


def world_cities(limit: int, per_country: int) -> list[dict]:
    """Step 2: international cities, biggest first, capped per country.

    cities15000 carries its codes inline in the alternatenames column - LON, TYO, NYC, DXB,
    SIN, BKK - so international names need no second download.
    """
    countries = {}
    for line in fetch("countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if not line.startswith("#") and "\t" in line:
            parts = line.split("\t")
            countries[parts[0]] = parts[4]

    cities = []
    with fetch("cities15000.zip").open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15 or parts[8] == "IN":       # India comes from the shapes DB
                continue
            name, alts, lat, lng, country = parts[1], parts[3], parts[4], parts[5], parts[8]
            population = int(parts[14] or 0)
            if not name.isascii() or population < MIN_CITY_POP or not re.fullmatch(r"[A-Za-z][A-Za-z .'-]+", name):
                continue
            cities.append((population, name, lat, lng, country,
                           _clean_codes(a for a in alts.split(",") if a.isupper())))

    cities.sort(key=lambda c: -c[0])
    rows, per, seen = [], {}, set()
    for population, name, lat, lng, country, codes in cities:
        if len(rows) >= limit:
            break
        if name.lower() in seen or per.get(country, 0) >= per_country:
            continue
        seen.add(name.lower())
        per[country] = per.get(country, 0) + 1
        rows.append({"name": name, "level": "city", "lat": lat, "lng": lng, "in_india": 0,
                     "parents": countries.get(country, country), "codes": codes})
    print(f"  world     cities: {len(rows)} across {len(per)} countries "
          f"({sum(bool(r['codes']) for r in rows)} with codes)")
    return rows


def india_codes(rows: list[dict]) -> dict[str, list[str]]:
    """Step 3: shortcuts for the Indian names, matched on name, position *and* feature class.

    Name alone is not enough - India has many places called Angara, and attaching Hyderabad's
    HYD to a village that happens to share a name is worse than having no code at all. Only
    the names actually in `rows` are indexed, so the 70 MB dump streams instead of loading.
    """
    wanted: dict[str, list[dict]] = {}
    for row in rows:
        if str(row["in_india"]) == "1":
            wanted.setdefault(fold(row["name"]), []).append(row)
    if not wanted:
        return {}

    # name -> geonameid. Ranked (right feature class, then population), so a state matches the
    # admin division rather than whichever city sits nearest its centroid.
    best: dict[str, tuple[int, int, str]] = {}
    with fetch("IN.zip").open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue
            keys = {fold(parts[1]), fold(parts[2])} & wanted.keys()
            if not keys:
                continue
            lat, lng, population = float(parts[4]), float(parts[5]), int(parts[14] or 0)
            for key in keys:
                for row in wanted[key]:
                    radius, want_class = LEVEL_MATCH.get(row["level"], (50.0, "P"))
                    near = math.hypot((float(row["lat"]) - lat) * 111,
                                      (float(row["lng"]) - lng) * 111 * math.cos(math.radians(lat)))
                    score = (int(parts[6] == want_class), population)
                    if near <= radius and score >= best.get(row["name"], (-1, -1, ""))[:2]:
                        best[row["name"]] = (*score, parts[0])

    by_id: dict[str, set] = {}
    with fetch("alternatenames/IN.zip").open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 8 and parts[2] in CODE_LANGS:
                by_id.setdefault(parts[1], set()).add(parts[3])

    codes = {}
    for name, (_, _, geonameid) in best.items():
        # a code identical to the name it stands for teaches the resolver nothing
        found = [c for c in _clean_codes(by_id.get(geonameid, ())) if fold(c) != fold(name)]
        if found:
            codes[name] = found
    print(f"  india     matched {len(best)}/{len(wanted)} names to GeoNames, "
          f"{len(codes)} carry a code")
    return codes


def misspellings(rng: random.Random, name: str, low: int, high: int) -> list[str]:
    """3-8 distinct keyboard slips of one place name, for the span tagger to train on.

    Villages are the names users get wrong most, and the tagger only ever saw them spelled
    perfectly. Each variant misspells one word of the name; the rest stays intact so the
    result still looks like the place it came from.
    """
    words = [m for m in re.finditer(r"[A-Za-z]{%d,}" % TYPO_MIN_LEN, name)]
    if not words:
        return []
    out: list[str] = []
    wanted = rng.randint(low, high)
    for _ in range(wanted * 6):                      # slips collide; try harder than `wanted`
        if len(out) >= wanted:
            break
        word = rng.choice(words)
        broken = name[:word.start()] + _misspell(rng, word.group()) + name[word.end():]
        if broken != name and broken not in out:
            out.append(broken)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/locations.csv")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--eval-share", type=int, default=5,
                        help="every Nth name is reserved for the eval split (default 5 = 20%%)")
    parser.add_argument("--no-db", action="store_true",
                        help="keep the names already in --out instead of querying shapes")
    parser.add_argument("--skip-world", action="store_true", help="no international cities")
    parser.add_argument("--skip-codes", action="store_true", help="no GeoNames download")
    parser.add_argument("--skip-typos", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    out = ROOT / args.out
    rng = random.Random(args.seed)

    # 1. names
    if args.no_db:
        if not out.exists():
            sys.exit(f"--no-db needs an existing {out}")
        rows = [{**r, "in_india": int(r["in_india"])} for r in csv.DictReader(out.open())]
        rows = [{k: v for k, v in r.items() if k in
                 {"name", "level", "parents", "lat", "lng", "in_india"}} for r in rows]
        print(f"  reusing   {len(rows)} names from {out.name}")
    else:
        rows = from_db(read_env(ROOT / args.env))

    # 2. world - quota computed from the India count, so the >=80% assertion cannot be tripped
    if not args.skip_world:
        rows = [r for r in rows if r["level"] != "city"]          # rebuilt every run
        india = sum(r["in_india"] for r in rows)
        outside = len(rows) - india
        allowed = max(int(india * WORLD_SHARE / (1 - WORLD_SHARE)) - outside, 0)
        rows += world_cities(allowed, PER_COUNTRY)

    # 3. codes
    codes = {r["name"]: r.get("codes", []) for r in rows}
    if not args.skip_codes:
        codes.update(india_codes(rows))

    # 4. typos
    for row in rows:
        row["codes"] = "|".join(codes.get(row["name"], []))
        row["misspellings"] = "" if args.skip_typos else "|".join(
            misspellings(rng, row["name"], *TYPO_RANGE))

    # Deterministic hold-out slice: eval-only names, so the eval split keeps unseen entities.
    # Misspellings ride along in their name's row and inherit its split - as separate rows a
    # broken spelling of an eval name would leak straight into train.
    rows.sort(key=lambda r: r["name"].lower())
    for i, row in enumerate(rows):
        row["split"] = "eval" if i % args.eval_share == 0 else "train"

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "level", "parents", "lat", "lng",
                                                "in_india", "split", "codes", "misspellings"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    india = sum(r["in_india"] for r in rows)
    levels = {lvl: sum(r["level"] == lvl for r in rows) for lvl in list(QUOTAS) + ["city"]}
    coded = sum(bool(r["codes"]) for r in rows)
    typo_total = sum(len(r["misspellings"].split("|")) if r["misspellings"] else 0 for r in rows)
    print(f"\nWrote {len(rows)} names -> {out}")
    print(f"  inside India: {india} ({india / len(rows):.1%})   levels: {levels}")
    print(f"  with codes:   {coded} ({coded / len(rows):.1%})   misspellings: {typo_total}")
    print(f"  reserved for eval split: {sum(r['split'] == 'eval' for r in rows)}")
    if india / len(rows) < 0.8:
        sys.exit("FAILED: under 80% of the vocabulary is in India - test_dataset.py asserts this")


if __name__ == "__main__":
    main()
