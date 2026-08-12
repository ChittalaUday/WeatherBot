"""
Samples real place names from the `shapes` schema into data/locations.csv, which
src/build_dataset.py uses as its LOCATION vocabulary.

READ-ONLY BY CONSTRUCTION: every statement below is a SELECT against shapes.* and the
connection is opened with default_transaction_read_only=on, so the server rejects any
write even if this file is edited carelessly. Only `name` and the shape centroid are read;
identifiers (`unq`, `*_id`) are never exported.

Mix is village-heavy (weather queries are asked about villages, and villages carry their
own centroid column), topped up with blocks, districts and states, and >=80% of the names
fall inside the India bounding box. A fifth of the names is reserved for the eval split so
the held-out entity vocabulary stays held out.

Usage:  python src/fetch_locations.py [--out data/locations.csv]
        (needs psql on PATH + DB_* in .env; without the CSV the dataset builder falls
         back to its built-in city list)
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# lat/lng envelope for India (minx, miny, maxx, maxy)
INDIA_BBOX = "st_makeenvelope(68.0, 6.0, 97.5, 37.6, 4326)"

# level -> (rows inside India, rows outside India). Village-heavy on purpose. shapes.village,
# .block and .district hold India only, so the non-India remainder (kept under 20%, Rule:
# >=80% of locations inside India) comes from foreign states and countries.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/locations.csv")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--eval-share", type=int, default=5,
                        help="every Nth name is reserved for the eval split (default 5 = 20%%)")
    args = parser.parse_args()

    env = read_env(ROOT / args.env)
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
                             "in_india": int(inside),
                             "parents": " | ".join(clean)})
                got += 1
            print(f"  {level:9s} {'inside' if inside else 'outside':7s} India: {got}/{limit}")

    # Deterministic hold-out slice: eval-only names, so the eval split keeps unseen entities.
    rows.sort(key=lambda r: r["name"].lower())
    for i, row in enumerate(rows):
        row["split"] = "eval" if i % args.eval_share == 0 else "train"

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["name", "level", "parents", "lat", "lng", "in_india", "split"])
        writer.writeheader()
        writer.writerows(rows)

    india = sum(r["in_india"] for r in rows)
    levels = {lvl: sum(r["level"] == lvl for r in rows) for lvl in QUOTAS}
    print(f"Wrote {len(rows)} names -> {out}")
    print(f"  inside India: {india} ({india / len(rows):.1%})   levels: {levels}")
    print(f"  reserved for eval split: {sum(r['split'] == 'eval' for r in rows)}")


if __name__ == "__main__":
    main()
