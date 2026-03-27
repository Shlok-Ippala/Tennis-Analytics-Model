"""
ETL pipeline: raw ATP match CSVs → fully feature-engineered match-level CSV.

Uses DuckDB for efficient SQL-based transformation. Produces one row per match
with Player A (lower ID) vs Player B (higher ID), all delta features, rolling
win percentages, missing indicators, and tournament metadata.

Output: data/clean/atp_matches_features.csv
"""

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "tennis_atp-master"
OUTPUT_PATH = ROOT / "data" / "clean" / "atp_matches_features.csv"
FIRST_YEAR = 2005
LAST_YEAR = 2024

# No tournament level filtering — matches the original notebook behavior

# Match stat columns in the raw data (w_ prefix for winner, l_ prefix for loser)
MATCH_STAT_COLS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]


def csv_paths_for_years(input_dir: Path, first_year: int, last_year: int) -> list[Path]:
    return [input_dir / f"atp_matches_{y}.csv" for y in range(first_year, last_year + 1)]


def build_match_level_sql() -> str:
    """Build the SQL to transform raw match data into match-level format."""

    # Player A/B assignment based on lower ID
    def a_or_b(winner_col: str, loser_col: str, is_a: bool) -> str:
        if is_a:
            return f"CASE WHEN winner_id < loser_id THEN {winner_col} ELSE {loser_col} END"
        else:
            return f"CASE WHEN winner_id < loser_id THEN {loser_col} ELSE {winner_col} END"

    # Build player A/B columns for basic attributes
    basic_attrs = [
        ("id", "winner_id", "loser_id"),
        ("name", "winner_name", "loser_name"),
        ("rank", "winner_rank", "loser_rank"),
        ("rank_points", "winner_rank_points", "loser_rank_points"),
        ("age", "winner_age", "loser_age"),
        ("ht", "winner_ht", "loser_ht"),
        ("hand", "winner_hand", "loser_hand"),
        ("seed", "winner_seed", "loser_seed"),
    ]

    select_parts = [
        "tourney_id",
        "tourney_name",
        "tourney_date",
        "surface",
        "tourney_level",
        "round",
        "draw_size",
        "best_of",
        "minutes",
        "score",
    ]

    # Player A and B basic attributes
    for attr, w_col, l_col in basic_attrs:
        select_parts.append(f"{a_or_b(w_col, l_col, True)} AS player_A_{attr}")
        select_parts.append(f"{a_or_b(w_col, l_col, False)} AS player_B_{attr}")

    # Player A and B match stats
    for stat in MATCH_STAT_COLS:
        select_parts.append(f"{a_or_b(f'w_{stat}', f'l_{stat}', True)} AS player_A_{stat}")
        select_parts.append(f"{a_or_b(f'w_{stat}', f'l_{stat}', False)} AS player_B_{stat}")

    # Delta features for baseline attributes
    baseline_deltas = ["rank", "rank_points", "age", "ht"]
    for attr in baseline_deltas:
        a_expr = a_or_b(f"winner_{attr}", f"loser_{attr}", True)
        b_expr = a_or_b(f"winner_{attr}", f"loser_{attr}", False)
        select_parts.append(f"({a_expr}) - ({b_expr}) AS delta_{attr}")

    # Delta features for match stats
    for stat in MATCH_STAT_COLS:
        a_expr = a_or_b(f"w_{stat}", f"l_{stat}", True)
        b_expr = a_or_b(f"w_{stat}", f"l_{stat}", False)
        select_parts.append(f"({a_expr}) - ({b_expr}) AS delta_{stat}")

    # Missing value indicators for baseline deltas
    for attr in baseline_deltas:
        a_expr = a_or_b(f"winner_{attr}", f"loser_{attr}", True)
        b_expr = a_or_b(f"winner_{attr}", f"loser_{attr}", False)
        select_parts.append(
            f"CASE WHEN ({a_expr}) IS NULL OR ({b_expr}) IS NULL THEN 1 ELSE 0 END AS delta_{attr}_missing"
        )

    # Missing value indicators for match stats
    for stat in MATCH_STAT_COLS:
        a_expr = a_or_b(f"w_{stat}", f"l_{stat}", True)
        b_expr = a_or_b(f"w_{stat}", f"l_{stat}", False)
        select_parts.append(
            f"CASE WHEN ({a_expr}) IS NULL OR ({b_expr}) IS NULL THEN 1 ELSE 0 END AS delta_{stat}_missing"
        )

    # Target: 1 if Player A won
    select_parts.append("CASE WHEN winner_id < loser_id THEN 1 ELSE 0 END AS target")

    # Year extracted for easy filtering
    select_parts.append("CAST(tourney_date / 10000 AS INTEGER) AS year")

    # Row number for unique join key
    select_parts.append("ROW_NUMBER() OVER (ORDER BY tourney_date, tourney_id, match_num) AS match_row_id")

    select_clause = ",\n    ".join(select_parts)

    return f"""
  SELECT
    {select_clause}
  FROM raw
  WHERE tourney_date >= {FIRST_YEAR}0101
    AND winner_id IS NOT NULL
    AND loser_id IS NOT NULL
  ORDER BY tourney_date, tourney_id
"""


def build_rolling_win_pct_sql() -> str:
    """Build SQL to compute rolling 52-week, surface-specific 52-week, and 10-week win percentages.

    Uses proper DATE arithmetic (INTERVAL) instead of integer subtraction on YYYYMMDD.
    """
    return """
-- Step 1: Unpivot matches into a player-level history table with proper dates
CREATE TABLE player_matches AS
SELECT
    STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')::DATE AS match_date,
    surface,
    player_A_id AS player_id,
    target AS won
FROM matches
UNION ALL
SELECT
    STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')::DATE AS match_date,
    surface,
    player_B_id AS player_id,
    1 - target AS won
FROM matches;

-- Step 2: Add proper date column to matches for window comparisons
ALTER TABLE matches ADD COLUMN match_date DATE;
UPDATE matches SET match_date = STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')::DATE;

-- Step 3: Compute rolling win percentages using proper date intervals
CREATE TABLE win_pct_52w AS
SELECT
    m.match_row_id,
    -- Player A 52-week (365 days)
    (SELECT AVG(pm.won) FROM player_matches pm
     WHERE pm.player_id = m.player_A_id
       AND pm.match_date < m.match_date
       AND pm.match_date >= m.match_date - INTERVAL '365 days'
    ) AS player_A_win_pct_52w,
    (SELECT COUNT(*) FROM player_matches pm
     WHERE pm.player_id = m.player_A_id
       AND pm.match_date < m.match_date
       AND pm.match_date >= m.match_date - INTERVAL '365 days'
    ) AS player_A_matches_52w,
    -- Player B 52-week
    (SELECT AVG(pm.won) FROM player_matches pm
     WHERE pm.player_id = m.player_B_id
       AND pm.match_date < m.match_date
       AND pm.match_date >= m.match_date - INTERVAL '365 days'
    ) AS player_B_win_pct_52w,
    (SELECT COUNT(*) FROM player_matches pm
     WHERE pm.player_id = m.player_B_id
       AND pm.match_date < m.match_date
       AND pm.match_date >= m.match_date - INTERVAL '365 days'
    ) AS player_B_matches_52w,
    -- Player A surface-specific 52-week
    (SELECT AVG(pm.won) FROM player_matches pm
     WHERE pm.player_id = m.player_A_id
       AND pm.surface = m.surface
       AND pm.match_date < m.match_date
       AND pm.match_date >= m.match_date - INTERVAL '365 days'
    ) AS player_A_win_pct_52w_surface,
    (SELECT COUNT(*) FROM player_matches pm
     WHERE pm.player_id = m.player_A_id
       AND pm.surface = m.surface
       AND pm.match_date < m.match_date
       AND pm.match_date >= m.match_date - INTERVAL '365 days'
    ) AS player_A_matches_52w_surface,
    -- Player B surface-specific 52-week
    (SELECT AVG(pm.won) FROM player_matches pm
     WHERE pm.player_id = m.player_B_id
       AND pm.surface = m.surface
       AND pm.match_date < m.match_date
       AND pm.match_date >= m.match_date - INTERVAL '365 days'
    ) AS player_B_win_pct_52w_surface,
    (SELECT COUNT(*) FROM player_matches pm
     WHERE pm.player_id = m.player_B_id
       AND pm.surface = m.surface
       AND pm.match_date < m.match_date
       AND pm.match_date >= m.match_date - INTERVAL '365 days'
    ) AS player_B_matches_52w_surface,
    -- Player A 10-week (70 days)
    (SELECT AVG(pm.won) FROM player_matches pm
     WHERE pm.player_id = m.player_A_id
       AND pm.match_date < m.match_date
       AND pm.match_date >= m.match_date - INTERVAL '70 days'
    ) AS player_A_win_pct_10w,
    (SELECT COUNT(*) FROM player_matches pm
     WHERE pm.player_id = m.player_A_id
       AND pm.match_date < m.match_date
       AND pm.match_date >= m.match_date - INTERVAL '70 days'
    ) AS player_A_matches_10w,
    -- Player B 10-week
    (SELECT AVG(pm.won) FROM player_matches pm
     WHERE pm.player_id = m.player_B_id
       AND pm.match_date < m.match_date
       AND pm.match_date >= m.match_date - INTERVAL '70 days'
    ) AS player_B_win_pct_10w,
    (SELECT COUNT(*) FROM player_matches pm
     WHERE pm.player_id = m.player_B_id
       AND pm.match_date < m.match_date
       AND pm.match_date >= m.match_date - INTERVAL '70 days'
    ) AS player_B_matches_10w
FROM matches m;
"""


def build_final_join_sql() -> str:
    """Join match data with rolling win percentages and export."""
    return """
SELECT
    m.*,
    -- 52-week rolling win %
    w.player_A_win_pct_52w,
    w.player_B_win_pct_52w,
    COALESCE(w.player_A_win_pct_52w, 0) - COALESCE(w.player_B_win_pct_52w, 0) AS delta_win_pct_52w,
    CASE WHEN w.player_A_matches_52w > 0 THEN 1 ELSE 0 END AS has_history_A,
    CASE WHEN w.player_B_matches_52w > 0 THEN 1 ELSE 0 END AS has_history_B,
    -- Surface-specific 52-week
    w.player_A_win_pct_52w_surface,
    w.player_B_win_pct_52w_surface,
    COALESCE(w.player_A_win_pct_52w_surface, 0) - COALESCE(w.player_B_win_pct_52w_surface, 0) AS delta_win_pct_52w_surface,
    CASE WHEN w.player_A_matches_52w_surface > 0 THEN 1 ELSE 0 END AS has_surface_history_A,
    CASE WHEN w.player_B_matches_52w_surface > 0 THEN 1 ELSE 0 END AS has_surface_history_B,
    -- 10-week rolling win %
    w.player_A_win_pct_10w,
    w.player_B_win_pct_10w,
    COALESCE(w.player_A_win_pct_10w, 0) - COALESCE(w.player_B_win_pct_10w, 0) AS delta_win_pct_10w,
    CASE WHEN w.player_A_matches_10w > 0 THEN 1 ELSE 0 END AS has_10w_history_A,
    CASE WHEN w.player_B_matches_10w > 0 THEN 1 ELSE 0 END AS has_10w_history_B
FROM matches m
JOIN win_pct_52w w ON m.match_row_id = w.match_row_id
ORDER BY m.tourney_date, m.tourney_id
"""


def main() -> None:
    if not INPUT_DIR.is_dir():
        print(f"Error: input directory not found: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    csv_files = [p for p in csv_paths_for_years(INPUT_DIR, FIRST_YEAR, LAST_YEAR) if p.exists()]
    if not csv_files:
        print(f"Error: no atp_matches_XXXX.csv files found in {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)
    file_list = [p.as_posix() for p in csv_files]
    print(f"Loading {len(file_list)} files (atp_matches_{FIRST_YEAR}.csv .. atp_matches_{LAST_YEAR}.csv) ...")

    conn = duckdb.connect(":memory:")

    # Load raw data
    files_sql = ", ".join("'" + f.replace("'", "''") + "'" for f in file_list)
    conn.execute(
        f"CREATE TABLE raw AS SELECT * FROM read_csv_auto([{files_sql}], union_by_name = true)"
    )
    raw_count = conn.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    print(f"Loaded {raw_count:,} raw match records")

    # Create match-level table with all features
    match_sql = build_match_level_sql()
    conn.execute(f"CREATE TABLE matches AS {match_sql}")
    match_count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    print(f"Created {match_count:,} match-level records (after filtering)")

    # Compute rolling win percentages
    print("Computing rolling win percentages (52-week, surface, 10-week)...")
    rolling_sql = build_rolling_win_pct_sql()
    for statement in rolling_sql.split(";"):
        # Strip comments and whitespace
        lines = [line for line in statement.strip().splitlines() if not line.strip().startswith("--")]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            conn.execute(cleaned)
    print("Rolling win percentages computed")

    # Final join and export
    final_sql = build_final_join_sql()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(f"COPY ({final_sql}) TO '{OUTPUT_PATH.as_posix()}' WITH (HEADER, DELIMITER ',')")
    conn.close()

    n = sum(1 for _ in open(OUTPUT_PATH)) - 1
    print(f"Wrote {n:,} rows to {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
