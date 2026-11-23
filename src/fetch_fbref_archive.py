"""Stage 1a-bis - recover FBref's advanced stats from the Wayback Machine.

On 2026-01-20 FBref lost its Opta licence and stopped serving every advanced
player stat: passing, progression, chance creation, defensive actions, xG.
The live pages still list each player, but those cells are now empty, which
leaves the similarity model with almost nothing to work with.

The Internet Archive saved copies of the same pages from before that date.
This stage downloads those copies and fills the empty cells of the live table
with the archived values. The live table stays the backbone: it supplies the
player list, the basic stats that never went away, and team possession.

**Pinned snapshots.** Every archived page used is listed, with its capture
timestamp, in data/fbref_archive_snapshots.csv. A Wayback capture never
changes, so anyone cloning the repo downloads exactly the same bytes. To
rebuild that list from the archive's index, run

    python -m src.fetch_fbref_archive --discover

which searches for captures taken after each season ended and before the
removal date, downloads them, keeps only complete ones, and rewrites the CSV.
Add `--season 2223` (repeatable) to search only that season and keep the
existing pins for the others.

**Archive-only seasons.** A season in config.SEASONS that the live table
lacks (2022-23: soccerdata's browser driver can be blocked by Windows
application control, and the archive has the whole season anyway) is built
from the archive alone. Its archived standard page becomes the player list -
the "backbone" that stands in for the live rows - and an archived Big 5 squad
page supplies team possession. Everything downstream (alignment, renamed
players, the overlay, the row filter) then treats it like any other season.

**Parsing.** The archived HTML is FBref's own page, so it goes through the
same table parser soccerdata uses for the live site, then through
fetch_fbref.flatten_columns. That way archived and live columns get identical
names and the feature patterns in config.py match both.

**Licence.** The underlying numbers are Opta's. Downloaded pages stay under
data/raw/ and data/interim/, both git-ignored. Only this code and the pinned
snapshot list belong in the repo.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import time
import urllib.parse

import pandas as pd
import requests
from lxml import etree, html

from . import config, paths
from .fbref_extra import BIG_FIVE_DICT  # also registers the renamed "Bundesliga"
from .fetch_fbref import IDENTITY_STATS, flatten_columns, parquet_safe

# soccerdata's private helpers: the exact parser it applies to live FBref
# pages. Reusing them is what makes archived column names match the live ones.
from soccerdata._config import TEAMNAME_REPLACEMENTS
from soccerdata._common import standardize_colnames
from soccerdata.fbref import _concat, _fix_nation_col, _parse_table

logger = logging.getLogger(__name__)

#: FBref removed the Opta stats on 2026-01-20. Captures after this are empty.
REMOVAL_CUTOFF = "20260119"

WAYBACK_RAW = "https://web.archive.org/web/{timestamp}id_/{url}"
CDX = "https://web.archive.org/cdx/search/cdx"

#: Seconds between archive requests. The Wayback Machine is a free public
#: service; there is no reason to hit it harder than this.
REQUEST_GAP = 2.0

#: URL slug of each stat type. FBref calls the standard table "stats".
PAGES = {
        "standard": "stats",
        "shooting": "shooting",
        "passing": "passing",
        "passing_types": "passing_types",
        "gca": "gca",
        "defense": "defense",
        "possession": "possession",
        "misc": "misc",
}

#: FBref competition id and URL slug per league, for the per-league pages.
LEAGUE_PAGES = {
        "ENG-Premier League": (9, "Premier-League"),
        "ESP-La Liga": (12, "La-Liga"),
        "ITA-Serie A": (11, "Serie-A"),
        "GER-Bundesliga": (20, "Bundesliga"),
        "FRA-Ligue 1": (13, "Ligue-1"),
}

#: Season key -> (FBref's season label, last matchday of the Big 5, the date
#: after which a season-less URL may already show the next season).
SEASON_WINDOWS = {
        "2223": ("2022-2023", "20230605", "20230815"),
        "2324": ("2023-2024", "20240603", "20240815"),
        "2425": ("2024-2025", "20250602", "20250815"),
}

#: A stat column per page that was always filled before the removal. If it is
#: mostly empty, the capture was taken after the removal or is truncated.
SENTINELS = {
        "standard": "standard__playing_time__min",
        "shooting": "shooting__standard__sh",
        "passing": "passing__total__cmp",
        "passing_types": "passing_types__pass_types__live",
        "gca": "gca__sca__sca",
        "defense": "defense__tackles__tkl",
        "possession": "possession__touches__touches",
        "misc": "misc__performance__crdy",
}

#: Minimum share of the live table's players, per league-season, a capture
#: must contain to count as complete. Truncated captures fall far short.
MIN_ROW_SHARE = 0.9

SNAPSHOT_COLUMNS = ["season", "stat_type", "scope", "timestamp", "url", "rows"]

#: Pinned-row stat type for the Big 5 squad possession page (team possession,
#: needed only for archive-only seasons).
TEAM_STAT = "team_possession"

#: Big 5 clubs per season. A squad page with fewer than 90% of these is truncated.
BIG5_TEAMS = 96


# --- downloading -------------------------------------------------------------

def _session() -> requests.Session:
        session = requests.Session()
        session.headers["User-Agent"] = config.USER_AGENT
        return session


def raw_path(season: str, stat_type: str, scope: str, timestamp: str):
        slug = scope.replace(" ", "_")
        return paths.FBREF_ARCHIVE_DIR / f"{season}_{stat_type}_{slug}_{timestamp}.html"


def download(session: requests.Session, row: dict) -> str:
        """Return the archived page's HTML, from disk if it was fetched before.

        The `id_` form of a Wayback URL returns the page exactly as captured,
        without the archive's toolbar. Some captures were stored gzipped and
        come back that way without a Content-Encoding header, so the bytes are
        checked for the gzip magic number and decompressed by hand.
        """
        target = raw_path(row["season"], row["stat_type"], row["scope"], str(row["timestamp"]))
        if target.exists():
                return target.read_text(encoding="utf-8")

        url = WAYBACK_RAW.format(timestamp=row["timestamp"], url=row["url"])
        for attempt in range(1, 4):
                try:
                        response = session.get(url, timeout=180)
                        response.raise_for_status()
                        break
                except requests.RequestException as exc:
                        if attempt == 3:
                                raise
                        logger.warning("Archive request failed (%s); retrying.", exc)
                        time.sleep(10 * attempt)
        time.sleep(REQUEST_GAP)

        body = response.content
        if body[:2] == b"\x1f\x8b":
                body = gzip.decompress(body)
        text = body.decode("utf-8", errors="replace")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return text


# --- parsing -----------------------------------------------------------------

def parse_page(text: str, stat_type: str, scope: str, season: str) -> pd.DataFrame:
        """One archived page to a raw soccerdata-style frame (before _concat).

        Mirrors the two branches of soccerdata's read_player_season_stats: the
        Big 5 page carries the player table directly, while a single league's
        page hides it inside an HTML comment.
        """
        tree = html.parse(io.StringIO(text))
        for elem in tree.xpath("//td[@data-stat='comp_level']//span"):
                elem.getparent().remove(elem)

        if scope == "big5":
                tables = tree.xpath(f"//table[@id='stats_{stat_type}']")
                if not tables:
                        raise ValueError("no player table on the page - capture probably truncated")
                frame = _parse_table(tables[0])
                frame[("Unnamed: league", "league")] = (
                        frame.xs("Comp", axis=1, level=1).squeeze().map(BIG_FIVE_DICT)
                )
                frame.drop("Comp", axis=1, level=1, inplace=True)
        else:
                comments = tree.xpath(f"//comment()[contains(.,'div_stats_{stat_type}')]")
                if not comments:
                        raise ValueError("no player table on the page - capture probably truncated")
                inner = etree.fromstring(comments[0].text, etree.HTMLParser(recover=True))
                tables = inner.xpath(f"//table[contains(@id, 'stats_{stat_type}')]")
                if not tables:
                        raise ValueError("no player table inside the comment")
                frame = _parse_table(tables[0])
                frame[("Unnamed: league", "league")] = scope

        frame[("Unnamed: season", "season")] = season
        return _fix_nation_col(frame)


def parse_team_page(text: str, season: str) -> pd.DataFrame:
        """The Big 5 squad possession page to (league, season, team) -> possession.

        The column is named `standard__flat__poss`, the name the live squad
        standard table produces, so merge.attach_team_possession finds it.
        """
        tree = html.parse(io.StringIO(text))
        for elem in tree.xpath("//td[@data-stat='comp_level']//span"):
                elem.getparent().remove(elem)
        tables = tree.xpath("//table[@id='stats_teams_possession_for']")
        if not tables:
                raise ValueError("no squad table on the page - capture probably truncated")
        raw = _parse_table(tables[0])
        frame = pd.DataFrame({
                "league": raw.xs("Comp", axis=1, level=1).squeeze().map(BIG_FIVE_DICT),
                "season": season,
                "team": raw.xs("Squad", axis=1, level=1).squeeze().replace(TEAMNAME_REPLACEMENTS),
                "standard__flat__poss": pd.to_numeric(raw.xs("Poss", axis=1, level=1).squeeze(), errors="coerce"),
        })
        frame = frame.dropna(subset=["league", "standard__flat__poss"])
        return frame.set_index(["league", "season", "team"]).sort_index()


def finish(frames: list[pd.DataFrame], stat_type: str) -> pd.DataFrame:
        """Combine raw frames exactly as soccerdata does, then flatten."""
        df = _concat(frames, key=["league", "season"])
        df = df[df.Player != "Player"]
        df = (
                df.drop("Matches", axis=1, level=0, errors="ignore")
                .drop("Rk", axis=1, level=0, errors="ignore")
                .rename(columns={"Squad": "team"})
                .replace({"team": TEAMNAME_REPLACEMENTS})
                .pipe(standardize_colnames, cols=["Player", "Nation", "Pos", "Age", "Born"])
                .set_index(["league", "season", "team", "player"])
                .sort_index()
        )
        df = df[df.index.get_level_values("league").notna()]
        flat = flatten_columns(df, stat_type)
        return flat.loc[~flat.index.duplicated()]


def check_page(frame: pd.DataFrame, stat_type: str, text: str, season_label: str) -> tuple[int, float]:
        """Rows in a parsed page, and the share with the sentinel stat filled."""
        flat = finish([frame], stat_type)
        sentinel = SENTINELS[stat_type]
        if sentinel not in flat.columns:
                return len(flat), 0.0
        filled = pd.to_numeric(flat[sentinel], errors="coerce").notna().mean()
        if season_label not in text[:20000]:
                logger.warning("Page title does not mention %s; treating as wrong season.", season_label)
                return len(flat), 0.0
        return len(flat), float(filled)


# --- aligning to the live table ----------------------------------------------

#: Live columns used to re-identify a player whose display name changed.
LIVE_BORN = "standard__flat__born"
LIVE_NINETIES = "standard__playing_time__90s"
LIVE_MINUTES = "standard__playing_time__min"


def _team_map(archived: pd.MultiIndex, live: pd.MultiIndex) -> dict[tuple, str]:
        """Map each archived (league, season, team) to the live team name.

        FBref has renamed several clubs' short names since the archive was
        taken ("Betis" -> "Real Betis", "Paris S-G" -> "Paris SG"). Within a
        league-season the right live club is the one sharing the most player
        names; at least three shared names are required, so a stray
        coincidence cannot remap a club.
        """
        live_squads: dict[tuple, set[str]] = {}
        for league, season, team, player in live:
                live_squads.setdefault((league, season, team), set()).add(player)

        archived_squads: dict[tuple, set[str]] = {}
        for league, season, team, player in archived:
                archived_squads.setdefault((league, season, team), set()).add(player)

        mapping: dict[tuple, str] = {}
        for (league, season, team), players in archived_squads.items():
                if (league, season, team) in live_squads:
                        mapping[(league, season, team)] = team
                        continue
                overlaps = {
                        live_team: len(players & live_players)
                        for (l, s, live_team), live_players in live_squads.items()
                        if (l, s) == (league, season)
                }
                best = max(overlaps, key=overlaps.get, default=None)
                if best is not None and overlaps[best] >= 3:
                        mapping[(league, season, team)] = best
                else:
                        logger.warning("No live club matches archived %s %s %s", league, season, team)
        return mapping


def align_to_live(flat: pd.DataFrame, live: pd.DataFrame, stat_type: str) -> pd.DataFrame:
        """Relabel archived rows with the live table's (league, season, team, player).

        Clubs are mapped first (see _team_map). A player still unmatched after
        that has usually been renamed ("Andrew Robertson" -> "Andy Robertson"),
        so within the same club they are matched on birth year plus 90s
        played, which is unique within a squad in practice. Anything still
        unmatched is dropped and counted.
        """
        team_of = _team_map(flat.index, live.index)
        keys = flat.index.to_frame(index=False)
        keys["team"] = [
                team_of.get((l, s, t)) for l, s, t in zip(keys["league"], keys["season"], keys["team"])
        ]

        live_keys = set(live.index)
        exact = [tuple(k) in live_keys for k in keys.itertuples(index=False)]

        # Fingerprint unmatched rows by birth year and 90s played, as a string
        # so it can sit in an index level without tuple-indexing surprises.
        def fingerprint(born, nineties):
                b = pd.to_numeric(pd.Series(born), errors="coerce")
                n = pd.to_numeric(pd.Series(nineties), errors="coerce").round(1)
                return [f"{x:.0f}|{y:.1f}" if pd.notna(x) and pd.notna(y) else "" for x, y in zip(b, n)]

        live_frame = live.index.to_frame(index=False)
        live_frame["fp"] = fingerprint(live[LIVE_BORN].to_numpy(), live[LIVE_NINETIES].to_numpy())
        live_fp = live_frame.groupby(["league", "season", "team", "fp"])["player"].agg(list)

        born_col = f"{stat_type}__flat__born"
        nineties_col = f"{stat_type}__flat__90s" if f"{stat_type}__flat__90s" in flat else LIVE_NINETIES
        fps = fingerprint(flat[born_col].to_numpy(), flat[nineties_col].to_numpy())

        players = keys["player"].tolist()
        renamed = 0
        for i, is_exact in enumerate(exact):
                if is_exact or keys.at[i, "team"] is None or not fps[i]:
                        continue
                candidates = live_fp.get((keys.at[i, "league"], keys.at[i, "season"], keys.at[i, "team"], fps[i]), [])
                if len(candidates) == 1:
                        players[i] = candidates[0]
                        renamed += 1
        keys["player"] = players

        out = flat.copy()
        out.index = pd.MultiIndex.from_frame(keys)
        matched = [k in live_keys for k in out.index]
        dropped = len(out) - sum(matched)
        out = out[matched]
        out = out.loc[~out.index.duplicated()]
        logger.info(
                "  %s: %s rows matched live (%s via renamed player), %s dropped",
                stat_type, len(out), renamed, dropped,
        )
        return out


# --- discovery ---------------------------------------------------------------

def _cdx(session: requests.Session, url: str, start: str, end: str) -> list[str]:
        params = {
                "url": url, "from": start, "to": end,
                "filter": "statuscode:200", "output": "json", "fl": "timestamp",
        }
        for attempt in range(1, 4):
                try:
                        response = session.get(CDX, params=params, timeout=90)
                        response.raise_for_status()
                        time.sleep(1)
                        return [row[0] for row in response.json()[1:]]
                except (requests.RequestException, json.JSONDecodeError):
                        time.sleep(5 * attempt)
        logger.warning("Archive index lookup failed for %s", url)
        return []


def _candidates(session: requests.Session, season: str, stat_type: str) -> dict[str, list[tuple[str, str]]]:
        """Every capture per scope, newest first, as (timestamp, url)."""
        label, season_end, window_end = SEASON_WINDOWS[season]
        page = PAGES[stat_type]
        urls: dict[str, list[tuple[str, str, str]]] = {
                "big5": [
                        (f"fbref.com/en/comps/Big5/{label}/{page}/players/{label}-Big-5-European-Leagues-Stats", season_end, REMOVAL_CUTOFF),
                        (f"fbref.com/en/comps/Big5/{page}/players/Big-5-European-Leagues-Stats", season_end, window_end),
                ],
        }
        for league, (comp_id, slug) in LEAGUE_PAGES.items():
                urls[league] = [
                        (f"fbref.com/en/comps/{comp_id}/{label}/{page}/{label}-{slug}-Stats", season_end, REMOVAL_CUTOFF),
                        (f"fbref.com/en/comps/{comp_id}/{page}/{slug}-Stats", season_end, window_end),
                ]

        found: dict[str, list[tuple[str, str]]] = {}
        for scope, specs in urls.items():
                captures = [
                        (timestamp, "https://" + url)
                        for url, start, end in specs
                        for timestamp in _cdx(session, url, start, end)
                ]
                found[scope] = sorted(captures, reverse=True)
        return found


def _live_counts(seasons=None) -> pd.Series:
        """Players per (league, season) who played, in the live table.

        This is the completeness bar for an archived page. Only players with
        minutes count: the live Premier League 2023-24 table also lists 126
        squad members who never played, which the archived pages never did.

        A season with no live rows (an archive-only season, before its
        backbone exists) gets each league's median over the other seasons -
        squad sizes barely move from year to year, and the bar only has to
        tell a complete page from a truncated one.
        """
        live = live_table()[[LIVE_MINUTES]]
        played = pd.to_numeric(live[LIVE_MINUTES], errors="coerce").gt(0)
        counts = played.groupby(level=["league", "season"]).sum()
        typical = counts.groupby(level="league").median()
        have = set(counts.index.get_level_values("season"))
        extra = {
                (league, season): int(n)
                for season in (seasons or []) if season not in have
                for league, n in typical.items()
        }
        if extra:
                counts = pd.concat([counts, pd.Series(extra)]).rename_axis(["league", "season"])
        return counts


def _first_complete(session, season, stat_type, scope, captures, expected) -> dict | None:
        """Newest capture of one scope that holds a complete, pre-removal table."""
        label = SEASON_WINDOWS[season][0]
        for timestamp, url in captures:
                row = {"season": season, "stat_type": stat_type, "scope": scope,
                       "timestamp": timestamp, "url": url}
                try:
                        text = download(session, row)
                        frame = parse_page(text, stat_type, scope, season)
                        rows, filled = check_page(frame, stat_type, text, label)
                except Exception as exc:  # noqa: BLE001 - a bad capture just means try the next
                        logger.info("    %s %s: unusable (%s)", scope, timestamp, exc)
                        continue
                if rows >= MIN_ROW_SHARE * expected and filled >= 0.9:
                        logger.info("    %s %s: %s rows, complete", scope, timestamp, rows)
                        return {**row, "rows": rows}
                logger.info("    %s %s: %s/%s rows, %.0f%% filled - skipped", scope, timestamp, rows, expected, 100 * filled)
        return None


def _first_team_page(session: requests.Session, season: str) -> dict | None:
        """Newest complete capture of the Big 5 squad possession page."""
        label, season_end, window_end = SEASON_WINDOWS[season]
        specs = [
                (f"fbref.com/en/comps/Big5/{label}/possession/squads/{label}-Big-5-European-Leagues-Stats", season_end, REMOVAL_CUTOFF),
                ("fbref.com/en/comps/Big5/possession/squads/Big-5-European-Leagues-Stats", season_end, window_end),
        ]
        captures = sorted(
                ((ts, "https://" + url) for url, start, end in specs for ts in _cdx(session, url, start, end)),
                reverse=True,
        )
        logger.info("%s %s", season, TEAM_STAT)
        for timestamp, url in captures:
                row = {"season": season, "stat_type": TEAM_STAT, "scope": "big5", "timestamp": timestamp, "url": url}
                try:
                        text = download(session, row)
                        teams = parse_team_page(text, season)
                except Exception as exc:  # noqa: BLE001 - a bad capture just means try the next
                        logger.info("    big5 %s: unusable (%s)", timestamp, exc)
                        continue
                if label in text[:20000] and len(teams) >= MIN_ROW_SHARE * BIG5_TEAMS:
                        logger.info("    big5 %s: %s teams, complete", timestamp, len(teams))
                        return {**row, "rows": len(teams)}
                logger.info("    big5 %s: %s teams - skipped", timestamp, len(teams))
        return None


def discover(seasons: list[str] | None = None) -> pd.DataFrame:
        """Search the archive and pin one complete capture per page.

        For each season and stat type, a complete Big 5 page is preferred,
        because it covers all five leagues in one request. Otherwise the
        newest complete capture of each league's own page is used, so one
        missing league does not throw away the other four.

        `seasons` limits the search to those season keys and keeps the
        existing pins for every other season, so adding a season does not
        re-search (and possibly re-pin) the ones already settled.
        """
        unknown = sorted(set(seasons or []) - set(SEASON_WINDOWS))
        if unknown:
                raise SystemExit(f"No archive window for {unknown}; add it to SEASON_WINDOWS.")
        counts = _live_counts(list(seasons or SEASON_WINDOWS))
        session = _session()
        pinned: list[dict] = []
        if seasons and paths.FBREF_ARCHIVE_SNAPSHOTS.exists():
                kept = pd.read_csv(paths.FBREF_ARCHIVE_SNAPSHOTS, dtype=str)
                pinned = kept[~kept["season"].isin(seasons)].to_dict("records")

        archive_only = archive_only_seasons()
        for season in seasons or SEASON_WINDOWS:
                if season in archive_only:
                        choice = _first_team_page(session, season)
                        if choice:
                                pinned.append(choice)
                        else:
                                logger.warning("%s: no complete squad page; possession adjustment will be skipped", season)
                big5_expected = int(counts.xs(season, level="season").sum())
                for stat_type in PAGES:
                        logger.info("%s %s", season, stat_type)
                        found = _candidates(session, season, stat_type)

                        choice = _first_complete(session, season, stat_type, "big5", found["big5"], big5_expected)
                        if choice:
                                pinned.append(choice)
                                continue

                        for league in LEAGUE_PAGES:
                                expected = int(counts.get((league, season), 0))
                                choice = _first_complete(session, season, stat_type, league, found[league], expected)
                                if choice:
                                        pinned.append(choice)
                                else:
                                        logger.warning("    %s: no complete capture", league)

        snapshots = pd.DataFrame(pinned, columns=SNAPSHOT_COLUMNS)
        snapshots = snapshots.sort_values("season", kind="stable")
        snapshots.to_csv(paths.FBREF_ARCHIVE_SNAPSHOTS, index=False)
        logger.info("Pinned %s captures in %s", len(snapshots), paths.FBREF_ARCHIVE_SNAPSHOTS)
        return snapshots


# --- the stage ---------------------------------------------------------------

def fetch_archived_stats() -> pd.DataFrame:
        """Download and parse every pinned capture into one wide table."""
        if not paths.FBREF_ARCHIVE_SNAPSHOTS.exists():
                raise SystemExit(
                        f"{paths.FBREF_ARCHIVE_SNAPSHOTS} is missing. "
                        "Run `python -m src.fetch_fbref_archive --discover` to build it."
                )
        if not paths.FBREF_WIDE.exists():
                raise SystemExit("Run `python run_pipeline.py --stage fetch-fbref` first.")
        snapshots = pd.read_csv(paths.FBREF_ARCHIVE_SNAPSHOTS, dtype=str)
        session = _session()

        by_stat: dict[str, list[pd.DataFrame]] = {}
        teams: list[pd.DataFrame] = []
        for row in snapshots.to_dict("records"):
                logger.info("Archive: %s %s (%s, %s)", row["season"], row["stat_type"], row["scope"], row["timestamp"])
                text = download(session, row)
                if row["stat_type"] == TEAM_STAT:
                        teams.append(parse_team_page(text, row["season"]))
                        continue
                frame = parse_page(text, row["stat_type"], row["scope"], row["season"])
                by_stat.setdefault(row["stat_type"], []).append(frame)

        write_backbone(by_stat.get("standard", []))
        if teams:
                pd.concat(teams).to_parquet(paths.FBREF_ARCHIVE_TEAMS)
        else:
                paths.FBREF_ARCHIVE_TEAMS.unlink(missing_ok=True)

        live = live_table()
        flats: list[pd.DataFrame] = []
        for stat_type, frames in by_stat.items():
                flat = align_to_live(finish(frames, stat_type), live, stat_type)
                # Identity comes from the live table; keep only the stats.
                drop = [c for c in flat.columns if c.rsplit("__", 1)[-1] in IDENTITY_STATS]
                flats.append(flat.drop(columns=drop))

        archived = pd.concat(flats, axis=1)
        return archived.loc[:, ~archived.columns.duplicated()]


def archive_only_seasons() -> list[str]:
        """Configured seasons with an archive window but no rows in the live table."""
        if not paths.FBREF_WIDE.exists():
                raise SystemExit("Run `python run_pipeline.py --stage fetch-fbref` first.")
        live_seasons = set(
                pd.read_parquet(paths.FBREF_WIDE, columns=[LIVE_MINUTES]).index.get_level_values("season")
        )
        return [s for s in config.SEASONS if s in SEASON_WINDOWS and s not in live_seasons]


def write_backbone(standard_frames: list[pd.DataFrame]) -> None:
        """Save archived standard-page rows for archive-only seasons.

        These rows keep their identity columns (nation, position, age, born),
        because for these seasons there is no live row to take them from.
        """
        seasons = archive_only_seasons()
        frames = [f for f in standard_frames if f[("Unnamed: season", "season")].isin(seasons).any()]
        if not seasons or not frames:
                paths.FBREF_ARCHIVE_BACKBONE.unlink(missing_ok=True)
                if seasons:
                        logger.warning("No archived standard page for %s; those seasons stay empty.", seasons)
                return
        backbone = finish(frames, "standard")
        backbone = backbone[backbone.index.get_level_values("season").isin(seasons)]
        parquet_safe(backbone).to_parquet(paths.FBREF_ARCHIVE_BACKBONE)
        logger.info("Archive-only seasons %s: %s player-seasons from the archived standard page", seasons, len(backbone))


def live_table() -> pd.DataFrame:
        """The live FBref table plus the backbone rows of archive-only seasons."""
        if not paths.FBREF_WIDE.exists():
                raise SystemExit("Run `python run_pipeline.py --stage fetch-fbref` first.")
        live = pd.read_parquet(paths.FBREF_WIDE)
        if paths.FBREF_ARCHIVE_BACKBONE.exists():
                backbone = pd.read_parquet(paths.FBREF_ARCHIVE_BACKBONE)
                live = pd.concat([live, backbone[~backbone.index.isin(live.index)]]).sort_index()
                # Archived ages parse as numbers, live ones as "29-034" text; a
                # column mixing both cannot be written to parquet downstream.
                live = parquet_safe(live)
        return live


def load_team_table() -> pd.DataFrame:
        """Live team stats plus archived possession for archive-only seasons."""
        teams = pd.read_parquet(paths.FBREF_TEAM) if paths.FBREF_TEAM.exists() else None
        if paths.FBREF_ARCHIVE_TEAMS.exists():
                archived = pd.read_parquet(paths.FBREF_ARCHIVE_TEAMS)
                if teams is not None:
                        archived = archived[~archived.index.isin(teams.index)]
                        teams = pd.concat([teams, archived])
                else:
                        teams = archived
        if teams is None:
                raise FileNotFoundError(paths.FBREF_TEAM)
        return teams


def overlay(live: pd.DataFrame, archived: pd.DataFrame) -> pd.DataFrame:
        """Fill the live table's empty cells with archived values.

        The live table decides which player-seasons exist; archived rows that
        match none of them are dropped and counted. Where both have a value,
        the archived one wins - they come from the same Opta feed, and the
        archived copy is the one taken before anything was removed.
        """
        aligned = archived.reindex(live.index)
        unmatched = len(archived.index.difference(live.index))
        if unmatched:
                logger.info("%s archived rows match no live player-season and are ignored", unmatched)

        combined = aligned.combine_first(live)
        extra = [c for c in combined.columns if c not in live.columns]
        return combined[list(live.columns) + extra]


def fill_gaps(table: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
        """Fill only the empty cells of `table` from `extra` (see fetch_kaggle).

        Unlike overlay(), nothing already present is overwritten: the live
        table and the archive stay authoritative, and the Kaggle copy only
        supplies what neither has.
        """
        aligned = extra.reindex(table.index)
        out = table.copy()
        filled = 0
        for column in aligned.columns:
                if column in out.columns:
                        before = out[column].isna()
                        out[column] = out[column].where(~before, aligned[column])
                        filled += int((before & aligned[column].notna()).sum())
                else:
                        out[column] = aligned[column]
                        filled += int(aligned[column].notna().sum())
        logger.info("Filled %s empty cells from %s", filled, paths.FBREF_KAGGLE.name)
        return out


def coverage_report(table: pd.DataFrame) -> pd.DataFrame:
        """Share of player-seasons with each sentinel stat, per league-season."""
        present = [s for s in SENTINELS.values() if s in table.columns]
        filled = table[present].apply(pd.to_numeric, errors="coerce").notna()
        report = filled.groupby(level=["season", "league"]).mean().round(3)
        report.columns = [c.split("__", 1)[0] for c in report.columns]
        return report


def load_player_table() -> pd.DataFrame:
        """The FBref player table merge.py should use: live, plus archive if fetched."""
        live = live_table()
        if not paths.FBREF_ARCHIVE.exists():
                logger.warning(
                        "No archived FBref stats (%s); advanced stats will be empty. "
                        "Run `python run_pipeline.py --stage fetch-fbref-archive`.",
                        paths.FBREF_ARCHIVE.name,
                )
                return live
        combined = overlay(live, pd.read_parquet(paths.FBREF_ARCHIVE))
        if paths.FBREF_KAGGLE.exists():
                combined = fill_gaps(combined, pd.read_parquet(paths.FBREF_KAGGLE))
        if config.REQUIRE_ADVANCED_STATS:
                combined = keep_complete_rows(combined)
        return combined


#: Pages that only exist in the archive. A row needs all of them to be modelled.
ADVANCED_PAGES = ("passing", "passing_types", "gca", "defense", "possession")


def keep_complete_rows(table: pd.DataFrame) -> pd.DataFrame:
        """Drop player-seasons missing any advanced stat page.

        Completeness is judged by each page's sentinel column. Players who
        never played are dropped too, since they have no stats on any page.
        Logs what survives per league-season, so a season that vanished
        entirely (2025-26, with no archived data) is visible in the run log.
        """
        sentinels = [SENTINELS[page] for page in ADVANCED_PAGES]
        missing = [s for s in sentinels if s not in table.columns]
        if missing:
                raise SystemExit(f"Archived table lacks {missing}; rerun fetch-fbref-archive.")

        complete = table[sentinels].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        summary = complete.groupby(level=["league", "season"]).agg(["sum", "size"])
        logger.info(
                "Rows with every advanced stat page, per league-season:\n%s",
                summary.rename(columns={"sum": "kept", "size": "of"}).to_string(),
        )
        logger.info("Keeping %s of %s player-seasons.", int(complete.sum()), len(complete))
        return table[complete]


def run(force: bool = False) -> pd.DataFrame:
        paths.ensure_dirs()
        if paths.FBREF_ARCHIVE.exists() and not force:
                logger.info("%s exists; use --force to rebuild.", paths.FBREF_ARCHIVE)
                return pd.read_parquet(paths.FBREF_ARCHIVE)

        archived = parquet_safe(fetch_archived_stats())
        archived.to_parquet(paths.FBREF_ARCHIVE)
        logger.info("Wrote %s (%s rows, %s columns)", paths.FBREF_ARCHIVE, *archived.shape)

        report = coverage_report(load_player_table())
        report.to_csv(paths.FBREF_ARCHIVE_COVERAGE)
        logger.info("Stat coverage after the overlay:\n%s", report.to_string())
        return archived


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("--force", action="store_true", help="re-parse even if the output exists")
        parser.add_argument("--discover", action="store_true", help="search the archive and rewrite the pinned snapshot list")
        parser.add_argument("--season", action="append", help="with --discover: search only this season key (repeatable)")
        args = parser.parse_args()
        discover(args.season) if args.discover else run(force=args.force)
