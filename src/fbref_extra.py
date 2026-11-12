"""A widened FBref reader.

soccerdata 1.9.0 only exposes five player-season stat types - `standard`,
`shooting`, `playing_time`, `keeper` and `misc`. That leaves out `passing`,
`passing_types`, `gca`, `defense` and `possession`, which between them hold
almost everything that describes *how* a player plays rather than how much he
scores: progressive passes, tackles by third, take-ons, touches by zone,
shot-creating actions.

Verified against the installed version:

    >>> import soccerdata.fbref as f, inspect
    >>> "passing" in inspect.getsource(f.FBref.read_player_season_stats)
    False

This subclass re-implements the one method with a wider allow-list. It is a
faithful copy of the upstream body - same caching, same rate limiting, same
column handling - so it inherits `self.get()` and benefits from soccerdata's
throttling rather than hitting FBref directly.

If a future soccerdata release adds these pages natively, delete this module
and set `USE_EXTRA_READER = False` in fetch_fbref.py.
"""

from __future__ import annotations

import logging

import pandas as pd
import soccerdata as sd
from lxml import etree, html

# Internals reused from soccerdata. Confirmed present in 1.9.0; the import
# fails loudly on a version where they have moved, which is the behaviour we
# want - a silent fallback would produce a half-empty feature set.
from soccerdata.fbref import (
        BIG_FIVE_DICT,
        FBREF_API,
        _concat,
        _fix_nation_col,
        _parse_table,
)
from soccerdata._common import standardize_colnames
from soccerdata._config import TEAMNAME_REPLACEMENTS

logger = logging.getLogger(__name__)

# stat_type -> FBref URL path segment. Most are identical; these are the ones
# where the page slug differs from the table suffix.
# --- upstream data fix -----------------------------------------------------
# FBref renamed the German top flight's display name to plain "Bundesliga",
# but soccerdata 1.9.0's BIG_FIVE_DICT still keys it as "Fussball-Bundesliga"
# (with an eszett). On the Big-5-combined pages the lookup therefore misses and
# EVERY German player comes back with league = NaN - silently, because a failed
# .map() produces NaN rather than raising.
#
# Patched in place rather than copied, because the inherited
# read_team_season_stats() reads this global directly, and team possession has
# to join to player rows on (league, season, team) for the possession
# adjustment in features.py to work for Bundesliga sides.
#
# setdefault is deliberate: additive, so the original key keeps working if
# FBref ever reverts. Re-check when bumping soccerdata.
BIG_FIVE_DICT.setdefault("Bundesliga", "GER-Bundesliga")

PAGE_OVERRIDES: dict[str, str] = {
        "standard": "stats",
        "playing_time": "playingtime",
        "keeper": "keepers",
        "keeper_adv": "keepersadv",
}

SUPPORTED_STAT_TYPES: tuple[str, ...] = (
        "standard",
        "shooting",
        "passing",
        "passing_types",
        "gca",
        "defense",
        "possession",
        "playing_time",
        "misc",
        "keeper",
        "keeper_adv",
)


class FBrefWide(sd.FBref):
        """FBref reader that can also fetch passing, defence and possession pages."""

        @classmethod
        def _all_leagues(cls) -> dict[str, str]:
                """Delegate the league table to FBref.

                soccerdata builds this with

                    {k: v[cls.__name__] for k, v in LEAGUE_DICT.items()
                     if cls.__name__ in v}

                so it keys off the CLASS NAME. Subclassing therefore breaks it:
                nothing is registered under "FBrefWide", and every league except
                the Big 5 combined entry silently disappears, giving

                    ValueError: Invalid league 'ENG-Premier League'.
                    Valid leagues are: ['Big 5 European Leagues Combined']

                Worse, it is order-dependent. The result is cached on the class
                as `_all_leagues_dict`, and `hasattr` walks the MRO - so if
                anything touches `sd.FBref` first in the same process, the
                subclass inherits a correct table and the bug vanishes.
                """
                return sd.FBref._all_leagues()

        def read_player_season_stats(self, stat_type: str = "standard") -> pd.DataFrame:
                if stat_type not in SUPPORTED_STAT_TYPES:
                        raise TypeError(
                                f"Invalid stat_type {stat_type!r}; expected one of {SUPPORTED_STAT_TYPES}"
                        )

                page = PAGE_OVERRIDES.get(stat_type, stat_type)
                filemask = "players_{}_{}_{}.html"

                seasons = self.read_seasons()
                frames: list[pd.DataFrame] = []

                for (lkey, skey), season in seasons.iterrows():
                        big_five = lkey == "Big 5 European Leagues Combined"
                        filepath = self.data_dir / filemask.format(lkey, skey, stat_type)
                        url = (
                                FBREF_API
                                + "/".join(season.url.split("/")[:-1])
                                + f"/{page}"
                                + ("/players/" if big_five else "/")
                                + season.url.split("/")[-1]
                        )

                        reader = self.get(url, filepath)
                        tree = html.parse(reader)

                        # Strip the competition-level icons, which otherwise end up inside
                        # the parsed cell text.
                        for elem in tree.xpath("//td[@data-stat='comp_level']//span"):
                                elem.getparent().remove(elem)

                        if big_five:
                                (html_table,) = tree.xpath(f"//table[@id='stats_{stat_type}']")
                                df_table = _parse_table(html_table)
                                competitions = df_table.xs("Comp", axis=1, level=1).squeeze()
                                mapped = competitions.map(BIG_FIVE_DICT)

                                # Canary. A failed .map() yields NaN rather than
                                # raising, so an FBref rename would otherwise drop a
                                # whole league without a word - which is exactly how
                                # the "Bundesliga" case above went unnoticed.
                                unmapped = sorted(set(competitions[mapped.isna()].dropna()))
                                if unmapped:
                                        logger.error(
                                                "Unmapped FBref competition names %s in %s - those players "
                                                "will have league = NaN. Add them to BIG_FIVE_DICT in "
                                                "src/fbref_extra.py.",
                                                unmapped, stat_type,
                                        )

                                df_table[("Unnamed: league", "league")] = mapped
                                df_table[("Unnamed: season", "season")] = skey
                                df_table.drop("Comp", axis=1, level=1, inplace=True)
                        else:
                                # Single-league pages wrap the table in an HTML comment.
                                comments = tree.xpath(
                                        f"//comment()[contains(.,'div_stats_{stat_type}')]"
                                )
                                if not comments:
                                        logger.warning(
                                                "No %s table found for %s %s - skipping", stat_type, lkey, skey
                                        )
                                        continue
                                parser = etree.HTMLParser(recover=True)
                                (html_table,) = etree.fromstring(comments[0].text, parser).xpath(
                                        f"//table[contains(@id, 'stats_{stat_type}')]"
                                )
                                df_table = _parse_table(html_table)
                                df_table[("Unnamed: league", "league")] = lkey
                                df_table[("Unnamed: season", "season")] = skey

                        df_table = _fix_nation_col(df_table)
                        frames.append(df_table)

                if not frames:
                        raise RuntimeError(
                                f"No {stat_type} tables parsed. Check LEAGUES and SEASONS in src/config.py."
                        )

                df = _concat(frames, key=["league", "season"])
                df = df[df.Player != "Player"]  # FBref repeats the header mid-table

                for col in ("Matches", "Rk"):
                        if col in df.columns.get_level_values(0):
                                df = df.drop(col, axis=1, level=0)

                return (
                        df.rename(columns={"Squad": "team"})
                        .replace({"team": TEAMNAME_REPLACEMENTS})
                        .pipe(standardize_colnames, cols=["Player", "Nation", "Pos", "Age", "Born"])
                        .set_index(["league", "season", "team", "player"])
                        .sort_index()
                )
