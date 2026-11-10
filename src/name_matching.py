"""Join FBref players to Transfermarkt players by name.

This is where the project quietly fails if you let it. "Sergio Reguilon" vs
"Sergio Reguilón"; "Emerson" vs "Emerson Palmieri"; players with three given
names rendered differently by each source. A 70% match rate still looks fine
in a notebook and will have silently removed exactly the recognisable players
a case study needs.

The join runs in four layers, ordered by how much work each does:

1. Normalise - unicode decomposition, transliteration, accent strip, casefold.
2. Block - only compare players whose birth years agree within a year.
   Blocking does more work here than the scorer: it cuts the candidate space
   by orders of magnitude AND prevents most false positives. Without it,
   "Emerson" would score 100 against every Emerson among 50,000 players.
3. Fuzzy match inside each block with rapidfuzz token_set_ratio.
4. Manual overrides - a hand-maintained CSV for the residual, checked into git.

Anything scoring between the reject and accept thresholds is written to
reports/name_match_review.csv for a human rather than guessed at. The match
rate goes to reports/name_match_summary.json; report it in the README, because
an honest 94% says more about the work than a silent 100% ever could.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process

from . import config, paths

logger = logging.getLogger(__name__)

TRANSLITERATE = str.maketrans({
        "Ø": "O", "ø": "o", "Đ": "D", "đ": "d", "Ð": "D", "ð": "d",
        "Ł": "L", "ł": "l", "ß": "ss", "Æ": "AE", "æ": "ae",
        "Œ": "OE", "œ": "oe", "Þ": "TH", "þ": "th", "İ": "I", "ı": "i",
})


def normalise_name(name: object) -> str:
        """Strip a name down to comparable ASCII tokens.

        >>> normalise_name("Sergio Reguilón")
        'sergio reguilon'
        >>> normalise_name("N'Golo Kanté")
        'ngolo kante'
        >>> normalise_name("Pierre-Emerick Aubameyang")
        'pierre emerick aubameyang'

        Returns "" for None / NaN so callers never have to guard.

        Two details carry the whole function:

        Apostrophes are deleted, not spaced. Transfermarkt writes "Ngolo", and
        token_set_ratio compares token *sets*, so splitting the name into
        {n, golo} instead of {ngolo} costs the match. Hyphens, by contrast, do
        become spaces - both sources split those the same way.

        TRANSLITERATE is applied BEFORE the ascii encode. NFKD decomposes marks
        that sit *on* a letter (é, ü, ç) but not letters whose stroke is part
        of the glyph (ø, ł, đ, ß), so those would otherwise be dropped outright
        by .encode("ascii", "ignore"): "Ødegaard" -> "degaard". That costs real
        matches whenever the two sources spell the name differently - "Weiß" vs
        "Weiss" scores 75 without it and 100 with it.
        """

        if pd.isna(name):
                return ""

        name = str(name)

        name = unicodedata.normalize("NFKD", name)
        name = "".join(ch for ch in name if not unicodedata.combining(ch))
        name = name.translate(TRANSLITERATE)
        name = name.encode("ascii", "ignore").decode("ascii")
        name = re.sub(r"[\'’`´]", "", name)
        name = re.sub(r"[^a-zA-Z0-9]+", " ", name)
        name = re.sub(r"\s+", " ", name).strip().casefold()
        return name


def season_end_year(season: object) -> int:
        """Calendar year a soccerdata season key ends in.

        >>> season_end_year("2425"), season_end_year("2024-2025")
        (2025, 2025)

        Accepts soccerdata's compact form ("2425") and the long form
        ("2024-2025"); raises ValueError on anything else.

        The raise matters more than it looks. A season key that silently parsed
        to the wrong year would pick market values twelve months out of date,
        and nothing downstream would flag it - the numbers would just be
        quietly wrong.
    
        """

        digits = re.sub(r"\D", "", str(season))

        if len(digits) == 4:
                return 2000 + int(digits[2:])

        elif len(digits) == 8:
                return int(digits[4:])

        else: 
                raise ValueError(f"Invalid season key: {season}")


def season_midpoint(season: object) -> pd.Timestamp:
        """Mid-season date used to pick a contemporaneous market value.

        Using today's value for a 2023-24 season would mean predicting the past
        from the present, and the undervalued list would really be a list of
        players who have since got worse.

        Returns 1 January of the season's end year.
    
        """
        return pd.Timestamp(year=season_end_year(season), month=1, day=1)


def load_overrides() -> dict[str, int]:
        """Read the hand-maintained override table.

        Maps a normalised FBref name to a Transfermarkt player_id. Columns:
        `fbref_player`, `tm_player_id`, optional `note`. See
        data/manual_name_overrides.csv.

        Returns {normalised_fbref_name: tm_player_id}, or {} if the file is
        absent. Written by hand from reports/name_match_review.csv, and takes
        precedence over any fuzzy result.
    
        """
        if not paths.NAME_OVERRIDES.exists():
                return {}
        rows = {}
        fbref_players = pd.read_csv(paths.NAME_OVERRIDES, comment="#").dropna(subset=["fbref_player", "tm_player_id"])

        for player in fbref_players.itertuples(index=False):
                rows[normalise_name(player.fbref_player)] = int(player.tm_player_id)

        return rows


def match_players(fbref_players: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
        """Match one row per distinct (FBref player, birth year).

        `fbref_players` needs columns `player` and `born`. `tm` needs `player_id`,
        `name`, `birth_year` and `sub_position`.

        Returns one row per input with the best Transfermarkt candidate, its
        score, and a `decision` of accepted / review / rejected / override /
        no_candidates. Columns are fixed, because merge.py reads them by name:
        fbref_player, fbref_player_norm, fbref_born, match_score, decision,
        tm_player_id, tm_name, tm_birth_year, tm_sub_position.

        Every input exits by exactly one of five paths, recorded in `decision`:

            override        a human entry in manual_name_overrides.csv won
            no_candidates   missing birth year, or no one in the blocks
            accepted        score >= config.NAME_MATCH_ACCEPT
            review          score >= config.NAME_MATCH_REJECT
            rejected        below both

        The blocking index is built once, up front, as birth_year -> row
        positions in `tm`, and the normalised Transfermarkt names are computed
        once outside the loop. Both matter: without blocking this is 2,500 x
        50,000 comparisons, and normalising per comparison is the difference
        between twenty seconds and twenty minutes.

        Overrides are checked first, because a human decision always beats a
        fuzzy score. An override naming an id absent from `tm` raises rather
        than falling through to fuzzy matching, so a typo in the CSV cannot
        hide as a plausible-looking match.
    
        """
        tm = tm.reset_index(drop=True)
        tm_norm = tm["name"].apply(normalise_name)

        blocks = {}
        for position, birth_year in enumerate(tm["birth_year"]):
                if pd.notna(birth_year):
                        blocks.setdefault(int(birth_year), []).append(position)

        overrides = load_overrides()
        id_to_pos = {player_id: pos for pos, 
                     player_id in enumerate(tm["player_id"])
                     if pd.notna(player_id)}
        
        rows = []
        for player in fbref_players.itertuples(index = False):
                query = normalise_name(player.player)
                row = {
                        "fbref_player": player.player,
                        "fbref_player_norm": query,
                        "fbref_born": player.born,
                        "match_score": 0,
                        "decision": "no_candidates",
                        "tm_player_id": None,
                        "tm_name": None,
                        "tm_birth_year": None,
                        "tm_sub_position": None,
                }

                if query in overrides:
                        player_id = overrides[query]
                        if player_id not in id_to_pos:
                                raise ValueError(f"Override for {player_id}, not found in Transfermarkt data")
                        candidate = tm.iloc[id_to_pos[player_id]]

                        row.update({
                                "match_score": 100,
                                "decision": "override",
                                "tm_player_id": candidate["player_id"],
                                "tm_name": candidate["name"],
                                "tm_birth_year": candidate["birth_year"],
                                "tm_sub_position": candidate["sub_position"],
                        })
                        rows.append(row)
                        continue


                if pd.isna(player.born) or not query:
                        rows.append(row)
                        continue

                born = int(player.born)
                candidates = [
                        position 
                        for year in range(
                                born - config.BIRTH_YEAR_TOLERANCE,
                                born + config.BIRTH_YEAR_TOLERANCE + 1
                        )
                        for position in blocks.get(year, [])
                ]
                if not candidates:
                        rows.append(row)
                        continue

                choices = {position: tm_norm[position] 
                           for position in candidates
                           if tm_norm[position]}
                if not choices:
                        rows.append(row)
                        continue

                best = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)
                if best is None:
                        rows.append(row)
                        continue

                score, position = best[1], best[2]
                candidate = tm.iloc[position]

                if score >= config.NAME_MATCH_ACCEPT:
                        decision = "accepted"
                elif score >= config.NAME_MATCH_REJECT:
                        decision = "review"
                else:
                        decision = "rejected"

                row.update({
                        "match_score": score,
                        "decision": decision,
                        "tm_player_id": candidate["player_id"],
                        "tm_name": candidate["name"],
                        "tm_birth_year": candidate["birth_year"],
                        "tm_sub_position": candidate["sub_position"],
                })
                rows.append(row)

        return pd.DataFrame(rows,  columns=[
                "fbref_player", 
                "fbref_player_norm",
                "fbref_born",
                "match_score", 
                "decision",
                "tm_player_id", 
                "tm_name", 
                "tm_birth_year", 
                "tm_sub_position"])


def write_reports(matches: pd.DataFrame) -> dict:
        """Persist the review band and a summary. Both belong in git.

        `matches` is the frame returned by match_players.

        Writes two artefacts, both of which belong in git as evidence that the
        join behaved: the review band (everything between the reject and accept
        thresholds) for hand-adjudication, and a summary carrying the match
        rate. Returns the summary dict.

        Warns below an 85% match rate. That number belongs in the README -
        "94% of qualifying player-seasons matched to a market value" tells a
        reader the work was done honestly, and a low rate means the players
        worth writing about may simply be absent.
    
        """
        paths.ensure_dirs()

        review = matches[matches["decision"] == "review"].sort_values("match_score", ascending=False)
        review.to_csv(paths.MATCH_REVIEW, index=False)

        total = len(matches)
        matched = len(matches[matches["decision"].isin(["accepted", "override"])])
        match_rate = matched / total if total > 0 else 0.0

        summary = {
                "total": total,
                "matched": matched,
                "match_rate": match_rate,
                "decisions": {str(decision): int(count) 
                              for decision, count in matches["decision"].value_counts().items()},
                "thresholds": {
                        "accept": config.NAME_MATCH_ACCEPT,
                        "reject": config.NAME_MATCH_REJECT,
                },
        }

        paths.MATCH_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")


        if match_rate < 0.85:
                logger.warning(
                                "Match rate %.2f%% is below 85%%, check %s before trusting downstream data.",
                                match_rate * 100,
                                paths.MATCH_REVIEW,
                        )


        return summary
