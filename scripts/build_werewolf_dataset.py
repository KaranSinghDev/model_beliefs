"""Build datasets/werewolf_games.json from Kaggle's "Werewolf Gameplay" dataset.

Source: kaggle.com/datasets/kaggle/werewolf-dataset (Kaggle Agent Arena / Game Arena, CC BY 4.0,
~31k games, ~61 GB, one JSON replay per game named <EpisodeId>.json). Single files can be
fetched from Kaggle's public API without a login:
    https://www.kaggle.com/api/v1/datasets/download/kaggle/werewolf-dataset?file_name=<name>.json
and the file list from
    https://www.kaggle.com/api/v1/datasets/list/kaggle/werewolf-dataset?pageSize=200[&pageToken=...]
If that ever stops working, download some <EpisodeId>.json files by hand from the Kaggle page
into data/raw/werewolf/games/ and run with --no-download.

Replay schema used here (kaggle-environments "werewolf", schema_version 1):
  info.GAME_END.all_players_and_role  {player: role}  (true roles; Werewolf/Villager/Seer/Doctor)
  info.MODERATOR_OBSERVATION          list of steps, each a list of {"data_type", "json_str"};
      json_str -> {event_name, day, phase, detailed_phase, description, public, visible_to,
                   data, source, created_at}
  An event is observable to player P if public is true or P is in visible_to.
  Event types used: game_start (own role), inspect_result (Seer), heal_action / heal_result
  (Doctor), discussion (public chat; data.message), vote_action in the Day phase (day votes),
  elimination (exile / night kill, with revealed role), moderator_announcement (e.g. "no one
  was eliminated"). Model reasoning fields and raw prompts are never shown.

Each case = one cut of one game, seen by a focal non-werewolf player who is alive at the cut:
their private evidence + the public transcript up to that point. Target = another alive player
whose role has not been publicly revealed. label = 1 if the target is a Werewolf. Labels are
balanced by first drawing the label, then a target of that class.

Usage:
    python scripts/build_werewolf_dataset.py                 # download 60 games, build ~500 cases
    python scripts/build_werewolf_dataset.py --no-download   # use what is in data/raw/werewolf/games
"""
import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAMES = ROOT / "data" / "raw" / "werewolf" / "games"
TEMPLATE_FILE = ROOT / "datasets" / "werewolf.json"
OUT = ROOT / "datasets" / "werewolf_games.json"
API = "https://www.kaggle.com/api/v1/datasets"
REF = "kaggle/werewolf-dataset"


# ---------------------------------------------------------------- download

def _get(url, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "model-beliefs-research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_files(pages):
    names, token = [], None
    for _ in range(pages):
        q = {"pageSize": 200}
        if token:
            q["pageToken"] = token
        d = json.loads(_get(f"{API}/list/{REF}?{urllib.parse.urlencode(q)}", 60))
        names += [f["name"] for f in d.get("datasetFiles", []) if f["name"].endswith(".json")]
        token = d.get("nextPageToken") or d.get("nextPageTokenNullable")
        if not token:
            break
    return names


def download(n_games, pages, rng):
    GAMES.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in GAMES.glob("*.json")}
    if len(have) >= n_games:
        return
    names = [n for n in list_files(pages) if n not in have]
    rng.shuffle(names)
    for name in names[: n_games - len(have)]:
        url = f"{API}/download/{REF}?{urllib.parse.urlencode({'file_name': name})}"
        data = _get(url)
        json.loads(data)  # must be a JSON replay
        (GAMES / name).write_bytes(data)
        print("downloaded", name, f"{len(data) / 1e6:.1f} MB")
        time.sleep(0.5)


# ---------------------------------------------------------------- parsing

def load_game(path):
    g = json.loads(path.read_text())
    events = [json.loads(e["json_str"]) for step in g["info"]["MODERATOR_OBSERVATION"] for e in step]
    end = g["info"].get("GAME_END") or {}
    roles = end.get("all_players_and_role") or {}
    if not roles:  # fall back to the private role announcements
        roles = {e["data"]["player_id"]: e["data"]["role"] for e in events
                 if e["event_name"] == "game_start" and e.get("data")}
    players = end.get("all_players") or []
    roster = [p["id"] if isinstance(p, dict) else p for p in players] or list(roles)
    models = {p["id"]: (p.get("agent") or {}).get("display_name") for p in players if isinstance(p, dict)}
    if end.get("terminated_with_agent_error"):
        return None
    return {"id": str(g["info"].get("EpisodeId", path.stem)), "events": events, "roles": roles,
            "roster": roster, "models": models, "config": g.get("configuration", {})}


def tag(e):
    return f"[Day {e['day']} | {e['phase']}]"


def seen_by(e, p):
    return e.get("public") or p in (e.get("visible_to") or [])


def is_chat(e):
    return e["event_name"] == "discussion" and e.get("public") and isinstance(e.get("data"), dict)


def render(e, focal):
    """(kind, text) for an event the focal player can see, or None to skip it."""
    n, d, desc = e["event_name"], e.get("data") or {}, (e.get("description") or "").strip()
    if n == "game_start" and d.get("player_id") == focal:
        return "private", f'{tag(e)} Your team is "{d["team"]}". Your role is "{d["role"]}". {d.get("rule_of_role", "")}'.strip()
    if n == "inspect_result":
        return "private", f"{tag(e)} You inspected {d['target_id']}. {d['target_id']} is a {d['role']}."
    if n == "heal_action":
        return "private", f"{tag(e)} You chose to protect {d['target_id']} tonight."
    if n == "heal_result":
        return "private", f"{tag(e)} {desc}"
    if is_chat(e):
        return "public", f"{tag(e)} {d.get('actor_id', e.get('source'))}: {d.get('message', '').strip()}"
    if n == "vote_action" and e["phase"] == "Day":
        return "public", f"{tag(e)} {desc}"
    if n == "elimination":
        return "public", f"{tag(e)} {desc}"
    if n == "moderator_announcement" and e.get("public") and len(desc) < 200 and "eliminated" in desc:
        return "public", f"{tag(e)} {desc}"
    return None


def eliminated_before(events, i):
    out = {}
    for e in events[:i]:
        if e["event_name"] == "elimination" and isinstance(e.get("data"), dict):
            d = e["data"]
            pid = d.get("elected_player_id") or d.get("eliminated_player_id") or d.get("player_id")
            if not pid:
                m = re.search(r'player "([^"]+)"', e.get("description", ""))
                pid = m and m.group(1)
            role = next((v for k, v in d.items() if "role" in k and isinstance(v, str)), None)
            if pid:
                out[pid] = role
    return out


def cut_points(game):
    """Indices i (cut = events[:i]) during day discussion, after a chat message or right before the vote."""
    ev, cuts = game["events"], []
    for i, e in enumerate(ev):
        if e["event_name"] == "game_end":
            break
        if e["phase"] != "Day" or e["day"] < 1:
            continue
        if is_chat(e):
            n_chat = sum(1 for x in ev[:i + 1] if is_chat(x) and x["day"] == e["day"])
            cuts.append((i + 1, f"Day {e['day']}, discussion, after {n_chat} message(s) so far today"))
        elif e["event_name"] == "phase_change" and e.get("public") and "Moving to day vote" in (e.get("description") or ""):
            cuts.append((i, f"Day {e['day']}, discussion has ended, before the day vote"))
    return cuts


def setup_line(game):
    counts = Counter(game["roles"].values())
    order = ["Villager", "Werewolf", "Doctor", "Seer"]
    cstr = ", ".join(f"{r} {counts[r]}" for r in order + sorted(set(counts) - set(order)) if counts[r])
    cfg = game["config"]
    reveal = {"role": "role", "team": "team"}
    parts = []
    dl, nl = cfg.get("day_exile_reveal_level", "role"), cfg.get("night_elimination_reveal_level", "role")
    if dl in reveal and dl == nl:
        parts.append(f"Exiled or night-eliminated players have their {reveal[dl]} revealed.")
    else:
        if dl in reveal:
            parts.append(f"Exiled players have their {reveal[dl]} revealed.")
        if nl in reveal:
            parts.append(f"Night-eliminated players have their {reveal[nl]} revealed.")
    return (f"[Game setup] {len(game['roster'])} players: {', '.join(game['roster'])}. "
            f"Role counts: {cstr}. " + " ".join(parts)).strip()


def make_context(game, i, focal, target, when):
    ev = game["events"]
    dead = eliminated_before(ev, i)
    alive = [p for p in game["roster"] if p not in dead]
    revealed = ", ".join(f"{p}: {r}" for p, r in dead.items() if r) or "none"
    private, public = [], [setup_line(game)]
    for e in ev[:i]:
        if not seen_by(e, focal):
            continue
        r = render(e, focal)
        if r:
            (private if r[0] == "private" else public).append(r[1])
    head = (f"GAME STATE AT THE CUT\nFocal player: {focal}\nYour private role: {game['roles'][focal]}\n"
            f"Target player: {target}\nCurrent point: {when}\nAlive players: {', '.join(alive)}\n"
            f"Publicly revealed roles: {revealed}")
    return f"{head}\n\nYOUR PRIVATE EVIDENCE\n" + "\n".join(private) + "\n\nPUBLIC EVIDENCE\n" + "\n".join(public)


# ---------------------------------------------------------------- sampling

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=60)
    ap.add_argument("--n-cases", type=int, default=500)
    ap.add_argument("--list-pages", type=int, default=5, help="pages of 200 files to sample games from")
    ap.add_argument("--max-chars", type=int, default=30000, help="skip cuts whose context is longer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if not args.no_download:
        download(args.n_games, args.list_pages, rng)
    paths = sorted(GAMES.glob("*.json"))[: args.n_games] if args.no_download else sorted(GAMES.glob("*.json"))
    games = [g for g in (load_game(p) for p in paths) if g and g["roles"]]
    if not games:
        raise SystemExit(f"no games found in {GAMES}")

    per_game = -(-args.n_cases // len(games))
    cases, seen = [], set()
    for game in games:
        ev, roles = game["events"], game["roles"]
        cuts = cut_points(game)
        made, tries = 0, 0
        while made < per_game and tries < 200 and cuts:
            tries += 1
            i, when = rng.choice(cuts)
            dead = eliminated_before(ev, i)
            alive = [p for p in game["roster"] if p not in dead]
            focals = [p for p in alive if roles[p] != "Werewolf"]
            if not focals:
                continue
            focal = rng.choice(focals)
            want = rng.random() < 0.5
            pool = [p for p in alive if p != focal and (roles[p] == "Werewolf") == want]
            if not pool:
                continue
            target = rng.choice(pool)
            key = (game["id"], i, focal, target)
            if key in seen:
                continue
            ctx = make_context(game, i, focal, target, when)
            if len(ctx) > args.max_chars:
                continue
            seen.add(key)
            cases.append({
                "id": f"wwk_{game['id']}_{len(cases) + 1:04d}",
                "label": int(roles[target] == "Werewolf"),
                "vars": {"target": target},
                "game_id": game["id"], "focal": focal, "focal_role": roles[focal],
                "day": ev[i - 1]["day"],
                "focal_model": game["models"].get(focal), "target_model": game["models"].get(target),
                "context": ctx,
            })
            made += 1

    rng.shuffle(cases)
    cases = cases[: args.n_cases]
    tmpl = json.loads(TEMPLATE_FILE.read_text())
    out = {
        "domain": "werewolf_games",
        "event": tmpl["event"],
        "source": (
            "Kaggle 'Werewolf Gameplay' (kaggle.com/datasets/kaggle/werewolf-dataset, Kaggle Agent Arena, "
            f"CC BY 4.0). {len(games)} games, cut at day-discussion points from a focal non-werewolf player's "
            "view (their private evidence + public transcript so far). label = 1 if target is a Werewolf. "
            f"Built by scripts/build_werewolf_dataset.py (seed {args.seed})."
        ),
        "preamble": tmpl["preamble"],
        "context_label": tmpl["context_label"],
        "templates": tmpl["templates"],
        "cases": cases,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    n1 = sum(c["label"] for c in cases)
    L = sorted(len(c["context"]) for c in cases)
    print(f"wrote {OUT}: {len(cases)} cases from {len(games)} games; label 1: {n1}, label 0: {len(cases) - n1}")
    print("focal roles:", Counter(c["focal_role"] for c in cases), "days:", Counter(c["day"] for c in cases))
    print(f"context chars: median {L[len(L) // 2]}, max {L[-1]}")


if __name__ == "__main__":
    main()
