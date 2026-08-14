"""The web interface: the framework-free API layer, and the HTTP shim over it.

The load-bearing test is :func:`test_no_response_ever_leaks_the_opponents_cards`. Anything
in the JSON is in the browser, and someone will read it.
"""

import json
import pathlib
import random
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import catan.topology as T
from catan import action_space, rules
from catan.dev_cards import DevCard
from catan.resources import DEV_CARD_COST, Resource
from catan.state import Phase

api = pytest.importorskip("interfaces.web.api")
from interfaces.web.server import Handler, safe_path  # noqa: E402


def fresh_game(seed=5, **kwargs):
    return api.Game(seed=seed, **kwargs)


def all_options(view):
    board = [i for targets in view["actions"]["board"].values() for i in targets.values()]
    return board + [entry["index"] for entry in view["actions"]["panel"]]


def play_out(game, limit=4000, rng=None):
    """Click through a game the way the client would, returning the final view."""
    rng = rng or random.Random(0)
    view = game.view()
    for _ in range(limit):
        if view["done"]:
            break
        options = all_options(view)
        assert options, f"nothing to do in {view['phase']}"
        view = game.play(rng.choice(options))
    return view


def watch_out(game, limit=9000, rng=None):
    """The same, but for a paced game: watch the opponent a move at a time.

    Exactly what the browser does, minus the second between moves. It draws on the same
    ``rng`` in the same order as :func:`play_out`, since ``advance`` makes no choice of its
    own — so the two produce the same game from the same seed.
    """
    rng = rng or random.Random(0)
    view = game.view()
    for _ in range(limit):
        if view["done"]:
            break
        if view["awaitingOpponent"]:
            view = game.advance()
            continue
        options = all_options(view)
        assert options, f"nothing to do in {view['phase']}"
        view = game.play(rng.choice(options))
    return view


# =========================================================================== #
# HIDDEN INFORMATION — the leak test                                          #
# =========================================================================== #

def test_no_response_ever_leaks_the_opponents_cards():
    """Walk a whole game and inspect every response the human would receive."""
    game = fresh_game(seed=7)
    rng = random.Random(7)
    view = game.view()

    for _ in range(1500):
        opponent = next(p for p in view["players"] if not p["you"])
        if not view["done"]:
            assert "hand" not in opponent, "the opponent's hand composition leaked"
            assert "dev" not in opponent, "the opponent's development cards leaked"
            assert "victoryPoints" not in opponent, "hidden victory points leaked"
            assert "tradeRates" not in opponent
        # counts are public — cards are countable at a real table
        assert "handCount" in opponent and "devCount" in opponent

        # neither deck may appear anywhere in the payload
        blob = json.dumps(view)
        assert "devDeckOrder" not in blob and "diceDeck" not in blob

        if view["done"]:
            break
        board = [i for t in view["actions"]["board"].values() for i in t.values()]
        panel = [entry["index"] for entry in view["actions"]["panel"]]
        view = game.play(rng.choice(board + panel))


@pytest.mark.parametrize("seat", [1, 2])
def test_no_response_leaks_the_other_seats_cards_whichever_seat_asks(seat):
    """The same walk, from either side of the board.

    ⚠️ ``view`` used to take no seat at all and reveal seat 1 unconditionally, so every
    caller received the same payload no matter who was asking — correct for a local
    single-player app and, the moment this is served over a network, a way to read a
    stranger's cards by asking for their game. The filter is now parameterised, and the leak
    test is parameterised with it: a rule that only holds for seat 1 is not a rule. See
    ``docs/audit-2026-08-05-public-arena.md`` §B8.
    """
    game = fresh_game(seed=13)
    rng = random.Random(13)
    view = game.view(seat=seat)

    for _ in range(1500):
        assert view["you"] == seat
        mine = next(p for p in view["players"] if p["id"] == seat)
        theirs = next(p for p in view["players"] if p["id"] != seat)

        assert mine["you"] is True and theirs["you"] is False
        assert "hand" in mine, "you must always see your own cards"
        if not view["done"]:
            assert "hand" not in theirs, f"seat {seat} was shown the other hand"
            assert "dev" not in theirs
            assert "victoryPoints" not in theirs, "hidden victory points leaked"
            assert "tradeRates" not in theirs
        assert "handCount" in theirs and "devCount" in theirs

        if view["done"]:
            break
        # Seat 1 is the one with a person behind it, so it is the one that can `play`; the
        # point here is what seat 2 is *shown*, not that it can move.
        if game.awaiting_opponent:
            game.advance()
        else:
            board = [i for t in game.view()["actions"]["board"].values() for i in t.values()]
            panel = [entry["index"] for entry in game.view()["actions"]["panel"]]
            game.play(rng.choice(board + panel))
        view = game.view(seat=seat)


@pytest.mark.parametrize("seat", [1, 2])
def test_the_statistics_panel_is_filtered_by_seat_too(seat):
    """The other response a person can ask for, and the one that reports on the *history* of
    the game rather than its position — so it is the one place where "is this public?" has to
    be asked about a running total rather than about a card.

    Each seat sees its own hidden victory-point sources and not the other's. Asking as seat 2
    must not be a way to turn seat 1's cards over.
    """
    game = fresh_game()
    game.state.dev_cards[1][DevCard.VICTORY_POINT] = 3
    game.state.dev_cards[2][DevCard.VICTORY_POINT] = 2

    panel = game.statistics(seat=seat)
    mine = next(p for p in panel["players"] if p["id"] == seat)
    theirs = next(p for p in panel["players"] if p["id"] != seat)

    assert mine["you"] is True and theirs["you"] is False
    assert any(row.get("hidden") for row in mine["pointSources"]), \
        "you must see your own hidden victory point cards"
    assert mine["victoryPointsComplete"] is True
    assert not any(row.get("hidden") for row in theirs["pointSources"]), \
        f"seat {seat} was shown the other seat's hidden victory points"
    assert theirs["victoryPointsComplete"] is False, \
        "the panel must say it does not know rather than imply a complete total"


def test_only_the_public_score_is_shown_while_the_game_runs():
    game = fresh_game()
    game.state.dev_cards[2][DevCard.VICTORY_POINT] = 3

    opponent = next(p for p in game.view()["players"] if not p["you"])
    assert opponent["publicVictoryPoints"] == rules.public_victory_points(game.state, 2)
    assert "victoryPoints" not in opponent


def test_the_opponents_cards_are_revealed_once_the_game_ends():
    """Hidden during play, shown at the end so the player can see what beat them."""
    game = fresh_game(seed=11)
    view = play_out(game)
    if not view["done"]:
        pytest.skip("game did not finish inside the step budget")
    opponent = next(p for p in view["players"] if not p["you"])
    assert "hand" in opponent and "dev" in opponent


def test_you_always_see_your_own_cards():
    game = fresh_game()
    you = next(p for p in game.view()["players"] if p["you"])
    assert set(you["hand"]) == {r.name.lower() for r in Resource}
    assert "victoryPoints" in you and "tradeRates" in you


# =========================================================================== #
# THE VIEW                                                                    #
# =========================================================================== #

def test_a_new_game_waits_for_the_human_to_place_a_settlement():
    view = fresh_game().view()
    assert view["you"] == api.HUMAN == 1
    assert view["yourTurn"] is True
    assert view["phase"] == Phase.SETUP_SETTLEMENT.name
    assert "settlement" in view["phaseHint"].lower()


def test_setup_offers_every_free_vertex_as_a_click_target():
    view = fresh_game().view()
    targets = view["actions"]["board"]["BUILD_SETTLEMENT"]
    assert len(targets) == T.NUM_VERTICES
    assert set(targets) == {str(v) for v in range(1, T.NUM_VERTICES + 1)}


def test_every_offered_index_is_actually_legal():
    """The client can only click what it is given, so what it is given must be right."""
    game = fresh_game(seed=3)
    rng = random.Random(3)
    view = game.view()
    for _ in range(200):
        if view["done"]:
            break
        legal = set(game.info["legal"])
        offered = [i for t in view["actions"]["board"].values() for i in t.values()]
        offered += [entry["index"] for entry in view["actions"]["panel"]]
        assert offered, f"nothing offered in {view['phase']}"
        assert set(offered) <= legal, "an illegal index was offered to the client"
        view = game.play(rng.choice(offered))


def test_board_targets_and_panel_actions_do_not_overlap():
    """A click target and a button for the same move would be two ways to do one thing."""
    game = fresh_game(seed=3)
    rng = random.Random(1)
    view = game.view()
    for _ in range(150):
        if view["done"]:
            break
        board = {i for t in view["actions"]["board"].values() for i in t.values()}
        panel = {entry["index"] for entry in view["actions"]["panel"]}
        assert not (board & panel)
        view = game.play(rng.choice(sorted(board | panel)))


def test_the_view_carries_what_a_player_needs_to_see():
    view = fresh_game().view()
    for key in ("lastRoll", "turn", "phase", "phaseHint", "bank", "devDeck",
                "robber", "board", "pieces", "players", "actions", "log",
                "victoryTarget", "handLimit", "winner", "done", "awaitingOpponent"):
        assert key in view, f"missing {key}"
    assert len(view["board"]["tiles"]) == T.NUM_TILES
    assert len(view["board"]["harbours"]) == 9


def test_the_dice_are_reported_every_turn():
    game = fresh_game(seed=4)
    assert game.view()["lastRoll"] is None       # nothing rolled yet
    view = play_out(game, limit=60)
    assert view["lastRoll"] is not None
    assert 2 <= view["lastRoll"] <= 12


def test_the_robber_and_tile_numbers_are_visible():
    view = fresh_game().view()
    flagged = [t for t in view["board"]["tiles"] if t["robber"]]
    assert len(flagged) == 1 and flagged[0]["id"] == view["robber"]
    desert = [t for t in view["board"]["tiles"] if t["resource"] == "desert"]
    assert len(desert) == 1 and desert[0]["number"] is None
    assert all(t["number"] is not None
               for t in view["board"]["tiles"] if t["resource"] != "desert")


def test_pieces_appear_once_they_are_built():
    game = fresh_game()
    view = game.view()
    assert view["pieces"]["buildings"] == []

    vertex = next(iter(view["actions"]["board"]["BUILD_SETTLEMENT"]))
    view = game.play(view["actions"]["board"]["BUILD_SETTLEMENT"][vertex])
    mine = [b for b in view["pieces"]["buildings"] if b["player"] == api.HUMAN]
    assert [b["vertex"] for b in mine] == [int(vertex)]
    assert mine[0]["kind"] == "settlement"


# =========================================================================== #
# THE LOG                                                                     #
# =========================================================================== #

def test_the_log_says_what_happened():
    game = fresh_game(seed=4)
    view = play_out(game, limit=120)
    assert view["log"], "nothing was logged"
    text = " ".join(view["log"])
    assert "rolled" in text
    assert "You" in text and "Opponent" in text


def test_the_log_reports_the_opponents_moves_not_just_yours():
    game = fresh_game(seed=4)
    view = play_out(game, limit=200)
    assert any(line.startswith("Opponent") for line in view["log"])


def test_production_is_narrated():
    game = fresh_game(seed=4)
    view = play_out(game, limit=400)
    assert any(" got " in line for line in view["log"]), "no payout was reported"


# =========================================================================== #
# PLAYING                                                                     #
# =========================================================================== #

def test_playing_advances_the_game_and_lets_the_opponent_reply():
    game = fresh_game()
    before = game.view()
    index = next(iter(before["actions"]["board"]["BUILD_SETTLEMENT"].values()))
    after = game.play(index)
    assert after["turn"] >= before["turn"]
    assert after["pieces"]["buildings"]


def test_the_human_is_never_asked_to_move_for_the_opponent():
    game = fresh_game(seed=6)
    rng = random.Random(6)
    view = game.view()
    for _ in range(300):
        if view["done"]:
            break
        assert view["yourTurn"] is True
        assert view["currentPlayer"] == api.HUMAN
        board = [i for t in view["actions"]["board"].values() for i in t.values()]
        panel = [entry["index"] for entry in view["actions"]["panel"]]
        view = game.play(rng.choice(board + panel))


@pytest.mark.parametrize("bad", [-1, 9999])
def test_an_out_of_range_index_is_refused(bad):
    game = fresh_game()
    with pytest.raises(ValueError):
        game.play(bad)


def test_an_illegal_index_is_refused():
    game = fresh_game()
    legal = set(game.info["legal"])
    illegal = next(i for i in range(action_space.NUM_ACTIONS) if i not in legal)
    with pytest.raises(ValueError):
        game.play(illegal)


def test_playing_after_the_game_is_over_is_refused():
    game = fresh_game(seed=11)
    view = play_out(game)
    if not view["done"]:
        pytest.skip("game did not finish inside the step budget")
    with pytest.raises(ValueError):
        game.play(0)


def test_a_seed_reproduces_the_whole_game():
    first = play_out(fresh_game(seed=21), rng=random.Random(1))
    second = play_out(fresh_game(seed=21), rng=random.Random(1))
    assert first["log"] == second["log"]
    assert first["winner"] == second["winner"]


@pytest.mark.parametrize("opponent", sorted(api.OPPONENTS))
@pytest.mark.parametrize("rules_name", sorted(api.RULESETS))
def test_every_opponent_and_ruleset_can_be_played(opponent, rules_name):
    game = fresh_game(seed=2, opponent=opponent, rules_name=rules_name)
    view = play_out(game, limit=300)
    assert view["log"]


def test_an_unknown_opponent_or_ruleset_is_refused():
    with pytest.raises(ValueError):
        api.Game(opponent="genius")
    with pytest.raises(ValueError):
        api.Game(rules_name="chess")


# =========================================================================== #
# WATCHING THE OPPONENT                                                       #
# =========================================================================== #
#
# The engine decides a whole turn in well under a millisecond, so a reply played inside
# `play` arrives as one repaint: the board is simply different afterwards and the player
# never sees the bot place a settlement, move the robber or play a card. A paced game hands
# the decisions back one at a time instead, and the client spaces them out.


def paced_game(seed=5, **kwargs):
    return api.Game(seed=seed, paced=True, **kwargs)


def until_the_opponent_owes_a_move(game, rng, limit=200):
    """Play the human's moves until the opponent has one outstanding."""
    view = game.view()
    for _ in range(limit):
        if view["done"] or view["awaitingOpponent"]:
            return view
        view = game.play(rng.choice(all_options(view)))
    pytest.fail("the opponent was never owed a move")


def test_the_view_says_when_the_opponent_owes_a_move():
    game = paced_game()
    assert game.view()["awaitingOpponent"] is False, "the human places first"

    view = until_the_opponent_owes_a_move(game, random.Random(0))
    assert view["awaitingOpponent"] is True
    assert view["yourTurn"] is False
    assert view["currentPlayer"] != api.HUMAN
    assert not all_options(view), "nothing may be offered while it is not your move"


def test_advancing_plays_exactly_one_of_the_opponents_decisions():
    """One per call is the whole point: it is what lets the client draw each of them, and
    what keeps what is drawn in the order it happened."""
    game = paced_game()
    until_the_opponent_owes_a_move(game, random.Random(0))

    asked = []
    agent = game.opponent

    def counting(observation, info):
        asked.append(info["player"])
        return agent(observation, info)

    game.opponent = counting
    game.advance()
    assert len(asked) == 1
    assert asked == [3 - api.HUMAN]


def test_advancing_when_it_is_your_move_does_nothing():
    """The client calls it without first working out whether it should — that decision
    belongs to the server, which has the engine's `info`."""
    game = paced_game()
    before = game.view()
    assert before["awaitingOpponent"] is False
    assert game.advance() == before


def test_advancing_after_the_game_is_over_does_nothing():
    game = paced_game(seed=11)
    view = watch_out(game)
    if not view["done"]:
        pytest.skip("game did not finish inside the step budget")
    assert game.advance() == view


def test_an_unpaced_game_still_gets_the_whole_reply_in_one_call():
    """The default, and what every test, benchmark and script driving api.Game relies on:
    nothing calls `advance`, so pacing them would leave the opponent stuck."""
    game = fresh_game()
    rng = random.Random(0)
    view = game.view()
    for _ in range(60):
        if view["done"]:
            break
        assert view["awaitingOpponent"] is False, "the opponent was left mid-reply"
        view = game.play(rng.choice(all_options(view)))


def test_pacing_changes_nothing_about_the_game_itself():
    """It is presentation and only presentation. The same seed and the same clicks must
    produce the same game whether the opponent's reply arrives in one call or a move at a
    time — otherwise a game played in the browser is not the game the recording replays."""
    straight = play_out(fresh_game(seed=21), rng=random.Random(1))
    paced = watch_out(paced_game(seed=21), rng=random.Random(1))
    assert paced["log"] == straight["log"]
    assert paced["winner"] == straight["winner"]
    assert paced["turn"] == straight["turn"]


def test_a_paced_game_records_the_same_moves(tmp_path, monkeypatch):
    """A recording is the seed plus the moves, and it is what a real game becomes training
    data from. Pacing must not drop, duplicate or reorder a decision in it."""
    from interfaces.web import recorder

    monkeypatch.setattr(recorder, "GAMES", tmp_path)
    straight = fresh_game(seed=21, record_game=True)
    play_out(straight, rng=random.Random(1))
    paced = paced_game(seed=21, record_game=True)
    watch_out(paced, rng=random.Random(1))

    assert paced.recorder.moves == straight.recorder.moves
    assert paced.recorder.result == straight.recorder.result
    assert recorder.verify(paced.recorder.path) is not False


# =========================================================================== #
# GEOMETRY                                                                    #
# =========================================================================== #

def test_geometry_covers_the_whole_board():
    geometry = api.geometry()
    assert len(geometry["tiles"]) == T.NUM_TILES
    assert len(geometry["vertices"]) == T.NUM_VERTICES
    assert len(geometry["roads"]) == T.NUM_ROADS
    assert {t["id"] for t in geometry["tiles"]} == set(range(1, T.NUM_TILES + 1))


def test_geometry_matches_the_png_renderer():
    """The browser and the image must agree, which they do by sharing one lattice."""
    from interfaces.render import Geometry

    plan = Geometry(hex_width=110)
    geometry = api.geometry(hex_width=110)
    assert (geometry["width"], geometry["height"]) == plan.size
    for spot in geometry["vertices"]:
        assert (spot["x"], spot["y"]) == plan.vertex(spot["id"])


def test_road_geometry_names_its_endpoints():
    geometry = api.geometry()
    from interfaces.render import Geometry
    plan = Geometry(hex_width=110)
    for road in geometry["roads"]:
        first, second = (plan.vertex(v) for v in T.ROAD_VERTICES[road["id"]])
        assert (road["x1"], road["y1"]) == first
        assert (road["x2"], road["y2"]) == second


def test_geometry_is_json_serialisable():
    json.dumps(api.geometry())


def test_the_offered_opponents_are_exactly_the_ones_that_work():
    """The page builds its dropdown from this, so anything listed must be startable.

    It used to be a hardcoded list in index.html, which offered `learned` whether or not a
    champion had loaded — and after an observation change it has not, so the option was
    clickable and answered with a 400.
    """
    offered = api.geometry()["opponents"]
    assert offered, "nothing to choose from"
    assert [entry["name"] for entry in offered] == [
        name for name in api.OPPONENT_LABELS if name in api.OPPONENTS
    ]
    for entry in offered:
        assert entry["label"], entry["name"]
        api.Game(opponent=entry["name"], seed=1)          # would raise if unknown

    defaults = [entry["name"] for entry in offered if entry["default"]]
    assert defaults == [api.DEFAULT_OPPONENT], "exactly one option must start selected"


def test_an_opponent_that_did_not_load_is_not_offered(monkeypatch):
    """The case this exists for: a champion trained against a different encoder."""
    monkeypatch.setitem(api.OPPONENT_LABELS, "learned", "Learned (strongest)")
    monkeypatch.delitem(api.OPPONENTS, "learned", raising=False)
    assert "learned" not in [entry["name"] for entry in api.opponents()]


# =========================================================================== #
# THE HTTP SHIM                                                               #
# =========================================================================== #

@pytest.fixture(scope="module")
def server(tmp_path_factory):
    # A game reached over HTTP is recorded, because that is how a person plays. The test
    # suite reaches it over HTTP too, so its games are sent somewhere disposable — the
    # point of recording is the handful of real games, and burying them under the suite's
    # would defeat it.
    from interfaces.web import recorder

    original = recorder.GAMES
    recorder.GAMES = tmp_path_factory.mktemp("recorded")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        recorder.GAMES = original


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as response:
        return response.status, response.read(), response.headers


@pytest.mark.parametrize("path,content_type", [
    ("/", "text/html"),
    ("/app/app.js", "text/javascript"),
    ("/app/app.css", "text/css"),
    ("/images/tiles/wood.png", "image/png"),
    ("/images/ressources/wood.png", "image/png"),
    ("/images/robber/robber.png", "image/png"),
])
def test_static_files_are_served(server, path, content_type):
    status, body, headers = request(server, path)
    assert status == 200
    assert headers["Content-Type"].startswith(content_type)
    assert len(body) > 0


def test_a_whole_game_can_be_played_over_http(server):
    """Exactly the client's loop: play a move, then watch the opponent's replies arrive one
    at a time. A game reached over HTTP is paced, because a person is watching it."""
    _, body, _ = request(server, "/api/game", {"seed": 8})
    view = json.loads(body)
    game_id = view["gameId"]
    rng = random.Random(8)
    advances = 0

    for _ in range(400):
        if view["done"]:
            break
        if view["awaitingOpponent"]:
            _, body, _ = request(server, f"/api/game/{game_id}/advance", {})
            advances += 1
        else:
            _, body, _ = request(server, f"/api/game/{game_id}/action",
                                 {"index": rng.choice(all_options(view))})
        view = json.loads(body)

    assert advances, "the opponent never moved"
    assert view["log"]
    _, body, _ = request(server, f"/api/game/{game_id}")
    assert json.loads(body)["gameId"] == game_id


def test_advancing_a_missing_game_is_a_404(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(server, "/api/game/999999/advance", {})
    assert caught.value.code == 404


def test_the_geometry_endpoint_works(server):
    _, body, _ = request(server, "/api/geometry")
    assert len(json.loads(body)["vertices"]) == T.NUM_VERTICES


def test_an_illegal_action_is_a_400(server):
    _, body, _ = request(server, "/api/game", {"seed": 1})
    game_id = json.loads(body)["gameId"]
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(server, f"/api/game/{game_id}/action", {"index": 0})
    assert caught.value.code == 400
    assert "legal" in json.loads(caught.value.read())["error"]


def test_a_missing_game_is_a_404(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(server, "/api/game/999999")
    assert caught.value.code == 404


def test_the_statistics_endpoint_works(server):
    """It has to be matched before the bare game route, which would otherwise read
    "3/stats" as the game id and answer 404."""
    _, body, _ = request(server, "/api/game", {"seed": 3})
    game_id = json.loads(body)["gameId"]

    _, body, _ = request(server, f"/api/game/{game_id}/stats")
    report = json.loads(body)
    assert report["players"] and "dice" in report
    assert [int(total) for total in report["dice"]["totals"]] == list(range(2, 13))


def test_the_statistics_of_a_missing_game_are_a_404(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(server, "/api/game/999999/stats")
    assert caught.value.code == 404


def test_an_unknown_route_is_a_404(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(server, "/nope")
    assert caught.value.code == 404


def test_a_non_integer_action_is_a_400(server):
    _, body, _ = request(server, "/api/game", {"seed": 1})
    game_id = json.loads(body)["gameId"]
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(server, f"/api/game/{game_id}/action", {"index": "one"})
    assert caught.value.code == 400


@pytest.mark.parametrize("path", [
    "/images/../../../catan/rules.py",
    "/app/../../../catan/rules.py",
    "/images/....//....//rules.py",
])
def test_path_traversal_is_refused(server, path):
    """A served path comes from the network, so `..` must be refused, not trusted."""
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(server, path)
    assert caught.value.code == 404


def test_safe_path_rejects_escapes(tmp_path):
    (tmp_path / "inside.txt").write_text("ok")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")

    assert safe_path(tmp_path, "inside.txt") is not None
    assert safe_path(tmp_path, "../outside.txt") is None
    assert safe_path(tmp_path, "missing.txt") is None


# =========================================================================== #
# ARTWORK                                                                     #
# =========================================================================== #

def test_the_client_is_told_which_picture_goes_with_which_name():
    """The art set predates the code: wheat is drawn by ``weat.png`` and ore by
    ``stone.png``. The client must not have to know that — a second copy of the mapping in
    JavaScript is exactly the kind of divergence this interface exists to avoid."""
    art = api.geometry()["art"]
    assert art["resources"]["wheat"] == "weat"
    assert art["resources"]["ore"] == "stone"
    assert set(art["resources"]) == set(api.RESOURCE_NAMES)


def test_piece_colours_match_the_png_renderer():
    """One asset set, drawn the same way on screen and in a saved image."""
    from interfaces.render import PLAYER_COLOURS

    colours = api.geometry()["art"]["colours"]
    for slot, colour in enumerate(PLAYER_COLOURS):
        assert colours[slot + 1] == colour


def test_sprite_scales_come_from_the_renderer():
    from interfaces import render

    scales = api.geometry()["art"]["scales"]
    assert scales["settlement"] == render.SETTLEMENT_SCALE
    assert scales["city"] == render.CITY_SCALE
    assert scales["roadLength"] == render.ROAD_LENGTH_SCALE
    assert scales["spot"] == render.SPOT_SCALE
    assert scales["robber"] == render.ROBBER_SCALE


def test_where_the_number_token_goes_comes_from_the_renderer():
    """The tile art leaves a blank panel for the token, and it is not in the middle of the
    hex. Both renderers need that figure, so it is served rather than written twice — the
    browser and a saved PNG putting a number in different places is precisely the drift this
    interface exists to remove. ``tests/test_render.py`` measures it against the art."""
    from interfaces import render

    assert api.geometry()["art"]["offsets"]["number"] == render.NUMBER_OFFSET
    assert "art.offsets" in client_code(), "the client must be told, not guess"


def test_where_the_robber_stands_comes_from_the_renderer():
    """Its scale and offset are one design decision, not two: together they are what keeps
    the robber off the number it blocks. ``tests/test_render.py`` measures that against the
    art; this only insists the browser reads the same three numbers rather than its own, so
    the two renderers cannot disagree about where the robber is standing."""
    from interfaces import render

    art = api.geometry()["art"]
    assert art["scales"]["robber"] == render.ROBBER_SCALE
    assert art["offsets"]["robber"] == render.ROBBER_OFFSET
    assert art["aspects"]["robber"] == render.ROBBER_ASPECT


def test_every_asset_the_client_can_ask_for_exists():
    """The client builds image URLs from the served mapping, so every combination it can
    produce has to be a real file. A missing one is an invisible piece, not an error."""
    art = api.geometry()["art"]
    images = pathlib.Path(__file__).resolve().parents[1] / "interfaces" / "static" / "images"

    missing = []
    for name in art["resources"].values():
        if not (images / "tiles" / f"{name}.png").is_file():
            missing.append(f"tiles/{name}.png")
    for name in art["cards"].values():
        if not (images / "ressources" / f"{name}.png").is_file():
            missing.append(f"ressources/{name}.png")
    for colour in art["colours"].values():
        for folder, pattern in (("roads", "{}_road.png"),
                                ("settlements", "{}.png"),
                                ("cities", "{}_city.png")):
            if not (images / folder / pattern.format(colour)).is_file():
                missing.append(f"{folder}/{pattern.format(colour)}")
    if not (images / "spots" / "circle.png").is_file():
        missing.append("spots/circle.png")
    if not (images / "robber" / "robber.png").is_file():
        missing.append("robber/robber.png")
    for number in list(range(2, 7)) + list(range(8, 13)):
        if not (images / "numbers" / f"{number}.png").is_file():
            missing.append(f"numbers/{number}.png")

    assert not missing, missing


def test_a_card_is_served_for_every_resource():
    """One per resource, and keyed the way a hand is: the panels look a card up by the same
    name the view reports a count under, so a gap here is a resource that draws as nothing."""
    from catan.resources import Resource

    assert set(api.geometry()["art"]["cards"]) == {r.name.lower() for r in Resource}


def test_the_card_shape_is_measured_from_the_card_art():
    """The browser sizes a card from this one number, so it has to be the art's own shape
    rather than a figure someone typed. All five are cropped to the same box; if a re-cut
    set is not, one ratio can no longer describe them and the client needs telling per card.
    """
    from PIL import Image

    images = (pathlib.Path(__file__).resolve().parents[1]
              / "interfaces" / "static" / "images" / "ressources")

    shapes = {}
    for name in api.geometry()["art"]["cards"].values():
        with Image.open(images / f"{name}.png") as card:
            shapes[name] = card.size

    assert len(set(shapes.values())) == 1, f"the cards are not one shape: {shapes}"
    width, height = next(iter(shapes.values()))
    assert api.CARD_ASPECT == pytest.approx(width / height)
    assert api.geometry()["art"]["aspects"]["card"] == pytest.approx(width / height)


def test_the_card_art_is_cropped_to_the_card():
    """The set arrived as 1920x1080 renders with the card afloat in white, which is the one
    defect this cannot see by looking at the shape: the padding was centred, so the aspect
    ratio was of the *canvas*. The client sizes the file, so a card in a white field draws
    smaller than asked with a margin nobody chose and no rule of the game to blame.

    Any ink at all counts, since a margin from a render is flat #FFFFFF and differs by zero.
    """
    from PIL import Image, ImageChops

    images = (pathlib.Path(__file__).resolve().parents[1]
              / "interfaces" / "static" / "images" / "ressources")

    for name in api.geometry()["art"]["cards"].values():
        with Image.open(images / f"{name}.png") as opened:
            card = opened.convert("RGB")
        white = Image.new("RGB", card.size, (255, 255, 255))
        drawn = ImageChops.difference(card, white).convert("L").getbbox()
        assert drawn is not None, f"{name}.png is blank"
        assert drawn == (0, 0, *card.size), f"{name}.png has white padding around it"


APP_JS = (pathlib.Path(__file__).resolve().parents[1]
          / "interfaces" / "web" / "static" / "app.js")


def client_code():
    """app.js with its comments stripped.

    Comments may name anything freely — that is where the explanation belongs, and removing
    them is what makes these tests about the behaviour rather than about the prose.
    """
    return "\n".join(
        line for line in APP_JS.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith(("//", "/*", "*"))
    )


def test_the_client_does_not_hard_code_the_art_mapping():
    """If the JavaScript ever writes "weat" or a colour name in *code*, the mapping has
    been duplicated and the two copies will drift."""
    for literal in ('"weat"', "'weat'", "weat.png", "stone.png", "wheat.png", "ore.png",
                    '"red_road', '"blue_road', "settlements/red", "cities/blue"):
        assert literal not in client_code(), f"{literal} is hard-coded in app.js"


def test_the_client_is_told_the_shape_of_a_sprite():
    """The number the browser needs that the PNG renderer never did: it fits a sprite by
    width and lets the file supply the rest, while an SVG <image> takes both. Written in
    JavaScript it would be a guess at the proportions of a file the client never opens, and
    a sprite drawn to a guessed ratio is a stretched sprite — which reads as bad art rather
    than as a bug, so nothing fails and nobody looks."""
    assert "art.aspects" in client_code(), "the client must be told, not guess"
    assert "art.cards" in client_code(), "the card file names must come from the server"


def test_the_client_says_who_won_rather_than_only_hinting_it():
    """The board goes quiet by itself when a game ends — nothing to place, no actions — so
    without an announcement the result is one grey line in the hint bar, which is genuinely
    easy to miss when the last move was the opponent's.

    ``done`` and ``winner`` have been in the payload since the interface was built and the
    client read neither.
    """
    code = client_code()
    assert "view.done" in code, "the client must notice the game ended"
    assert "view.winner" in code, "and which side won"
    assert "You won!" in code and "You lost" in code


def test_the_result_is_announced_once_per_game_not_once_per_repaint():
    """`render` runs after every decision, so an overlay rebuilt there would restart its
    own animation on every repaint after the game ended."""
    code = client_code()
    assert "celebrated" in code


def test_a_watched_game_is_told_who_won_rather_than_congratulated():
    """There is no "you" in a game nobody is playing."""
    assert "view.watching" in client_code()


def test_the_animation_can_be_turned_off_by_the_operating_system():
    """Nobody should be made motion-sick by a scoreboard. The result still appears and
    still says who won; only the movement goes."""
    css = (pathlib.Path(__file__).resolve().parents[1]
           / "interfaces" / "web" / "static" / "app.css").read_text(encoding="utf-8")
    assert "prefers-reduced-motion" in css


def test_the_client_asks_for_the_statistics_only_when_they_are_wanted():
    """Its own request, made on a click. Folding the report into `view` would compute it
    four times a turn for a panel nobody has opened."""
    code = client_code()
    assert "/stats" in code
    assert "openStats" in code


def test_the_client_has_no_stray_control_characters():
    """A ``\b`` written inside a template literal is a backspace, not a word boundary. It
    produces a pattern that matches nothing while reading as though it should work."""
    client = APP_JS.read_text(encoding="utf-8")
    for bad in ("\b", "\f", "\v", "\x00"):
        assert bad not in client, f"control character {bad!r} in app.js"


def test_the_client_spaces_the_opponents_moves_a_second_apart():
    """Without a delay the opponent's whole reply is one repaint and the player never sees
    what it did. This is the interface's pacing and nothing else's — training never loads
    app.js, so no agent is ever made to wait for it."""
    code = client_code()
    assert "OPPONENT_MOVE_MS = 1000" in code
    assert "/advance" in code, "the client must ask for one move at a time"
    assert "awaitingOpponent" in code, "and let the server say when one is owed"


# =========================================================================== #
# SEEING YOUR OWN CARDS                                                       #
# =========================================================================== #

def test_you_can_see_every_development_card_you_hold():
    """Including the ones you cannot play. A Victory Point card never appears as an action
    button and a card bought this turn is not playable until the turn ends, so without
    this the only evidence you hold them is a count."""
    from catan.dev_cards import DevCard

    game = api.Games().new(opponent="hard", seed=5)
    state = game.state
    state.dev_cards[api.HUMAN][DevCard.VICTORY_POINT] = 2
    state.dev_cards[api.HUMAN][DevCard.KNIGHT] = 1
    state.dev_cards_new[api.HUMAN][DevCard.KNIGHT] = 1

    you = next(p for p in game.view()["players"] if p["you"])
    assert you["dev"]["victory_point"] == 2
    assert you["dev"]["knight"] == 1
    # and which of them arrived this turn, so "held" and "usable" stay distinguishable
    assert you["devNew"]["knight"] == 1
    assert you["devNew"]["victory_point"] == 0


def test_an_opponents_development_cards_stay_hidden_while_the_game_runs():
    from catan.dev_cards import DevCard

    game = api.Games().new(opponent="hard", seed=5)
    game.state.dev_cards[2][DevCard.MONOPOLY] = 3

    view = game.view()
    assert not view["done"]
    opponent = next(p for p in view["players"] if not p["you"])
    assert "dev" not in opponent and "devNew" not in opponent
    assert opponent["devCount"] == 3, "the count is public; the composition is not"
    # Not a blob search: your *own* dev dict lists every card name, including the ones you
    # hold none of, so "monopoly" appears in the payload legitimately. What must not appear
    # is any per-card figure for the opponent.
    assert not any(key.startswith("dev") and isinstance(value, dict)
                   for key, value in opponent.items())


def test_buying_a_card_says_which_one_you_drew():
    """The count going up does not tell you what you bought."""
    import random

    game = api.Games().new(opponent="hard", seed=6)
    rng = random.Random(1)
    for _ in range(600):
        view = game.view()
        if view["done"]:
            break
        panel = {e["type"]: e["index"] for e in view["actions"]["panel"]}

        if "BUY_DEV_CARD" in panel:
            game.play(panel["BUY_DEV_CARD"])
            break

        # Stock the hand for the *next* turn rather than hoping a random walk affords a
        # card. What the log names is the point of the test, and waiting for luck tied it to
        # the exact order of the legal actions — narrowing the pre-roll list sent the walk
        # somewhere else and the game finished with no purchase ever made.
        #
        # Before ending the turn, not during one: `play` recomputes what is legal, and
        # `game.info["legal"]` is a snapshot that a mid-turn hand would contradict.
        # Production only ever adds to a hand, so the cost survives the roll; if a 7 steals
        # part of it, the next turn stocks it again.
        if "END_TURN" in panel:
            game.state.hands[api.HUMAN] = list(DEV_CARD_COST)
            game.play(panel["END_TURN"])
            continue

        # Whatever is left is a discard or a robber move, where the choice does not matter.
        choices = [i for group in view["actions"]["board"].values() for i in group.values()]
        choices += list(panel.values())
        if not choices:
            break
        game.play(rng.choice(choices))

    named = [line for line in game.log if line.startswith("You bought a development card:")]
    assert named, game.log[-6:]
    card = named[-1].split(":")[-1].strip()
    assert card.replace(" ", "_") in api.DEV_CARD_NAMES, card


def test_an_opponents_purchase_is_never_named():
    """The same log both players read. Naming their card would hand over the game."""
    import random

    game = api.Games().new(opponent="hard", seed=2)
    rng = random.Random(4)
    for _ in range(600):
        view = game.view()
        if view["done"]:
            break
        choices = [i for group in view["actions"]["board"].values() for i in group.values()]
        choices += [e["index"] for e in view["actions"]["panel"]]
        if not choices:
            break
        game.play(rng.choice(choices))

    theirs = [line for line in game.log if line.startswith("Opponent bought")]
    assert theirs, "the opponent never bought one; pick another seed"
    for line in theirs:
        assert line == "Opponent bought a development card", line


def test_the_card_a_steal_took_is_named():
    """In a two-player game both sides see the card move, so naming it is not a leak — and
    a robber that silently changed a count would look broken."""
    import random

    game = api.Games().new(opponent="hard", seed=6)
    rng = random.Random(1)
    for _ in range(600):
        view = game.view()
        if view["done"]:
            break
        choices = [i for group in view["actions"]["board"].values() for i in group.values()]
        choices += [e["index"] for e in view["actions"]["panel"]]
        if not choices:
            break
        game.play(rng.choice(choices))

    steals = [line for line in game.log if " stole " in line]
    assert steals, "no steal happened; pick another seed"
    for line in steals[:5]:
        assert any(name in line for name in api.RESOURCE_NAMES), line


def test_the_drawn_card_is_not_recorded_on_the_event():
    """`info["events"]` is handed to agents, so a card id there would put hidden
    information where an opponent could read it. The web layer works it out instead."""
    from catan.events import EventKind

    game = api.Games().new(opponent="hard", seed=6)
    for event in game.info.get("events", ()):
        if event.kind is EventKind.BOUGHT_DEV:
            assert event.position == 0 and event.other == 0, event


# =========================================================================== #
# Watching two agents play                                                    #
# =========================================================================== #

def test_a_watched_game_puts_an_agent_in_the_human_seat():
    game = api.Game(opponent="hard", watch="greedy", seed=11)

    assert game.watching
    assert game.watcher is not None
    assert game.paced, "a spectated game must hand back one decision at a time"
    assert game.awaiting_opponent, "nobody is playing, so every decision is owed"
    assert game.info["turn"] == 0, (
        "the constructor must not play the game: when both seats are agents the "
        "'until it is the human's turn' loop has no stopping condition"
    )


def test_a_watched_game_advances_one_decision_at_a_time():
    game = api.Game(opponent="hard", watch="greedy", seed=11)
    movers = []
    for _ in range(12):
        movers.append(game.info["player"])
        game.advance()
    assert set(movers) == {1, 2}, "both seats must actually move"


def test_a_watched_game_refuses_to_be_played():
    game = api.Game(opponent="hard", watch="greedy", seed=11)
    with pytest.raises(ValueError, match="watched, not played"):
        game.play(0)


def test_a_watched_game_reaches_an_end():
    game = api.Game(opponent="hard", watch="greedy", seed=3)
    for _ in range(20_000):
        if game.info["done"]:
            break
        game.advance()
    assert game.info["done"]
    assert game.info["winner"] in (1, 2, None)


def test_a_watched_view_says_so_and_carries_the_pace():
    game = api.Game(opponent="hard", watch="greedy", seed=11)
    payload = api.view(game)

    assert payload["watching"] is True
    assert payload["watchedBy"] == "greedy"
    assert payload["paceMs"] == api.WATCH_PACE_MS
    assert payload["yourTurn"] is False, "there is no 'you' in a watched game"
    assert payload["awaitingOpponent"] is True


def test_an_ordinary_game_is_not_watched():
    game = api.Game(opponent="hard", seed=11)
    payload = api.view(game)

    assert game.watching is False
    assert payload["watching"] is False
    assert payload["paceMs"] is None
    assert payload["watchedBy"] is None


def test_a_watched_game_hides_the_opponent_exactly_as_a_played_one_does():
    """Watching must not be a way to see more than a seat-1 player would."""
    game = api.Game(opponent="hard", watch="greedy", seed=7)
    for _ in range(40):
        if game.info["done"]:
            break
        game.advance()

    payload = api.view(game)
    opponent = [p for p in payload["players"] if p["id"] != payload["you"]][0]
    assert "cards" not in opponent
    assert "hand" not in opponent
    assert "handCount" in opponent


def test_an_unknown_watcher_is_refused():
    with pytest.raises(ValueError, match="unknown agent"):
        api.Game(opponent="hard", watch="telepathy", seed=1)


def test_a_watched_game_is_driven_over_http_and_is_not_recorded(server):
    """The client's existing loop drives a whole agent-versus-agent game unchanged, and
    games/ stays for the games a person actually played."""
    _, body, _ = request(server, "/api/game",
                         {"opponent": "hard", "watch": "greedy", "seed": 21})
    view = json.loads(body)
    game_id = view["gameId"]

    assert view["watching"] is True
    assert view["paceMs"] == api.WATCH_PACE_MS
    assert view["yourTurn"] is False
    from interfaces.web import server as web_server

    assert web_server.GAMES.get(game_id).recorder is None, "two bots are not a game to keep"

    for _ in range(2_000):
        if view["done"]:
            break
        assert view["awaitingOpponent"], "a watched game always owes a decision"
        _, body, _ = request(server, f"/api/game/{game_id}/advance", {})
        view = json.loads(body)

    assert view["done"], "the watched game never finished"
    assert view["log"], "nothing was written down to watch"


def test_a_watched_game_names_the_agents_instead_of_saying_you():
    """There is no "you" in a game nobody is playing, and two bots have to be told apart."""
    game = api.Game(opponent="hard", watch="greedy", seed=4242)
    for _ in range(2_000):
        if game.info["done"]:
            break
        game.advance()

    payload = api.view(game)
    text = " ".join(payload["log"])
    assert "You " not in text and "Opponent " not in text, text[:200]
    assert "greedy" in text or "hard" in text
    assert payload["phaseHint"].startswith(("greedy", "hard"))


def test_an_ordinary_game_still_addresses_the_player():
    game = api.Game(opponent="hard", seed=4242)
    game.play(game.info["legal"][0])
    text = " ".join(api.view(game)["log"])
    assert "You " in text
