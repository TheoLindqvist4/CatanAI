/* CatanIA — the browser client.
 *
 * It renders and reports clicks. It holds no rules: no legality, no board generation, no
 * scoring. Every one of those lives in Python, where it is tested — the last time this
 * project had board logic in JavaScript it was a second implementation that could disagree
 * with the engine.
 *
 * So the whole client is: fetch the static geometry once, draw whatever state the server
 * sends, and POST an action index when the player clicks something the server marked legal.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

/* Tints, used only for accents — the pieces themselves are the painted assets, whose colour
 * names arrive from the server in `geometry.art.colours` so the board and the PNG renderer
 * cannot drift apart. */
const PLAYER_COLOURS = { 1: "#d64545", 2: "#3b6fd4", 3: "#e08b2a", 4: "#4aa564" };

/* How long the opponent's moves are spaced out on screen.
 *
 * The engine decides a whole turn in well under a millisecond, so without this the
 * opponent's reply — four setup placements, or a roll and a robber move and three builds —
 * lands as a single repaint and the board has simply changed. A second apiece makes it a
 * sequence you can follow: the server plays one decision per request, so what you watch is
 * the order it actually happened in.
 *
 * This is the interface's own pacing and nothing else's. Training never loads this file.
 *
 * A *watched* game — both seats played by agents — is paced by the server instead, which
 * sends `paceMs`. One number in one place: the pace is a property of the game being watched,
 * not of this file. */
const OPPONENT_MOVE_MS = 1000;

const state = {
  geometry: null,
  view: null,
  gameId: null,
  mode: null,        // which board action type is armed
  busy: false,
  epoch: 0,          // bumped by a new game, so a running watch knows it is stale
  // The game whose result has already been announced. `render` runs after every single
  // decision, so without this the overlay would be rebuilt — and replay its animation —
  // on every repaint after the game ended.
  celebrated: null,
};

/** Asset file name for a player's pieces, e.g. 1 -> "red". */
const colourOf = (player) => (state.geometry.art.colours[player] || "black");

/** Sprite sizes, taken from the PNG renderer so both draw the board the same. */
const scale = (name) => state.geometry.art.scales[name];

/** Where a sprite sits relative to its tile's centre, in hex heights. Served rather than
 *  written here, for the same reason the sizes are: two renderers placing the same asset
 *  differently is a divergence nobody notices until they compare a screenshot to the game. */
const offset = (name) => state.geometry.art.offsets[name];

/** A sprite's own proportions, width over height. The PNG renderer scales by width alone and
 *  lets the file supply the rest; an SVG <image> takes both, so the ratio is served rather
 *  than written here, where it would be a guess at the shape of a file this never opens. */
const aspect = (name) => state.geometry.art.aspects[name];

/** Asset file name for a resource *card*, which is a different picture from the tile: the
 *  board shows a forest, the card in your hand shows one tree. Served, like every other
 *  asset name, so the client never has to know how a file is spelled. */
const cardImage = (name) => state.geometry.art.cards[name] || name;

/** Does this word name a resource? Asked of every word in a button label, to illustrate the
 *  ones that do — so it has to be the served list rather than one written here. */
const isResource = (name) => Boolean(state.geometry.art.cards[name]);

/** How tall a resource card is drawn: in a hand or the bank supply, and inline in a button
 *  label, where it has to sit in a line of text without setting the line height. */
const CARD_HEIGHT = { hand: 46, inline: 21 };

/** A picture of a resource card, as an <img> for the side panels.
 *
 *  Only the height is chosen here; the width follows from the art's own proportions, which
 *  the server sends. Passing both would be this file's own guess at the shape of a file it
 *  cannot see, and a card at the wrong ratio is a stretched card. */
function resourceCard(name, height) {
  const width = Math.round(height * aspect("card"));
  return `<img class="res-card" src="/images/ressources/${cardImage(name)}.png"` +
         ` alt="${name}" title="${name}" width="${width}" height="${height}">`;
}

/** How a development card reads on screen, and whether it can be played at all. */
const DEV_CARDS = {
  knight: { label: "Knight", icon: "⚔" },
  victory_point: { label: "Victory point", icon: "★", never: true },
  road_building: { label: "Road building", icon: "─" },
  year_of_plenty: { label: "Year of plenty", icon: "☘" },
  monopoly: { label: "Monopoly", icon: "◆" },
};

/** Your development cards, including the ones you cannot play.
 *
 * Worth showing in full: a Victory Point card is never playable but decides the game, and a
 * card bought this turn is unplayable only until the turn ends. Both are things you want to
 * see while deciding a move, and neither appears among the action buttons.
 */
function devCards(player) {
  if (!player.dev) {
    // An opponent's composition is hidden; only the count is public.
    const backs = Array.from({ length: Math.min(player.devCount, 10) },
      () => '<span class="dev-back"></span>').join("");
    return player.devCount
      ? `<div class="hand dev-hand">${backs}<span class="count">${player.devCount} dev</span></div>`
      : '<div class="muted">no development cards</div>';
  }

  const fresh = player.devNew || {};
  const chips = [];
  for (const [name, count] of Object.entries(player.dev)) {
    if (!count) continue;
    const meta = DEV_CARDS[name] || { label: name, icon: "?" };
    const pending = fresh[name] || 0;
    // Bought this turn: held, but not playable until the turn ends.
    if (count - pending > 0) chips.push(devChip(meta, count - pending, meta.never));
    if (pending > 0) chips.push(devChip(meta, pending, true, "bought this turn"));
  }
  return chips.length
    ? `<div class="hand dev-hand">${chips.join("")}</div>`
    : '<div class="muted">no development cards</div>';
}

function devChip(meta, count, waiting, why) {
  const note = why || (meta.never ? "counts toward victory, never played" : "");
  const title = `${meta.label}${count > 1 ? ` x${count}` : ""}${note ? ` — ${note}` : ""}`;
  return `<span class="dev-chip${waiting ? " dev-waiting" : ""}" title="${title}">` +
         `<span class="dev-icon">${meta.icon}</span>${meta.label}` +
         `${count > 1 ? `<b>${count}</b>` : ""}</span>`;
}

/** A resource count as the card itself with a number on it, rather than "wood 3". */
function resourceChip(name, count, muted) {
  return `<span class="chip${muted ? " chip-empty" : ""}" title="${name}: ${count}">` +
         `${resourceCard(name, CARD_HEIGHT.hand)}<b>${count}</b></span>`;
}

/* ---------------------------------------------------------------- server */

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `request failed: ${response.status}`);
  return body;
}

const getJSON = (path) => api(path);
const postJSON = (path, payload) =>
  api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/* ----------------------------------------------------------------- game */

/* Both of these hold `busy` until the opponent has finished replying, which is now a matter
 * of seconds rather than milliseconds — long enough that a click would otherwise land, be
 * refused by the server as out of turn, and replace the hint with an error the player did
 * nothing to deserve.
 *
 * `epoch` is what makes New game safe to press at any moment: a request already in flight
 * finds the epoch changed when it lands and drops its answer, instead of drawing the
 * abandoned game over the new one. */

async function newGame() {
  const epoch = ++state.epoch;
  state.busy = true;
  try {
    const seedField = document.getElementById("seed").value;
    const view = await postJSON("/api/game", {
      opponent: document.getElementById("opponent").value,
      // Empty string means "I am playing"; anything else names the agent that takes my
      // seat, which turns this into a game to watch rather than one to play.
      watch: (document.getElementById("watch") || {}).value || null,
      rules: document.getElementById("rules").value,
      seed: seedField === "" ? null : Number(seedField),
    });
    if (state.epoch !== epoch) return;
    state.view = view;
    state.gameId = view.gameId;
    state.mode = null;
    state.celebrated = null;
    closeOverlay("result");
    closeOverlay("stats-modal");
    render();
    await watchOpponent();
  } finally {
    if (state.epoch === epoch) state.busy = false;
  }
}

async function play(index) {
  if (state.busy) return;
  const epoch = state.epoch;
  state.busy = true;
  try {
    const view = await postJSON(`/api/game/${state.gameId}/action`, { index });
    if (state.epoch !== epoch) return;
    state.view = view;
    state.mode = null;
    render();
    await watchOpponent();
  } catch (error) {
    // The server refuses anything illegal, so this means the two disagreed — say so
    // loudly rather than leaving a click that quietly does nothing.
    if (state.epoch === epoch) setHint(`⚠ ${error.message}`);
  } finally {
    if (state.epoch === epoch) state.busy = false;
  }
}

/** Watch the opponent play, one move a second, drawing the board after each.
 *
 * The server decides whether a move is owed (`awaitingOpponent`) and plays exactly one per
 * request, so this neither knows the rules nor can get the order wrong: what is drawn is
 * the sequence the engine produced.
 *
 * Bounded rather than `while (true)`, in the same spirit as the server's own loop — a bug
 * that left the opponent moving forever should stop and say so, not poll until the tab is
 * closed. `epoch` is the other guard: a game started mid-watch retires the old watcher
 * rather than letting it draw the previous game over the new board.
 */
async function watchOpponent() {
  const epoch = state.epoch;
  for (let moves = 0; moves < 1000; moves += 1) {
    if (state.epoch !== epoch || !state.view || !state.view.awaitingOpponent) return;
    await sleep(state.view.paceMs || OPPONENT_MOVE_MS);
    if (state.epoch !== epoch) return;
    const view = await postJSON(`/api/game/${state.gameId}/advance`);
    if (state.epoch !== epoch) return;
    state.view = view;
    render();
  }
  setHint("⚠ the opponent would not stop moving");
}

/* --------------------------------------------------------------- drawing */

function el(name, attrs, parent) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs || {})) {
    node.setAttribute(key, value);
  }
  if (parent) parent.appendChild(node);
  return node;
}

function boardTargets(type) {
  return (state.view && state.view.actions.board[type]) || {};
}

/** Board action types offered right now, in the order they should appear. */
function availableModes() {
  const order = ["BUILD_SETTLEMENT", "BUILD_ROAD", "BUILD_CITY", "MOVE_ROBBER"];
  return order.filter((type) => Object.keys(boardTargets(type)).length > 0);
}

function render() {
  const view = state.view;
  if (!view) return;

  // With one board action available there is nothing to choose between, so arm it
  // automatically — during setup that means you just click a spot.
  const modes = availableModes();
  if (state.mode && !modes.includes(state.mode)) state.mode = null;
  if (!state.mode && modes.length === 1) state.mode = modes[0];

  drawBoard();
  drawStatus();
  drawModes(modes);
  drawPanel();
  drawLog();
  setHint(view.phaseHint);

  // Last, so the board underneath is already the final position when the result lands on
  // top of it.
  if (view.done && state.celebrated !== view.gameId) {
    state.celebrated = view.gameId;
    showResult(view);
  }
}

function drawBoard() {
  const { geometry, view } = state;
  const svg = document.getElementById("board");
  svg.setAttribute("viewBox", `0 0 ${geometry.width} ${geometry.height}`);
  svg.innerHTML = "";

  const tileById = Object.fromEntries(view.board.tiles.map((t) => [t.id, t]));
  const buildingAt = Object.fromEntries(view.pieces.buildings.map((b) => [b.vertex, b]));
  const roadOn = Object.fromEntries(view.pieces.roads.map((r) => [r.road, r.player]));

  el("rect", { x: 0, y: 0, width: geometry.width, height: geometry.height, fill: "#244e78" }, svg);

  // tiles, then number tokens on top
  for (const spot of geometry.tiles) {
    const tile = tileById[spot.id];
    el("image", {
      href: `/images/tiles/${tile.resource}.png`,
      x: spot.x - geometry.hexWidth / 2,
      y: spot.y - geometry.hexHeight / 2,
      width: geometry.hexWidth,
      height: geometry.hexHeight,
      "data-tile": spot.id,
    }, svg);
  }
  for (const spot of geometry.tiles) {
    const tile = tileById[spot.id];
    if (tile.number === null) continue;
    // The art has a blank panel cut out of it for the token, and it is not in the middle of
    // the hex — the sheaf or the tree is drawn above it. `offsets.number` is where its
    // centre is; see interfaces/render.py::NUMBER_OFFSET.
    const size = geometry.hexWidth * scale("number");
    el("image", {
      href: `/images/numbers/${tile.number}.png`,
      x: spot.x - size / 2,
      y: spot.y - size / 2 + geometry.hexHeight * offset("number"),
      width: size,
      height: size,
    }, svg);
  }

  drawHarbours(svg);

  // The robber, standing above the number token rather than over it — see
  // render.ROBBER_OFFSET, which is why the offset is served rather than assumed.
  const robber = geometry.tiles.find((t) => t.id === view.robber);
  if (robber) {
    const width = geometry.hexWidth * scale("robber");
    const height = width / aspect("robber");
    el("image", {
      href: "/images/robber/robber.png",
      x: robber.x - width / 2,
      y: robber.y - height / 2 + geometry.hexHeight * offset("robber"),
      width, height,
      class: "piece",
    }, svg);
  }

  drawRoads(svg, roadOn);
  drawVertices(svg, buildingAt);
}

function drawHarbours(svg) {
  const { geometry, view } = state;
  const byId = Object.fromEntries(geometry.roads.map((r) => [r.id, r]));
  const midX = geometry.width / 2;
  const midY = geometry.height / 2;

  for (const harbour of view.board.harbours) {
    const road = byId[harbour.road];
    if (!road) continue;
    // push the marker out to sea so it does not sit on top of the land
    const dx = road.cx - midX;
    const dy = road.cy - midY;
    const span = Math.hypot(dx, dy) || 1;
    const push = geometry.hexWidth * 0.3;
    const x = road.cx + (dx / span) * push;
    const y = road.cy + (dy / span) * push;

    el("circle", {
      cx: x, cy: y, r: geometry.hexWidth * 0.155,
      fill: "#f7f0d6", stroke: "#46341c", "stroke-width": 2,
    }, svg);
    const label = el("text", {
      x, y, "text-anchor": "middle", "dominant-baseline": "central",
      "font-size": geometry.hexWidth * 0.11, fill: "#2d2109",
    }, svg);
    label.textContent = harbour.kind.replace(" ", " ");
  }
}

function drawRoads(svg, roadOn) {
  const { geometry } = state;
  const targets = boardTargets("BUILD_ROAD");
  const armed = state.mode === "BUILD_ROAD";

  for (const road of geometry.roads) {
    const owner = roadOn[road.id];
    const index = targets[road.id];

    if (owner) {
      // The road assets are painted vertically and the geometry gives the angle from
      // vertical, which is the same convention interfaces/render.py uses for the PNG.
      const length = geometry.edge * scale("roadLength");
      const width = length * (56 / 225);          // the sprites are 56x225
      el("image", {
        href: `/images/roads/${colourOf(owner)}_road.png`,
        x: road.cx - width / 2, y: road.cy - length / 2,
        width, height: length,
        transform: `rotate(${road.angle} ${road.cx} ${road.cy})`,
      }, svg);
    } else if (armed && index !== undefined) {
      el("line", {
        x1: road.x1, y1: road.y1, x2: road.x2, y2: road.y2,
        stroke: "#ffe680", "stroke-width": geometry.hexWidth * 0.05,
        "stroke-linecap": "round", "stroke-dasharray": "6 5", class: "target",
      }, svg);
    }

    if (armed && index !== undefined) {
      // a fat invisible line, so the click target is bigger than the drawn road
      const hit = el("line", {
        x1: road.x1, y1: road.y1, x2: road.x2, y2: road.y2,
        stroke: "transparent", "stroke-width": geometry.hexWidth * 0.16,
        class: "hit",
      }, svg);
      hit.addEventListener("click", () => play(index));
    }
  }
}

function drawVertices(svg, buildingAt) {
  const { geometry } = state;
  const settlementTargets = boardTargets("BUILD_SETTLEMENT");
  const cityTargets = boardTargets("BUILD_CITY");
  const mode = state.mode;

  for (const spot of geometry.vertices) {
    const building = buildingAt[spot.id];
    if (building) {
      const city = building.kind === "city";
      const colour = colourOf(building.player);
      // sprites are 226x201 (cities) and 160x133 (settlements)
      const width = geometry.hexWidth * scale(city ? "city" : "settlement");
      const height = width * (city ? 201 / 226 : 133 / 160);
      el("image", {
        href: city ? `/images/cities/${colour}_city.png`
                   : `/images/settlements/${colour}.png`,
        x: spot.x - width / 2, y: spot.y - height / 2,
        width, height,
        class: "piece",
      }, svg);
    }

    const index =
      mode === "BUILD_SETTLEMENT" ? settlementTargets[spot.id]
      : mode === "BUILD_CITY" ? cityTargets[spot.id]
      : undefined;
    if (index === undefined) continue;

    const marker = geometry.hexWidth * scale("spot");
    el("image", {
      href: "/images/spots/circle.png",
      x: spot.x - marker / 2, y: spot.y - marker / 2,
      width: marker, height: marker,
      class: "target",
    }, svg);
    const hit = el("circle", {
      cx: spot.x, cy: spot.y, r: geometry.hexWidth * 0.16,
      fill: "transparent", class: "hit",
    }, svg);
    hit.addEventListener("click", () => play(index));
  }

  if (mode === "MOVE_ROBBER") {
    const targets = boardTargets("MOVE_ROBBER");
    for (const spot of geometry.tiles) {
      const index = targets[spot.id];
      if (index === undefined) continue;
      el("circle", {
        cx: spot.x, cy: spot.y, r: geometry.hexWidth * 0.3,
        fill: "rgba(255,230,128,0.28)", stroke: "#fff2ad",
        "stroke-width": 3, class: "target",
      }, svg);
      const hit = el("circle", {
        cx: spot.x, cy: spot.y, r: geometry.hexWidth * 0.34,
        fill: "transparent", class: "hit",
      }, svg);
      hit.addEventListener("click", () => play(index));
    }
  }
}

/* ---------------------------------------------------------------- panels */

function drawStatus() {
  const view = state.view;
  document.getElementById("dice").textContent =
    view.lastRoll === null ? "–" : view.lastRoll;
  document.getElementById("turn-label").textContent = `turn ${view.turn}`;
  document.getElementById("phase-label").textContent =
    `${view.phase.toLowerCase().replace(/_/g, " ")} · first to ${view.victoryTarget}`;

  const holder = document.getElementById("players");
  holder.innerHTML = "";
  for (const player of view.players) {
    const card = document.createElement("div");
    card.className = "player" + (player.you ? " you" : "");
    card.style.borderLeftColor = PLAYER_COLOURS[player.id];

    const badges = [];
    if (player.largestArmy) badges.push("largest army");
    if (player.longestRoadHolder) badges.push("longest road");

    const points = player.victoryPoints === undefined
      ? `${player.publicVictoryPoints} public vp`
      : `${player.victoryPoints} vp`;

    // Your own hand is shown card by card; an opponent's is hidden information, so all
    // that can be drawn is the right number of card backs.
    let cards;
    if (player.hand) {
      const chips = Object.entries(player.hand)
        .map(([name, n]) => resourceChip(name, n, n === 0))
        .join("");
      cards = `<div class="hand">${chips}</div>`;
    } else {
      const backs = Array.from({ length: Math.min(player.handCount, 14) },
        () => '<span class="card-back"></span>').join("");
      cards = `<div class="hand hidden-hand">${backs}` +
              `<span class="count">${player.handCount}</span></div>`;
    }

    card.innerHTML = `
      <div class="player-head">
        <strong>${player.you ? "You" : "Opponent"}</strong>
        <span>${points}</span>
      </div>
      ${cards}
      ${devCards(player)}
      <div class="muted">${player.knights} knights · road ${player.longestRoad}</div>
      <div class="muted">left: ${player.settlementsLeft}s ${player.citiesLeft}c
        ${player.roadsLeft}r</div>
      ${badges.length ? `<div class="badges">${badges.join(" · ")}</div>` : ""}
    `;
    holder.appendChild(card);
  }
}

function drawModes(modes) {
  const holder = document.getElementById("modes");
  holder.innerHTML = "";
  const labels = {
    BUILD_SETTLEMENT: "Settlement",
    BUILD_ROAD: "Road",
    BUILD_CITY: "City",
    MOVE_ROBBER: "Move robber",
  };
  if (!modes.length) {
    holder.innerHTML = '<span class="muted">nothing to place</span>';
    return;
  }
  for (const mode of modes) {
    const button = document.createElement("button");
    button.textContent = `${labels[mode]} (${Object.keys(boardTargets(mode)).length})`;
    button.className = state.mode === mode ? "active" : "";
    button.addEventListener("click", () => {
      state.mode = state.mode === mode ? null : mode;
      render();
    });
    holder.appendChild(button);
  }
}

function drawPanel() {
  const holder = document.getElementById("panel");
  holder.innerHTML = "";
  const actions = state.view.actions.panel;
  if (!actions.length) {
    holder.innerHTML = '<span class="muted">nothing available</span>';
  }
  for (const action of actions) {
    const button = document.createElement("button");
    // Illustrate whichever resources the label names, so a row of trades can be read at a
    // glance instead of parsed. Word by word rather than by regular expression: a `\b` in
    // a template literal is a backspace character, not a word boundary, and the resulting
    // pattern matches nothing while looking entirely correct.
    button.innerHTML = action.label
      .split(" ")
      .map((word) => {
        const name = word.toLowerCase();
        return isResource(name)
          ? `${resourceCard(name, CARD_HEIGHT.inline)}${word}`
          : word;
      })
      .join(" ");
    if (action.type === "END_TURN") button.className = "primary";
    button.addEventListener("click", () => play(action.index));
    holder.appendChild(button);
  }

  const view = state.view;
  document.getElementById("supply").innerHTML =
    `<div class="hand">` +
    Object.entries(view.bank).map(([name, n]) => resourceChip(name, n, n === 0)).join("") +
    `</div><div class="muted">dev deck ${view.devDeck}</div>`;
}

function drawLog() {
  const list = document.getElementById("log");
  list.innerHTML = "";
  const lines = state.view.log.slice().reverse();
  lines.forEach((line, index) => {
    const item = document.createElement("li");
    item.textContent = line;
    const classes = [];
    if (line.startsWith("You")) classes.push("mine");
    // A steal or a draw is the outcome a player most wants confirmed, and the easiest to
    // miss at the end of a long log.
    if (line.includes(" stole ") || line.includes("development card:")) {
      classes.push("notable");
    }
    if (index === 0) classes.push("latest");
    item.className = classes.join(" ");
    list.appendChild(item);
  });
}

function setHint(text) {
  document.getElementById("hint").textContent = text;
}

/* -------------------------------------------------------------- overlays */

function openOverlay(id, html) {
  const holder = document.getElementById(id);
  holder.innerHTML = html;
  holder.hidden = false;
  holder.setAttribute("role", "dialog");
  holder.setAttribute("aria-modal", "true");
  // Clicking the dimmed area behind the sheet closes it, which is what everyone tries
  // first. Clicks inside the sheet must not, so this only fires on the backdrop itself.
  holder.onclick = (event) => { if (event.target === holder) closeOverlay(id); };
  for (const button of holder.querySelectorAll("[data-close]")) {
    button.addEventListener("click", () => closeOverlay(id));
  }
  // So Escape and Enter land somewhere sensible rather than on whatever was focused
  // before the game ended.
  const first = holder.querySelector("button");
  if (first) first.focus();
}

function closeOverlay(id) {
  const holder = document.getElementById(id);
  holder.hidden = true;
  holder.innerHTML = "";
}

/* ---------------------------------------------------------- the result */

/** Announce the end of the game.
 *
 * The board goes quiet by itself when a game ends — nothing left to place, no actions
 * offered — so without this the result is one grey line in the hint bar and it is genuinely
 * easy to miss that you have won.
 *
 * Purely a matter of the interface. Training never loads this file, and the server is not
 * asked for anything extra: `done`, `winner` and both final scores are already in the view
 * that drew the last move.
 */
function showResult(view) {
  const you = view.players.find((player) => player.you);
  const them = view.players.find((player) => !player.you);

  // There is no "you" in a game nobody is playing, so a watched game is told who won
  // rather than congratulated.
  let outcome;
  let heading;
  let badge;
  if (view.winner === null) {
    outcome = "neutral";
    heading = "No winner";
    badge = "◇";
  } else if (view.watching) {
    outcome = "neutral";
    heading = `${view.winner === view.you ? view.watchedBy : view.opponent} wins`;
    badge = "◆";
  } else if (view.winner === view.you) {
    outcome = "won";
    heading = "You won!";
    badge = "🏆";
  } else {
    outcome = "lost";
    heading = "You lost";
    badge = "🛡";
  }

  const side = (player, name) => `
    <div class="score-side" style="border-top-color: ${PLAYER_COLOURS[player.id]}">
      <div class="who">${name}</div>
      <div class="points">${player.victoryPoints}</div>
    </div>`;

  const names = view.watching
    ? { you: view.watchedBy, them: view.opponent }
    : { you: "You", them: "Opponent" };

  // The confetti is a sibling of the sheet, not a child: it falls the height of the
  // window, and the sheet is a scroll box that would otherwise be its containing block.
  openOverlay("result", `
    ${outcome === "won" ? confetti() : ""}
    <div class="sheet result ${outcome}">
      <span class="result-badge">${badge}</span>
      <h2>${heading}</h2>
      <div class="result-sub">${view.turn} turns · first to ${view.victoryTarget}</div>
      <div class="result-score">
        ${side(you, names.you)}
        <span class="score-dash">–</span>
        ${side(them, names.them)}
      </div>
      <div class="result-actions">
        <button data-stats>Game stats</button>
        <button class="primary" data-new-game>New game</button>
        <button data-close>Close</button>
      </div>
    </div>
  `);

  const holder = document.getElementById("result");
  holder.querySelector("[data-stats]")
    .addEventListener("click", () => openStats().catch(report));
  holder.querySelector("[data-new-game]").addEventListener("click", () => {
    closeOverlay("result");
    newGame().catch(report);
  });
}

/** Decoration, and nothing else: hidden from assistive technology and inert to the mouse.
 *
 * Each piece gets its own column, delay, duration and colour, because forty identical
 * pieces falling in step reads as a loading bar rather than as a celebration. */
function confetti(pieces = 40) {
  const colours = ["#d8b44a", "#4aa564", "#3b6fd4", "#d64545", "#e8e8ea"];
  const spans = [];
  for (let i = 0; i < pieces; i += 1) {
    const left = Math.random() * 100;
    const delay = Math.random() * 1.6;
    const duration = 2.4 + Math.random() * 1.8;
    const colour = colours[i % colours.length];
    spans.push(
      `<i style="left:${left.toFixed(2)}%;background:${colour};` +
      `animation-delay:${delay.toFixed(2)}s;animation-duration:${duration.toFixed(2)}s"></i>`);
  }
  return `<div class="confetti" aria-hidden="true">${spans.join("")}</div>`;
}

/* ------------------------------------------------------- the statistics */

/** Fetch and draw the whole game in numbers.
 *
 * Its own request rather than another block on the view: it is asked for by a click and not
 * four times a turn, so nothing here is on the path of a move.
 */
async function openStats() {
  if (!state.gameId) return;
  // The same guard every other request carries: New game is pressable at any moment, and a
  // report that lands afterwards belongs to a game that is no longer on the board.
  const epoch = state.epoch;
  const report = await getJSON(`/api/game/${state.gameId}/stats`);
  if (state.epoch !== epoch) return;
  openOverlay("stats-modal", `
    <div class="sheet stats-sheet">
      <div class="sheet-head">
        <div>
          <h2>Game statistics</h2>
          <div class="muted">${report.turns} turns · ${report.rolls} rolls</div>
        </div>
        <button class="close" data-close title="Close">✕</button>
      </div>
      <div class="sheet-body">
        ${diceSection(report)}
        ${playerSection(report)}
        ${pointsSection(report)}
      </div>
    </div>
  `);
}

/** How often each total has come up, and who rolled it.
 *
 * One row per total from 2 to 12, always — a number that never came up keeps its row,
 * because a gap in the middle of a histogram reads as missing data rather than as never.
 *
 * The reference tick is what a fair deck of 36 would have given over the same number of
 * rolls. Without it a bar chart of dice says only "6 came up a lot", which is what 6 is
 * supposed to do.
 */
function diceSection(report) {
  const totals = report.dice.totals;
  const expected = report.dice.expected;
  const byPlayer = report.dice.byPlayer;
  const ids = report.players.map((player) => player.id);

  // Scaled to whichever is longer, so the reference tick is never off the end of the track.
  const scale = Math.max(
    1,
    ...Object.values(totals),
    ...Object.values(expected)) * 1.02;

  const rows = Object.keys(totals).map((total) => {
    const count = totals[total];
    const segments = ids
      .filter((id) => byPlayer[id][total] > 0)
      .map((id) => `<span class="seg" style="width:${(byPlayer[id][total] / scale) * 100}%;` +
                   `background:${PLAYER_COLOURS[id]}"></span>`)
      .join("");

    // The whole row in one hover: the split between the players is the part a colour alone
    // would be carrying, so it is said in words here as well.
    const split = report.players
      .map((player) => `${player.name} ${byPlayer[player.id][total]}`)
      .join(", ");

    return `
      <div class="dice-row${Number(total) === 7 ? " seven" : ""}">
        <span class="total">${total}</span>
        <span class="dice-track"
              title="${total}: rolled ${count} (${split}) · expected about ${expected[total]}">
          <span class="dice-bar">${segments}</span>
          <span class="dice-expected" style="left:${(expected[total] / scale) * 100}%"></span>
        </span>
        <span class="count">${count}</span>
      </div>`;
  }).join("");

  const keys = report.players
    .map((player) => `<span><i style="background:${PLAYER_COLOURS[player.id]}"></i>` +
                     `${player.name} rolled</span>`)
    .join("");

  return `
    <section class="stats-section">
      <h3>Dice distribution</h3>
      <div class="legend">
        ${keys}
        <span><i class="expected"></i>expected over ${report.rolls} rolls</span>
      </div>
      <div class="dice-chart">${rows}</div>
      <div class="points-note">
        ${report.dice.balanced
          ? "This ruleset deals a shuffled deck of all 36 dice pairs, so the totals track the expectation closely by design."
          : "Plain dice: each roll is independent, so a run away from the expectation is luck rather than a bug."}
      </div>
    </section>`;
}

function playerSection(report) {
  return `
    <section class="stats-section">
      <h3>Per player</h3>
      <div class="stat-columns">
        ${report.players.map(playerStats).join("")}
      </div>
    </section>`;
}

/** A row of resource cards with a count on each, the way a hand reads everywhere else. */
function resourceRow(amounts) {
  const cards = Object.entries(amounts)
    .map(([name, count]) =>
      `<span class="res-count${count ? "" : " zero"}" title="${name}: ${count}">` +
      `${resourceCard(name, CARD_HEIGHT.inline)}${count}</span>`)
    .join("");
  return `<div class="res-row">${cards}</div>`;
}

function statRow(label, value, title) {
  return `<div class="stat-row"${title ? ` title="${title}"` : ""}>` +
         `<span>${label}</span><b>${value}</b></div>`;
}

function playerStats(entry) {
  const cards = (n) => `${n} card${n === 1 ? "" : "s"}`;
  return `
    <div class="stat-card" style="border-left-color:${PLAYER_COLOURS[entry.id]}">
      <h4>${entry.name}</h4>

      ${statRow("Rolls made", entry.rolled)}
      ${statRow("7s rolled", entry.sevensRolled,
                "Sevens this player rolled themselves")}
      ${statRow("Knights played", entry.knights)}
      ${statRow("Robber moved", entry.robberMoves,
                "From a 7 or from playing a Knight")}

      <div class="stat-group">
        <h5>Caught by a 7</h5>
        ${statRow("Times over the limit", entry.sevenOuts,
                  "How many separate 7s cost them cards")}
        ${statRow("Cards discarded", cards(entry.cardsDiscarded))}
        ${entry.cardsDiscarded ? resourceRow(entry.discarded) : ""}
      </div>

      <div class="stat-group">
        <h5>Blocked by the robber</h5>
        ${statRow("Production denied", cards(entry.blockedTotal),
                  "Cards the tile under the robber would have paid them")}
        ${entry.blockedTotal ? resourceRow(entry.blocked) : ""}
      </div>

      <div class="stat-group">
        <h5>Cards produced (${entry.producedTotal})</h5>
        ${resourceRow(entry.produced)}
      </div>

      <div class="stat-group">
        <h5>Other movements</h5>
        ${statRow("Stolen by them", cards(sum(entry.stolenBy)))}
        ${statRow("Stolen from them", cards(sum(entry.stolenFrom)))}
        ${statRow("Taken by monopoly", cards(sum(entry.monopolised)))}
        ${statRow("Development cards bought", entry.devBought)}
        ${statRow("Cards spent building", cards(sum(entry.spent)))}
      </div>
    </div>`;
}

const sum = (amounts) => Object.values(amounts).reduce((a, b) => a + b, 0);

/** Where every point came from, per player.
 *
 * The question a losing scoreboard actually raises: 15–3 says nothing about whether the
 * game was lost to cities or to a road nobody contested.
 */
function pointsSection(report) {
  const sources = ["settlements", "cities", "longest road", "largest army",
                   "victory point cards"];
  const rowFor = (name) => {
    const cells = report.players.map((player) => {
      const found = player.pointSources.find((row) => row.source === name);
      if (!found) return `<td class="num zero" title="hidden until the game ends">?</td>`;
      const detail = found.count && found.count !== found.points
        ? ` <span class="muted">(${found.count})</span>` : "";
      const held = found.held ? " held" : "";
      return `<td class="num${found.points ? held : " zero"}">${found.points}${detail}</td>`;
    }).join("");
    return `<tr><td class="source">${name}</td>${cells}</tr>`;
  };

  const totals = report.players
    .map((player) => `<td class="num">${player.victoryPoints}` +
                     `${player.victoryPointsComplete ? "" : "+"}</td>`)
    .join("");

  const incomplete = report.players.some((player) => !player.victoryPointsComplete);

  return `
    <section class="stats-section">
      <h3>Where the points came from</h3>
      <table class="points-table">
        <thead>
          <tr>
            <th>Source</th>
            ${report.players.map((player) => `<th class="num">${player.name}</th>`).join("")}
          </tr>
        </thead>
        <tbody>
          ${sources.map(rowFor).join("")}
          <tr class="total"><td>Total</td>${totals}</tr>
        </tbody>
      </table>
      ${incomplete ? `<div class="points-note">A Victory Point card is held rather than
        played, so it stays hidden until the game ends — the totals marked
        <b>+</b> are the public score and may be short.</div>` : ""}
    </section>`;
}

/* ------------------------------------------------------------------ boot */

const report = (error) => setHint(`⚠ ${error.message}`);

/* The opponent list comes from the server, because which opponents exist is not fixed: the
 * learned one appears only when a champion actually loads, and a champion trained against a
 * different observation layout does not. A hardcoded <option> would still be clickable and
 * the server would refuse it with a 400 the player did nothing to deserve. */
function fillOpponents(available) {
  const select = document.getElementById("opponent");
  select.replaceChildren();
  for (const { name, label, default: isDefault } of available) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = label;
    option.selected = isDefault;
    select.append(option);
  }

  /* The same list again for your own seat, plus an empty first entry meaning "I play".
   * Two agents can then be put against each other and watched — the champion against the
   * heuristic, or against the previous lineage — without any of it being decided here. */
  const watch = document.getElementById("watch");
  if (!watch) return;
  watch.replaceChildren();
  const playing = document.createElement("option");
  playing.value = "";
  playing.textContent = "I play";
  playing.selected = true;
  watch.append(playing);
  for (const { name, label } of available) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `Watch ${label}`;
    watch.append(option);
  }
}

async function start() {
  state.geometry = await getJSON("/api/geometry");
  // The stylesheet draws an opponent's card backs, the one card here not coming from a
  // file. Handing it the ratio and the height leaves it nothing to guess: two hands in one
  // panel have to be the same deck, and a back sized by its own pair of pixel counts stops
  // matching the moment either of these changes.
  const style = document.documentElement.style;
  style.setProperty("--card-aspect", aspect("card"));
  style.setProperty("--card-height", `${CARD_HEIGHT.hand}px`);
  fillOpponents(state.geometry.opponents);
  // Not `newGame` itself: it now waits for the opponent, so a failure part way through is
  // a rejected promise the click handler would drop on the floor.
  document.getElementById("new-game")
    .addEventListener("click", () => newGame().catch(report));
  document.getElementById("stats")
    .addEventListener("click", () => openStats().catch(report));
  // Escape closes whichever overlay is open, because every other dialog on the web does.
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    closeOverlay("stats-modal");
    closeOverlay("result");
  });
  await newGame();
}

start().catch(report);
