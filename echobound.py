"""
Echobound - an echolocation maze game built with pygame.

Core idea: the player cannot see the maze. Pressing SPACE emits a sound wave
(a ring of rays) that travels outward, reflects off walls, and briefly lights
up nearby surfaces before the screen fades back to black. The player must
navigate a procedurally generated maze to the exit by memory, while an enemy
that hunts by sound chases them using A* pathfinding.

Algorithms used (mapped to course lectures):
  - Recursive Backtracker maze generation        (Lecture 3)
  - Raycasting expanding "sonar" wave-front        (Lecture 9)
  - A* pathfinding - used two opposite ways:       (Lecture 4)
      * the enemy chases the last position it HEARD you ping
      * the exit "beacon" guides you (closer = higher pitch + faster)
  - BFS shortest-path distance for the exit beacon (Lecture 4)
  - AABB / grid collision detection               (Lecture 5)
  - Stereo panning + distance falloff for footsteps (Lecture 7)
  - Frame-to-frame interpolation (smooth movement) (Lecture 1)
  - Standard input/update/render game loop         (Lecture 1)

Mechanics: pinging costs energy and has a cooldown (no spamming); the enemy
hunts by sound rather than omniscience; sound generated procedurally (no assets).

Run normally:         python echobound.py
Deterministic maze:   python echobound.py --seed 7
Tune difficulty:      python echobound.py --render 320 --fade 3 --enemy-speed 14
Generate screenshots: python echobound.py --shots
Web build (pygbag):   pip install pygbag && pygbag echobound.py  (then open localhost:8000)

The game loop is async (await asyncio.sleep(0) each frame) and all audio is
optional/guarded, so it runs in the browser via pygbag/WebAssembly as well as
on the desktop. No external asset files or local file writes are used.
"""

import sys
import math
import random
import heapq
import asyncio
import argparse
from collections import deque

import pygame

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CELL = 32                 # pixel size of one maze cell
COLS, ROWS = 21, 15       # maze dimensions in cells (odd numbers work best)
WIDTH, HEIGHT = COLS * CELL, ROWS * CELL
FPS = 60

# Recursive backtracker makes long winding corridors with few junctions, which
# is dull for echolocation. "Braiding" knocks out extra walls to add loops and
# branch points, giving the player real navigation decisions (Lecture 3).
BRAID_RATE = 0.10         # chance to open each eligible interior wall

NUM_RAYS = 180            # rays cast per echo "ping"
ECHO_SPEED = 6.0          # how fast the echo ring expands (pixels/frame)
ECHO_MAX = 260            # max echo radius before it dies
FADE_PER_FRAME = 6        # how fast lit surfaces fade back to black

# Ping economy: restores tension by preventing "ping spamming"
ENERGY_MAX = 100.0        # full energy meter
PING_COST = 20.0          # energy spent per echo
COOLDOWN_FRAMES = 30      # min frames between pings (~0.5s at 60 FPS)
ENERGY_REGEN = 0.2        # energy recovered per frame

# Smooth movement (#4): pixel positions interpolate toward the logical cell
MOVE_LERP = 0.25          # fraction of remaining distance covered per frame

# Expanding wave-front (#5): only walls near the ring's radius light up
WAVE_THICKNESS = 26       # pixels of tolerance around the current radius

# Enemy perception (#6): the enemy hunts by sound, not omniscience
ENEMY_MOVE_FRAMES = 18    # frames per enemy step (smaller = faster)
ENEMY_HEAR_CELLS = 8      # how far (in cells) a ping is heard by the enemy

# Difficulty presets (PCG parameter tuning, Lecture 8): (name, enemy_base, fade_base)
DIFFICULTIES = [("Easy", 26, 4), ("Normal", 18, 6), ("Hard", 12, 8)]

# Audio
SAMPLE_RATE = 44100
HEAR_RANGE = 6 * CELL     # pixels within which the enemy's footsteps are audible
STEP_INTERVAL = 22        # frames between footstep sounds
BEACON_MIN_INTERVAL = 16  # frames between exit beeps when very close
BEACON_MAX_INTERVAL = 110 # frames between exit beeps when far away
BEACON_MAX_DIST = 28      # path length (in cells) treated as "far" for the beacon

SEED = None               # base maze seed (set via --seed); None = random each run

BLACK = (0, 0, 0)
PLAYER_COLOR = (90, 200, 255)
ENEMY_COLOR = (255, 80, 90)
EXIT_COLOR = (120, 255, 140)


# ---------------------------------------------------------------------------
# Maze generation - Recursive Backtracker (Lecture 3)
# ---------------------------------------------------------------------------
def generate_maze(cols, rows):
    """Return a 2D grid: True = wall, False = open. Recursive backtracker."""
    grid = [[True for _ in range(cols)] for _ in range(rows)]
    stack = [(1, 1)]
    grid[1][1] = False
    while stack:
        cx, cy = stack[-1]
        neighbors = []
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < cols - 1 and 1 <= ny < rows - 1 and grid[ny][nx]:
                neighbors.append((nx, ny, dx, dy))
        if neighbors:
            nx, ny, dx, dy = random.choice(neighbors)
            grid[cy + dy // 2][cx + dx // 2] = False   # knock down wall between
            grid[ny][nx] = False
            stack.append((nx, ny))
        else:
            stack.pop()

    # Braiding: open extra walls that sit between two corridors, adding loops
    # and junctions so the maze isn't just one long winding path (Lecture 3).
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if not grid[y][x] or random.random() >= BRAID_RATE:
                continue
            horiz = not grid[y][x - 1] and not grid[y][x + 1]
            vert = not grid[y - 1][x] and not grid[y + 1][x]
            if horiz or vert:          # only open walls that connect two passages
                grid[y][x] = False
    return grid


# ---------------------------------------------------------------------------
# A* pathfinding for the enemy (Lecture 4)
# ---------------------------------------------------------------------------
def astar(grid, start, goal):
    """Grid A* with Manhattan heuristic. Returns next step toward goal, or None."""
    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(h(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path, return the first step after start
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path[1] if len(path) > 1 else None
        cx, cy = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid) and not grid[ny][nx]:
                tentative = g + 1
                if tentative < g_score.get((nx, ny), 1e9):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative
                    f = tentative + h((nx, ny), goal)
                    heapq.heappush(open_set, (f, tentative, (nx, ny)))
    return None


def bfs_dist(grid, start, goal):
    """Shortest path length (in cells) between two open cells, or None."""
    if start == goal:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        (cx, cy), d = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in seen:
                continue
            if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid) and not grid[ny][nx]:
                if (nx, ny) == goal:
                    return d + 1
                seen.add((nx, ny))
                q.append(((nx, ny), d + 1))
    return None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Glow sprites - soft radial light with distance falloff (Lecture 10).
# Instead of flat dots we blit pre-rendered glows so lit surfaces, the player
# and the enemy read as light blooming out of the darkness.
# ---------------------------------------------------------------------------
_WALL_GLOWS = None        # list of intensity buckets for lit wall points
_PLAYER_GLOW = None
_ENEMY_GLOW = None
_EXIT_GLOW = None
WALL_GLOW_BUCKETS = 8


def make_glow(radius, color, max_alpha=255):
    """Return an SRCALPHA surface with a quadratic radial alpha falloff."""
    size = radius * 2 + 1
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    for r in range(radius, 0, -1):
        t = 1.0 - r / radius                       # 0 at edge, 1 at center
        a = int(max_alpha * t * t)                 # quadratic falloff (Lec 10)
        pygame.draw.circle(surf, (*color, a), (radius, radius), r)
    return surf


def build_glow_assets():
    """Pre-render glow sprites once a display/surface exists."""
    global _WALL_GLOWS, _PLAYER_GLOW, _ENEMY_GLOW, _EXIT_GLOW
    if _WALL_GLOWS is not None:
        return
    _WALL_GLOWS = []
    for i in range(WALL_GLOW_BUCKETS):
        a = int(60 + 195 * i / (WALL_GLOW_BUCKETS - 1))
        _WALL_GLOWS.append(make_glow(7, (150, 190, 255), a))
    _PLAYER_GLOW = make_glow(16, PLAYER_COLOR, 200)
    _ENEMY_GLOW = make_glow(18, ENEMY_COLOR, 220)
    _EXIT_GLOW = make_glow(16, EXIT_COLOR, 210)


# ---------------------------------------------------------------------------
# Audio: procedurally-generated tones (no asset files needed). Degrades
# gracefully to silence if numpy or an audio device is unavailable.
# ---------------------------------------------------------------------------
class Audio:
    def __init__(self):
        # All audio is optional: browsers (pygbag/WebAssembly) have limited mixer
        # support and only allow sound after a user gesture. Any failure here must
        # leave self.ok == False so the game runs silently instead of crashing.
        self.ok = False
        self._beacon_cache = {}
        self.step_sound = None
        self.ping_sound = None
        if not _HAS_NUMPY:
            return
        try:
            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
            self.step_sound = self._make_step()
            self.ping_sound = self._make_ping()
            self.ok = True
        except Exception:           # noqa: BLE001 - any audio error -> stay silent
            self.ok = False

    def _make(self, freq, ms, vol=0.5, kind="sine"):
        """Build a short pygame Sound (stereo int16) for a tone or noise burst."""
        n = int(SAMPLE_RATE * ms / 1000)
        if kind == "noise":
            wave = np.random.uniform(-1.0, 1.0, n)
        else:
            t = np.arange(n)
            wave = np.sin(2 * np.pi * freq * t / SAMPLE_RATE)
        # short attack/release envelope to avoid clicks
        env = np.ones(n)
        a = max(1, int(n * 0.12))
        env[:a] = np.linspace(0, 1, a)
        env[-a:] = np.linspace(1, 0, a)
        samples = (wave * env * vol * 32767).astype(np.int16)
        stereo = np.ascontiguousarray(np.column_stack([samples, samples]))
        return pygame.sndarray.make_sound(stereo)

    def _make_step(self):
        """A soft, low 'thud' footstep: a low sine with a percussive decay,
        plus a little muffled (smoothed) noise for texture. Much gentler than
        a raw white-noise burst."""
        ms = 110
        n = int(SAMPLE_RATE * ms / 1000)
        t = np.arange(n)
        decay = np.exp(-t / (n * 0.28))          # fast percussive decay
        body = np.sin(2 * np.pi * 110 * t / SAMPLE_RATE)   # low "thump"
        # muffled noise: smooth white noise so it's a soft "puff", not "ssss"
        noise = np.random.uniform(-1.0, 1.0, n)
        k = 24
        noise = np.convolve(noise, np.ones(k) / k, mode="same")
        wave = body * 0.8 + noise * 0.35
        samples = (wave * decay * 0.5 * 32767).astype(np.int16)
        stereo = np.ascontiguousarray(np.column_stack([samples, samples]))
        return pygame.sndarray.make_sound(stereo)

    def _make_ping(self):
        """A drum hit (kick/tom) for emitting an echo: a low pitch that drops
        fast with a percussive decay, plus a tiny attack 'click'. Warm, not shrill."""
        ms = 220
        n = int(SAMPLE_RATE * ms / 1000)
        t = np.arange(n)
        tt = t / SAMPLE_RATE
        # frequency sweeps down quickly (~180 Hz -> ~55 Hz): the classic drum "boom"
        freq = 55.0 + 130.0 * np.exp(-tt * 22.0)
        phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
        body = np.sin(phase)
        decay = np.exp(-t / (n * 0.22))           # tight percussive decay
        # short noise transient for the "thwack" of the stick hitting the skin
        click = np.zeros(n)
        cl = max(1, int(n * 0.03))
        click[:cl] = np.random.uniform(-1.0, 1.0, cl) * np.linspace(1, 0, cl)
        wave = body * decay * 0.95 + click * 0.25
        samples = (wave * 0.55 * 32767).astype(np.int16)
        stereo = np.ascontiguousarray(np.column_stack([samples, samples]))
        return pygame.sndarray.make_sound(stereo)

    def play_ping(self):
        if not self.ok:
            return
        try:
            self.ping_sound.play()
        except Exception:           # noqa: BLE001
            pass

    def play_step(self, left, right):
        if not self.ok:
            return
        try:
            ch = self.step_sound.play()
            if ch:
                ch.set_volume(clamp(left, 0, 1), clamp(right, 0, 1))
        except Exception:           # noqa: BLE001
            pass

    def play_beacon(self, closeness):
        """closeness in 0..1: closer to exit = higher pitch + louder."""
        if not self.ok:
            return
        try:
            pitch = int(300 + closeness * 500)
            key = pitch // 25 * 25             # cache tones in 25 Hz buckets
            snd = self._beacon_cache.get(key)
            if snd is None:
                snd = self._make(key or 300, 90, vol=0.5, kind="sine")
                self._beacon_cache[key] = snd
            ch = snd.play()
            if ch:
                v = clamp(0.12 + closeness * 0.5, 0, 1)
                ch.set_volume(v, v)
        except Exception:           # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Echo: a ring of rays that reflect off walls (Lecture 9 raycasting)
# ---------------------------------------------------------------------------
class Echo:
    def __init__(self, origin):
        self.origin = origin
        self.radius = 0.0
        self.alive = True
        # precompute ray directions
        self.dirs = [(math.cos(2 * math.pi * i / NUM_RAYS),
                      math.sin(2 * math.pi * i / NUM_RAYS))
                     for i in range(NUM_RAYS)]
        self._hits = None   # cached first-wall hit per ray (px, py, dist) or None

    def update(self):
        self.radius += ECHO_SPEED
        if self.radius > ECHO_MAX:
            self.alive = False

    def _compute_hits(self, grid):
        """Cast every ray once and record where it first meets a wall.
        Walls don't move, so this is cached for the echo's whole lifetime."""
        self._hits = []
        ox, oy = self.origin
        for dx, dy in self.dirs:
            dist = 0
            hit = None
            while dist <= ECHO_MAX:
                px = ox + dx * dist
                py = oy + dy * dist
                cx, cy = int(px // CELL), int(py // CELL)
                if cx < 0 or cy < 0 or cx >= COLS or cy >= ROWS:
                    break
                if grid[cy][cx]:
                    hit = (px, py, dist)
                    break
                dist += 4
            self._hits.append(hit)

    def lit_points(self, grid):
        """Light only the walls whose distance is near the ring's CURRENT radius,
        so the echo reads as a sonar ring expanding outward (Lecture 9)."""
        if self._hits is None:
            self._compute_hits(grid)
        points = []
        for hit in self._hits:
            if hit is None:
                continue
            px, py, dist = hit
            if abs(dist - self.radius) < WAVE_THICKNESS:
                b = max(0, 255 - int(dist / ECHO_MAX * 255))
                points.append((px, py, b))
        return points


def stereo_pan(player, source):
    """Lecture 7: compute left/right volume from horizontal offset.
    Returns (left, right) in 0..1. (Used for enemy footstep cues.)"""
    dx = source[0] - player[0]
    norm = max(-1.0, min(1.0, dx / WIDTH))
    left = max(0.0, min(1.0, 0.5 - norm / 2))
    right = max(0.0, min(1.0, 0.5 + norm / 2))
    return left, right


# ---------------------------------------------------------------------------
# Main game
# ---------------------------------------------------------------------------
class Game:
    def __init__(self, audio=None, level=1, base_seed=0, enemy_base=None, fade_base=None):
        self.audio = audio
        self.level = level
        # deterministic per-level maze; harder params as the level climbs (#7/#9)
        if base_seed:
            random.seed(base_seed + level)
        enemy_base = ENEMY_MOVE_FRAMES if enemy_base is None else enemy_base
        fade_base = FADE_PER_FRAME if fade_base is None else fade_base
        self.enemy_move_frames = max(8, enemy_base - (level - 1) * 2)
        self.fade = min(14, fade_base + (level - 1))        # memory fades faster
        self.grid = generate_maze(COLS, ROWS)
        self.player = [1, 1]                       # logical grid coords
        self.exit = [COLS - 2, ROWS - 2]
        self.grid[self.exit[1]][self.exit[0]] = False
        self.enemy = [COLS - 2, 1]
        self.grid[1][COLS - 2] = False
        self.echoes = []
        # persistent "memory" surface that holds faded light
        self.light = pygame.Surface((WIDTH, HEIGHT))
        self.light.fill(BLACK)
        self.enemy_timer = 0
        self.won = False
        self.caught = False
        self.last_pan = (0.5, 0.5)
        self.energy = ENERGY_MAX
        self.ping_timer = 0
        # smooth movement (#4): pixel positions ease toward the logical cells
        self.player_px = list(self.center(self.player))
        self.enemy_px = list(self.center(self.enemy))
        # enemy perception (#6): it only knows the last spot it heard a ping
        self.enemy_known = None
        self.enemy_goal = None
        # audio cue timers
        self.beacon_timer = 0
        self.step_timer = 0

    def center(self, cell):
        return (cell[0] * CELL + CELL // 2, cell[1] * CELL + CELL // 2)

    @staticmethod
    def _cell_dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _random_open_cell(self):
        while True:
            x, y = random.randint(1, COLS - 2), random.randint(1, ROWS - 2)
            if not self.grid[y][x]:
                return (x, y)

    def player_moving(self):
        tx, ty = self.center(self.player)
        return math.hypot(self.player_px[0] - tx, self.player_px[1] - ty) > 1.5

    def can_ping(self):
        return self.ping_timer == 0 and self.energy >= PING_COST

    def try_ping(self):
        """Emit an echo only if off cooldown and enough energy. Returns success."""
        if not self.can_ping():
            return False
        self.echoes.append(Echo(tuple(self.center(self.player))))
        self.energy -= PING_COST
        self.ping_timer = COOLDOWN_FRAMES
        if self.audio:
            self.audio.play_ping()
        # the enemy hears nearby pings -> updates its last-known target (#6)
        if self._cell_dist(self.enemy, self.player) <= ENEMY_HEAR_CELLS:
            self.enemy_known = list(self.player)
        return True

    def emit_echo(self):
        self.echoes.append(Echo(tuple(self.center(self.player))))

    def try_move(self, dx, dy):
        if self.player_moving():          # input lock until the slide finishes (#4)
            return
        nx, ny = self.player[0] + dx, self.player[1] + dy
        if 0 <= nx < COLS and 0 <= ny < ROWS and not self.grid[ny][nx]:  # collision (Lec 5)
            self.player = [nx, ny]
            if self.player == self.exit:
                self.won = True

    @staticmethod
    def _lerp_px(px, target):
        px[0] += (target[0] - px[0]) * MOVE_LERP
        px[1] += (target[1] - px[1]) * MOVE_LERP

    def _step_enemy(self):
        # decide a goal: chase the last heard position, else wander (#6)
        if self.enemy_known is not None:
            goal = tuple(self.enemy_known)
        else:
            if self.enemy_goal is None or tuple(self.enemy) == self.enemy_goal:
                self.enemy_goal = self._random_open_cell()
            goal = self.enemy_goal
        step = astar(self.grid, tuple(self.enemy), goal)
        if step:
            self.enemy = list(step)
        # arrived at the last-known spot but the player isn't here -> forget it
        if self.enemy_known is not None and tuple(self.enemy) == tuple(self.enemy_known):
            self.enemy_known = None
        self.last_pan = stereo_pan(self.player_px, self.enemy_px)

    def _update_beacon(self):
        # exit beacon (#2): reuse pathfinding so closer = higher pitch + faster
        self.beacon_timer -= 1
        if self.beacon_timer > 0:
            return
        d = bfs_dist(self.grid, tuple(self.player), tuple(self.exit))
        if d is None:
            self.beacon_timer = BEACON_MAX_INTERVAL
            return
        closeness = clamp(1 - d / BEACON_MAX_DIST, 0.0, 1.0)
        self.beacon_timer = int(lerp(BEACON_MAX_INTERVAL, BEACON_MIN_INTERVAL, closeness))
        if self.audio:
            self.audio.play_beacon(closeness)

    def _update_footsteps(self):
        # enemy footsteps (#3): stereo pan + volume falls off with distance
        d = math.hypot(self.player_px[0] - self.enemy_px[0],
                       self.player_px[1] - self.enemy_px[1])
        if d >= HEAR_RANGE:
            return
        gain = clamp(1 - d / HEAR_RANGE, 0.0, 1.0)
        self.step_timer -= 1
        if self.step_timer <= 0:
            self.step_timer = STEP_INTERVAL
            left, right = stereo_pan(self.player_px, self.enemy_px)
            if self.audio:
                self.audio.play_step(left * gain, right * gain)

    def update(self):
        # ping economy: tick down cooldown, slowly regenerate energy
        self.ping_timer = max(0, self.ping_timer - 1)
        self.energy = min(ENERGY_MAX, self.energy + ENERGY_REGEN)

        # ease pixel positions toward their logical cells (#4)
        self._lerp_px(self.player_px, self.center(self.player))
        self._lerp_px(self.enemy_px, self.center(self.enemy))

        # fade the whole light surface toward black (echo memory decays)
        fade = pygame.Surface((WIDTH, HEIGHT))
        fade.set_alpha(self.fade)
        fade.fill(BLACK)
        self.light.blit(fade, (0, 0))

        # advance echoes and bloom lit wall points onto the light surface
        build_glow_assets()
        for echo in self.echoes:
            echo.update()
            for (px, py, b) in echo.lit_points(self.grid):
                idx = min(WALL_GLOW_BUCKETS - 1, int(b / 256 * WALL_GLOW_BUCKETS))
                g = _WALL_GLOWS[idx]
                self.light.blit(g, (int(px) - g.get_width() // 2,
                                    int(py) - g.get_height() // 2))
        self.echoes = [e for e in self.echoes if e.alive]

        # enemy hunts via A* (Lecture 4); moves on a timer so it's beatable
        self.enemy_timer += 1
        if self.enemy_timer >= self.enemy_move_frames:
            self.enemy_timer = 0
            self._step_enemy()
        if self.enemy == self.player:
            self.caught = True

        self._update_beacon()
        self._update_footsteps()

    @staticmethod
    def _blit_centered(screen, sprite, cx, cy):
        screen.blit(sprite, (cx - sprite.get_width() // 2,
                             cy - sprite.get_height() // 2))

    def draw(self, screen):
        build_glow_assets()
        screen.fill(BLACK)
        screen.blit(self.light, (0, 0))
        # player is always faintly visible (soft glow + bright core)
        px, py = int(self.player_px[0]), int(self.player_px[1])
        self._blit_centered(screen, _PLAYER_GLOW, px, py)
        pygame.draw.circle(screen, (220, 245, 255), (px, py), 4)
        # enemy only glows when an echo is currently near it (heard, not seen)
        ex, ey = int(self.enemy_px[0]), int(self.enemy_px[1])
        for echo in self.echoes:
            ox, oy = echo.origin
            if abs(math.hypot(ex - ox, ey - oy) - echo.radius) < 30:
                self._blit_centered(screen, _ENEMY_GLOW, ex, ey)
                pygame.draw.circle(screen, (255, 200, 200), (ex, ey), 4)
                break
        # exit glows green when lit
        gx, gy = self.center(self.exit)
        if self.light.get_at((gx, gy))[0] > 20:
            self._blit_centered(screen, _EXIT_GLOW, gx, gy)
            pygame.draw.circle(screen, (220, 255, 220), (gx, gy), 4)

        # energy bar (HUD): full = ready to ping, drains on each ping
        bar_w, bar_h, margin = 120, 10, 8
        frac = self.energy / ENERGY_MAX
        pygame.draw.rect(screen, (40, 40, 50), (margin, margin, bar_w, bar_h))
        fill = PLAYER_COLOR if self.can_ping() else (150, 150, 160)
        pygame.draw.rect(screen, fill, (margin, margin, int(bar_w * frac), bar_h))
        pygame.draw.rect(screen, (90, 90, 110), (margin, margin, bar_w, bar_h), 1)


# ---------------------------------------------------------------------------
# Menu / help screen (Lecture 1: game-state machine)
# ---------------------------------------------------------------------------
def draw_menu(screen, font, small_font, selected):
    screen.fill((6, 8, 14))
    title = font.render("ECHOBOUND", True, PLAYER_COLOR)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 40))

    lines = [
        "You cannot see the maze - you can only hear it.",
        "",
        "SPACE  emit an echo to light up nearby walls (costs energy)",
        "WASD / Arrows  move",
        "Listen: the beacon rises in pitch as you near the green exit.",
        "Footsteps (left/right) warn you the red enemy is closing in.",
        "Reach the exit. Don't get caught.",
    ]
    y = 110
    for ln in lines:
        surf = small_font.render(ln, True, (210, 215, 225))
        screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, y))
        y += 30

    y += 6
    pick = small_font.render("Difficulty (Up/Down to change):", True, (170, 180, 200))
    screen.blit(pick, (WIDTH // 2 - pick.get_width() // 2, y))
    y += 34
    for i, (name, _, _) in enumerate(DIFFICULTIES):
        chosen = (i == selected)
        color = EXIT_COLOR if chosen else (150, 155, 170)
        label = ("> %s <" % name) if chosen else name
        surf = small_font.render(label, True, color)
        screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, y))
        y += 30

    start = small_font.render("Press ENTER to start", True, (235, 240, 255))
    screen.blit(start, (WIDTH // 2 - start.get_width() // 2, HEIGHT - 40))


def draw_pause(screen, font, small_font):
    """Translucent overlay drawn over the frozen game (paused state)."""
    veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    veil.fill((0, 0, 0, 170))
    screen.blit(veil, (0, 0))
    title = font.render("PAUSED", True, (235, 240, 255))
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 80))
    lines = [
        "SPACE  echo    WASD / Arrows  move",
        "H / Esc  resume      R  restart      M  main menu",
    ]
    y = HEIGHT // 2 - 10
    for ln in lines:
        surf = small_font.render(ln, True, (210, 215, 225))
        screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, y))
        y += 32


# ---------------------------------------------------------------------------
# Interactive entry point
# ---------------------------------------------------------------------------
async def run():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Echobound")
    # Stop SDL text-input mode so a CJK/IME doesn't swallow letter keys (e.g. R)
    try:
        pygame.key.stop_text_input()
    except (AttributeError, pygame.error):
        pass
    clock = pygame.time.Clock()
    audio = Audio()
    font = pygame.font.Font(None, 48)
    small_font = pygame.font.Font(None, 28)

    # game-state machine (Lecture 1): MENU <-> PLAYING(+win/caught) transitions
    state = "menu"
    selected = 1                          # default difficulty = Normal
    base_seed = SEED if SEED is not None else random.randrange(1 << 30)
    level = 1
    game = None

    def new_game(lvl, fresh_seed=False):
        nonlocal base_seed
        if fresh_seed:
            base_seed = random.randrange(1 << 30)
        _, e_base, f_base = DIFFICULTIES[selected]
        return Game(audio, level=lvl, base_seed=base_seed,
                    enemy_base=e_base, fade_base=f_base)

    running = True
    while running:  # input -> update -> render (Lecture 1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if state == "menu":
                    if event.key in (pygame.K_UP, pygame.K_w, pygame.K_LEFT, pygame.K_a):
                        selected = (selected - 1) % len(DIFFICULTIES)
                    elif event.key in (pygame.K_DOWN, pygame.K_s, pygame.K_RIGHT, pygame.K_d):
                        selected = (selected + 1) % len(DIFFICULTIES)
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        selected = event.key - pygame.K_1
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                        level = 1
                        game = new_game(level, fresh_seed=True)
                        state = "playing"
                elif state == "paused":
                    if event.key in (pygame.K_h, pygame.K_ESCAPE):
                        state = "playing"                   # resume the same game
                    elif event.key == pygame.K_m:
                        state = "menu"                      # quit to the help menu
                    elif event.key == pygame.K_r:           # restart from level 1
                        level = 1
                        game = new_game(level, fresh_seed=True)
                        state = "playing"
                else:  # playing
                    game_over = game.won or game.caught
                    if event.key == pygame.K_h and not game_over:
                        state = "paused"                    # freeze, keep progress
                    elif event.key == pygame.K_r:           # full restart -> level 1
                        level = 1
                        game = new_game(level, fresh_seed=True)
                    elif game_over and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER,
                                                     pygame.K_SPACE):
                        if game.won:                        # advance to a harder level (#7)
                            level += 1
                            game = new_game(level)
                        else:                               # caught -> start over
                            level = 1
                            game = new_game(level, fresh_seed=True)
                    elif not game_over and event.key == pygame.K_SPACE:
                        game.try_ping()

        if state == "menu":
            draw_menu(screen, font, small_font, selected)
            pygame.display.flip()
            await asyncio.sleep(0)        # yield to the browser event loop (pygbag)
            clock.tick(FPS)
            continue

        game_over = game.won or game.caught
        # held-key movement polled each frame -> smooth continuous walking (#4).
        # Reading the key state directly also bypasses any IME letter-key capture.
        if state == "playing" and not game_over and not game.player_moving():
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                game.try_move(0, -1)
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                game.try_move(0, 1)
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                game.try_move(-1, 0)
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                game.try_move(1, 0)

        # freeze updates when paused or when the round is over
        if state == "playing" and not game_over:
            game.update()
        game.draw(screen)

        # HUD: level + difficulty (top-right), help hint (bottom-left)
        hud = small_font.render("Level %d  -  %s" % (game.level, DIFFICULTIES[selected][0]),
                                True, (200, 210, 230))
        screen.blit(hud, (WIDTH - hud.get_width() - 8, 8))
        tip = small_font.render("H = pause", True, (120, 130, 150))
        screen.blit(tip, (8, HEIGHT - tip.get_height() - 6))

        if game.won:
            msg = font.render("LEVEL %d CLEARED" % game.level, True, EXIT_COLOR)
            screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 20))
            hint = small_font.render("Press Enter for the next level", True, (220, 220, 220))
            screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, HEIGHT // 2 + 24))
        elif game.caught:
            msg = font.render("CAUGHT", True, ENEMY_COLOR)
            screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 20))
            hint = small_font.render("Press R or Enter to play again", True, (220, 220, 220))
            screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, HEIGHT // 2 + 24))

        if state == "paused":
            draw_pause(screen, font, small_font)

        pygame.display.flip()
        await asyncio.sleep(0)            # yield to the browser event loop (pygbag)
        clock.tick(FPS)
    pygame.quit()


# ---------------------------------------------------------------------------
# Headless screenshot generator (for the proposal's testing section)
# ---------------------------------------------------------------------------
def shots():
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.Surface((WIDTH, HEIGHT))
    random.seed(7)
    game = Game()

    # Shot 1: a single fresh echo, mid-expansion
    game.emit_echo()
    for _ in range(18):
        game.update()
    game.draw(screen)
    pygame.image.save(screen, "shot1_echo.png")

    # Shot 2: player walks a path, pinging as they go -> memory map builds up.
    # Slow the fade so earlier echoes persist and the explored trail is visible.
    global FADE_PER_FRAME
    FADE_PER_FRAME = 1
    walk = [(0, 1)] * 5 + [(1, 0)] * 4 + [(0, 1)] * 3
    for dx, dy in walk:
        game.emit_echo()
        for _ in range(6):
            game.update()
        game.try_move(dx, dy)
    for _ in range(20):
        game.update()
    game.draw(screen)
    pygame.image.save(screen, "shot2_explored.png")

    # Shot 3: full-reveal debug view (what the maze actually looks like)
    debug = pygame.Surface((WIDTH, HEIGHT))
    debug.fill((15, 15, 20))
    for y in range(ROWS):
        for x in range(COLS):
            if game.grid[y][x]:
                pygame.draw.rect(debug, (70, 70, 90),
                                 (x*CELL, y*CELL, CELL, CELL))
    pygame.draw.circle(debug, PLAYER_COLOR, game.center(game.player), 7)
    pygame.draw.circle(debug, ENEMY_COLOR, game.center(game.enemy), 7)
    pygame.draw.circle(debug, EXIT_COLOR, game.center(game.exit), 7)
    pygame.image.save(debug, "shot3_debug.png")
    pygame.quit()
    print("saved shot1_echo.png, shot2_explored.png, shot3_debug.png")


def _apply_cli_args():
    """Parse command-line options for difficulty/PCG tuning (#7/#9).
    Skipped entirely in the browser (pygbag/WebAssembly), where there is no
    real argv and argparse could misbehave."""
    if sys.platform == "emscripten":
        return False
    p = argparse.ArgumentParser(description="Echobound - an echolocation maze game.")
    p.add_argument("--shots", action="store_true",
                   help="render headless screenshots instead of playing")
    p.add_argument("--seed", type=int, default=None,
                   help="maze seed for deterministic generation (#9)")
    p.add_argument("--render", type=int, default=None, metavar="PX",
                   help="echo render distance in pixels (#7)")
    p.add_argument("--fade", type=int, default=None, metavar="N",
                   help="light fade per frame; lower = longer memory (#7)")
    p.add_argument("--enemy-speed", type=int, default=None, metavar="FRAMES",
                   help="frames per enemy step; lower = faster enemy (#7)")
    args = p.parse_args()

    # PCG / difficulty parameter tuning (#7): override module globals up front
    global ECHO_MAX, FADE_PER_FRAME, ENEMY_MOVE_FRAMES, SEED
    if args.render is not None:
        ECHO_MAX = args.render
    if args.fade is not None:
        FADE_PER_FRAME = args.fade
    if args.enemy_speed is not None:
        ENEMY_MOVE_FRAMES = args.enemy_speed
    if args.seed is not None:
        SEED = args.seed
        random.seed(args.seed)
    return args.shots


async def main():
    """Async entry point required by pygbag for browser/WebAssembly play.
    Works unchanged for normal desktop runs via `python echobound.py`."""
    want_shots = _apply_cli_args()
    if want_shots:
        shots()
        return
    await run()


# pygbag runs this module and drives the asyncio loop; this also works as a
# plain desktop launch (`python echobound.py`).
if __name__ == "__main__":
    asyncio.run(main())
