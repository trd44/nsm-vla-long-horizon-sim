"""Task definitions for reproducing the paper's neuro-symbolic results, plus the
planning and evaluation helpers shared by the experiment driver and the tests.

Each PAPER_TASKS entry pins one row of the paper's Tables 3-4 to an environment
configuration, a PDDL domain, a goal and the published numbers.

Advancement is not defined formally in the paper (it only says progress is
"tracked using detectors"). Here it is the fraction of plan moves, a move being a
PICK/PLACE pair, whose effect was verified by the simulator detector.
"""
import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paper section 4.3: spawn noise is uniform in +/-2.5 cm on x and y.
SPAWN_NOISE = 0.025
# Paper sections 3.2 and 4.4: terminate after 600 steps without sub-task progress.
NO_PROGRESS_LIMIT = 600

HANOI_COLORS = {"blue cube": "cube1", "red cube": "cube2", "green cube": "cube3", "yellow cube": "cube4"}
TOWER_COLORS = {"blue cube": "cube0", "red cube": "cube1", "green cube": "cube2", "yellow cube": "cube3"}

# Workcell map: the pegs, platforms and bins are fixed fixtures (no jitter in any paper
# task), so vision mode takes their positions from this map and never from the
# simulator. tests/test_task_configs.py checks it against the environments.
_PEGS = {"peg1": (0.0, -0.18, 0.8), "peg2": (0.0, 0.02, 0.8), "peg3": (0.0, 0.22, 0.8)}
_TOWER_PLATFORM = {"platform": (0.0, 0.24, 0.805)}
_SORT_BINS = {"platform1": (-0.15, -0.2, 0.806), "platform2": (-0.15, 0.2, 0.806)}
_SORT_BINS_SWAPPED = {"platform1": (-0.15, 0.2, 0.806), "platform2": (-0.15, -0.2, 0.806)}
_SORT3_BINS = {"bin0": (-0.15, -0.2, 0.806), "bin1": (-0.15, 0.0, 0.806), "bin2": (-0.15, 0.2, 0.806)}
# Sorting tasks: the colour (YOLO class) of the cubes each area takes. With type_match()
# from the symbolic state this gives every PDDL cube its colour.
_SORT_COLORS = {"platform1": "blue cube", "platform2": "red cube"}
_SORT3_COLORS = {"bin0": "red cube", "bin1": "green cube", "bin2": "blue cube"}


def _hanoi_goal(state, detector):
    cubes = list(detector.objects)
    goals = [f"on {cubes[-1]} peg3"]
    goals += [f"on {upper} {lower}" for upper, lower in zip(cubes[-2::-1], cubes[:0:-1])]
    return goals


def _sorting_goal(state, detector):
    goals = []
    for cube in detector.objects:
        targets = [key[key.index(",") + 1:-1] for key, value in state.items()
                   if value and key.startswith(f"type_match({cube},")]
        if len(targets) != 1:
            raise ValueError(f"{cube} must match exactly one bin, got {targets}")
        goals.append(f"on {cube} {targets[0]}")
    return goals


def _tower_goal(state, detector):
    env = detector.env
    active = sorted(env.cubes[i].name for i in range(env.active_num_cubes))
    by_size = sorted(active, key=lambda cube: env.cube_sizes[int(cube.replace("cube", ""))])
    goals = [f"on {upper} {lower}" for upper, lower in zip(by_size, by_size[1:])]
    goals.append(f"on {by_size[-1]} platform")
    return goals


@dataclass(frozen=True)
class TaskConfig:
    name: str
    env_name: str
    env_kwargs: dict
    pddl_dir: str
    predicates: Tuple[str, ...]
    goal_builder: Callable
    paper_success: Optional[float]
    paper_advancement: Optional[float]
    # YOLO class -> PDDL cube where colour identifies the cube. None when several
    # cubes share a colour, so identity has to come from relational matching.
    color_to_pddl: Optional[Dict[str, str]]
    expected_plan_len: Optional[int] = None
    detector: str = ""
    planning_mode: int = 0
    notes: str = ""
    # Fixed fixture positions used by vision mode (see the workcell map above).
    static_positions: Optional[Dict[str, Tuple[float, float, float]]] = None
    # Sorting tasks: area -> colour of the cubes that belong there.
    area_colors: Optional[Dict[str, str]] = None


def _task(name, env_name, env_kwargs, pddl, predicates, goal, success, advancement, colors, plan_len, detector, notes="",
          static=None, area_colors=None):
    return TaskConfig(name=name, env_name=env_name, env_kwargs=env_kwargs, pddl_dir=f"planning/PDDL/{pddl}/",
                      predicates=predicates, goal_builder=goal, paper_success=success,
                      paper_advancement=advancement, color_to_pddl=colors, expected_plan_len=plan_len,
                      detector=detector, notes=notes, static_positions=static, area_colors=area_colors)


_HANOI_PRED = ("on", "clear", "grasped")
_TOWER_PRED = ("on", "clear", "grasped", "smaller")
_SORT_PRED = ("on", "clear", "grasped", "type_match")

PAPER_TASKS: Dict[str, TaskConfig] = {
    "sort_2color": _task(
        "sort_2color", "CubeSorting", {"cube_placement_noise": SPAWN_NOISE}, "cubesorting", _SORT_PRED,
        _sorting_goal, 100.0, 100.0, None, 8, "CubeSortingDetector",
        "Colours are i.i.d. fair draws per reset, so 1 episode in 8 is all one colour.",
        static=_SORT_BINS, area_colors=_SORT_COLORS),
    "tower_3_set": _task(
        "tower_3_set", "HeightStacking",
        {"min_cubes": 3, "max_cubes": 3, "cube_placement_noise": SPAWN_NOISE, "deterministic_reset": True},
        "heightstacking", _TOWER_PRED, _tower_goal, 100.0, 100.0, TOWER_COLORS, 6, "HeightStackingDetector",
        static=_TOWER_PLATFORM),
    "hanoi_3": _task(
        "hanoi_3", "Hanoi", {"cube_init_pos_noise_uniform": SPAWN_NOISE}, "hanoi", _HANOI_PRED,
        _hanoi_goal, 100.0, 100.0, {k: v for k, v in HANOI_COLORS.items() if v != "cube4"}, 14, "HanoiDetector",
        static=_PEGS),
    "sort_2color_swapped": _task(
        "sort_2color_swapped", "CubeSorting", {"cube_placement_noise": SPAWN_NOISE, "swap_bins": True},
        "cubesorting", _SORT_PRED, _sorting_goal, 100.0, 100.0, None, 8, "CubeSortingDetector",
        static=_SORT_BINS_SWAPPED, area_colors=_SORT_COLORS),
    "sort_3color": _task(
        "sort_3color", "AssemblyLineSorting", {"num_bins": 3, "cube_placement_noise": SPAWN_NOISE},
        "assemblylinesorting", _SORT_PRED, _sorting_goal, 86.0, 96.5, None, 8, "AssemblyLineSortingDetector",
        "Colours are fixed (red, green, blue, red); only the spawn order is shuffled.",
        static=_SORT3_BINS, area_colors=_SORT3_COLORS),
    "tower_3_random": _task(
        "tower_3_random", "HeightStacking",
        {"min_cubes": 3, "max_cubes": 3, "cube_placement_noise": SPAWN_NOISE, "deterministic_reset": False},
        "heightstacking", _TOWER_PRED, _tower_goal, 100.0, 100.0, TOWER_COLORS, 6, "HeightStackingDetector",
        static=_TOWER_PLATFORM),
    "tower_4_set": _task(
        "tower_4_set", "HeightStacking",
        {"min_cubes": 4, "max_cubes": 4, "cube_placement_noise": SPAWN_NOISE, "deterministic_reset": True},
        "heightstacking", _TOWER_PRED, _tower_goal, 97.0, 99.5, TOWER_COLORS, 8, "HeightStackingDetector",
        static=_TOWER_PLATFORM),
    "hanoi_4": _task(
        "hanoi_4", "Hanoi4x3", {"cube_init_pos_noise_uniform": SPAWN_NOISE}, "hanoi4x3", _HANOI_PRED,
        _hanoi_goal, 78.0, 84.4, HANOI_COLORS, 30, "HanoiDetector",
        static=_PEGS),
}


def legacy_task(env_name: str) -> TaskConfig:
    """Config for the driver's legacy --env flag: no paper numbers and no spawn noise override."""
    legacy = {
        "Hanoi": ("hanoi", _HANOI_PRED, _hanoi_goal, HANOI_COLORS, "HanoiDetector"),
        "Hanoi4x3": ("hanoi4x3", _HANOI_PRED, _hanoi_goal, HANOI_COLORS, "HanoiDetector"),
        "CubeSorting": ("cubesorting", _SORT_PRED, _sorting_goal, None, "CubeSortingDetector"),
        "HeightStacking": ("heightstacking", _TOWER_PRED, _tower_goal, TOWER_COLORS, "HeightStackingDetector"),
        "AssemblyLineSorting": ("assemblylinesorting", _SORT_PRED, _sorting_goal, None, "AssemblyLineSortingDetector"),
    }
    pddl, predicates, goal, colors, detector = legacy[env_name]
    return _task(f"legacy_{env_name}", env_name, {}, pddl, predicates, goal, None, None, colors, None, detector)


# A cube with this many others below it sits above the default home height (z=0.958).
# The policies only ever saw the target below the gripper, so they would drive into the
# tower from there, and the regressor was fit at default home, so only those picks lift.
TALL_PICK_LEVEL = 3
TALL_PICK_HOME_Z = 1.03


def cubes_below(state: Dict[str, bool], cube: str) -> int:
    """Number of cubes stacked under `cube` in the symbolic state."""
    count = 0
    current = cube
    while True:
        support = next((key[key.index(",") + 1:-1] for key, value in state.items()
                        if value and key.startswith(f"on({current},")), None)
        if support is None or "cube" not in support:
            return count
        count += 1
        current = support


LEGACY_ENVS = ("Hanoi", "Hanoi4x3", "CubeSorting", "HeightStacking", "AssemblyLineSorting")


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def make_env(task: TaskConfig, env_kwargs_override: Optional[dict] = None, **make_kwargs):
    """Build the task's environment on Kinova3 without rendering unless asked for."""
    import robosuite as suite

    arguments = dict(
        robots="Kinova3",
        controller_configs=suite.load_controller_config(default_controller="OSC_POSITION"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        reward_shaping=True,
        control_freq=20,
        horizon=20000,
    )
    arguments.update(make_kwargs)
    arguments.update(task.env_kwargs)
    arguments.update(env_kwargs_override or {})
    return suite.make(env_name=task.env_name, **arguments)


def make_detector(task: TaskConfig, env):
    from robosuite.utils import detector as detectors

    return getattr(detectors, task.detector)(env)


# ---------------------------------------------------------------------------
# Detector-based effects and success
# ---------------------------------------------------------------------------

def _inside_cubesorting_bin(detector, cube: str, platform: str) -> bool:
    """CubeSorting's in() is a 10 cm radius, which also accepts a cube resting just
    outside the wall (centre about 8.4 cm from the bin centre); require the inner walls too."""
    env = detector.env
    cube_pos = env.sim.data.body_xpos[env.cube_body_ids[int(cube.replace("cube", ""))]]
    bin_pos = env.platform1_pos if platform == "platform1" else env.platform2_pos
    inner = env.platform_size - env.wall_thickness
    return bool(detector.in_platform(cube, platform)
                and abs(cube_pos[0] - bin_pos[0]) < inner and abs(cube_pos[1] - bin_pos[1]) < inner)


def _supported(detector, cube: str, location: str) -> bool:
    """Whether cube rests on location. Bins are tested with in(), since on() is false
    for valid placements: it fails for a cube stacked inside a bin or near a bin corner."""
    if location.startswith("bin"):
        return bool(detector.in_bin(cube, location))
    if location in ("platform1", "platform2"):
        return _inside_cubesorting_bin(detector, cube, location)
    return bool(detector.on(cube, location))


def place_effect_holds(detector, cube: str, location: str) -> bool:
    """Effect of PLACE cube location, read from the simulator."""
    return _supported(detector, cube, location) and not detector.grasped(cube)


def failed_goals(detector, goals: Sequence[str]) -> List[str]:
    failed = []
    for goal in goals:
        _, cube, location = goal.split()
        if not place_effect_holds(detector, cube, location):
            failed.append(goal)
    return failed


def goals_hold(detector, goals: Sequence[str]) -> bool:
    return not failed_goals(detector, goals)


def parse_moves(plan: Sequence[str]) -> List[Tuple[str, str, str]]:
    """Group a PICK/PLACE plan into (cube, source, destination) moves."""
    if len(plan) % 2:
        raise ValueError(f"Plan is not made of PICK/PLACE pairs: {plan}")
    moves = []
    for pick, place in zip(plan[0::2], plan[1::2]):
        pick_op, cube, source = pick.lower().split()
        place_op, placed, destination = place.lower().split()
        if pick_op != "pick" or place_op != "place" or cube != placed:
            raise ValueError(f"Plan is not made of PICK/PLACE pairs: {pick}, {place}")
        moves.append((cube, source, destination))
    return moves


def missing_preconditions(state: dict, detector, move: Tuple[str, str, str]) -> List[str]:
    """Move preconditions, as in the PDDL domains, that the current state does not satisfy.

    "table" and the HeightStacking platform are static in problem_save.pddl and have
    no groundings, so they are read from the cubes' on() predicates instead.
    """
    cube, source, destination = move
    missing = []
    if any(detector.grasped(other) for other in detector.objects):
        missing.append("free-gripper")
    if not state.get(f"clear({cube})", False):
        missing.append(f"clear({cube})")
    if source == "table":
        if any(value and key.startswith(f"on({cube},") for key, value in state.items()):
            missing.append(f"on({cube},table)")
    elif not _supported(detector, cube, source):
        missing.append(f"on({cube},{source})")
    if f"clear({destination})" in state:
        if not state[f"clear({destination})"]:
            missing.append(f"clear({destination})")
    elif any(value and key.endswith(f",{destination})") and key.startswith("on(") for key, value in state.items()):
        missing.append(f"clear({destination})")
    return missing


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

class PlannerError(RuntimeError):
    """The planner returned something other than a plan or a verdict of unsolvable."""


def _tokenize(text: str) -> list:
    text = re.sub(r";[^\n]*", "", text.lower())
    return re.findall(r"[()]|[^\s()]+", text)


def _parse_sexpr(tokens: list):
    stack, current = [], []
    for token in tokens:
        if token == "(":
            stack.append(current)
            current = []
        elif token == ")":
            parent = stack.pop()
            parent.append(current)
            current = parent
        else:
            current.append(token)
    return current[0]


def _literals(expr) -> Tuple[List[list], List[list]]:
    """Split an expression into (positive atoms, negated atoms)."""
    if not expr:
        return [], []
    if expr[0] == "and":
        positive, negative = [], []
        for sub in expr[1:]:
            pos, neg = _literals(sub)
            positive += pos
            negative += neg
        return positive, negative
    if expr[0] == "not":
        return [], [expr[1]]
    return [expr], []


def parse_domain(path: str) -> Dict[str, dict]:
    """Read the STRIPS actions of a domain file."""
    with open(path) as file:
        domain = _parse_sexpr(_tokenize(file.read()))
    actions = {}
    for section in domain[1:]:
        if section[0] != ":action":
            continue
        fields = {section[i]: section[i + 1] for i in range(2, len(section) - 1, 2)}
        parameters = [token for token in fields[":parameters"] if token.startswith("?")]
        pre_pos, pre_neg = _literals(fields[":precondition"])
        add, delete = _literals(fields[":effect"])
        actions[section[1]] = dict(parameters=parameters, pre=pre_pos, pre_neg=pre_neg, add=add, delete=delete)
    return actions


def parse_problem(path: str) -> Tuple[set, set]:
    """Read the (init, goal) atom sets of a problem file."""
    with open(path) as file:
        problem = _parse_sexpr(_tokenize(file.read()))
    atoms = lambda items: {" ".join(item) for item in items}
    init, goal = set(), set()
    for section in problem[1:]:
        if section[0] == ":init":
            init = atoms(section[1:])
        elif section[0] == ":goal":
            goal = atoms(_literals(section[1])[0])
    return init, goal


def simulate_plan(plan: Sequence[str], init: set, goal: set, domain_path: str) -> Tuple[bool, set, str]:
    """Apply a plan in the STRIPS domain. Returns (goal reached, final state, error)."""
    actions = parse_domain(domain_path)
    state = set(init)
    for step in plan:
        name, *arguments = step.lower().split()
        if name not in actions or len(arguments) != len(actions[name]["parameters"]):
            return False, state, f"not an action of the domain: {step}"
        action = actions[name]
        binding = dict(zip(action["parameters"], arguments))
        ground = lambda atoms: {" ".join(binding.get(token, token) for token in atom) for atom in atoms}
        missing = ground(action["pre"]) - state
        if missing:
            return False, state, f"{step}: unmet preconditions {sorted(missing)}"
        state = (state - ground(action["delete"])) | ground(action["add"])
    unmet = set(goal) - state
    return not unmet, state, f"unmet goals {sorted(unmet)}" if unmet else ""


def init_predicates(state: dict, task: TaskConfig) -> Dict[str, bool]:
    """State atoms for the PDDL init section.

    A released cube inside a bin is reported as on(cube,bin) even when the detector's
    on() misses it, otherwise replanning would move sorted cubes again.
    """
    init = {key: True for key, value in state.items()
            if value and key.split("(")[0] in task.predicates and "ref" not in key}
    for key, value in state.items():
        if value and key.startswith("in("):
            cube = key[3:key.index(",")]
            if not state.get(f"grasped({cube})", False):
                init[f"on{key[2:]}"] = True
    return init


def plan_for_state(state: dict, detector, task: TaskConfig, goals: Optional[Sequence[str]] = None):
    """Plan from a detector state.

    Returns (plan, goals, init atoms, goal atoms). plan is False when the planner
    proves the state unsolvable, and [] when the goals already hold. Each process
    writes its own problem file, so parallel runs cannot read each other's problems.
    """
    from planning.planner import add_predicates_to_pddl, call_planner, define_goal_in_pddl

    goals = list(goals) if goals is not None else task.goal_builder(state, detector)
    pddl_dir = os.path.join(REPO_ROOT, task.pddl_dir)
    problem = f"problem_run_{os.getpid()}.pddl"
    problem_path = os.path.join(pddl_dir, problem)
    predicates = init_predicates(state, task)
    try:
        add_predicates_to_pddl(pddl_dir, predicates, problem_name=problem)
        define_goal_in_pddl(pddl_dir, goals, problem_name=problem)
        _drop_stale_table_support(problem_path, predicates)
        init, goal = parse_problem(problem_path)
        if goal <= init:
            return [], goals, init, goal
        plan, _ = call_planner(pddl_dir, problem=problem, mode=task.planning_mode)
    finally:
        if os.path.exists(problem_path):
            os.remove(problem_path)
    if plan is False:
        return False, goals, init, goal
    reached, _, error = simulate_plan(plan, init, goal, os.path.join(pddl_dir, "domain.pddl"))
    if not reached:
        raise PlannerError(f"Planner output is not a valid plan ({error}). Is the FF binary runnable "
                           f"on this machine? See NSM_FF_BIN. Output: {plan}")
    return plan, goals, init, goal


def _drop_stale_table_support(problem_path: str, predicates: Dict[str, bool]) -> None:
    """problem_save.pddl declares every cube on the table; remove that for a cube that has moved."""
    moved = {key[3:key.index(",")] for key in predicates if key.startswith("on(")}
    if not moved:
        return
    with open(problem_path) as file:
        lines = file.readlines()
    with open(problem_path, "w") as file:
        file.writelines(line for line in lines
                        if not (line.split("(on ")[-1].split(" ")[0] in moved and "table)" in line))


# ---------------------------------------------------------------------------
# Episode bookkeeping
# ---------------------------------------------------------------------------

class EpisodeTimeout(Exception):
    """No sub-task advanced within the paper's limit of env steps."""


class ProgressMonitor:
    """Counts env steps and raises EpisodeTimeout after `limit` steps without progress.

    A sub-task is a plan operator: the PICK advances when the detector first sees the
    cube grasped, the PLACE when it first sees the cube released on its destination.
    """

    def __init__(self, detector=None, limit: int = NO_PROGRESS_LIMIT):
        self.detector = detector
        self.limit = limit
        self.start_episode()

    def start_episode(self):
        self.steps = 0
        self.last_progress = 0
        self.armed = False
        self.move = None
        self.stage = 0
        # Where the episode is, so a timeout raised mid-skill can still be attributed.
        self.skill = None
        self.failed_skill = None

    def arm(self):
        self.last_progress = self.steps
        self.armed = True

    def disarm(self):
        self.armed = False

    def set_move(self, cube: str, destination: str):
        self.move = (cube, destination)
        self.stage = 0

    def _advance(self):
        self.stage += 1
        self.last_progress = self.steps

    def on_step(self):
        self.steps += 1
        if not self.armed:
            return
        if self.move is not None:
            cube, destination = self.move
            if self.stage == 0 and self.detector.grasped(cube):
                self._advance()
            elif self.stage == 1 and place_effect_holds(self.detector, cube, destination):
                self._advance()
        if self.steps - self.last_progress > self.limit:
            raise EpisodeTimeout(f"no sub-task progress within {self.limit} env steps")


def wilson_ci(successes: float, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion, as fractions."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(records: Sequence[dict], task: TaskConfig) -> dict:
    """Aggregate episode records and compare them with the paper's values."""
    n = len(records)
    successes = sum(bool(record["success"]) for record in records)
    advancement = [float(record["advancement"]) for record in records]
    low, high = wilson_ci(successes, n)
    mean_adv = float(np.mean(advancement)) if n else 0.0
    # Advancement is a per-episode fraction rather than a binomial outcome, so its
    # interval is the normal approximation on the mean.
    sem = float(np.std(advancement, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    summary = {
        "task": task.name,
        "episodes": n,
        "successes": successes,
        "success_pct": 100.0 * successes / n if n else None,
        "success_wilson95_pct": [100.0 * low, 100.0 * high],
        "advancement_pct": 100.0 * mean_adv,
        "advancement_ci95_pct": [100.0 * max(0.0, mean_adv - 1.96 * sem), 100.0 * min(1.0, mean_adv + 1.96 * sem)],
        "paper_success_pct": task.paper_success,
        "paper_advancement_pct": task.paper_advancement,
        "unsolvable_resets": sum(record.get("unsolvable_resets", 0) for record in records),
        "mean_env_steps": float(np.mean([record["env_steps"] for record in records])) if n else None,
        "mean_wall_s": float(np.mean([record["wall_s"] for record in records])) if n else None,
    }
    if task.paper_success is not None:
        summary["paper_success_in_ci"] = 100.0 * low <= task.paper_success <= 100.0 * high
    return summary
