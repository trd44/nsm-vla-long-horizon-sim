"""Checks the eight paper task configurations without rendering or running policies.

Metric-FF must be runnable; on machines where the tracked binary is not, set NSM_FF_BIN.
Run with: pytest tests/test_task_configs.py
"""
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pytest

from planning.task_configs import (
    PAPER_TASKS,
    REPO_ROOT,
    EpisodeTimeout,
    ProgressMonitor,
    failed_goals,
    goals_hold,
    make_detector,
    make_env,
    missing_preconditions,
    parse_moves,
    plan_for_state,
    simulate_plan,
    summarize,
    wilson_ci,
)

SEEDS = range(5)
NOISE = 0.025
PAPER_VALUES = {
    "sort_2color": (100, 100), "tower_3_set": (100, 100), "hanoi_3": (100, 100), "sort_2color_swapped": (100, 100),
    "sort_3color": (86, 96.5), "tower_3_random": (100, 100), "tower_4_set": (97, 99.5), "hanoi_4": (78, 84.4),
}
PLAN_LENGTHS = {
    "sort_2color": 8, "tower_3_set": 6, "hanoi_3": 14, "sort_2color_swapped": 8,
    "sort_3color": 8, "tower_3_random": 6, "tower_4_set": 8, "hanoi_4": 30,
}


@pytest.fixture(autouse=True)
def repo_cwd(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)


_ENVS = {}


def get_env(name):
    if name not in _ENVS:
        env = make_env(PAPER_TASKS[name])
        _ENVS[name] = (env, make_detector(PAPER_TASKS[name], env))
    return _ENVS[name]


def reset(name, seed, steps=4):
    env, detector = get_env(name)
    np.random.seed(seed)
    env.reset()
    for _ in range(steps):
        env.step(np.zeros(env.action_dim))
    return env, detector


def groundings(detector):
    return detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)


def cube_objects(env):
    if hasattr(env, "cubes"):
        return list(env.cubes)
    cubes = [env.cube1, env.cube2, env.cube3]
    return cubes + [env.cube4] if env.env_id == "Hanoi4x3" else cubes


def cube_xy(env):
    return {cube.name: env.sim.data.body_xpos[env.sim.model.body_name2id(cube.root_body)][:2].copy()
            for cube in cube_objects(env)}


def teleport(env, cube, xyz):
    env.sim.data.set_joint_qpos(cube.joints[0], np.concatenate([np.asarray(xyz, float), [1, 0, 0, 0]]))
    qvel = env.sim.model.get_joint_qvel_addr(cube.joints[0])
    env.sim.data.qvel[qvel[0]:qvel[1]] = 0


def settle(env, steps=30):
    env.sim.forward()
    for _ in range(steps):
        env.step(np.zeros(env.action_dim))


def nominal_slots(env, task):
    """Nominal (x, y) spawn of each cube for row tasks, or the tower centre for Hanoi."""
    if task.env_name in ("Hanoi", "Hanoi4x3"):
        return [(0.0, -0.18)]
    if task.env_name == "CubeSorting":
        return [(0.0, y) for y in np.linspace(env.line_y_start, env.line_y_end, env.num_cubes)]
    if task.env_name == "AssemblyLineSorting":
        return [(env.assembly_line_x, y)
                for y in np.linspace(env.assembly_line_y_start, env.assembly_line_y_end, env.num_cubes)]
    return [(env.line_x, y) for y in np.linspace(env.line_y_start, env.line_y_end, env.active_num_cubes)]


@pytest.mark.parametrize("name", list(PAPER_TASKS))
def test_config_fields(name):
    task = PAPER_TASKS[name]
    assert (task.paper_success, task.paper_advancement) == PAPER_VALUES[name]
    assert task.expected_plan_len == PLAN_LENGTHS[name]
    assert os.path.exists(os.path.join(REPO_ROOT, task.pddl_dir, "domain.pddl"))
    assert get_env(name)[0].robots[0].name == "Kinova3"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", list(PAPER_TASKS))
def test_initial_plan_reaches_goal(name, seed):
    task = PAPER_TASKS[name]
    _, detector = reset(name, seed)
    plan, goals, init, goal = plan_for_state(groundings(detector), detector, task)
    assert plan, f"no plan for {name} seed {seed}"
    assert len(plan) == PLAN_LENGTHS[name], plan
    reached, _, error = simulate_plan(plan, init, goal, os.path.join(REPO_ROOT, task.pddl_dir, "domain.pddl"))
    assert reached, error
    assert len(parse_moves(plan)) == len(plan) // 2
    assert failed_goals(detector, goals), "goals must not already hold at spawn"
    assert missing_preconditions(groundings(detector), detector, parse_moves(plan)[0]) == []


@pytest.mark.parametrize("name", list(PAPER_TASKS))
def test_spawn_within_noise_and_varies_by_seed(name):
    task = PAPER_TASKS[name]
    env, _ = get_env(name)
    spawns = []
    for seed in SEEDS:
        np.random.seed(seed)
        env.reset()
        spawn = cube_xy(env)
        slots = nominal_slots(env, task)
        if task.env_name in ("Hanoi", "Hanoi4x3"):
            positions = list(spawn.values())
            for position in positions:
                assert np.all(np.abs(position - slots[0]) <= NOISE + 1e-9), position
            assert max(np.abs(p - positions[0]).max() for p in positions) < 1e-6, "the tower must stay stacked"
        else:
            active = [spawn[f"cube{i}"] for i in range(len(slots))]
            used = set()
            for position in active:
                matches = [i for i, slot in enumerate(slots) if np.all(np.abs(position - slot) <= NOISE + 1e-9)]
                assert len(matches) == 1, (position, slots)
                used.add(matches[0])
            assert len(used) == len(slots), "each spawn slot is used exactly once"
        spawns.append(np.concatenate(list(spawn.values())))
    assert all(np.abs(spawns[0] - other).max() > 1e-3 for other in spawns[1:]), "the spawn must vary by seed"


@pytest.mark.parametrize("name,fixed", [("tower_3_set", True), ("tower_4_set", True), ("tower_3_random", False)])
def test_tower_order(name, fixed):
    env, _ = get_env(name)
    orders = set()
    for seed in SEEDS:
        np.random.seed(seed)
        env.reset()
        spawn = cube_xy(env)
        orders.add(tuple(sorted(spawn, key=lambda cube: spawn[cube][1])))
    assert (len(orders) == 1) == fixed, orders


def test_hanoi_noise_is_off_unless_requested():
    from robosuite.environments.manipulation.hanoi import Hanoi

    assert PAPER_TASKS["hanoi_3"].env_kwargs["cube_init_pos_noise_uniform"] == NOISE
    env = make_env(PAPER_TASKS["hanoi_3"], env_kwargs_override={"cube_init_pos_noise_uniform": 0.0})
    assert isinstance(env, Hanoi)
    np.random.seed(0)
    env.reset()
    assert np.allclose(cube_xy(env)["cube3"], [0.0, -0.18])


@pytest.mark.parametrize("name", ["hanoi_3", "hanoi_4"])
def test_on_peg_holds_at_noise_extremes(name):
    env, detector = get_env(name)
    np.random.seed(0)
    env.reset()
    for corner in ([1, 1], [-1, 1], [1, -1], [-1, -1]):
        offset = np.array(corner) * NOISE
        for cube in cube_objects(env):
            body = env.sim.model.body_name2id(cube.root_body)
            position = env.sim.data.body_xpos[body].copy()
            position[:2] = np.array([0.0, -0.18]) + offset
            teleport(env, cube, position)
        settle(env, 4)
        state = groundings(detector)
        base = cube_objects(env)[-1].name
        assert state[f"on({base},peg1)"], corner


def test_bins_swapped():
    plain, _ = get_env("sort_2color")
    swapped, _ = get_env("sort_2color_swapped")
    for env in (plain, swapped):
        np.random.seed(0)
        env.reset()
    body_y = lambda env, platform: env.sim.data.body_xpos[env.sim.model.body_name2id(platform.root_body)][1]
    rgba = lambda env, platform: env.sim.model.geom_rgba[env.sim.model.geom_name2id(platform.visual_geoms[0])][:3]
    assert body_y(plain, plain.platform1) == pytest.approx(-0.2) and body_y(plain, plain.platform2) == pytest.approx(0.2)
    assert body_y(swapped, swapped.platform1) == pytest.approx(0.2) and body_y(swapped, swapped.platform2) == pytest.approx(-0.2)
    assert swapped.platform1_pos[1] == pytest.approx(0.2) and swapped.platform2_pos[1] == pytest.approx(-0.2)
    # The colours stay with the bins: blue is platform1 in both layouts.
    for env in (plain, swapped):
        assert np.allclose(rgba(env, env.platform1), [0, 0, 1])
        assert np.allclose(rgba(env, env.platform2), [1, 0, 0])


def test_swapped_cube_goal_targets_same_colored_bin():
    env, detector = reset("sort_2color_swapped", 0)
    plan, goals, _, _ = plan_for_state(groundings(detector), detector, PAPER_TASKS["sort_2color_swapped"])
    for goal in goals:
        _, cube, platform = goal.split()
        size = env.cube_sizes[int(cube.replace("cube", ""))]
        assert platform == ("platform1" if size == "small" else "platform2")


@pytest.mark.parametrize("name", ["tower_3_set", "tower_3_random", "tower_4_set"])
def test_tower_color_mapping_matches_env(name):
    env, _ = get_env(name)
    task = PAPER_TASKS[name]
    rgba = {"blue cube": [0, 0, 1, 1], "red cube": [1, 0, 0, 1], "green cube": [0, 1, 0, 1], "yellow cube": [1, 1, 0, 1]}
    for color, cube in task.color_to_pddl.items():
        index = int(cube.replace("cube", ""))
        if index < env.max_cubes:
            assert env.cube_colors[index] == rgba[color], (color, cube)
    sizes = [env.cube_sizes[i] for i in range(env.max_cubes)]
    assert sizes == sorted(sizes)


@pytest.mark.parametrize("name", ["hanoi_3", "hanoi_4"])
def test_hanoi_color_mapping_is_smallest_first(name):
    env, _ = get_env(name)
    mapping = PAPER_TASKS[name].color_to_pddl
    order = ["blue cube", "red cube", "green cube", "yellow cube"][:len(cube_objects(env))]
    assert [mapping[color] for color in order] == [cube.name for cube in cube_objects(env)]
    assert sorted(mapping) == sorted(order)
    assert env.cube1.size[0] < env.cube2.size[0] < env.cube3.size[0]


@pytest.mark.parametrize("name", ["sort_2color", "sort_2color_swapped", "sort_3color"])
def test_sorting_has_no_color_mapping(name):
    assert PAPER_TASKS[name].color_to_pddl is None


def place_goal_state(env, detector, name, goals):
    """Teleport the cubes straight to the goal configuration."""
    task = PAPER_TASKS[name]
    by_name = {cube.name: cube for cube in cube_objects(env)}
    half = lambda cube: float(by_name[cube].size[2])
    if task.env_name in ("Hanoi", "Hanoi4x3", "HeightStacking"):
        cubes = sorted(by_name, key=half, reverse=True)
        base = (np.array(env.pegs_xy_center[2][:2]) if task.env_name != "HeightStacking" else env.platform_pos[:2])
        height = 0.8 + 0.002
        if task.env_name == "HeightStacking":
            height = env.platform_pos[2] + 0.001
        for cube in cubes:
            teleport(env, by_name[cube], [base[0], base[1], height + half(cube)])
            height += 2 * half(cube) + 0.0005
    else:
        counts = {}
        for goal in goals:
            _, cube, target = goal.split()
            center = (env.platform1_pos if target == "platform1" else env.platform2_pos) if target.startswith("platform") \
                else env.bin_positions[int(target.replace("bin", ""))]
            spot = counts.get(target, 0)
            counts[target] = spot + 1
            offset = [(-0.03, -0.03), (0.03, 0.03), (-0.03, 0.03), (0.03, -0.03)][spot]
            teleport(env, by_name[cube], [center[0] + offset[0], center[1] + offset[1], center[2] + 0.03])
    settle(env)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("name", list(PAPER_TASKS))
def test_teleported_goal_state_is_success(name, seed):
    task = PAPER_TASKS[name]
    env, detector = reset(name, seed)
    state = groundings(detector)
    _, goals, _, _ = plan_for_state(state, detector, task)
    assert not goals_hold(detector, goals), "spawn must not count as success"
    place_goal_state(env, detector, name, goals)
    assert failed_goals(detector, goals) == []


def test_fallen_cube_is_not_success():
    env, detector = reset("hanoi_3", 0)
    _, goals, _, _ = plan_for_state(groundings(detector), detector, PAPER_TASKS["hanoi_3"])
    place_goal_state(env, detector, "hanoi_3", goals)
    assert goals_hold(detector, goals)
    teleport(env, env.cube1, [0.0, 0.02, 0.85])
    settle(env)
    assert failed_goals(detector, goals) == ["on cube1 cube2"]


def test_four_cube_pile_in_one_bin_is_success():
    """Detector on() is false for the top of a pile; the goal check must not depend on it."""
    task = PAPER_TASKS["sort_2color"]
    env, detector = get_env("sort_2color")
    seed = next(s for s in range(50) if _single_color_seed(env, s))
    env, detector = reset("sort_2color", seed)
    _, goals, _, _ = plan_for_state(groundings(detector), detector, task)
    target = goals[0].split()[2]
    center = env.platform1_pos if target == "platform1" else env.platform2_pos
    half = float(env.cubes[0].size[2])
    for level, cube in enumerate(env.cubes):
        teleport(env, cube, [center[0], center[1], center[2] + 0.002 + half + 2 * level * half + level * 0.0005])
    settle(env)
    state = groundings(detector)
    assert not state[f"on(cube3,{target})"], "pile top: this is why goals are not checked with on()"
    assert failed_goals(detector, goals) == []


def _single_color_seed(env, seed):
    np.random.seed(seed)
    env.reset()
    return len(set(env.cube_sizes)) == 1


def test_cube_outside_bin_wall_is_not_success():
    env, detector = reset("sort_2color", 0)
    _, goals, _, _ = plan_for_state(groundings(detector), detector, PAPER_TASKS["sort_2color"])
    place_goal_state(env, detector, "sort_2color", goals)
    assert goals_hold(detector, goals)
    _, cube, platform = goals[0].split()
    center = env.platform1_pos if platform == "platform1" else env.platform2_pos
    teleport(env, env.cubes[int(cube[4:])], [center[0], center[1] + 0.09, 0.83])
    settle(env)
    assert detector.in_platform(cube, platform), "this is the case the detector's in() alone accepts"
    assert goals[0] in failed_goals(detector, goals)


def test_replan_from_partial_state_skips_sorted_cubes():
    env, detector = reset("sort_3color", 0)
    task = PAPER_TASKS["sort_3color"]
    _, goals, _, _ = plan_for_state(groundings(detector), detector, task)
    _, cube, target = goals[0].split()
    center = env.bin_positions[int(target.replace("bin", ""))]
    teleport(env, env.cubes[int(cube[4:])], [center[0] + 0.04, center[1] + 0.04, center[2] + 0.03])
    settle(env)
    plan, _, _, _ = plan_for_state(groundings(detector), detector, task, goals=goals)
    assert len(plan) == 6, plan
    assert cube.upper() not in " ".join(plan)


def test_planner_garbage_is_rejected(monkeypatch):
    from planning import planner
    from planning.task_configs import PlannerError

    _, detector = reset("hanoi_3", 0)
    monkeypatch.setattr(planner.subprocess, "getoutput", lambda _: "ff: found legal plan as follows\nsh: ./ff: cannot execute\ntime spent:")
    with pytest.raises(PlannerError):
        plan_for_state(groundings(detector), detector, PAPER_TASKS["hanoi_3"])


def test_no_problem_files_left_behind():
    for task in PAPER_TASKS.values():
        leftovers = [f for f in os.listdir(os.path.join(REPO_ROOT, task.pddl_dir))
                     if f.startswith(f"problem_run_{os.getpid()}")]
        assert leftovers == []


class FakeDetector:
    def __init__(self):
        self.held = False
        self.placed = False

    def grasped(self, cube):
        return self.held

    def on(self, cube, location):
        return self.placed


def test_monitor_times_out_only_without_progress():
    detector = FakeDetector()
    monitor = ProgressMonitor(detector, limit=600)
    monitor.arm()
    monitor.set_move("cube1", "peg3")
    for _ in range(500):
        monitor.on_step()
    detector.held = True
    monitor.on_step()
    assert monitor.stage == 1
    for _ in range(500):
        monitor.on_step()
    detector.held, detector.placed = False, True
    monitor.on_step()
    assert monitor.stage == 2
    for _ in range(600):
        monitor.on_step()
    with pytest.raises(EpisodeTimeout):
        monitor.on_step()
    assert monitor.steps == 500 + 1 + 500 + 1 + 601


def test_monitor_disarmed_never_times_out():
    monitor = ProgressMonitor(FakeDetector(), limit=5)
    for _ in range(50):
        monitor.on_step()
    assert monitor.steps == 50


def test_wilson_and_summary():
    low, high = wilson_ci(50, 50)
    assert low == pytest.approx(0.9287, abs=1e-3) and high == pytest.approx(1.0)
    low, high = wilson_ci(39, 50)
    assert low == pytest.approx(0.6476, abs=1e-3) and high == pytest.approx(0.8723, abs=1e-3)
    records = [dict(success=i < 39, advancement=1.0 if i < 39 else 0.5, env_steps=10, wall_s=1.0, unsolvable_resets=i == 0)
               for i in range(50)]
    summary = summarize(records, PAPER_TASKS["hanoi_4"])
    assert summary["episodes"] == 50 and summary["successes"] == 39
    assert summary["success_pct"] == pytest.approx(78.0)
    assert summary["advancement_pct"] == pytest.approx(89.0)
    assert summary["unsolvable_resets"] == 1
    assert summary["paper_success_in_ci"] is True


@pytest.mark.parametrize("name", sorted(PAPER_TASKS))
def test_workcell_map_matches_simulator(name):
    """Vision mode takes fixture positions from the workcell map, never the simulator."""
    static = PAPER_TASKS[name].static_positions
    assert static
    for seed in (0, 7, 5000):
        _, detector = reset(name, seed, steps=1)
        sim = {k: np.asarray(v) for k, v in detector.get_all_objects_pos().items()
               if "cube" not in k and k not in ("gripper", "aperture")}
        assert sorted(sim) == sorted(static)
        for key, pos in static.items():
            assert np.array_equal(np.asarray(pos, dtype=float), sim[key]), (key, pos, sim[key])


@pytest.mark.parametrize("name", ["sort_2color", "sort_2color_swapped", "sort_3color"])
def test_sort_area_colours_match_cube_colours(name):
    """type_match() plus the task's area colours gives each cube the colour YOLO sees."""
    task = PAPER_TASKS[name]
    rgba_name = {(0, 0, 1): "blue cube", (1, 0, 0): "red cube", (0, 1, 0): "green cube"}
    for seed in range(4):
        env, detector = reset(name, seed, steps=1)
        state = groundings(detector)
        for index, cube in enumerate(detector.objects):
            areas = [a for a in task.area_colors if state.get(f"type_match({cube},{a})")]
            assert len(areas) == 1
            rgba = tuple(int(round(x)) for x in list(env.cubes[index].rgba)[:3])
            assert task.area_colors[areas[0]] == rgba_name[rgba], (cube, areas, rgba)
