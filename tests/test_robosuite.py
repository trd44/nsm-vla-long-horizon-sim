# A simple file that launches a robosuite environment and takes random actions for a few steps.
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.utils.detector import PatternReplicationDetector

if __name__ == "__main__":
    # Load the NutAssembly environment
    env = suite.make(
        env_name="PatternReplication",
        robots="Kinova3",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    # Wrap the environment with the GymWrapper
    env = GymWrapper(env)
    detector = PatternReplicationDetector(env)

    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, _,_, info = env.step(action)
        state = detector.get_pattern_replication_goal()
        print(state)
        print()
        env.render()
    env.close()