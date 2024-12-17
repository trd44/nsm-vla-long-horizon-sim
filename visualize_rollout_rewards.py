#%%
# add PYTHONPATH to the env path
import pandas as pd
import matplotlib.pyplot as plt

#%%
# plot a colored scatter plot of the collision penalty where each point has a different color based on the penalty
def plot_collision_penalty(collision_penalty, collision_closest_points):
    """Given the y, z coordinates of the closest points and their corresponding collision penalties, plot a colored scatter plot of the collision penalty where each point has a different color based on the penalty

    Args:
        collision_penalty (list): a list of collision penalties
        collision_closest_points (List[Tuple]): a list of tuples containing the y, z coordinates of the closest points
    """
    # plot the collision penalty
    plt.figure()
    plt.scatter([point[0] for point in collision_closest_points], [point[1] for point in collision_closest_points], c=collision_penalty, cmap='viridis')
    plt.colorbar()
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('Collision Penalty')
    plt.show()
#%%
rollout_path = 'learning/policies/nut_assembly/pick-up-nut-from-peg/seed_0/rollout_3.csv'
# load the csv file containing the rollout data
rollout_data = pd.read_csv(rollout_path)
rollout_data
print(rollout_data['closest_point'].head())
print(rollout_data['closest_point'].dtype)

# %%
# extract collision penalty and closest points where the x coordnate of the point is in the range [-0.01, 0.01]
# find the rows where the x coordnate of the point is in the range [-0.01, 0.01]
x_filter = rollout_data['closest_point'].apply(lambda p: -0.01 <= p[0] <= 0.02)
collision_penalty = rollout_data['collision_penalty'][x_filter]
closest_points = rollout_data['closest_point'][x_filter]
print(collision_penalty)
print(closest_points)

# %%
# plot the collision penalty
plot_collision_penalty(collision_penalty, collision_closest_points)
# %%
