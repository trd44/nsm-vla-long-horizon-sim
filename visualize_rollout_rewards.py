#%%
# add PYTHONPATH to the env path
import pandas as pd
import matplotlib.pyplot as plt

#%%
# plot a colored scatter plot of the collision penalty where each point has a different color based on the penalty
def plot_collision_penalty(collision_penalty, first_coords, second_coords):
    """Given the y, z coordinates of the closest points and their corresponding collision penalties, plot a colored scatter plot of the collision penalty where each point has a different color based on the penalty

    Args:
        collision_penalty (list): a list of collision penalties
        first_coords (list): a list of first coordinates of the closest points
        second_coords (list): a list of second coordinates of the closest points
    """
    # plot the collision penalty
    plt.figure()
    plt.scatter(first_coords, second_coords, c=collision_penalty, cmap='viridis')
    plt.colorbar()
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('Collision Penalty')
    plt.show()
#%%
rollout_path = 'learning/policies/nut_assembly/pick-up-nut-from-peg/seed_0/rollout_1.csv'
# load the csv file containing the rollout data
rollout_data = pd.read_csv(rollout_path)
rollout_data

# %%
# extract collision penalty and closest points where the x coordnate of the point is in the range [-0.01, 0.01]
# find the rows where the x coordnate of the point is in the range [-0.01, 0.01]
x_filter = rollout_data['closest_x'].apply(lambda x: -0.05 <= x <= 0.05)
collision_penalty = rollout_data['collision_penalty'][x_filter]
closest_y = rollout_data['closest_y'][x_filter]
closest_z = rollout_data['closest_z'][x_filter]
collision_penalty

# %%
# plot the collision penalty
plot_collision_penalty(collision_penalty, closest_y, closest_z)
# %%
