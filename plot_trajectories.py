import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import scienceplots

# Use the 'science' style
plt.style.use('science')

# Load the Q-table
Q = np.load('./logs/qtable_999.npy')

# Load the trajectories
trajectories = np.load('./logs/trajectories.npy')

# Load every state return npy
every_state_return = np.load('./logs/every_state_returns_ppo.npy')

x_ticks = ('-$\Delta_g$', '$0_g$', '$+\Delta_g$', '-$\Delta_{m_c}$', '$0_{m_c}$', '$+\Delta_{m_c}$', '-$\Delta_{m_p}$', '$0_{m_p}$', '$+\Delta_{m_p}$', '-$\Delta_l$', '$0_l$', '$+\Delta_l$', '-$\Delta_{F_m}$', '$0_{F_m}$', '$+\Delta_{F_m}$')
plain_text_x_ticks = ('-delta_gravity', 'zero_gravity', '+delta_gravity', '-delta_masscart', 'zero_masscart', '+delta_masscart', '-delta_masspole', 'zero_masspole', '+delta_masspole', '-delta_length', 'zero_length', '+delta_length', '-delta_force_mag', 'zero_force_mag', '+delta_force_mag')

# Create a 243 by 5 matrix to store the state information
matrix = np.zeros((243, 5))

# Populate the matrix
for i in range(243):
    matrix[i, 0] = i // 81
    matrix[i, 1] = (i % 81) // 27
    matrix[i, 2] = (i % 27) // 9
    matrix[i, 3] = (i % 9) // 3
    matrix[i, 4] = i % 3

# Define state_x_ticks with appropriate labels
state_x_ticks = ('$g$', '$m_{cart}$', '$m_{pole}$', '$l$', '$F_{mag}$')
plain_text_state_x_ticks = ('gravity', 'masscart', 'masspole', 'length', 'force_mag')

# Create a figure with appropriate size
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the trajectories on the 243 by 15 grid
im1 = ax.imshow(Q, interpolation='nearest', aspect='auto', cmap='Blues')
ax.set_xticks(np.arange(0, 15))
ax.set_xticklabels(x_ticks, fontsize=7)
ax.set_title('Q(s,a): Adversary Agent')

# Define a list of colors for trajectories and corresponding labels
colors = ['#FF0000', '#FF4500', '#FFA500', '#FF6347', '#FF7F50']

labels = ['$traj_1$', '$traj_2$', '$traj_3$', '$traj_4$', '$traj_5$']

legend_patches = []

def get_state_string(matrix, state_index, state_x_ticks):
    state = matrix[state_index]
    state_string = ''
    for i in range(5):
        state_string += state_x_ticks[i] + ': ' + str(state[i]) + ', '
    return state_string[:-2]

for traj_id, traj in enumerate(trajectories):
    print('Traj ID:', traj_id)
    
    color = colors[traj_id % len(colors)]  # Assign a unique color to each trajectory
    
    for i in range(len(traj) - 1):
        print('Traj step:', i)

        # Extract (s, a, s') transitions
        (s, a, s_next) = traj[i]

        # Print the state, action, and next state
        print('s:', get_state_string(matrix, s, plain_text_state_x_ticks))
        print('a:', plain_text_x_ticks[a])
        print('r:', every_state_return[s_next])

        # print('s_next:', get_state_string(matrix, s_next, plain_text_state_x_ticks))

        # Calculate the grid cell coordinates
        x_start, y_start = a, s
        x_end, y_end = a, s_next

        # Create a rounded arrow representing the transition with the assigned color
        arrow = FancyArrowPatch((x_start, y_start), (x_end, y_end), color=color,
                                arrowstyle='->, head_width=0.4, head_length=0.4',
                                mutation_scale=15)
        ax.add_patch(arrow)

        # Annotate the arrow with trajectory ID and step
        arrow_text = f'Traj: {traj_id}, Step: {i}'
        ax.annotate(arrow_text, xy=(x_start, y_start), xytext=(x_start, y_start - 10),
                    color='black', fontsize=8, ha='center')

    # Create legend patches for each trajectory
    legend_patches.append(plt.Line2D([], [], color=color, label=labels[traj_id]))

    print('s_final:', get_state_string(matrix, s_next, plain_text_state_x_ticks))
    print('-' * 50)

# Set axis labels and title
ax.set_xlabel('Action')
ax.set_ylabel('State')
ax.set_title('Trajectories Over Q-table')

# Add a legend for trajectories
ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0, 1.13), ncol=3, title='Trajectories'
            , fancybox=True, shadow=True, framealpha=1)

fig.colorbar(im1, ax=ax)

# Save the figure
plt.tight_layout()
plt.savefig('./logs/trajectories.pdf', bbox_inches='tight')

# Show the plot
# plt.show()
