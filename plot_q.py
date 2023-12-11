# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import scienceplots

# # Use the 'science' style
# plt.style.use('science')

# # Load the Q-table
# q_table_path = './logs/qtable_999.npy'
# Q = np.load(q_table_path)

# x_ticks = ('$g_{min}$', '$g_{orig}$', '$g_{max}$', '$m_{cart,min}$', '$m_{cart,orig}$', '$m_{cart,max}$', '$m_{pole,min}$', '$m_{pole,orig}$', '$m_{pole,max}$', '$l_{min}$', '$l_{orig}$', '$l_{max}$', '$F_{mag,min}$', '$F_{mag,orig}$', '$F_{mag,max}$')

# # Create a 243 by 5 matrix to store the state information
# # The matrix starts as 00000 and ends with 22222
# # The first column is the gravity, second is masscart, third is masspole, fourth is length, and fifth is force_mag

# matrix = np.zeros((243, 5))

# # Populate the matrix
# for i in range(243):
#     matrix[i, 0] = i // 81
#     matrix[i, 1] = (i % 81) // 27
#     matrix[i, 2] = (i % 27) // 9
#     matrix[i, 3] = (i % 9) // 3
#     matrix[i, 4] = i % 3

# # Convert this matrix into yticks
# y_ticks = []
# tick_lib = {0: 'min', 1: 'orig', 2: 'max'}
# for i in range(243):
#     y_tick = ''
#     for j in range(5):
#         if matrix[i, j] == 0:
#             y_tick += tick_lib[0]
#         elif matrix[i, j] == 1:
#             y_tick += tick_lib[1]
#         else:
#             y_tick += tick_lib[2]
#     y_ticks.append(y_tick)

# # Create a figure with appropriate size
# fig, ax = plt.subplots(figsize=(10, 10))

# # Plot the Q-table
# im = ax.imshow(Q, interpolation='nearest', aspect='auto', cmap='magma')
# ax.set_xlabel('Action')
# ax.set_xticks(np.arange(0, 15))
# ax.set_xticklabels(x_ticks, rotation=90)
# ax.set_ylabel('State')
# # ax.set_yticks(np.arange(0, 243))
# # ax.set_yticklabels(y_ticks, fontsize=2)
# ax.set_title('Q(s,a): Adversary Agent')
# fig.colorbar(im)

# # Save the figure
# plt.savefig('./logs/q_table.pdf', bbox_inches='tight')

# # Show the figure
# plt.show()


# # Create the figure and axis
# fig, ax = plt.subplots()

# # Create a heatmap for the state table
# cax = ax.matshow(matrix, aspect="auto", cmap='magma')

# # Set labels and ticks
# state_x_ticks = ('gravity', 'masscart', 'masspole', 'length', 'force_mag')
# ax.set_xlabel('Environment Parameters')
# ax.set_ylabel('State Index')
# ax.set_xticks(np.arange(0, 5))
# ax.set_xticklabels(state_x_ticks)


# # Create a colormap for the three values (0, 1, 2)
# cmap = plt.get_cmap('magma', 3)


# # Create a legend
# legend_labels = [mpatches.Patch(color=cmap(i), label=tick_lib[i]) for i in range(3)]
# legend = ax.legend(handles=legend_labels, title='Values', loc='upper center', ncol=3,
#                    bbox_to_anchor=(0.5, 1.3))  # Adjust bbox_to_anchor to place the legend

# # Create a colorbar as a legend
# # cbar = plt.colorbar(cax, label='Values')
# plt.savefig('./logs/state_table.pdf', bbox_inches='tight')
# # Show the plot
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scienceplots

# Use the 'science' style
# plt.style.use('science')
plt.style.use('science')

# Load the Q-table
log_dir = './stored_rewards_logs/'
q_table_path = log_dir + '/qtable_99999.npy'
Q = np.load(q_table_path)

# x_ticks = ('$g_{min}$', '$g_{orig}$', '$g_{max}$', '$m_{cart,min}$', '$m_{cart,orig}$', '$m_{cart,max}$', '$m_{pole,min}$', '$m_{pole,orig}$', '$m_{pole,max}$', '$l_{min}$', '$l_{orig}$', '$l_{max}$', '$F_{mag,min}$', '$F_{mag,orig}$', '$F_{mag,max}$')
x_ticks = ('-$\Delta_g$', '$0_g$', '$+\Delta_g$', '-$\Delta_{m_c}$', '$0_{m_c}$', '$+\Delta_{m_c}$', '-$\Delta_{m_p}$', '$0_{m_p}$', '$+\Delta_{m_p}$', '-$\Delta_l$', '$0_l$', '$+\Delta_l$', '-$\Delta_{F_m}$', '$0_{F_m}$', '$+\Delta_{F_m}$')


# Create a 243 by 5 matrix to store the state information
# The matrix starts as 00000 and ends with 22222
# The first column is the gravity, second is masscart, third is masspole, fourth is length, and fifth is force_mag

matrix = np.zeros((243, 5))

# Populate the matrix
for i in range(243):
    matrix[i, 0] = i // 81
    matrix[i, 1] = (i % 81) // 27
    matrix[i, 2] = (i % 27) // 9
    matrix[i, 3] = (i % 9) // 3
    matrix[i, 4] = i % 3

# Convert this matrix into yticks
y_ticks = []
tick_lib = {0: 'min', 1: 'orig', 2: 'max'}
for i in range(243):
    y_tick = ''
    for j in range(5):
        if matrix[i, j] == 0:
            y_tick += tick_lib[0]
        elif matrix[i, j] == 1:
            y_tick += tick_lib[1]
        else:
            y_tick += tick_lib[2]
    y_ticks.append(y_tick)

# Define state_x_ticks with appropriate labels
state_x_ticks = ('$g$', '$m_{cart}$', '$m_{pole}$', '$l$', '$F_{mag}$')

# Create a figure with appropriate size and shared y-axis
fig, (ax_state, ax_q) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 3], 'wspace': 0.15})

# Plot the state table in the left subplot
cax = ax_state.matshow(matrix, aspect="auto", cmap='magma')
ax_state.set_xlabel('Environment Parameters')
ax_state.yaxis.tick_right()  # Move y-ticks to the right
ax_state.xaxis.tick_bottom()
ax_state.set_xticks(np.arange(0, 5))
ax_state.set_xticklabels(state_x_ticks)

# Create a colormap for the three values (0, 1, 2)
cmap = plt.get_cmap('magma', 3)

# Create a legend for the state table on the left side and rotated 90 degrees
legend_labels = [mpatches.Patch(color=cmap(i), label=tick_lib[i]) for i in range(3)]
legend = ax_state.legend(handles=legend_labels, title='Values', loc='upper left', ncol=3, bbox_to_anchor=(0, 1.13))
legend.get_title()

# Plot the Q-table in the right subplot
im1 = ax_q.imshow(Q, interpolation='nearest', aspect='auto', cmap='magma')
ax_q.set_xlabel('Action')
ax_q.set_xticks(np.arange(0, 15))
ax_q.set_xticklabels(x_ticks, fontsize=7)
ax_q.set_ylabel('State')  # Share the 'State' label
ax_q.set_title('Q(s,a): Adversary Agent')
fig.colorbar(im1, ax=ax_q)

# Save the figure
plt.tight_layout()
plt.savefig(log_dir +'q_and_state_table.pdf', bbox_inches='tight')

# Show the plot
plt.show()
