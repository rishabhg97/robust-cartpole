import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scienceplots

# Use the 'science' style
plt.style.use('science')

every_state_return = np.load('./logs/every_state_returns_ppo.npy')
robust_every_state_return = np.load('./logs/robust_every_state_returns.npy')

def plot_every_state_return(every_state_return, savefig_path):
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
    fig, (ax_state, ax_every_state_return) = plt.subplots(
        1, 2, figsize=(4, 12), gridspec_kw={'wspace': 0.0})  # Adjust the figsize and wspace

    # Plot the state table in the left subplot
    cax = ax_state.matshow(matrix, aspect="auto", cmap='magma')
    ax_state.set_xlabel('Environment Parameters')
    ax_state.yaxis.tick_right()  # Move y-ticks to the right
    ax_state.xaxis.tick_bottom()
    ax_state.set_xticks(np.arange(0, 5))
    ax_state.set_xticklabels(state_x_ticks, rotation=90)

    # Create a colormap for the three values (0, 1, 2)
    cmap = plt.get_cmap('magma', 3)

    # Create a legend for the state table on the left side and rotated 90 degrees
    legend_labels = [mpatches.Patch(color=cmap(
        i), label=tick_lib[i]) for i in range(3)]
    legend = ax_state.legend(handles=legend_labels, title='Values',
                            loc='upper left', ncol=3, bbox_to_anchor=(0, 1.07))
    legend.get_title()

    # Plot the Q-table in the right subplot
    every_state_return = every_state_return.reshape(-1, 1)
    every_state_return = np.concatenate(
        [every_state_return]*10, axis=1)

    im1 = ax_every_state_return.imshow(every_state_return,  cmap='magma')
    ax_every_state_return.set_xlabel('$r$')
    ax_every_state_return.set_ylabel('State')
    ax_every_state_return.set_xticks([])
    fig.colorbar(im1, ax=ax_every_state_return)

    # Set the title for the entire figure
    fig.suptitle('Adversary Reward For Reaching A State', fontsize=16)

    # Save the figure
    plt.tight_layout()
    plt.savefig(savefig_path, bbox_inches='tight')
    
    # Show the plot
    plt.show()


print('Every state return:', np.mean(every_state_return), np.std(every_state_return))
print('Robust every state return:', np.mean(robust_every_state_return), np.std(robust_every_state_return))

plot_every_state_return(every_state_return, './logs/every_state_return_ppo.pdf')
plot_every_state_return(robust_every_state_return, './logs/robust_every_state_return_ppo.pdf')