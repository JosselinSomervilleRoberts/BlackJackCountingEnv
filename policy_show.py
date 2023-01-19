import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import numpy as np


def show_sub_policy(fig, axs, idx_rox_ax, policy, reward, title, xlim, ylim, ylegend="Player\'s sum"):
    # make a color map of fixed colors
    action_idx_to_color = ['red', 'yellow', 'blue', 'green', 'grey', 'white']
    extent = [xlim[0] - 0.5, xlim[1] + 0.5, ylim[1] + 0.5, ylim[0] - 0.5]

    # Policy
    cmap = ListedColormap(action_idx_to_color)
    bounds=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5]
    norm = BoundaryNorm(bounds, cmap.N)
    axs[idx_rox_ax, 0].set_title(title + " policy")
    axs[idx_rox_ax, 0].imshow(policy, cmap=cmap, extent=extent, norm=norm)
    axs[idx_rox_ax, 0].set_xlabel('Dealer\'s card')
    axs[idx_rox_ax, 0].set_ylabel(ylegend)
    axs[idx_rox_ax, 0].set_xticks(np.arange(xlim[0], xlim[1]+1, 1))
    axs[idx_rox_ax, 0].set_yticks(np.arange(ylim[0], ylim[1]+1, 1))

    # reward
    cmap_reward = 'PiYG'
    norm_reward = Normalize(vmin=-2, vmax=2)
    axs[idx_rox_ax, 1].set_title(title+ ' avg reward')
    im2 = axs[idx_rox_ax, 1].imshow(reward, cmap=cmap_reward, extent=extent, norm = norm_reward)
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    axs[idx_rox_ax, 1].set_xlabel('Dealer\'s card')
    axs[idx_rox_ax, 1].set_ylabel(ylegend)
    axs[idx_rox_ax, 1].set_xticks(np.arange(xlim[0], xlim[1]+1, 1))
    axs[idx_rox_ax, 1].set_yticks(np.arange(ylim[0], ylim[1]+1, 1))

def show_policy(policy, expected_values, state_action_to_idx, nb_iterations):
    fig, axs = plt.subplots(nrows=3, ncols=2, gridspec_kw={'width_ratios':[1,1], 'height_ratios':[20 - 5 +1, 20 - 13 + 1, 10]})
    plt.subplots_adjust(hspace=0.1, wspace = 0.5, top=0.88)
    fig.set_figwidth(5)
    fig.set_figheight(10)
    fig.suptitle('Optimized policy')

    # Hard policy
    hard_policy = np.zeros((16, 10))
    hard_reward = np.zeros((16, 10))
    for player_sum in range(5, 20+1):
        for dealer_card in range(2, 11+1):
            state_double = (player_sum, dealer_card, 0, 0, 1)
            state_no_double = (player_sum, dealer_card, 0, 0, 0)
            state = None
            if state_double in policy: state = state_double
            elif state_no_double in policy: state = state_no_double
            action = 5
            if state is not None: action = policy[state]
            hard_policy[player_sum-5, dealer_card-2]  = action
            if state is not None: hard_reward[player_sum-5, dealer_card-2] = expected_values[state_action_to_idx[(state, action)]] / nb_iterations
    show_sub_policy(fig, axs, 0, hard_policy, hard_reward, 'Hard', [2, 11], [5, 20])

    # Soft policy
    soft_policy = np.zeros((8, 10))
    soft_reward = np.zeros((8, 10))
    for player_sum in range(13, 20+1):
        for dealer_card in range(2, 11+1):
            state_double = (player_sum, dealer_card, 1, 0, 1)
            state_no_double = (player_sum, dealer_card, 1, 0, 0)
            state = None
            if state_double in policy: state = state_double
            elif state_no_double in policy: state = state_no_double
            action = 5
            if state is not None: action = policy[state]
            soft_policy[player_sum-13, dealer_card-2]  = action
            if state is not None: soft_reward[player_sum-13, dealer_card-2] = expected_values[state_action_to_idx[(state, action)]] / nb_iterations
    show_sub_policy(fig, axs, 1, soft_policy, soft_reward, 'Soft', [2, 11], [13, 20])

    # Split policy
    split_policy = np.zeros((10, 10))
    split_reward = np.zeros((10, 10))
    for card in range(2, 11+1):
        for dealer_card in range(2, 11+1):
            player_sum = 2 * card
            if player_sum > 21: player_sum -= 10
            state_double = (player_sum, dealer_card, card==11, 1, 1)
            if state_double in policy:
                action = policy[state_double]
                split_policy[card-2, dealer_card-2] = action
                split_reward[card-2, dealer_card-2] = expected_values[state_action_to_idx[(state_double, action)]] / nb_iterations
            else: split_policy[card-2, dealer_card-2] = 5
    show_sub_policy(fig, axs, 2, split_policy, split_reward, 'Split', [2, 11], [2, 11], ylegend="Split card")

    # Colors
    values = np.unique(np.concatenate((hard_policy.ravel(), soft_policy.ravel(), split_policy.ravel())))
    actions_list = ['Hit', 'Stick', 'Double', 'Split', 'Surrender', 'no data']
    action_idx_to_color = ['red', 'yellow', 'blue', 'green', 'grey', 'white']
    patches = [ mpatches.Patch(color=action_idx_to_color[int(values[i])], label=actions_list[int(values[i])]) for i in range(len(values)) ]
    fig.legend(handles=patches, loc='upper left')

    plt.show()