from deck import DecksOfCards
from blackjack_env import BlackJackBettingEnv
from playing_env import BlackJackPlayingEnv, ACTION_STAND
from agent_policy import AgentPolicy
from policy_iteration_betting import format_betting_state
from policy_iteration_counting import format_state_counting
from tqdm import tqdm

def evaluate_policy(policy_player_name, policy, rules, list_money, list_true_count, N_EPISODES=10000, initial_money=100):
    decks = DecksOfCards(nb_decks=6, fraction_not_in_play=0.2)
    playing_env = BlackJackPlayingEnv(decks = decks, rules=rules)
    player_agent = AgentPolicy()
    player_agent.load(name=policy_player_name, format_state=format_state_counting, default_action=ACTION_STAND)
    betting_env = BlackJackBettingEnv(playing_env, player_agent, initial_money=initial_money, min_bet=rules["min_bet"], max_bet=rules["max_bet"])


    rewards = [0]
    prev_cum_reward = 0
    cum_rewards = [0]
    bets = []
    state, _ = betting_env.reset()
    player_money = [state[0]]
    for i in tqdm(range(N_EPISODES)):
        # Place the bet
        state = format_betting_state(state, list_money, list_true_count)
        action = policy[state]
        state, reward, done, _ = betting_env.step(action)

        # Update the rewards
        prev_cum_reward += reward
        cum_rewards.append(prev_cum_reward)
        rewards.append(reward)
        bets.append(action)
        player_money.append(state[0])

        if done:
            break

    plt.plot(player_money, label="Money")
    plt.plot(cum_rewards, label="Cumulative rewards")
    plt.plot(bets, label="Bets")
    plt.legend()
    plt.show()