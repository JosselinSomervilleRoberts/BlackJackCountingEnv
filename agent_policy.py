import pickle


def policy_and_reward_save(name, policy, reward, state_action_to_idx):
    with open('save/' + name + '_policy.pickle', 'wb') as f:
        pickle.dump(policy, f, pickle.HIGHEST_PROTOCOL)
    with open('save/' + name + '_reward.pickle', 'wb') as f:
        pickle.dump(reward, f, pickle.HIGHEST_PROTOCOL)
    with open('save/' + name + '_state.pickle', 'wb') as f:
        pickle.dump(state_action_to_idx, f, pickle.HIGHEST_PROTOCOL)

def policy_and_reward_load(name):
    try:
        policy, reward, state_action_to_idx = None, None, None
        with open('save/' + name + '_policy.pickle', 'rb') as f:
            policy = pickle.load(f)
        with open('save/' + name + '_reward.pickle', 'rb') as f:
            reward = pickle.load(f)
        with open('save/' + name + '_state.pickle', 'rb') as f:
            state_action_to_idx = pickle.load(f)
        return True, policy, reward, state_action_to_idx
    except:
        return False, None, None, None


class AgentPolicy:

    def __init__(self):
        self.policy = {}
        self.format_state = (lambda x: x)
        self.default_action = None

    def load(self, name, format_state, default_action):
        success, self.policy, self.reward, self.state_action_to_idx = policy_and_reward_load(name)
        if success: return self.load_from_policy(self.policy, format_state, default_action)
        raise Exception("Could not load policy " + name)

    def load_from_policy(self, policy, format_state, default_action):
        self.policy = policy
        self.format_state = format_state
        self.default_action = default_action
        return True

    def choose_action(self, state):
        action = self.default_action
        formated_state = self.format_state(state)
        if formated_state in self.policy: action = self.policy[formated_state]
        if action is None: raise Exception("No action for state " + str(state))
        return action

