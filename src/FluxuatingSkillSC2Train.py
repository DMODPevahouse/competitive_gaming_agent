'''

'''
# USEFUL LINKS FOR THE PROJECT
# Units PYSC2: https://github.com/deepmind/pysc2/blob/master/pysc2/lib/units.py
# Functions PYSC2: https://github.com/deepmind/pysc2/blob/master/pysc2/lib/actions.py

import os
os.environ['SC2PATH'] = 'D:\Games\StarCraft II'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
from s2clientprotocol import sc2api_pb2 as sc_pb

possible_results = {
    sc_pb.Victory: 1,
    sc_pb.Defeat: -1,
    sc_pb.Tie: 0,
    sc_pb.Undecided: 0,
}


# Reinforcment Learning Algorithm
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, competitive='none', alternate='none'):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.count = 0
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.competitive = competitive
        self.alternate = alternate

    # 90% Chooses preferred action and 10% randomly for extra possibilities
    def choose_action(self, observation, e_greedy=0.9):
        self.check_state_exist(observation)
        if np.random.uniform() < e_greedy and self.competitive == 'none':
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)
            if self.alternate is not 'none':
                self.competitive = self.alternate
        elif np.random.uniform() < e_greedy and self.competitive == 'neutral':
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.argmin(np.abs(state_action))].index)
            if self.alternate is not 'none':
                self.competitive = 'none'
        elif np.random.uniform() < e_greedy and self.competitive == 'half':
            state_action = self.q_table.loc[observation, :]
            max_val = np.max(state_action)
            half_max = max_val / 2
            closest_to_half_max = min(state_action, key=lambda x: abs(x - half_max))
            action = np.random.choice(state_action[state_action == closest_to_half_max].index)
            if self.alternate is not 'none':
                self.competitive = 'none'
        elif np.random.uniform() < e_greedy and self.competitive == 'quarter':
            state_action = self.q_table.loc[observation, :]
            max_val = np.max(state_action)
            quarter_max = max_val / 4
            closest_to_quarter_max = min(state_action, key=lambda x: abs(x - quarter_max))
            action = np.random.choice(state_action[state_action == closest_to_quarter_max].index)
            if self.alternate is not 'none':
                self.competitive = 'none'
        elif np.random.uniform() < e_greedy and self.competitive == 'second':
            state_action = self.q_table.loc[observation, :]
            action = state_action[state_action == np.max(state_action)].index[1]
            if self.alternate is not 'none':
                self.competitive = 'none'
        else:
            action = np.random.choice(self.actions)
        return action

    # Takes the state and action and update table accordingly to learn over time
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[
            s, a]  # Get the value that was given for taking the action when we were first in the state
        # Determine the maximum possible value across all actions in the current state
        # and then discount it by the decay rate (0.9) and add the reward we received (can be terminal or not)
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
        else:  # Reward from last step of game is better
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self,
                          state):  # Check to see if the state is in the QTable already, and if not it will add it with a value of 0 for all possible actions.
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


# Simple Agent definition (both random and learning agent use this)
class Agent(base_agent.BaseAgent):
    # Base actions both Smart and Random Agent can perform
    actions = ("do_nothing", "harvest_minerals", "build_supply_depot", "build_barracks", "train_marine", "attack")

    # HELPER FUNCTIONS
    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)  # Normalize the array

    # AGENT ACTIONS

    def step(self, obs):
        super(Agent, self).step(obs)
        if obs.first():
            command_center = self.get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_minerals(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
            mineral_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.BattleStationMineralField,
                                   units.Neutral.BattleStationMineralField750,
                                   units.Neutral.LabMineralField,
                                   units.Neutral.LabMineralField750,
                                   units.Neutral.MineralField,
                                   units.Neutral.MineralField750,
                                   units.Neutral.PurifierMineralField,
                                   units.Neutral.PurifierMineralField750,
                                   units.Neutral.PurifierRichMineralField,
                                   units.Neutral.PurifierRichMineralField750,
                                   units.Neutral.RichMineralField,
                                   units.Neutral.RichMineralField750
                               ]]
            scv = random.choice(idle_scvs)
            distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", scv.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self, obs):
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(supply_depots) == 0 and obs.observation.player.minerals >= 100 and len(scvs) > 0):
            supply_depot_xy = (22, 26) if self.base_top_left else (35, 42)
            distances = self.get_distances(obs, scvs, supply_depot_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scv.tag, supply_depot_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and len(
                barrackses) == 0 and obs.observation.player.minerals >= 150 and len(scvs) > 0):
            barracks_xy = (22, 21) if self.base_top_left else (35, 45)
            distances = self.get_distances(obs, scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100 and free_supply > 0):
            barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            distances = self.get_distances(obs, marines, attack_xy)
            marine = marines[np.argmax(distances)]
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt("now", marine.tag,
                                                   (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()


# Takes the list of actions of the base agent, chooses one at random and then execute it
class RandomAgent(Agent):
    def step(self, obs):
        super(RandomAgent, self).step(obs)
        action = random.choice(self.actions)
        return getattr(self, action)(obs)


# Similar to Random Agent but also initialize the QLearning Table to know which actions can perform and learn
class SmartAgent(Agent):
    def __init__(self, competitive='none', alternate='none'):
        super(SmartAgent, self).__init__()
        self.qtable = QLearningTable(self.actions, competitive=competitive, alternate=alternate)
        self.new_game()

    def reset(self):
        super(SmartAgent, self).reset()
        print(self.qtable.q_table)
        self.qtable.count += 1
        if self.qtable.count == 100:
            self.qtable.q_table.to_excel(r'QLearningTable_basic.xlsx', sheet_name='QLearningTable_basic', index=False)
        self.new_game()

    # Start the new game and store actions and states for the reinforcement learning
    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None

    # Takes all the values of the game we find important and then returning those in a tuple to feed into our machine learning algorithm
    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        queued_marines = (completed_barrackses[0].order_length if len(completed_barrackses) > 0 else 0)

        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100

        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.get_enemy_completed_units_by_type(obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)

        # Return tuple
        return (len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                queued_marines,
                free_supply,
                can_afford_supply_depot,
                can_afford_barracks,
                can_afford_marine,
                len(enemy_command_centers),
                len(enemy_scvs),
                len(enemy_idle_scvs),
                len(enemy_supply_depots),
                len(enemy_completed_supply_depots),
                len(enemy_barrackses),
                len(enemy_completed_barrackses),
                len(enemy_marines))

    # Gets the current state of the game, feeds the state into the QLearningTable and the QLearningTable chooses an action
    def step(self, obs):
        super(SmartAgent, self).step(obs)
        state = str(self.get_state(obs))
        action = self.qtable.choose_action(state)
        if self.previous_action is not None:
            self.qtable.learn(self.previous_state, self.previous_action, obs.reward,
                              'terminal' if obs.last() else state)
        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)



from collections import namedtuple
import random
import numpy as np


Transition = namedtuple("Transition", ["s", "a", "s_1", "r", "done"])


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, item):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        out = random.sample(self.memory, batch_size)
        batched = Transition(*zip(*out))
        s = np.array(list(batched.s))
        a = np.array(list(batched.a))
        # a = np.expand_dims(np.array(list(batched.a)), axis=1)
        s_1 = np.array(list(batched.s_1))
        r = np.expand_dims(np.array(list(batched.r)), axis=1)
        done = np.expand_dims(np.array(list(batched.done)), axis=1)
        return [s, a, s_1, r, done]

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        result = []
        for i in range(self.__len__()):
            result.append(self.memory[i].__str__() + " \n")
        return "".join(result)
        

class Epsilon(object):
    def __init__(self, start=1.0, end=0.01, update_increment=0.01):
        self._start = start
        self._end = end
        self._update_increment = update_increment
        self._value = self._start
        self.isTraining = True
    
    def increment(self, count=1):
        self._value = max(self._end, self._value - self._update_increment*count)
        return self
        
    def value(self):
        if not self.isTraining:
            return 0.0
        else:
            return self._value
  
import torch
import torch.nn as nn
import torch.optim as optim
gpu = torch.device('cuda:0')
from collections import deque
import copy

class DQNAgent(Agent):
    def __init__(self, competitive='none', alternate='none', training=True):
        super(DQNAgent, self).__init__()
        self.model = DQNModel().to(gpu)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-8)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayMemory(100000)
        self.batch_size = 256
        self.gamma = 0.99
        self.epsilon = Epsilon(start=0.9, end=0.1, update_increment=0.0001)
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.15
        self.update_target_frequency = 100
        self.competitive = competitive
        self.training = training
        self.new_game()
        self.alternate = alternate

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None

    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        queued_marines = (completed_barrackses[0].order_length if len(completed_barrackses) > 0 else 0)

        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100

        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.get_enemy_completed_units_by_type(obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)

        # Return tuple
        return (len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                queued_marines,
                free_supply,
                can_afford_supply_depot,
                can_afford_barracks,
                can_afford_marine,
                len(enemy_command_centers),
                len(enemy_scvs),
                len(enemy_idle_scvs),
                len(enemy_supply_depots),
                len(enemy_completed_supply_depots),
                len(enemy_barrackses),
                len(enemy_completed_barrackses),
                len(enemy_marines))

    def train(self):
        #batch = self.replay_buffer.sample(self.batch_size)
        states, actions,  next_states, rewards, dones = self.replay_buffer.sample(self.batch_size)

        # Convert actions to numerical representation
        action_dict = {action: i for i, action in enumerate(self.actions)}
        actions = [action_dict[action] for action in actions]

        states = torch.tensor(states, dtype=torch.float).to(gpu)
        actions = torch.tensor(actions, dtype=torch.long).to(gpu)
        rewards = torch.tensor(rewards, dtype=torch.float).to(gpu)
        next_states = torch.tensor(next_states, dtype=torch.float).to(gpu)
        dones = torch.tensor(dones, dtype=torch.float).to(gpu)
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            next_q_values, _ = next_q_values.max(1)

        targets = rewards + dones * self.gamma * next_q_values 
        self.epsilon.increment()
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_model()

    def update_target_model(self):
        if self.training:
            self.target_model.load_state_dict(self.model.state_dict())
            self.epsilon.increment()

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon._value:
            return random.choice(self.actions)
        else:
            if self.competitive == 'neutral':
                state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                q_values = self.model(state_tensor)
                # setting the action to the most neutral action
                abs_q_values = torch.abs(q_values)
                if self.alternate is not 'none':
                    self.competitive = 'none'
                return self.actions[torch.argmin(abs_q_values)]
            elif self.competitive == 'half':
                state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                q_values = self.model(state_tensor)
                # find max
                max_q_value_idx = torch.argmax(q_values)
                max_q_value = q_values[max_q_value_idx]
                # get half of max
                half_max_q_value = max_q_value / 2
                # find the closest value to that
                diffs = torch.abs(q_values - half_max_q_value)
                closest_idx = torch.argmin(diffs)
                if self.alternate is not 'none':
                    self.competitive = 'none'
                return self.actions[closest_idx]
            elif self.competitive == 'quarter':
                state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                q_values = self.model(state_tensor)
                # find max
                max_q_value_idx = torch.argmax(q_values)
                max_q_value = q_values[max_q_value_idx]
                # get quarter of max
                quarter_max_q_value = max_q_value / 4
                # find the closest value to that
                diffs = torch.abs(q_values - quarter_max_q_value)
                closest_idx = torch.argmin(diffs)
                if self.alternate is not 'none':
                    self.competitive = 'none'
                return self.actions[closest_idx]
            elif self.competitive == 'second':
                state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                q_values = self.model(state_tensor)
                # Get the top 2 actions
                top_2_actions = torch.topk(q_values, k=2)
                # Select the second best action
                second_best_action = self.actions[top_2_actions.indices[1]]
                if self.alternate is not 'none':
                    self.competitive = 'none'
                return second_best_action
            else:
                state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                q_values = self.model(state_tensor)
                if self.alternate is not 'none':
                    self.competitive = self.alternate
                return self.actions[torch.argmax(q_values)]

    def step(self, obs):
        super(DQNAgent, self).step(obs)
        state = self.get_state(obs)
        action = self.choose_action(state)
        if self.previous_action is not None:
            done = obs.reward - 10 > 0
            transition = Transition(self.previous_state, action, state, obs.reward - 10, done)
            #self.replay_buffer.append((self.previous_state, self.previous_action, obs.reward, state, obs.last()))
            self.replay_buffer.push(transition)
            if len(self.replay_buffer) > self.batch_size and self.training:
                self.train()
        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)

class DQNModel(nn.Module):
    def __init__(self):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(21, 256)  # increase the number of units
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
# class DQNAgent(Agent):
    # def __init__(self, competitive='none', alternate='none', training=True):
        # super(DQNAgent, self).__init__()
        # # self.model = DQNModel().to(gpu)
        # # self.target_model = DQNModel().to(gpu)
        # # self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)
        # # self.entropy_coeff = 0.01  # entropy coefficient
        # self.loss_fn = nn.SmoothL1Loss()
        # # self.replay_buffer = deque(maxlen=10000)
        # # self.batch_size = 32
        # # self.gamma = 0.99
        # # self.epsilon = 0.1
        # # self.epsilon_decay = 0.995
        # # self.min_epsilon = 0.01
        # # self.update_target_frequency = 1000
        # # self.competitive = competitive
        # # self.training = training
        # # self.new_game()
        # # self.alternate = alternate
        # self.model = DQNModel().to(gpu)
        # self.target_model = DQNModel().to(gpu)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # self.entropy_coeff = 0.01  # entropy coefficient
        # #self.loss_fn = nn.MSELoss()
        # self.replay_buffer = deque(maxlen=500000)
        # self.batch_size = 32
        # self.gamma = 0.99
        # self.epsilon = 0.1
        # self.epsilon_decay = 0.99
        # self.min_epsilon = 0.01
        # self.update_target_frequency = 10
        # self.competitive = competitive
        # self.training = training
        # self.new_game()
        # self.alternate = alternate

    # def new_game(self):
        # self.base_top_left = None
        # self.previous_state = None
        # self.previous_action = None

    # def get_state(self, obs):
        # scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        # idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        # command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        # supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        # completed_supply_depots = self.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
        # barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        # completed_barrackses = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        # marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        # queued_marines = (completed_barrackses[0].order_length if len(completed_barrackses) > 0 else 0)

        # free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
        # can_afford_supply_depot = obs.observation.player.minerals >= 100
        # can_afford_barracks = obs.observation.player.minerals >= 150
        # can_afford_marine = obs.observation.player.minerals >= 100

        # enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        # enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        # enemy_command_centers = self.get_enemy_units_by_type(obs, units.Terran.CommandCenter)
        # enemy_supply_depots = self.get_enemy_units_by_type(obs, units.Terran.SupplyDepot)
        # enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(obs, units.Terran.SupplyDepot)
        # enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        # enemy_completed_barrackses = self.get_enemy_completed_units_by_type(obs, units.Terran.Barracks)
        # enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)

        # # Return tuple
        # return (len(command_centers),
                # len(scvs),
                # len(idle_scvs),
                # len(supply_depots),
                # len(completed_supply_depots),
                # len(barrackses),
                # len(completed_barrackses),
                # len(marines),
                # queued_marines,
                # free_supply,
                # can_afford_supply_depot,
                # can_afford_barracks,
                # can_afford_marine,
                # len(enemy_command_centers),
                # len(enemy_scvs),
                # len(enemy_idle_scvs),
                # len(enemy_supply_depots),
                # len(enemy_completed_supply_depots),
                # len(enemy_barrackses),
                # len(enemy_completed_barrackses),
                # len(enemy_marines))

    # def train(self):
        # batch = random.sample(self.replay_buffer, self.batch_size)
        # states, actions, rewards, next_states, dones = zip(*batch)

        # # Convert actions to numerical representation
        # action_dict = {action: i for i, action in enumerate(self.actions)}
        # actions = [action_dict[action] for action in actions]

        # states = torch.tensor(states, dtype=torch.float).to(gpu)
        # actions = torch.tensor(actions, dtype=torch.long).to(gpu)
        # rewards = torch.tensor(rewards, dtype=torch.float).to(gpu)
        # next_states = torch.tensor(next_states, dtype=torch.float).to(gpu)
        # dones = torch.tensor(dones, dtype=torch.float).to(gpu)

        # q_values = self.model(states)
        # q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # with torch.no_grad():
            # next_q_values = self.target_model(next_states)
            # next_q_values, _ = next_q_values.max(1)

        # targets = rewards + self.gamma * next_q_values * (1 - dones)

        # # Compute entropy
        # q_values_softmax = torch.softmax(q_values, dim=0)
        # entropy = -torch.sum(q_values_softmax * torch.log(q_values_softmax + 1e-6)).mean()

        # # Compute loss
        # loss = self.loss_fn(q_values, targets) - self.entropy_coeff * entropy
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # self.update_target_model()


    # def update_target_model(self):
        # if self.training:
            # self.target_model.load_state_dict(self.model.state_dict())
            # self.epsilon *= self.epsilon_decay
            # self.epsilon = max(self.epsilon, self.min_epsilon)

    # def choose_action(self, state):
        # if np.random.uniform() < self.epsilon:
            # return random.choice(self.actions)
        # else:
            # if self.competitive == 'neutral':
                # state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                # q_values = self.model(state_tensor)
                # # setting the action to the most neutral action
                # abs_q_values = torch.abs(q_values)
                # if self.alternate is not 'none':
                    # self.competitive = 'none'
                # return self.actions[torch.argmin(abs_q_values)]
            # elif self.competitive == 'half':
                # state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                # q_values = self.model(state_tensor)
                # # find max
                # max_q_value_idx = torch.argmax(q_values)
                # max_q_value = q_values[max_q_value_idx]
                # # get half of max
                # half_max_q_value = max_q_value / 2
                # # find the closest value to that
                # diffs = torch.abs(q_values - half_max_q_value)
                # closest_idx = torch.argmin(diffs)
                # if self.alternate is not 'none':
                    # self.competitive = 'none'
                # return self.actions[closest_idx]
            # elif self.competitive == 'quarter':
                # state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                # q_values = self.model(state_tensor)
                # # find max
                # max_q_value_idx = torch.argmax(q_values)
                # max_q_value = q_values[max_q_value_idx]
                # # get quarter of max
                # quarter_max_q_value = max_q_value / 4
                # # find the closest value to that
                # diffs = torch.abs(q_values - quarter_max_q_value)
                # closest_idx = torch.argmin(diffs)
                # if self.alternate is not 'none':
                    # self.competitive = 'none'
                # return self.actions[closest_idx]
            # elif self.competitive == 'second':
                # state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                # q_values = self.model(state_tensor)
                # # Get the top 2 actions
                # top_2_actions = torch.topk(q_values, k=2)
                # # Select the second best action
                # second_best_action = self.actions[top_2_actions.indices[1]]
                # if self.alternate is not 'none':
                    # self.competitive = 'none'
                # return second_best_action
            # else:
                # state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                # q_values = self.model(state_tensor)
                # if self.alternate is not 'none':
                    # self.competitive = self.alternate
                # return self.actions[torch.argmax(q_values)]



    # def choose_action(self, state):
        # if np.random.uniform() < self.epsilon:
            # return random.choice(self.actions)
        # else:
            # if self.competitive == 'neutral':
                # state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                # q_values = self.model(state_tensor)
                # # setting the action to the most neutral action
                # abs_q_values = torch.abs(q_values)
                # if self.alternate is not 'none':
                    # self.competitive = 'none'
                # return self.actions[torch.argmin(abs_q_values)]
            # elif self.competitive == 'half':
                # state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                # q_values = self.model(state_tensor)
                # # find max
                # max_q_value_idx = torch.argmax(q_values)
                # max_q_value = q_values[max_q_value_idx]
                # # get half of max
                # half_max_q_value = max_q_value / 2
                # # find the closest value to that
                # diffs = torch.abs(q_values - half_max_q_value)
                # closest_idx = torch.argmin(diffs)
                # if self.alternate is not 'none':
                    # self.competitive = 'none'
                # return self.actions[closest_idx]
            # elif self.competitive == 'quarter':
                # state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                # q_values = self.model(state_tensor)
                # # find max
                # max_q_value_idx = torch.argmax(q_values)
                # max_q_value = q_values[max_q_value_idx]
                # # get quarter of max
                # quarter_max_q_value = max_q_value / 4
                # # find the closest value to that
                # diffs = torch.abs(q_values - quarter_max_q_value)
                # closest_idx = torch.argmin(diffs)
                # if self.alternate is not 'none':
                    # self.competitive = 'none'
                # return self.actions[closest_idx]
            # elif self.competitive == 'second':
                # state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                # q_values = self.model(state_tensor)
                # # Get the top 2 actions
                # top_2_actions = torch.topk(q_values, k=2)
                # # Select the second best action
                # second_best_action = self.actions[top_2_actions.indices[1]]
                # if self.alternate is not 'none':
                    # self.competitive = 'none'
                # return second_best_action
            # else:
                # state_tensor = torch.tensor(state, dtype=torch.float).to(gpu)
                # q_values = self.model(state_tensor)
                # if self.alternate is not 'none':
                    # self.competitive = self.alternate
                # return self.actions[torch.argmax(q_values)]


    # def step(self, obs):
        # super(DQNAgent, self).step(obs)
        # state = self.get_state(obs)
        # action = self.choose_action(state)
        # if self.previous_action is not None:
            # self.replay_buffer.append((self.previous_state, self.previous_action, obs.reward, state, obs.last()))
            # if len(self.replay_buffer) > self.batch_size and self.training:
                # self.train()
        # self.previous_state = state
        # self.previous_action = action
        # return getattr(self, action)(obs)


# class DQNModel(nn.Module):
    # def __init__(self):
        # super(DQNModel, self).__init__()
        # self.fc1 = nn.Linear(21, 128)  # input layer (21) -> hidden layer (128)
        # self.fc2 = nn.Linear(128, 128)  # hidden layer (128) -> hidden layer (128)
        # self.fc3 = nn.Linear(128, 6)  # hidden layer (128) -> output layer (6)

    # def forward(self, x):
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x


import time
import pickle
import json


def my_run_loop(agents, env, max_frames=0, max_episodes=0):
    episode_rewards = []
    episode_winners = []
    total_frames = 0
    total_episodes = 0
    start_time = time.time()

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
        agent.setup(obs_spec, act_spec)
    while not max_episodes or total_episodes < max_episodes:
        total_episodes += 1
        timesteps = env.reset()
        for a in agents:
            a.reset()
        while True:
            total_frames += 1
            actions = [agent.step(timestep)
                       for agent, timestep in zip(agents, timesteps)]
            if max_frames and total_frames >= max_frames:
                return
            if timesteps[0].last():
                break
            timesteps = env.step(actions)
        outcome = [0] * env._num_agents
        for i, o in enumerate(env._obs):
            player_id = o.observation.player_common.player_id
            for result in o.player_result:
                if result.player_id == player_id:
                    outcome[i] = possible_results.get(result.result, 0)
                    episode_rewards.append(outcome[i])
        if outcome[0] > 0:
            episode_winners.append(1)  # agent1 wins
        elif outcome[0] < 0:
            episode_winners.append(-1)  # agent2 wins
        else:
            episode_winners.append(0)  # draw
    return episode_rewards, episode_winners






# Run the game, create agents, set up instructions for the game
def main(unused_argv):
    # agent1 = SmartAgent(competitive='none', alternate='none')
    # agent2 = RandomAgent()
    # try:
        # # run_match(agent1, agent2, 1000)
        # with sc2_env.SC2Env(
                # map_name="Simple64",  # Choose the map
                # players=[sc2_env.Agent(sc2_env.Race.terran),
                         # sc2_env.Agent(sc2_env.Race.terran)],
                # agent_interface_format=features.AgentInterfaceFormat(
                    # action_space=actions.ActionSpace.RAW,
                    # use_raw_units=True,
                    # raw_resolution=64,
                # ),
                # step_mul=128,  # How fast it runs the game
                # disable_fog=True,  # Too see everything in the minimap
        # ) as env:
            # #run_loop.run_loop([agent1, agent2], env, max_episodes=1000)  # Control both agents
            # episode_rewards, episode_winners = my_run_loop([agent1, agent2], env, max_episodes=1000)  # Control both agents
            # print("Average episode reward:", np.mean(episode_rewards))
            # print("Win rate:", np.mean([1 if winner == 0 else 0 for winner in episode_winners]))
            # with open('simple_QL_Train_rewards.json', 'w') as f:
                # json.dump(episode_rewards, f)
            # with open('simple_QL_Train_winners.json', 'w') as f:
                # json.dump(episode_winners, f)
            # with open('simpleQLagent.pkl', 'wb') as f:
                # pickle.dump(agent1, f)

    # except KeyboardInterrupt:
        # pass
    agent3 = DQNAgent(competitive='none', alternate='none')
    agent4 = RandomAgent()
    # with open('simpleQLagent.pkl', 'rb') as f:
        # agent4 = pickle.load(f)
    try:
        # run_match(agent1, agent2, 1000)
        with sc2_env.SC2Env(
                map_name="Simple64",  # Choose the map
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                ),
                step_mul=128,  # How fast it runs the game
                disable_fog=True,  # Too see everything in the minimap
        ) as env:
            # run_loop.run_loop([agent1, agent2], env, max_episodes=1000)  # Control both agents
            episode_rewards, episode_winners = my_run_loop([agent3, agent4], env, max_episodes=1000)  # Control both agents
            print("Average episode reward:", np.mean(episode_rewards))
            print("Win rate:", np.mean([1 if winner == 0 else 0 for winner in episode_winners]))
            with open('deep_QL_Train_rewards.json', 'w') as f:
                json.dump(episode_rewards, f)
            with open('deep_QL_Train_winners.json', 'w') as f:
                json.dump(episode_winners, f)
            with open('DeepQLagent.pkl', 'wb') as f:
                pickle.dump(agent3, f)
            torch.save(agent3.model, 'DeepLearningAgent.pth')
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)