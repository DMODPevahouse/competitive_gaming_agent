import os
os.environ['SC2PATH'] = 'E:\games\StarCraft II'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import random
import numpy as np  # Mathematical functions
import pandas as pd  # Manipulate and analize of data
import os
from absl import app
import torch
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
from s2clientprotocol import sc2api_pb2 as sc_pb
import pickle
from FluxuatingSkillSC2Train import RandomAgent, my_run_loop, SmartAgent, QLearningTable, DQNAgent, DQNModel
import json

levels = ['neutral', 'half', 'quarter', 'second']
alter = ['no', 'yes']
players = ['dqn', 'ql', 'random']


def model_decision(decisions, changing, model):
    if model == "ql":
        with open('SimpleQLagent.pkl', 'rb') as f:
            model = pickle.load(f)
            model.qtable.competitive = decisions
            model.qtable.alternate = changing
    elif model == "dqn":
        with open('DeepQLagent.pkl', 'rb') as f:
            model = pickle.load(f)
        model.model = torch.load('DeepLearningAgent.pth')
        model.competitive = decisions
        model.alternate = changing
    elif model =="random":
        model = RandomAgent()
    return model

# Run the game, create agents, set up instructions for the game
def main(unused_argv):
    for player1 in players[0:2]:
        for level in levels:
            for alternate in alter:
                for player2 in players:
                    # reloading the model every time to prevent learning from as much of playing a factor
                    agent1 = model_decision(level, alternate, player1)
                    agent2 = model_decision('none', 'none', player2)
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
                            #run_loop.run_loop([agent1, agent2], env, max_episodes=1000)  # Control both agents
                            episode_rewards, episode_winners = my_run_loop([agent1, agent2], env, max_episodes=2)  # Control both agents
                            print("Average episode reward:", np.mean(episode_rewards))
                            print("Win rate:", np.mean([1 if winner == 0 else 0 for winner in episode_winners]))
                            filename_rewards = "{}_experiment_{}_{}_vs_{}.json".format(player1, level, alternate, player2)
                            filename_winners = "{}_experiment_{}_{}_vs_{}.json".format(player1, level, alternate, player2 )
                            filepath_rewards = os.path.join("results", filename_rewards)
                            filepath_winners = os.path.join("results", filename_winners)
                            with open(filepath_rewards, 'w') as f:
                                json.dump(episode_rewards, f)
                            with open(filepath_winners, 'w') as f:
                                json.dump(episode_winners, f)


                    except KeyboardInterrupt:
                        pass

if __name__ == "__main__":
    app.run(main)