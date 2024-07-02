# @author: Bastien Veuthey
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict
from random import randrange, shuffle, randint, choice
from matplotlib import style
import pickle
import time
import dill
import pandas as pd
import seaborn as sns
import sys


style.use("ggplot")

nodes = [1, 2, 3, 4]
                # [port, req, pkt_loss_req]
elephant_flows = [[1414, 40, 0.01], # p2p
                  [1415, 20, 0.06]] # video streaming
lc = 100 # link capacity
min_req_elephant_f = 100000
max_lc = lc
ec = lc * lc
pkt = 1
weighted_reward_params = (0.3,0.3,0.4)

G = nx.Graph()
G.add_nodes_from(nodes)
edges = [(1, 2, lc, ec, pkt), (1, 3, lc, ec, pkt), (2, 3, lc, ec, pkt), (3, 4, lc, ec, pkt), (4, 1, lc, ec, pkt)]
# Translate the link capacity to the link weight (only for visualization and representation with networkx)
for edge in edges:
    G.add_edge(edge[0], edge[1], capacity=edge[2], energy=edge[3], pk_loss=edge[4])
G.nodes[1]['pos'] = (0, 0)
G.nodes[2]['pos'] = (0, 1)
G.nodes[3]['pos'] = (1, 1)
G.nodes[4]['pos'] = (1, 0)
#pos = nx.get_node_attributes(G, 'pos')
#nx.draw_networkx(G, pos, node_size=300)
#arc_weight = nx.get_edge_attributes(G, 'weight')
#nx.draw_networkx_edge_labels(G, pos, edge_labels=arc_weight)

class Environment:
    def __init__(self):
        self.nodes = [1, 2, 3, 4] # 4 nodes of the network
        self.action_space = ['SWP', 'WSP'] # actions translates later to 0 and 1
        self.links = nx.get_edge_attributes(G, 'capacity') # link loads

    def getState(self):
        # get the link loads of the network
        self.links = nx.get_edge_attributes(G, 'capacity')
        self.pk_loss = nx.get_edge_attributes(G, 'pk_loss')
        self.energy = nx.get_edge_attributes(G, 'energy')
        return (self.links, self.energy, self.pk_loss)

    def step(self, flow, path):
        # execute the action and get the chosen path of the flow (install flow rules)
        c_res_old = self.execute(flow, path)
        # get the new state
        s_ = self.getState()
        # get the reward
        r = self.get_reward(flow, path, c_res_old)
        # check if the network is congested, if then has to stop the episode
        congested = self.is_congested()
        return s_, r, congested

    def is_congested(self):
        # check if all the links are congested
        if self.links[(1, 2)] <= 0 and \
                self.links[(2, 3)] <= 0 and \
                self.links[(3, 4)] <= 0 and \
                self.links[(1, 4)] <= 0 and \
                self.links[(1, 3)] <= 0:
            return True
        return False

    def erlang_b_pkt(self, u, c):
        import math
        h = u**c/math.factorial(c)
        l = 0
        for n in range(c):
            l += u**n/math.factorial(n)
        return h/l

    def execute(self, flow, path):
        c_res = lc + 1
        # update the link loads in the graph
        for i in range(len(path)-1):
            c_res = min(G[path[i]][path[i+1]]['capacity'], c_res)
            G[path[i]][path[i+1]]['capacity'] -= flow[2] # flow[2] is the requirement of the flow (and its size)
            if G[path[i]][path[i+1]]['capacity'] < 0:
                G[path[i]][path[i+1]]['capacity'] = 0
            
            G[path[i]][path[i+1]]['energy'] = lc*(lc - G[path[i]][path[i+1]]['capacity'])
            #TODO: add the update formula for packet loss
            G[path[i]][path[i+1]]['pk_loss'] = self.erlang_b_pkt(G[path[i]][path[i+1]]['capacity'], max_lc)

        return c_res

    def reset(self):
        # set the weights of the links in the graph to the link capacity
        links = nx.get_edge_attributes(G, 'capacity')
        for link, _ in links.items():
            links[link] = lc
        nx.set_edge_attributes(G, links, 'capacity') # update the graph with the new weights (link loads residual)

        energy = nx.get_edge_attributes(G, 'energy')
        for e, _ in energy.items():
            energy[e] = ec
        nx.set_edge_attributes(G, energy, 'energy')

        pktl = nx.get_edge_attributes(G, 'pk_loss')
        for e, _ in pktl.items():
            pktl[e] = pkt
        nx.set_edge_attributes(G, pktl, 'pk_loss') # update the graph with the new weights (link loads residual)

        return self.getState()

    def get_actions(self):
        return self.action_space

    def get_reward(self, flow, path, c_res_old):
        # calculate the total capacity residual of the path
        c_res = lc + 1
        r_util = 0
        r_e = 0
        r_loss = 0
        c_res_old = c_res_old*0.8
        if c_res_old < flow[2]:
            r_util = 1 - (flow[2] - c_res)/flow[2]
        else:
            r_util = c_res/(c_res - flow[2])

        e_p = 0

        for i in range(len(path)-1):
            e_p = e_p + G[path[i]][path[i+1]]['energy']
        
        r_e = e_p / len(path)


        pkt_loss_p = 1

        for i in range(len(path)-1):
            pkt_loss_p = pkt_loss_p*(1- G[path[i]][path[i+1]]['pk_loss'])

        if flow[3] < pkt_loss_p:
            r_loss = 1 - (pkt_loss_p - flow[3])/pkt_loss_p
        else:
            r_loss = (flow[3] - pkt_loss_p)/pkt_loss_p

        return weighted_reward_params[0]*r_util + weighted_reward_params[1]*r_e + weighted_reward_params[2]*r_loss

    def select_path(self, action, flow):
        # select the path according to the action
        if action == 0:
            path = self.swp(flow)
        else:
            path = self.wsp(flow)
        return path

    def swp(self, flow):
        # get the widest shortest paths
        paths = nx.all_simple_paths(G, source=flow[0], target=flow[1])
        paths = list(paths)
        capacities = []
        energies = []
        losses = []
        
        for i in range(len(paths)):
            capacities.append(lc + 1)
            e = 0
            l = 1
            for j in range(len(paths[i]) - 1):
                w = G[paths[i][j]][paths[i][j + 1]]['capacity']
                w = round(w)
                capacities[i] = min(w, capacities[i])

                e = e + G[paths[i][j]][paths[i][j + 1]]['energy']
                l = l * G[paths[i][j]][paths[i][j + 1]]['pk_loss']
            
            energies[i] = e
            losses[i] = l

        weighted_sum = weighted_reward_params[0]*capacities - weighted_reward_params[1]*energies - weighted_reward_params[2]*losses

                # capacities[i] = round(capacities[i])
        # choose the shortest path among the widest ones
        paths = [paths[i] for i in range(len(paths)) if 
                 (weighted_reward_params[0]*capacities[i] - weighted_reward_params[1]*energies[i] - weighted_reward_params[2]*losses[i]) 
                 == max(weighted_sum)]

        if (len(paths) > 1):
            length = len(paths[0])
            for i in range(len(paths)):
                if len(paths[i]) < length:
                    length = len(paths[i])
            paths = [paths[i] for i in range(len(paths)) if len(paths[i]) == length]
        path = paths[randint(0, len(paths) - 1)]
        return path

    def wsp(self, flow):
        paths = nx.all_simple_paths(G, source=flow[0], target=flow[1])
        paths = list(paths)
        capacities = []
        energies = []
        losses = []
        
        for i in range(len(paths)):
            capacities.append(lc + 1)
            e = 0
            l = 1
            for j in range(len(paths[i]) - 1):
                w = G[paths[i][j]][paths[i][j + 1]]['capacity']
                w = round(w)
                capacities[i] = min(w, capacities[i])

                e = e + G[paths[i][j]][paths[i][j + 1]]['energy']
                l = l * G[paths[i][j]][paths[i][j + 1]]['pk_loss']
            
            energies[i] = e
            losses[i] = l

        weighted_sum = weighted_reward_params[0]*capacities - weighted_reward_params[1]*energies - weighted_reward_params[2]*losses

                # capacities[i] = round(capacities[i])
        # choose the shortest path among the widest ones
        paths = [paths[i] for i in range(len(paths)) if 
                 (weighted_reward_params[0]*capacities[i] - weighted_reward_params[1]*energies[i] - weighted_reward_params[2]*losses[i]) 
                 == max(weighted_sum)]
        
        path = paths[randint(0, len(paths) - 1)]
        return path

    #TODO: revise this function later to make sure the correct first shortest path to be returned
    def spf(self, flow):
        # get the first shortest path
        path = nx.shortest_path(G, source=flow[0], target=flow[1])
        # update the link loads in the graph
        for i in range(len(path)-1):
            G[path[i]][path[i+1]]['capacity'] -= flow[2]
            if G[path[i]][path[i+1]]['capacity'] < 0:
                G[path[i]][path[i+1]]['capacity'] = 0

    def discretize(self, state, flow, path):
        freq = 1 # binary variable to check if the requirement for an elephant flow is satisfied
        # check if there is at least one link with residual capacity less than 10
        # if so, the requirement for the elephant flow is not satisfied
        for i in range(len(path) - 1):
            src = min(path[i], path[i+1])
            dst = max(path[i], path[i+1])
            link = (src, dst)
            c_res = state.get(link, [0, 0, 0])[0] # residual capacity of the link
            ener = state.get(link, [0, 0, 0])[1]
            loss = state.get(link, [0, 0, 0])[2]
            if c_res < flow[2] or ener < flow[3] or loss < flow[4]: # requirement for an elephant flow
                freq = 0 # the requirement is not satisfied
                break

        s_ = []
        s1_ = []
        s2_ = []
        print(f'########## state: {state}')
        for _, c_res  in state[0].items():
            ll = (lc - c_res) / lc # link load
            ener = lc*(lc - ll)
            loss = self.erlang_b_pkt(ll, max_lc)
            # round to 0.3 (low), 0.7 (medium), 1.0 (high)
            if ll <= 0.3:
                s_.append(0.3)
            elif ll <= 0.7:
                s_.append(0.7)
            else:
                s_.append(1.0)

            if ener <= 0.3:
                s1_.append(0.3)
            elif ener <= 0.7:
                s1_.append(0.7)
            else:
                s1_.append(1.0)

            if loss <= 0.05:
                s2_.append(0.05)
            elif loss <= 0.15:
                s2_.append(0.15)
            else:
                s2_.append(1.0)
            
            print(f'Inside the loop: c_res: {c_res}, ll: {ll}, energy: {ener}, loss: {loss}, s_: {s_}, s1_: {s1_}, s2_: {s2_}')
        
        s_ = tuple((s_, s1_, s2_))
        print(f'######s_: {s_}')
        return (s_, freq)

class QAgent:
    def __init__(self, environment, epsilon=1.0, alpha=0.01, gamma=0.9):
        self.n_action = len(environment.action_space) # number of actions
        self.q = defaultdict(self.init_q_table) # initialize the Q-table
        # (defaultdict is a dictionary that returns a default value if the key is not present)
        self.epsilon = epsilon # exploration rate
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor

    def act(self, observation):
        # choose an action according to the epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            action = randrange(self.n_action) # choose a random action (exploration)
        else:
            print(f'observation: {observation}, q: {self.q}')
            state_action = self.q[observation] # get the Q-values for the current state
            action = np.argmax(state_action) # choose the best action (exploitation)
        return action

    def init_q_table(self):
        return np.zeros(self.n_action)

    def save_q_table(self):
        with open("q-tables/qtable-{0}.pickle".format(int(time.time())), "wb") as f:
            pickle.dump(dict(self.q), f)

    def load_q_table(self, file):
        with open(file, "rb") as f:
            self.q = defaultdict(self.init_q_table, pickle.load(f))

def generate_flow_matrix(size):
    # matrix : [src, dst, req]
    # ratio 1:9 of the size for elephant and mice flows
    n_elephant = int(size * 0.1) # number of elephant flows (10%)
    i_elephant = 0
    flows = []
    for i in range(size):
        # random source and destination (not the same)
        src = np.random.randint(1, 5)
        dst = np.random.randint(1, 5)
        while dst == src:
            dst = np.random.randint(1, 5)
        # no more than 10 elephant flows, then only mice flows (< 10)
        if i_elephant < n_elephant:
            # get the requirement of a random flow in elephant_flows
            freq = elephant_flows[np.random.randint(0, len(elephant_flows))][1]
            if freq < min_req_elephant_f:
                min_req_elephant_f = freq
            i_elephant += 1
        else:
            freq = round(np.random.random(), 2) # random requirement for a mice flow between [0,1[
        flows.append([src, dst, freq])
    # flows list is shuffled to have a random order of flows
    shuffle(flows)
    return flows

def train():
    # directory for the output files
    # n_test = 17
    # 10 random seeds
    # seeds = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    # run_rewards = []
    env = Environment()
    agent = QAgent(env)
    n_episodes = 100000
    show_every = 5000
    # max_steps = 1000
    n_flow = 100  # number of flows per episode
    eps_limit = 0.001  # exploration will stop decrease after this limit
    eps_decay = (eps_limit / agent.epsilon) ** (1 / n_episodes)  # exponential decay for epsilon-greedy policy
    eps_delta = (agent.epsilon - eps_limit) / n_episodes  # delta for linear decay

    print("Agent epsilon: {0}, learning rate: {1} and discount factor: {2}".format(agent.epsilon, agent.alpha,
                                                                                   agent.gamma))
    print("Number of episodes: {0}".format(n_episodes))
    print("Number of flows: {0}".format(n_flow))
    print("Epsilon decay (exponential)")
    print("Epsilon limit: {0}".format(eps_limit))
    print("State space: {0}".format(env.getState()))
    print("Discrete state space for a mice flow [1,2,3]: {0}".format(env.discretize(env.getState(), [1, 2, 3], [])))
    print(
        "NOTE: The state space is discretized by dividing the link load by the link capacity (network load) in order to have a value between 0 and 1.")

    episode_rewards = []
    episode_losses = []
    cumulative_rewards = []
    for e in range(n_episodes):
        # print the mean reward every show_every episode
        if e % show_every == 0 and e != 0:
            print("Episode: {0}, epsilon is {1}".format(e, agent.epsilon))
            print("Mean reward over last {0} episodes was {1}".format(show_every, np.mean(episode_rewards[-show_every:])))
        state = env.reset() # reset the environment for a new episode
        state = env.discretize(state, [0, 0, 0], [])
        episode_reward = 0 # initialize the episode reward
        episode_loss = 0
        steps = 0 # initialize the number of steps
        congested = False # initialize the congestion flag (terminal state)
        #n_flow = np.random.randint(10, 1001) # number of flows for the episode
        flows = generate_flow_matrix(n_flow) # generate the flows for the episode
        # run the episode until congestion or all the flows are treated
        for i in range(n_flow):
            flow = flows[i] # get the flow of the current step
            if flow[2] < 1:  # mice flow
                env.spf(flow)  # choose the shortest path
            else:  # elephant flow
            #if flow[2] > 1:
                steps += 1
                action = agent.act(state) # choose an action according to the epsilon-greedy policy
                path = env.select_path(action, flow) # select the path for the flow
                next_state, reward, congested = env.step(flow, path) # take the action and observe the next state and reward
                next_state = env.discretize(next_state, flow, path) # discretize the next state
                episode_reward += reward # update the episode reward list
                # Q-learning update:
                # Q(S,A) <- Q(S,A) + alpha * (R + gamma * max_a Q(S',a) - Q(S,A))
                # TD_target = R + gamma * max_a Q(S',a)
                # TD_delta = TD_target - Q(S,A)
                # Q(S,A) <- Q(S,A) + alpha * TD_delta
                td_target = reward + agent.gamma * np.argmax(agent.q[next_state]) # calculate the TD target
                td_delta = td_target - agent.q[state][action] # calculate the TD delta
                agent.q[state][action] += agent.alpha * td_delta # update the Q-table
                state = next_state # update the state
                episode_loss += td_delta ** 2
                # check if the network is congested
                if congested:
                    break
        # get the mean reward for the episode
        #episode_reward /= steps
        #episode_loss /= steps
        # append the mean reward to the list of episode rewards
        episode_rewards.append(episode_reward)
        episode_losses.append(np.mean(episode_loss))
        cumulative_rewards.append(np.mean(episode_rewards))
        # decrease epsilon according to the decay rate
        agent.epsilon *= eps_decay
        #agent.epsilon -= eps_delta
        # decrease the learning rate according to the decay rate
        #agent.alpha -= lr_decay

    # get the moving average of the rewards
    moving_avg = np.convolve(episode_rewards, np.ones((show_every,))/show_every, mode='valid')
    # plot the rewards over the episodes
    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel("Reward {0}ma".format(show_every))
    plt.xlabel("episode #")
    plt.savefig("plots/rewards-{0}.png".format(time.time()))
    plt.show()

    moving_avg = np.convolve(cumulative_rewards, np.ones((show_every,))/show_every, mode='valid')
    # plot the rewards over the episodes
    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel("Average Reward {0}ma".format(show_every))
    plt.xlabel("episode #")
    plt.savefig("plots/cumulative_rewards-{0}.png".format(time.time()))
    plt.show()

    moving_avg_loss = np.convolve(episode_losses, np.ones((show_every,))/show_every, mode='valid')
    # plot the average losses over the episodes
    plt.plot([i for i in range(len(moving_avg_loss))], moving_avg_loss)
    plt.ylabel("Average Loss")
    plt.xlabel("episode #")
    plt.savefig("plots/average_losses-{0}.png".format(time.time()))
    plt.show()

    print("Size of the Q Table: " + str(agent.q.__len__()))
    wsp_q_table = {k: v for k, v in agent.q.items() if v[1] > v[0]}
    swp_q_table = {k: v for k, v in agent.q.items() if v[1] < v[0]}
    print("Size of the WSP Q Table: " + str(wsp_q_table.__len__()))
    print("Size of the SWP Q Table: " + str(swp_q_table.__len__()))
    agent.save_q_table()

def test():
    test_flows_matrix = [[1, 4, 40],
                         [1, 3, 20],
                         [1, 4, 40],
                         [4, 2, 40],
                         [1, 4, 20]]
    start_q_table = 'saving/utilization/min/square/200k/qtable-1690210270.pickle'
    if start_q_table is None:
        print("No Q-table to load")
        return
    env = Environment()
    agent = QAgent(env, epsilon=0.0)
    agent.load_q_table(start_q_table)
    test_flows = test_flows_matrix
    # state of the network:
    # [  0, 100, 95, 70]
    # [100,   0, 80,  0]
    # [ 95,  80,  0, 90]
    # [ 70,   0, 90,  0]
    G[1][2]['capacity'] = 100
    G[1][3]['capacity'] = 95
    G[1][4]['capacity'] = 90
    G[2][3]['capacity'] = 80
    G[3][4]['capacity'] = 90
    state = env.getState()
    state = env.discretize(state, [0, 0, 0], [])
    episode_rewards = []
    wsp_q_table = {k: v for k, v in agent.q.items() if v[1] > v[0]}
    swp_q_table = {k: v for k, v in agent.q.items() if v[1] < v[0]}
    for flow in test_flows:
        # Choose the best action for this state
        action = agent.act(state)
        # Execute the action and get the reward
        path = env.select_path(action, flow)
        print(path)
        next_state, reward, congested = env.step(flow, path)
        next_state = env.discretize(next_state, flow, path)

        # Update state and reward
        state = next_state
        episode_rewards.append(reward)

        if congested:
            break

    print("Rewards: " + str(episode_rewards))
    return episode_rewards

if __name__ == "__main__":
    #train()
    test()
