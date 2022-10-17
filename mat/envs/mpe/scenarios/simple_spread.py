import numpy as np
from mat.envs.mpe.core import World, Agent, Landmark
from mat.envs.mpe.scenario import BaseScenario
import os


class Scenario(BaseScenario):
    def make_world(self, args, rank):
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.num_agents = args.num_agents
        world.num_landmarks = args.num_landmarks  # 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
            agent.id = i
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        world.all_pos = np.zeros((world.num_agents,2))
        dir_name = os.path.dirname(os.path.abspath(__file__)) #2
        txt_file_path = os.path.join(dir_name, '{}_maps'.format(world.num_agents), "{}agent_simple_spread_map_{}.txt".format(world.num_agents,rank % 4))
        with open(txt_file_path, "r") as f:
            num = 0
            for line in f.readlines():
                line = line.strip('\n')  #去掉列表中每一个元素的换行符
                line = line.split()
                for i in range(2):
                    world.all_pos[num,i] = float(line[i])
                num += 1   
        f.close()
        world.all_land = np.zeros((world.num_agents,2))
        dir_name = os.path.dirname(os.path.abspath(__file__)) #2
        txt_file_path = os.path.join(dir_name, '{}_maps'.format(world.num_agents), "{}agent_simple_spread_land_{}.txt".format(world.num_agents,rank % 4))
        with open(txt_file_path, "r") as f:
            num = 0
            for line in f.readlines():
                line = line.strip('\n')  #去掉列表中每一个元素的换行符
                line = line.split()
                for i in range(2):
                    world.all_land[num,i] = float(line[i])
                num += 1   
        f.close()
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()

        world.assign_landmark_colors()

        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = world.all_pos[i].copy()/1.5#np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = world.all_land[i].copy()/1.5#0.8 * np.random.uniform(-1, +1, world.dim_p)#
            landmark.state.p_vel = np.zeros(world.dim_p)
        self.max_dis = self.max_distance(world)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        cover = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            rew -= min(dists)/self.max_dis
            
            if min(dists) <= world.agents[0].size + world.landmarks[0].size:
                cover += 1
                # give bonus for cover landmarks
                rew += 1
        # success bonus
        if cover == len(world.landmarks):
            rew += 4 * len(world.landmarks) 
        # rew = 0
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
        #              for a in world.agents]
        #     rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        
        id_vector = np.zeros(len(world.agents))
        id_vector[agent.id] = 1
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + [id_vector] + comm)

    def info(self, world):
        info = {}
        cover = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            if min(dists) <= world.agents[0].size + l.size:
                cover += 1
        info['success_rate'] = cover / world.num_landmarks      

        return info
    
    def max_distance(self, world):
        cost = np.zeros((len(world.agents), len(world.landmarks)))
        all_distance = []
        for agent_id, agent in enumerate(world.agents):
            for landmark_id, entity in enumerate(world.landmarks):  # world.entities:
                rel_dis = np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos)))
                cost[agent_id, landmark_id] = rel_dis
                
            
        # row_ind, col_ind = linear_sum_assignment(cost)
        al_max_dis = cost.max()
        # for a in world.agents:
        #     goal_id = col_ind[a.id]
        #     target_goal = world.landmarks[goal_id]
        #     dists += np.sqrt(np.sum(np.square(a.state.p_pos - target_goal.state.p_pos))) / al_max_dis
        return al_max_dis