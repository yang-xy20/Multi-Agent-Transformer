import numpy as np
from mat.envs.mpe.core import World, Agent, Landmark
from mat.envs.mpe.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment

class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        # set any world properties first
        world.name = 'ball'
        world.dim_c = 2
        now_agent_num = args.num_agents#2
        world.world_length = args.episode_length
        if now_agent_num==None:
            num_people = args.num_agents
            num_boxes = args.num_landmarks
            num_landmarks = args.num_landmarks
        else:
            num_people = now_agent_num
            num_boxes = now_agent_num
            num_landmarks = now_agent_num
        world.collaborative = True
        self.num_boxes = num_boxes
        self.num_people = num_people
        self.num_agents = num_people # deactivate "good" agent
        self.num_landmarks = num_landmarks + num_boxes
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.id = i
            agent.first_reach = False
            agent.collide = True
            agent.silent = True
            agent.adversary = True  # people.adversary = True     box.adversary = False
            agent.size = 0.1
            # agent.accel = 3.0 if agent.adversary else 5
            # agent.max_speed = 0.5 if agent.adversary else 0.5
            agent.action_callback = None  # box有action_callback 即不做动作

        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.reach = False
            landmark.size = 0.15
            landmark.cover = 0
            # landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) 
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.85, 0.35, 0.35]) if i < self.num_agents else np.array([0, 0, 0])
        # set random initial states
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-2.0, +2.0, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p) 
            landmark.reach = False      
        for agent in world.agents:
            agent.first_reach = False
            agent.state.p_pos = np.random.uniform(-2.0, +2.0, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # def info(self, world):
    #     num = 0
    #     success = False
    #     for i in range(self.num_boxes):
    #         l = world.landmarks[i+self.num_boxes]                
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents if a.first_reach]
    #         if len(dists) > 0 and min(dists) <= world.agents[0].size + world.landmarks[0].size:
    #             num = num + 1
    #     # success
    #     # if num==len(world.landmarks):
    #     #     success = True
    #     info_list = {'success_rate': num/self.num_boxes}
    #     return info_list

    
    def reward(self, agent, world):
    # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions  
        rew = 0
        cover = 0
        i = 0
        for land_id, l in enumerate(world.landmarks):
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
            agent_id = np.argmin(np.array(dists))
            if land_id < self.num_boxes:
                if min(dists) <= world.agents[0].size + world.landmarks[0].size \
                and  (not l.reach) and (not world.agents[agent_id].first_reach):
                    l.reach = True
                    world.agents[agent_id].first_reach = True
                    # give bonus for cover landmarks
                    rew += 4
            else:
                if min(dists) <= world.agents[0].size + world.landmarks[0].size \
                and world.agents[agent_id].first_reach:
                    cover += 1
                    # give bonus for cover landmarks
                    rew += 4

        if cover == self.num_boxes:
            rew += 4 * self.num_boxes            
        if agent.collide:
            for a in world.agents:
                if a != agent and self.is_collision(a, agent):
                    rew -= 1
            
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for land_id, entity in enumerate(world.landmarks):  # world.entities:
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

        id_vector = np.zeros(2)
        if agent.first_reach:
            id_vector = np.ones(2)
        
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + comm + other_pos)
    
    def info(self, world):
        num = 0
        success = False
        for i in range(self.num_boxes):
            l = world.landmarks[i+self.num_boxes]                
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents if a.first_reach]
            if len(dists) > 0 and min(dists) <= world.agents[0].size + world.landmarks[0].size:
                num = num + 1
        # success
        # if num==len(world.landmarks):
        #     success = True
        info_list = {'success_rate': num/self.num_boxes}
        return info_list
