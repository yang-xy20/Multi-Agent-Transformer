import numpy as np
from mat.envs.mpe.core import World, Agent, Landmark
from mat.envs.mpe.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment

class Scenario(BaseScenario):
    def make_world(self, args, rank):
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
        world.all_pos = np.zeros((self.num_agents*2,2))
        dir_name = os.path.dirname(os.path.abspath(__file__)) #2
        txt_file_path = os.path.join(dir_name, '{}_maps'.format(self.num_agents), "{}agent_push_ball_map_{}.txt".format(self.num_agents,rank % 4))
        with open(txt_file_path, "r") as f:
            num = 0
            for line in f.readlines():
                line = line.strip('\n')  #去掉列表中每一个元素的换行符
                line = line.split()
                for i in range(2):
                    world.all_pos[num,i] = float(line[i])
                num += 1   
        f.close()
        world.all_ball = np.zeros((self.num_agents,2))
        dir_name = os.path.dirname(os.path.abspath(__file__)) #2
        txt_file_path = os.path.join(dir_name, '{}_maps'.format(self.num_agents), "{}agent_push_ball_ball_{}.txt".format(self.num_agents,rank % 4))
        with open(txt_file_path, "r") as f:
            num = 0
            for line in f.readlines():
                line = line.strip('\n')  #去掉列表中每一个元素的换行符
                line = line.split()
                for i in range(2):
                    world.all_ball[num,i] = float(line[i])
                num += 1   
        f.close()
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
            landmark.state.p_pos = world.all_ball[i].copy()/1.5 if i<self.num_boxes else world.all_pos[2*(i-self.num_boxes)+1].copy()/1.5#np.random.uniform(-2.0, +2.0, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p) 
            landmark.reach = False      
        for agent in world.agents:
            agent.first_reach = False
            agent.state.p_pos = world.all_pos[2*i].copy()/1.5#np.random.uniform(-2.0, +2.0, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        _,_,_,self.max_distance = self.compute_macro_allocation(world)

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
        reach = 0
        for a in world.agents:
            if a.first_reach:
                reach += 1
        for land_id, l in enumerate(world.landmarks): 
            if land_id < self.num_boxes:
                if not l.reach:
                    dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents if not a.first_reach]
                    #rew -= min(dists)
                    if min(dists) <= world.agents[0].size + world.landmarks[0].size :
                        agent_id = np.argmin(np.array(dists))
                        l.reach = True
                        world.agents[agent_id].first_reach = True
                        # give bonus for cover landmarks
                        rew += 4
            else:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents if a.first_reach]
                dists += np.array([[np.sqrt(np.sum(np.square(a.state.p_pos - world.landmarks[box_id].state.p_pos)))+np.sqrt(np.sum(np.square(world.landmarks[box_id].state.p_pos - l.state.p_pos))) for a in world.agents if not a.first_reach] \
                for box_id in range(self.num_boxes) if not world.landmarks[box_id].reach]).reshape(-1).tolist()
                #rew -= min(dists)/self.max_distance
                agent_id = np.argmin(np.array(dists))
                if agent_id < reach:
                    if min(dists) <= world.agents[0].size + world.landmarks[0].size:
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
            if agent.first_reach:
                if land_id<self.num_boxes:
                    entity_pos.append([0,0])
                else:
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            else:
                if land_id<self.num_boxes and entity.reach:
                    entity_pos.append([1e6,1e6])
                else:
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

        id_vector = np.zeros((2))
        if agent.first_reach:
            id_vector[1] = 1
        else:
            id_vector[0] = 1
        
        
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [id_vector] + entity_pos + comm + other_pos)
    
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

    def compute_macro_allocation(self, world):
        cost_agent_box = np.zeros((self.num_agents, self.num_boxes))
        cost_box_land = np.zeros((self.num_boxes, self.num_boxes))
        for box_id in range(self.num_boxes):
            box = world.landmarks[box_id]
            for landmark_id in range(self.num_boxes):  # world.entities:
                rel_dis = np.sqrt(np.sum(np.square(world.landmarks[landmark_id+self.num_boxes].state.p_pos - box.state.p_pos)))
                cost_box_land[box_id, landmark_id] = rel_dis
            for people_id in range(self.num_people):
                people = world.agents[people_id]
                rel_dis = np.sqrt(np.sum(np.square(people.state.p_pos - box.state.p_pos)))
                cost_agent_box[people_id, box_id] = rel_dis
                
        bl_row_ind, bl_col_ind = linear_sum_assignment(cost_box_land)
        ab_row_ind, ab_col_ind = linear_sum_assignment(cost_agent_box)
        max_distance = cost_box_land.max()+cost_agent_box.max()
        
        dists = 0
        for box_id in range(self.num_boxes):
            box = world.landmarks[box_id]
            agent_id = ab_row_ind[box_id]
            land_id = bl_col_ind[box_id]
            target_agent = world.agents[agent_id]
            target_land = world.landmarks[land_id+self.num_boxes]
            dists += (np.sqrt(np.sum(np.square(box.state.p_pos - target_agent.state.p_pos)))+\
            np.sqrt(np.sum(np.square(box.state.p_pos - target_land.state.p_pos))))
        return ab_col_ind, bl_col_ind, dists, max_distance
