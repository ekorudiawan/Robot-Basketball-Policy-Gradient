import pygame 
import numpy as np 
import random 
import time 
import os
import matplotlib.pyplot as plt 
import matplotlib.style as style
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class BasketballEnv:
    def __init__(self, width=9, height=6, n_opponents=5, render=False, version=0):
        self.width = width
        self.height = height
        self.scale = 100 
        self.n_opponents = n_opponents
        # Color
        self.ball_color = (168, 92, 50)
        self.white_color = (255, 255, 255)
        self.field_color = (50, 168, 52)
        self.basket_color = (168, 58, 50)
        self.robot_color = (27, 34, 48)
        self.opponent_color = (84, 3, 3)
        self.after_shoot = 1
        self.ball_to_basket = 0
        self.robot_pos_static = True
        self.version = version
        self.total_shoot_success = 0
        self.total_shoot_fail = 0
        # PyGame
        self.render_ui = render 
        if self.render_ui:
            pygame.init()
            pygame.display.set_caption("Basketball-V0")
            self.window_size = [self.width*self.scale, self.height*self.scale]
            self.screen = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
            pygame.font.init() 
            self.font = pygame.font.SysFont('Comic Sans MS', 20)
            self.robot_text = self.font.render('ROBOT', False, (192, 197, 207))
            self.ball_text = self.font.render('BALL', False, (192, 197, 207))
            self.opp_text = self.font.render('OPP', False, (192, 197, 207))
        # Default Position
        self.basket_pos = [self.width-1, (self.height-1)//2]
        self.ball_pos = [0,(self.height-1)//2]

        if self.robot_pos_static:
            self.robot_pos = [0, 0]
        else:
            self.robot_pos = [random.randint(0, self.width-1), random.randint(0, self.height-1)]

        self.opponent_pos = []
        random.seed(42)
        for i in range(self.n_opponents):
            random.seed(time.clock())
            done = False
            while not done:
                pos = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
                if pos not in self.opponent_pos:
                    if pos != self.robot_pos and pos != self.ball_pos:
                        self.opponent_pos.append(pos)
                        done = True 

        self.action_dict = {0: "UP", 
                            1: "LEFT",
                            2: "DOWN",
                            3: "RIGHT",
                            4: "DRIBBLE-UP",
                            5: "DRIBBLE-LEFT",
                            6: "DRIBBLE-DOWN",
                            7: "DRIBBLE-RIGHT",
                            8: "SHOOT"}

    def target_empty(self):
        result = True
        for i in range(self.n_opponents):
            if self.opponent_pos[i][0] == self.robot_pos[0] and self.opponent_pos[i][1] == self.robot_pos[1]:
                result = False 
        return result

    def ball_out(self):
        if self.ball_pos[0] < 0 or self.ball_pos[0] >= self.width or self.ball_pos[1] < 0 or self.ball_pos[1] >= self.height:
            return True
        else:
            return False

    def leave_field(self):
        if self.robot_pos[0] < 0 or self.robot_pos[0] >= self.width or self.robot_pos[1] < 0 or self.robot_pos[1] >= self.height:
            return True
        else:
            return False
    
    def drible_possible(self):
        if self.robot_pos[0] == self.ball_pos[0] and self.robot_pos[1] == self.ball_pos[1]:
            return True
        else:
            return False

    def distance_to_basket(self):
        distance = np.sqrt((self.basket_pos[0]-self.robot_pos[0])**2 + (self.basket_pos[1]-self.robot_pos[1])**2)
        return distance

    def distance_ball_to_basket(self):
        distance = np.sqrt((self.basket_pos[0]-self.ball_pos[0])**2 + (self.basket_pos[1]-self.ball_pos[1])**2)
        return distance

    def distance_robot_to_ball(self):
        distance = np.sqrt((self.robot_pos[0]-self.ball_pos[0])**2 + (self.robot_pos[1]-self.ball_pos[1])**2)
        return distance

    def step(self, action):
        done = False
        reward = 0
        robot_prev_pos = self.robot_pos.copy()
        ball_prev_pos = self.ball_pos.copy()
        if self.pick_ball == False and self.drible_possible():
            self.pick_ball = True

        move = 0
        if self.version == 0:
            move = 1
        else:
            move_choice = [1,2]
            move = np.random.choice(move_choice, p=[0.6, 0.4])

        if self.action_dict[action] == "UP":
            self.robot_pos[1] -= move
        elif self.action_dict[action] == "LEFT":
            self.robot_pos[0] -= move
        elif self.action_dict[action] == "DOWN":
            self.robot_pos[1] += move
        elif self.action_dict[action] == "RIGHT":
            self.robot_pos[0] += move
        elif self.action_dict[action] == "DRIBBLE-UP":
            if self.drible_possible():
                self.robot_pos[1] -= move
                self.ball_pos[1] -= move
        elif self.action_dict[action] == "DRIBBLE-LEFT":
            if self.drible_possible():
                self.robot_pos[0] -= move
                self.ball_pos[0] -= move
        elif self.action_dict[action] == "DRIBBLE-DOWN":
            if self.drible_possible():
                self.robot_pos[1] += move
                self.ball_pos[1] += move
        elif self.action_dict[action] == "DRIBBLE-RIGHT":
            if self.drible_possible():
                self.robot_pos[0] += move
                self.ball_pos[0] += move
        elif self.action_dict[action] == "SHOOT":
            if self.drible_possible():
                # check distance
                dist = self.distance_to_basket() 
                shoot_choice = ["success","fail"]
                shoot_prob = 0
                shoot_reward_success = 0
                shoot_reward_fail = 0
                if dist < 1:
                    shoot_prob = 0.9
                    shoot_reward_success = 30
                    shoot_reward_fail = 1
                elif dist >= 1 and dist < 3:
                    shoot_prob = 0.66
                    shoot_reward_success = 40
                    shoot_reward_fail = 1
                elif dist >= 3 and dist < 4:
                    shoot_prob = 0.1
                    shoot_reward_success = 50
                    shoot_reward_fail = 1
                else:
                    shoot_prob = 0.0
                    shoot_reward_fail = -1

                shoot_result = np.random.choice(shoot_choice, p=[shoot_prob, 1-shoot_prob])
                if shoot_result == "success":
                    reward += shoot_reward_success 
                    self.ball_pos = self.basket_pos.copy()
                    self.total_shoot_success += 1
                    done = True

                else:
                    reward += shoot_reward_fail
                    self.ball_pos[0] = 0
                    self.ball_pos[1] = (self.height-1)//2
                    self.after_shoot = 2
                    self.total_shoot_fail += 1
                    self.pick_ball = False
                print("Shoot Distance :", "{:.2f}".format(dist)," Result :", shoot_result)

        if self.leave_field():
            reward = -1
            done = True
            
        # If target not empy, back to previous position
        if not self.target_empty():
            self.robot_pos = robot_prev_pos.copy()
            self.ball_pos = ball_prev_pos.copy()

        # If ball out 
        if self.ball_out():
            self.ball_pos[0] = 0
            self.ball_pos[1] = (self.height-1)//2
        
        holding_ball = 1 
        if self.drible_possible():
            holding_ball = 2

        if self.point1 == False and self.distance_robot_to_ball() < (self.height/2)*0.75:
            reward += 1
            self.point1 = True

        if self.point2 == False and self.distance_robot_to_ball() < (self.height/2)*0.5:
            reward += 1
            self.point2 = True

        if self.point3 == False and self.drible_possible() < 1:
            reward += 1
            self.point3 = True
        # print(self.distance_ball_to_basket())
        if self.point4 == False and self.distance_ball_to_basket() < self.width*0.8:
            reward += 1
            self.point4 = True 

        if self.point5 == False and self.distance_ball_to_basket() < self.width*0.7:
            reward += 1
            self.point5 = True 
        
        if self.point6 == False and self.distance_ball_to_basket() < self.width*0.6:
            reward += 1
            self.point6 = True 
        
        if self.point7 == False and self.distance_ball_to_basket() < self.width*0.5:
            reward += 1
            self.point7 = True 
        
        if self.point8 == False and self.distance_ball_to_basket() < self.width*0.4:
            reward += 1
            self.point8 = True 
        
        if self.point9 == False and self.distance_ball_to_basket() < self.width*0.3:
            reward += 1
            self.point9 = True 
        
        if self.point10 == False and self.distance_ball_to_basket() < self.width*0.2:
            reward += 1
            self.point10 = True

        dist_robot_to_basket = self.distance_to_basket()
        dist_ball_to_basket = self.distance_ball_to_basket()

        # observation = np.array([self.robot_pos[0], self.robot_pos[1], holding_ball, self.after_shoot, dist_robot_to_basket, dist_ball_to_basket, self.ball_pos[0], self.ball_pos[1]])
        # observation = np.array([self.robot_pos[0], self.robot_pos[1], holding_ball, self.after_shoot])
        observation = np.array([self.robot_pos[0], self.robot_pos[1], holding_ball, self.after_shoot, dist_robot_to_basket, dist_ball_to_basket, self.ball_pos[0], self.ball_pos[1]])
        return observation, done, reward

    def reset(self):
        self.total_shoot_success = 0
        self.total_shoot_fail = 0
        self.pick_ball = False
        self.point1 = False
        self.point2 = False
        self.point3 = False
        self.point4 = False
        self.point5 = False
        self.point6 = False
        self.point7 = False
        self.point8 = False
        self.point9 = False
        self.point10 = False
        # self.point11 = False 
        # self.point12 = False 
        if self.robot_pos_static:
            self.robot_pos = [0, 0]
        else:
            self.robot_pos = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
        
        if self.version == 2:
            self.opponent_pos = []
            random.seed(42)
            for i in range(self.n_opponents):
                random.seed(time.clock())
                done = False
                while not done:
                    pos = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
                    if pos not in self.opponent_pos:
                        if pos != self.robot_pos and pos != self.ball_pos:
                            self.opponent_pos.append(pos)
                            done = True 

        self.after_shoot = 1
        self.ball_pos = [0,(self.height-1)//2]
        holding_ball = 1
        self.ball_to_basket = self.distance_ball_to_basket()
        self.robot_to_ball = self.distance_robot_to_ball()
        dist_robot_to_basket = self.distance_to_basket()
        dist_ball_to_basket = self.distance_ball_to_basket()
        observation = np.array([self.robot_pos[0], self.robot_pos[1], holding_ball, self.after_shoot, dist_robot_to_basket, dist_ball_to_basket, self.ball_pos[0], self.ball_pos[1]])
        # observation = np.array([self.robot_pos[0], self.robot_pos[1], holding_ball, self.after_shoot])
        return observation

    def draw_grid_cell(self):
        for i in range(0, self.window_size[0], self.scale):
            for j in range(0, self.window_size[1], self.scale):
                pygame.draw.rect(self.screen, self.white_color, [i, j, self.scale, self.scale], 3)

    def draw_basket(self):
        pygame.draw.rect(self.screen, self.basket_color, [self.basket_pos[0]*self.scale, self.basket_pos[1]*self.scale, self.scale, self.scale])

    def draw_ball(self):
        x = (self.ball_pos[0]*self.scale) + int((3/4)*self.scale)
        y = (self.ball_pos[1]*self.scale) + int((1/4)*self.scale)
        r = self.scale//4
        pygame.draw.circle(self.screen, self.ball_color, [x, y], r)
        self.screen.blit(self.ball_text,(x-int((1/6)*self.scale), y-int((1/8)*self.scale)))

    def draw_robot(self):
        x = self.robot_pos[0]*self.scale
        y = self.robot_pos[1]*self.scale
        w = self.scale//2
        h = self.scale//2
        pygame.draw.rect(self.screen, self.robot_color, [x, y, w, h])
        self.screen.blit(self.robot_text,(x+int((1/32)*self.scale), y+int((2/16)*self.scale)))

    def draw_opponents(self):
        for i in range(len(self.opponent_pos)):
            x = (self.opponent_pos[i][0]*self.scale) + self.scale//2
            y = (self.opponent_pos[i][1]*self.scale) + self.scale//2
            w = self.scale//2
            h = self.scale//2
            pygame.draw.rect(self.screen, self.opponent_color, [x, y, w, h])
            self.screen.blit(self.opp_text,(x+int(1/8*self.scale), y+int(1/8*self.scale)))

    def render(self):
        self.screen.fill(self.field_color)
        self.draw_grid_cell()
        self.draw_basket()
        self.draw_ball()
        self.draw_robot()
        self.draw_opponents()
        self.clock.tick(60)
        pygame.display.flip()

class Policy(nn.Module):
    def __init__(self, device="", input_size=8, hidden_size=20, output_size=8):
        super(Policy, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

class PolicyGradient:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.95, epsilon_decay=0.99, epsilon_min=0.01, render=False, plot=False):
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min
        self.render = render
        self.env = BasketballEnv(width=9, height=6, n_opponents=5, render=render, version=0)
        self.n_states = self.env.width * self.env.height * 2 * 2
        self.n_actions = len(self.env.action_dict)
        use_cuda = True
        device = torch.device("cuda" if use_cuda else "cpu")
        self.policy = Policy(device=device, input_size=8, hidden_size=20, output_size=self.n_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.alpha)
        self.plot = plot

    def manual_action(self, state):
        action = -1
        pressed = pygame.key.get_pressed()
        while action == -1:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_LEFT:
                        action = 1
                    elif event.key == pygame.K_DOWN:
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        action = 3
                    elif event.key == pygame.K_w:
                        action = 4
                    elif event.key == pygame.K_a:
                        action = 5
                    elif event.key == pygame.K_x:
                        action = 6
                    elif event.key == pygame.K_d:
                        action = 7
                    elif event.key == pygame.K_s:
                        action = 8
        return action

    def learn(self, n_episodes, n_steps):
        reward_history = [] 
        avg_reward_history = []
        total_reward = 0 
        total_shoot_success = 0
        total_shoot_fail = 0
        for episode in range(1,n_episodes+1):
            saved_log_probs = []
            rewards = []
            state = self.env.reset()
            if self.render:
                self.env.render()
                time.sleep(0.01)
            sum_reward = 0
            done = False
            for i in range (n_steps):
                action, log_prob = self.policy.act(state)
                saved_log_probs.append(log_prob)
                state, done, reward = self.env.step(action)
                sum_reward += reward
                rewards.append(reward)
                if self.render and episode > (0.9*n_episodes):
                    self.env.render()
                    time.sleep(0.01)

            total_reward += sum_reward 
            avg_reward = total_reward / episode
            reward_history.append(sum_reward)
            avg_reward_history.append(avg_reward)
            total_shoot_success += self.env.total_shoot_success
            total_shoot_fail += self.env.total_shoot_fail
            print("Episode :", episode, "Sum Reward :", sum_reward, "Avg Reward :", avg_reward)

            # Update policy
            discounts = [self.gamma**i for i in range(len(rewards)+1)]
            R = sum([a*b for a, b in zip(discounts, rewards)])

            policy_loss = []
            for log_prob in saved_log_probs:
                policy_loss.append(-log_prob*R)

            policy_loss = torch.cat(policy_loss).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

        if self.plot:
            if total_shoot_success+total_shoot_fail != 0:
                print("Total Shoot Success :", total_shoot_success, " Percentage :", total_shoot_success/(total_shoot_success+total_shoot_fail))
                print("Total Shoot Fail :", total_shoot_fail, "Percentage :", total_shoot_fail/(total_shoot_success+total_shoot_fail))
            style.use('seaborn-poster') 
            fig, ax = plt.subplots()
            ax.plot(np.arange(0,n_episodes), avg_reward_history)
            ax.set_title("Episode VS Average Reward")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Average Reward")
            plt.show()

def main():
    pg = PolicyGradient(alpha=0.001, gamma=0.9, render=False, plot=True)
    pg.learn(5000, 50)
   
if __name__ == "__main__":
    main()