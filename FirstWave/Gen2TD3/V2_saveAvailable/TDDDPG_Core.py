import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ActorNet(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, device):
        super(ActorNet, self).__init__()

        self.device = device

        self.fc     = nn.Linear(state_dim, 200)
        self.hd     = nn.Linear(200, 400)
        self.hd2     = nn.Linear(400, 200)
        #self.hd3     = nn.Linear(200, 100)
        self.mu_layer = nn.Linear(200, action_dim)
        
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.hd.weight)   
        torch.nn.init.xavier_uniform_(self.hd2.weight)  
        #torch.nn.init.xavier_uniform_(self.hd3.weight)  
        torch.nn.init.xavier_uniform_(self.mu_layer.weight)

        self.PreluWeightA = torch.rand(400).to(self.device)

    def forward(self, s):
        s   = s.to(self.device)
        x   = (self.fc(s))
        x   = F.tanh(self.hd(x))
        x   = F.tanh(self.hd2(x))
        #x   = F.tanh(self.hd3(x))
        acc = self.mu_layer(x) #2 .0 * F.tanh(self.mu_layer(x))
        acc = acc.to('cpu')
        return acc

class CriticNet(nn.Module):

    def __init__(self, state_dim, action_dim, device):
        super(CriticNet, self).__init__()
        self.device = device

        self.fc = nn.Linear(state_dim+action_dim, 200)
        self.hd = nn.Linear(200, 400)
        self.hd2 = nn.Linear(400, 200)
        #self.hd3 = nn.Linear(200, 100)
        self.Q_layer = nn.Linear(200, 1)
        
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.hd.weight)
        torch.nn.init.xavier_uniform_(self.hd2.weight)
        #torch.nn.init.xavier_uniform_(self.hd3.weight)
        torch.nn.init.xavier_uniform_(self.Q_layer.weight)

        self.PreluWeightC = torch.rand(400).to(self.device)

    def forward(self, s, a):
        s   = s.to(self.device)
        #k   = k.to(device)
        a   = a.to(self.device)
        #s = torch.cat([s, k], dim=1)
        x = self.fc(torch.cat([s, a], dim=1))

        x = F.tanh(self.hd(x))
        x = F.tanh(self.hd2(x))
        #x = F.tanh(self.hd3(x))
        state_value = self.Q_layer(x)
        state_value = state_value.to('cpu')
        return state_value
    
class TD3:
    def __init__(self, lra, lrc1, lrc2, state_dim, action_dim, max_action, device):
        self.device = device

        self.actor = ActorNet(state_dim, action_dim, max_action, self.device).to(self.device)
        self.actor_target = ActorNet(state_dim, action_dim, max_action, self.device).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lra)
        
        self.critic_1 = CriticNet(state_dim, action_dim, self.device).to(self.device)
        self.critic_1_target = CriticNet(state_dim, action_dim, self.device).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lrc1)
        
        self.critic_2 = CriticNet(state_dim, action_dim, self.device).to(self.device)
        self.critic_2_target = CriticNet(state_dim, action_dim, self.device).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lrc2)
        
        self.max_action = max_action
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self, replay_buffer, sim_t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay, dt):
        n_iter = int(sim_t/dt)
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action_).to(self.device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to('cpu')
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to('cpu')
            
            # Select next action according to target policy:
            noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to('cpu')
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).to(self.device)
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()
            
            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            
            # Delayed policy updates:
            if i % policy_delay == 0:
                # Compute actor loss:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                    
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))
        
        torch.save(self.critic_1.state_dict(), '%s/%s_crtic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))
        
        torch.save(self.critic_2.state_dict(), '%s/%s_crtic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_1.load_state_dict(torch.load('%s/%s_crtic_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_2.load_state_dict(torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        
    def load_actor(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        self.size +=1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)        
        
      
        
