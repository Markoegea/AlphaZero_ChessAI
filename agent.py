from pathlib import Path
import copy
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_counts=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = visit_counts
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return -q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count+1)) * child.prior

    def expand(self, actions, policy):
        for action, prob in zip(actions, policy):
            if prob > 0:
                action = copy.deepcopy(action)
                child_state = copy.deepcopy(self.state)
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)
                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, copy.deepcopy(state), visit_counts=1)
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(root.state)).unsqueeze(dim=0).to(device)    
        )
        policy = policy.softmax(dim=1).squeeze(dim=0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        raw_moves = self.game.get_valid_moves(root.state)
        policy = policy.flatten()[:len(raw_moves)]
        policy /= np.sum(policy)
        root.expand(raw_moves, policy)
        
        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state)).unsqueeze(dim=0).to(device)
                )
                policy = policy.softmax(dim=1).squeeze(dim=0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy = policy.flatten()[:len(valid_moves)]
                policy /= np.sum(policy)

                value = value.cpu().item()
                
                node.expand(valid_moves, policy)
            
            node.backpropagate(value)

        action_probs = np.zeros(len(raw_moves))
        for index, child in enumerate(root.children):
            action_probs[index] = child.visit_count
        action_probs /= np.sum(action_probs)

        return action_probs

class AlphaZero:
    def __init__(self, model, optimizer, game, args, state_url = None):
        self.game = game
        self.args = args
        self.model = model(self.game, self.args['num_resBlocks'], self.args['num_hidden']).to(device)
        
        if state_url != None:
            self.model.load_state_dict(torch.load(state_url, map_location=device))
            
        self.optimizer = optimizer(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
        self.mcts = MCTS(self.game, self.args, self.model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:        
            valid_moves = self.game.get_valid_moves(state)
            
            raw_probs = self.mcts.search(state)
            action_probs = np.zeros(self.game.action_size)     
            action_probs[:raw_probs.shape[0]] = raw_probs
            
            memory.append((state, action_probs, player))
            temperature_action_probs = raw_probs ** ( 1/ self.args['temperature'])
            
            action = np.random.choice(valid_moves, p=(temperature_action_probs/sum(temperature_action_probs)))

            state = self.game.get_next_state(state, action, player)
            
            print(state)

            state = self.game.change_perspective(state, -1)   

            value, is_terminal = self.game.get_value_and_terminated(state, action)
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)
            
    def playerPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            print('\n', state, '\n')
            valid_moves = self.game.get_valid_moves(state)
            action_probs = np.zeros(self.game.action_size)
            
            if (player == 1):
                print("valid_moves: ", {i:vm for i,vm in enumerate(valid_moves)})
                action = int(input(f'{player}:'))

                if action >= len(valid_moves):
                    print("action not valid")
                    continue
                
                action_probs[action] = 1.0
                
                action = valid_moves[action]
            else:
                raw_probs = self.mcts.search(state)   
                
                action_probs[:raw_probs.shape[0]] = raw_probs
                
                temperature_action_probs = raw_probs ** ( 1/ self.args['temperature'])
                action = np.random.choice(valid_moves, p=(temperature_action_probs/sum(temperature_action_probs)))
             
                print(action)
                
            memory.append((state, action_probs, player))

            state = self.game.get_next_state(state, action, player)
            
            state = self.game.change_perspective(state, -1)

            value, is_terminal = self.game.get_value_and_terminated(state, action)
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)

    def train(self, memory, train_losses):
        random.shuffle(memory)

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx: min(len(memory)-1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1,1) 

            state = torch.tensor(state, dtype=torch.float32).to(device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32).to(device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32).to(device)

            out_policy, out_value = self.model(state)
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            train_losses.append(loss.item())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def learn(self, playType, numOfPlays):
        train_losses = []
        for iteration in tqdm(range(self.args['num_iterations'])):
            memory = []

            self.model.eval()
            with torch.inference_mode():
                for selfPlay_iteration in tqdm(range(numOfPlays)):
                    memory += playType()

            self.model.train()
            for epoch in tqdm(range(self.args['num_epochs'])):
                self.train(memory, train_losses)
                
            self.save_model(iteration)
        return train_losses

    def save_model(self,iteration):
        info = str(self.args['num_resBlocks']) + '_' + str(self.args['num_hidden'])+ '_' + str(self.game)
        pathModel = Path(f'models/model_{info}')
        pathModel.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), pathModel/f'model_{iteration}_{self.game}.pth')

        pathOptimizer = Path(f'optimizers/model_{info}')
        pathOptimizer.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), pathOptimizer/f'optimizer_{iteration}_{self.game}.pth')