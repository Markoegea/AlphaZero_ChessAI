from pathlib import Path
import datetime as dt
import copy
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, move_count=0, prior=0, visit_counts=0):
        self.game = game
        self.args = args
        self.state = state
        self.action_taken = action_taken
        self.move_count = move_count
        
        self.parent = parent
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
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count+1)) * child.prior

    def expand(self, actions, policy):
        for action, prob in zip(actions, policy):
            if prob > 0:
                action = copy.deepcopy(action)
                child_state = copy.deepcopy(self.state)
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)
                child = Node(self.game, self.args, child_state, self, action, self.move_count+1, prob)
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
            torch.tensor(self.game.get_encoded_state(root.state)).to(device)    
        )
        policy = policy.softmax(dim=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        raw_moves = self.game.get_valid_moves(root.state)
        policy = policy.flatten()[:len(raw_moves)]
        policy /= np.sum(policy)
        root.expand(raw_moves, policy)
        
        for search in tqdm(range(self.args['num_searches'])):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            # 60 are the maximum moves, by 2 we have two consider the two players move
            if node.move_count >= (60 * 2):
                value, is_terminal = 0, True
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state)).to(device)
                )
                policy = policy.softmax(dim=1).cpu().numpy()
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
            
        self.optimizer = optimizer(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        self.mcts = MCTS(self.game, self.args, self.model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        moves_count = 0

        while True:        
            valid_moves = self.game.get_valid_moves(state)
            print(len(valid_moves), valid_moves, state)

            raw_probs = self.mcts.search(state)
            action_probs = np.zeros(self.game.action_size)     
            action_probs[:raw_probs.shape[0]] = raw_probs
            
            memory.append((state, action_probs, player))
            temperature_action_probs = raw_probs ** ( 1/ self.args['temperature'])
            
            action = np.random.choice(valid_moves, p=(temperature_action_probs/sum(temperature_action_probs)))

            state = self.game.get_next_state(state, action, player)
            moves_count += 1

            state = self.game.change_perspective(state, -1)   

            value, is_terminal = self.game.get_value_and_terminated(state, action)
            if moves_count > 60:
                value, is_terminal = 0, True
                
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
        moves_count = 0
        
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
            moves_count += 1
            
            state = self.game.change_perspective(state, -1)

            value, is_terminal = self.game.get_value_and_terminated(state, action)
            if moves_count >= 60 and not is_terminal:
                value, is_terminal = 0, True

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
            state, policy_targets, value_targets = np.array(state).squeeze(), np.array(policy_targets), np.array(value_targets).reshape(-1,1) 

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
        date = dt.datetime.now().strftime("%Y-%m-%d")
        
        pathModel = Path(f'models/model_lineal_{info}')
        pathModel.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), pathModel/f'model_{iteration}_{self.game}_{date}.pth')

class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        
        self.memory = []
        self.root = None
        self.node = None
        
        self.moves_count = 0      
            
class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.inference_mode()
    def search(self, states, spGames):
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states)).to(device)
        )
        policy = policy.softmax(dim=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            raw_moves = self.game.get_valid_moves(states[i])

            spg_policy = spg_policy.flatten()[:len(raw_moves)]
            spg_policy /= np.sum(spg_policy)
            
            spg.root = Node(self.game, self.args, copy.deepcopy(states[i]), visit_counts=1)
            spg.root.expand(raw_moves, spg_policy)
            
        for search in tqdm(range(self.args['num_searches'])):
            for spg in spGames:
                spg.node = None
                node = spg.root
                
                while node.is_fully_expanded():
                    node = node.select()
                    
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)
                
                # 60 are the maximum moves, by 2 we have two consider the two players move
                if node.move_count > (60 * 2):
                    value, is_terminal = 0, True
                    
                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node
                    
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
            
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states)).to(device)
                )
                policy = policy.softmax(dim=1).cpu().numpy()
                value = value.cpu().numpy()
                
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                
                raw_moves = self.game.get_valid_moves(node.state)
                spg_policy = spg_policy.flatten()[:len(raw_moves)]
                spg_policy /= np.sum(spg_policy)
                
                node.expand(raw_moves, spg_policy)
                node.backpropagate(spg_value)        
    
class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args, state_url = None):
        self.game = game
        self.args = args
        self.model = model(self.game, self.args['num_resBlocks'], self.args['num_hidden']).to(device)
        
        if state_url != None:
            self.model.load_state_dict(torch.load(state_url, map_location=device))

        self.optimizer = optimizer(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        self.mcts = MCTSParallel(self.game, self.args, self.model)
        
    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            
            self.mcts.search(states, spGames)
            
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                
                valid_moves = self.game.get_valid_moves(spg.state)
                
                action_probs = np.zeros(self.game.action_size)
                for index, child in enumerate(spg.root.children):
                    action_probs[index] = child.visit_count
                action_probs /= np.sum(action_probs)
                
                spg.memory.append((spg.state, action_probs, player))
                
                temperature_action_probs = (action_probs ** (1 / self.args['temperature']))[:len(valid_moves)]
                action = np.random.choice(valid_moves, p=(temperature_action_probs/sum(temperature_action_probs)))
                
                spg.state = self.game.get_next_state(spg.state, action, player) 
                spg.moves_count += 1
                
                spg.state = self.game.change_perspective(spg.state, -1)
                
                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)
                if spg.moves_count >= 60 and not is_terminal:
                    value, is_terminal = 0, True
                    
                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]
            player = self.game.get_opponent(player)
        return return_memory
    
    def train(self, memory, train_losses):
        random.shuffle(memory)
        
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx: min(len(memory)-1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = np.array(state).squeeze(), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)     
            
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
    
    def learn(self):
        train_losses = []
        for iteration in tqdm(range(self.args['num_iterations'])):
            memory = []

            self.model.eval()
            with torch.inference_mode():
                for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games'])):
                    memory += self.selfPlay()

            self.model.train()
            for epoch in tqdm(range(self.args['num_epochs'])):
                self.train(memory, train_losses)
                
            self.save_model(iteration)
        return train_losses   
    
    def save_model(self,iteration):
        info = str(self.args['num_resBlocks']) + '_' + str(self.args['num_hidden'])+ '_' + str(self.game)
        date = dt.datetime.now().strftime("%Y-%m-%d")
        
        pathModel = Path(f'models/model_parallel_{info}')
        pathModel.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), pathModel/f'model_{iteration}_{self.game}_{date}.pth')