#External libraries
import torch
import matplotlib.pyplot as plt
#Internal libraries
from game import *
from model import *
from agent import MCTS, AlphaZero, AlphaZeroParallel

#Runtime variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = {
    'C': 2,
    'num_searches': 600, # 600
    'num_iterations': 2, # 8
    'num_selfPlay_iterations': 20, # 500    
    'num_parallel_games': 10,
    'num_playerPlay_iterations': 5,
    
    'num_epochs':50,
    'batch_size':32,
    'lr': 0.00001,
    'weight_decay': 0.0001,
    
    'num_resBlocks': 8,
    'num_hidden' : 128,
    
    'temperature':1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
}

#Init game TicTacToe or ConnectFour
game = Chess()
player = 1

def play_alone(player, game):
    state = game.get_initial_state()
    while True:
        print(state)
        valid_moves = game.get_valid_moves(state)
        print("valid_moves", {i:vm for i,vm in enumerate(valid_moves)})
        action = int(input(f'{player}:'))

        if action >= len(valid_moves):
            print("action not valid")
            continue

        action = valid_moves[action]

        state = game.get_next_state(state, action, player)
    
        state = game.change_perspective(state, -1)
        
        value, is_terminal = game.get_value_and_terminated(state, action)
        if is_terminal:
            if value == 1:
                print(player, 'won')
            else:
                print('draw')
            break

        player = game.get_opponent(player)        
        
def play_with_machine(game, player, args, device):
    model = ResNet(game, args['num_resBlocks'], args['num_hidden']).to(device)
    model.load_state_dict(torch.load('models/model_8_128_Chess_Game/model_1_Chess_Game.pth', map_location=device))
    mcts = MCTS(game, args, model)
    state = game.get_initial_state()
    
    model.eval()
    with torch.inference_mode():
        while True:
            print(state)
            if player == 1:
                valid_moves = game.get_valid_moves(state)
                print("valid_moves: ", {i:vm for i,vm in enumerate(valid_moves)})
                action = int(input(f'{player}:'))

                if action >= len(valid_moves):
                    print("action not valid")
                    continue
            else:
                valid_moves = game.get_valid_moves(state)
                mcts_probs = mcts.search(state)
                action = mcts_probs.argmax().item()
                print(f'{player}:{action}')

            action = valid_moves[action]

            state = game.get_next_state(state, action, player)
            
            state = game.change_perspective(state, -1)

            value, is_terminal = game.get_value_and_terminated(state, action)
            if is_terminal:
                print(state)
                if value == 1:
                    print(player, 'won')
                else:
                    print('draw')
                break

            player = game.get_opponent(player)

def train_model(game, args):
    state_dict = 'models/model_8_128_Chess_Game/model_01_Chess_Game.pth'
    model = ResNet
    optimizer = torch.optim.AdamW
    alphaZero = AlphaZero(model, optimizer, game, args, state_dict)
    train_losses = alphaZero.learn(playType=alphaZero.selfPlay, numOfPlays=args['num_selfPlay_iterations'])
    plot_loss(train_losses=train_losses)
    
def train_model_in_parallel(game, args):
    state_dict = None
    model = ResNet
    optimizer = torch.optim.AdamW
    alphaZeroParallel = AlphaZeroParallel(model, optimizer, game, args, state_dict)
    train_losses = alphaZeroParallel.learn()
    plot_loss(train_losses=train_losses)
    
# A function that allow me to play with the machine, save our moves and train with it
def train_by_playing(game, args):
    state_dict = 'models/model_8_128_Chess_Game/model_lineal_01_Chess_Game.pth'
    model = ResNet
    optimizer = torch.optim.AdamW
    alphaZero = AlphaZero(model, optimizer, game, args, state_dict)
    train_losses = alphaZero.learn(playType=alphaZero.playerPlay, numOfPlays=args['num_playerPlay_iterations'])
    plot_loss(train_losses=train_losses)
        
def plot_loss(train_losses):
    epochs = range(len(train_losses))

    plt.figure(figsize=(8, 11))

    plt.plot(epochs, train_losses, label='train_loss')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

def plot_model_predictions(state, device):
    torch.manual_seed(42)
    
    action = Move((6,0), (5,0), state)
    state = game.get_next_state(state, action, player)
    print(state)

    encoded_state = game.get_encoded_state(state)
    print(encoded_state.shape)

    tensor_state = torch.tensor(encoded_state).to(device)
    print(tensor_state.shape)

    model = ResNet(game, args['num_resBlocks'], args['num_hidden']).to(device)
    model.load_state_dict(torch.load('models/model_8_128_Chess_Game/model_1_Chess_Game.pth',map_location=device))

    model.eval()
    with torch.inference_mode():
        policy, value = model(tensor_state)
        policy = policy.softmax(dim=1).squeeze(0).detach().cpu().numpy()
        value = value.cpu().item()
        print(policy, value)
        plt.bar(range(game.action_size), policy)
        plt.show()

if __name__ == '__main__':
    menu = """
    Menu:
    1. Play Chess game in cooperative mode.
    2. Play Chess game against the machine.
    3. Train the model playing AI vs AI.
    4. Train the model playing AI vs AI in parallel.
    5. Train the model playing with you.
    6. Plot the model predictions given a state
    7. Exit
    Enter your choice (1-7):
    """
    choice = input(menu)
    if choice == '1':
        play_alone(player, game)
    elif choice == '2':
        play_with_machine(game, player, args, device) 
    elif choice == '3':
        train_model(game, args)
    elif choice == '4':
        train_model_in_parallel(game, args)
    elif choice == '5':
       train_by_playing(game, args) 
    elif choice == '6':
        state = game.get_initial_state()
        plot_model_predictions(state, device)
    elif choice == '7':
            print("Exiting the program. Goodbye!")