from genericpath import isfile
from logger import *
from idk import *
from agent import *
from torchvision import transforms as T
import ludopy
from ludopy.visualizer import *
import time
LOGDIR ="runs/dqn"
def get_state_format(moment):
    state = draw_basic_board()
    draw_moment(state, moment)
    state = np.transpose(state, (2, 0, 1))
    # print(state.shape)
    state = torch.tensor(state).float()
    resize_fn = T.Resize((128, 128))
    state_resized = resize_fn(state)
    # state_resized = np.transpose(state_resized, (1, 2, 0))
    return state_resized

def play_against_previous(previous_file, new: DQN, save=False):
    game = ludopy.Game(ghost_players=[1, 3])
    previous = DQN(state_dim=(128, 128, 3), action_dim=4)
    checkpoint = torch.load(previous_file)
    previous.net.main.load_state_dict(checkpoint["main"])
    previous.net.target.load_state_dict(checkpoint["target"])
    there_is_a_winner = False
    current = [1, new]
    next = [-1, previous]
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, reward, there_is_a_winner), player_i = game.get_observation()
        pieces = game.get_piece_hist()[-1]
        moment = pieces, dice, game.current_player, game.round
        state = get_state_format(moment)
        action = -1            
        if len(move_pieces):
            action = current[1].act(state, move_pieces)
        dice, move_pieces, player_pieces, enemy_pieces, reward, there_is_a_winner = game.answer_observation(action)
        current, next = next, current
    if save:
        game.save_hist_video(f"game_video.mp4")
    return current[0] == 1

def learning_loop(game: ludopy.Game, agent: DQN, logger: Logger, nb_ep = 1000, nb_games = 20):
    start_time = time.time()
    for ep in range(nb_ep):
        print("episode:", ep)
        done = False
        game.reset()
        there_is_a_winner = False 
        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, reward, there_is_a_winner), player_i = game.get_observation()
            pieces = game.get_piece_hist()[-1]
            moment = pieces, dice, game.current_player, game.round
            state = get_state_format(moment)
            action = -1            
            if len(move_pieces):
                action = agent.act(state, move_pieces)

            # next_state, reward, done, info = env.step(action) #generate obs
            dice, move_pieces, player_pieces, enemy_pieces, reward, there_is_a_winner = game.answer_observation(action)
            # next_state = dice, move_pieces, player_pieces, enemy_pieces
            next_pieces = game.get_piece_hist()[-1]
            next_moment = next_pieces, dice, game.current_player, game.round
            next_state = get_state_format(next_moment)
            agent.collect_experience(state, action, next_state, int(reward), there_is_a_winner)
            q, loss = agent.learn()
            logger.log_step(reward, loss, q)
            state = next_state
        agent.memory[-2][3] = [-1]
        elapsed_time = time.time() - start_time
        logger.log_episode(agent.exploration_rate, elapsed_time)
    checkpoint = {
            'main': agent.net.main.state_dict(),
            'target': agent.net.target.state_dict(),
        }
    file = "checkpoint.pth.tar"
    if isfile(file):
        wins = 0
        for i in range(nb_games):
            wins += play_against_previous(file, agent)
        if wins/nb_games > 0.5:
            agent.save_checkpoint(checkpoint, best=True)
    agent.save_checkpoint(checkpoint)

if __name__ == "__main__":
    #create env
    environment = ludopy.Game()
    #create agent
    agent = DQN(environment, state_dim=(128, 128, 3), action_dim=4)
    #create logging
    logger = Logger(use_tensorboard=True, log_dir=LOGDIR)
    #call learning loop
    learning_loop(environment, agent, logger)