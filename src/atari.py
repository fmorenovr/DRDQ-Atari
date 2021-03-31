import time
import gym
import argparse
import numpy as np
import atari_py
import datetime
from game_models.dqn_game_model import DQNTrainer, DQNSolver
from game_models.ddqn_game_model import DDQNTrainer, DDQNSolver
from game_models.drqn_game_model import DRQNTrainer, DRQNSolver
from game_models.ddrqn_game_model import DDRQNTrainer, DDRQNSolver
from game_models.ge_game_model import GETrainer, GESolver
from utils.gym_wrappers import MainGymWrapper

FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)

HALF_MM = 500000
ONE_MM = 1000000
TWO_MM = 2000000
THREE_MM = 3000000
FOUR_MM = 4000000
FIVE_MM = 5000000
TEN_MM = 10000000


class Atari:
  def __init__(self, game_name, game_mode, render, total_step_limit, total_run_limit, clip, getDate, cnn_mode):
    #game_name, game_mode, render, total_step_limit, total_run_limit, clip = self._args()
    env_name = game_name + "Deterministic-v4"  # Handles frame skipping (4) at every iteration
    env = MainGymWrapper.wrap(gym.make(env_name))
    self._main_loop(self._game_model(game_mode, game_name, env.action_space.n, getDate, cnn_mode), env, render, total_step_limit, total_run_limit, clip)

  def _main_loop(self, game_model, env, render, total_step_limit, total_run_limit, clip):
    if isinstance(game_model, GETrainer):
      game_model.genetic_evolution(env)

    run = 0
    total_step = 0
    while True:
      if total_run_limit is not None and run >= total_run_limit:
        print ("Reached total run limit of: " + str(total_run_limit))
        exit(0)

      run += 1
      current_state = env.reset()
      step = 0
      score = 0
      while True:
        if total_step >= total_step_limit:
          print ("Reached total step limit of: " + str(total_step_limit))
          exit(0)
        total_step += 1
        step += 1

        if render:
          env.render()

        action = game_model.move(current_state)
        next_state, reward, terminal, info = env.step(action)
        
        if clip:
          np.sign(reward)
        score += reward
        game_model.remember(current_state, action, reward, next_state, terminal)
        current_state = next_state
        
        # update accuracy, q, loss -> logger
        game_model.step_update(total_step)

        # call looger.run -> base_game_model
        if terminal:
          game_model.save_run(score, step, run)
          break

  def _game_model(self, game_mode,game_name, action_space, getDate, cnn_mode):
    if game_mode == "drqn_training":
      return DRQNTrainer(game_name, INPUT_SHAPE, action_space, getDate, cnn_mode)
    elif game_mode == "drqn_testing":
      return DRQNSolver(game_name, INPUT_SHAPE, action_space, getDate, cnn_mode)
    elif game_mode == "ddrqn_training":
      return DDRQNTrainer(game_name, INPUT_SHAPE, action_space, getDate, cnn_mode)
    elif game_mode == "ddrqn_testing":
      return DDRQNSolver(game_name, INPUT_SHAPE, action_space, getDate, cnn_mode)
    elif game_mode == "ddqn_training":
      return DDQNTrainer(game_name, INPUT_SHAPE, action_space, getDate, cnn_mode)
    elif game_mode == "ddqn_testing":
      return DDQNSolver(game_name, INPUT_SHAPE, action_space, getDate, cnn_mode)
    elif game_mode == "dqn_training":
      return DQNTrainer(game_name, INPUT_SHAPE, action_space, getDate, cnn_mode)
    elif game_mode == "dqn_testing":
      return DQNSolver(game_name, INPUT_SHAPE, action_space, getDate, cnn_mode)
    elif game_mode == "ge_training":
      return GETrainer(game_name, INPUT_SHAPE, action_space, getDate, cnn_mode)
    elif game_mode == "ge_testing":
      return GESolver(game_name, INPUT_SHAPE, action_space, getDate, cnn_mode)
    else:
      print ("Unrecognized mode. Use --help")
      exit(1)


def main():
  cnn_mode = "mse"
  # game_name, game_mode, rendering, total_steps, total_run_limit, clip, date
  currentDate = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
  #currentDate = "2019-01-27_18-37"
  Atari("SpaceInvaders", "ge_training", False, THREE_MM, None, True, currentDate, cnn_mode)

if __name__ == "__main__":
  main()
