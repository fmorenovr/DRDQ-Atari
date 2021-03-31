import numpy as np
import os
import random
import shutil
from statistics import mean
from game_models.base_game_model import BaseGameModel
from convolutional_networks.neural_networks import RecurrentConvolutionalNeuralNetwork

GAMMA = 0.99
MEMORY_SIZE = 900000
BATCH_SIZE = 32
TRAINING_FREQUENCY = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 40000
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 50000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.02
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS

class DDRQNGameModel(BaseGameModel):
  def __init__(self, game_name, mode_name, input_shape, action_space, logger_path, model_path, cnn_mode):
    BaseGameModel.__init__(self, game_name,
                           mode_name,
                           logger_path,
                           input_shape,
                           action_space)
    self.model_path = model_path
    self.ddrqn = RecurrentConvolutionalNeuralNetwork(self.input_shape, action_space, cnn_mode).model
    if os.path.isfile(self.model_path):
      self.ddrqn.load_weights(self.model_path)

  def _save_model(self):
    self.ddrqn.save_weights(self.model_path)


class DDRQNSolver(DDRQNGameModel):
  def __init__(self, game_name, input_shape, action_space, getDate, cnn_mode):
    testing_model_path = "./output/neural_nets/" + game_name + "/ddrqn/model/" + getDate + "/model.h5"
    assert os.path.exists(os.path.dirname(testing_model_path)), "No testing model in: " + str(testing_model_path)
    DDRQNGameModel.__init__(self,
                           game_name,
                           "DDRQN testing",
                           input_shape,
                           action_space,
                           "./output/logs/" + game_name + "/ddrqn/testing/" + getDate + "/",
                           testing_model_path,
                           cnn_mode)

  def move(self, state):
    if np.random.rand() < EXPLORATION_TEST:
      return random.randrange(self.action_space)
    q_values = self.ddrqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
    return np.argmax(q_values[0])


class DDRQNTrainer(DDRQNGameModel):
  def __init__(self, game_name, input_shape, action_space, getDate, cnn_mode):
    DDRQNGameModel.__init__(self,
                           game_name,
                           "DDRQN training",
                           input_shape,
                           action_space,
                           "./output/logs/" + game_name + "/ddrqn/training/" + getDate + "/",
                           "./output/neural_nets/" + game_name + "/ddrqn/model/" + getDate + "/model.h5",
                           cnn_mode)

    if os.path.exists(os.path.dirname(self.model_path)):
      shutil.rmtree(os.path.dirname(self.model_path), ignore_errors=True)
    os.makedirs(os.path.dirname(self.model_path))

    self.ddrqn_target = RecurrentConvolutionalNeuralNetwork(self.input_shape, action_space, cnn_mode).model
    self._reset_target_network()
    self.epsilon = EXPLORATION_MAX
    self.memory = []

  def move(self, state):
    if np.random.rand() < self.epsilon or len(self.memory) < REPLAY_START_SIZE:
      return random.randrange(self.action_space)
    q_values = self.ddrqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
    return np.argmax(q_values[0])

  def remember(self, current_state, action, reward, next_state, terminal):
    self.memory.append({"current_state": current_state,
                        "action": action,
                        "reward": reward,
                        "next_state": next_state,
                        "terminal": terminal})
    if len(self.memory) > MEMORY_SIZE:
      self.memory.pop(0)

  def step_update(self, total_step):
    if len(self.memory) < REPLAY_START_SIZE:
      return

    if total_step % TRAINING_FREQUENCY == 0:
      loss, accuracy, average_max_q, reward = self._train()
      self.logger.add_loss(loss)
      self.logger.add_accuracy(accuracy)
      self.logger.add_q(average_max_q)
      self.logger.add_reward(reward)

    self._update_epsilon()

    if total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0:
      self._save_model()

    if total_step % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
      self._reset_target_network()
      #print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon))
      #print('{{"metric": "total_step", "value": {}}}'.format(total_step))
      print("--------------------------------------")
      print("Epsilon: " + str(self.epsilon))
      print("Total Step: " + str(total_step))
      print("--------------------------------------")

  def _train(self):
    batch = np.asarray(random.sample(self.memory, BATCH_SIZE))
    if len(batch) < BATCH_SIZE:
      return

    current_states = []
    q_values = []
    max_q_values = []

    for entry in batch:
      current_state = np.expand_dims(np.asarray(entry["current_state"]).astype(np.float64), axis=0)
      current_states.append(current_state)
      
      next_state = np.expand_dims(np.asarray(entry["next_state"]).astype(np.float64), axis=0)
      next_state_prediction = self.ddrqn_target.predict(next_state).ravel()
      next_q_value = np.max(next_state_prediction)
      q = list(self.ddrqn.predict(current_state)[0])
      
      if entry["terminal"]:
        q[entry["action"]] = entry["reward"]
      else:
        q[entry["action"]] = entry["reward"] + GAMMA * next_q_value
      q_values.append(q)
      max_q_values.append(np.max(q))

    fit = self.ddrqn.fit(np.asarray(current_states).squeeze(),
                        np.asarray(q_values).squeeze(),
                        batch_size=BATCH_SIZE,
                        verbose=0)
    loss = fit.history["loss"][0]
    accuracy = fit.history["acc"][0]
    return loss, accuracy, mean(max_q_values), entry["reward"]

  def _update_epsilon(self):
    self.epsilon -= EXPLORATION_DECAY
    self.epsilon = max(EXPLORATION_MIN, self.epsilon)

  def _reset_target_network(self):
    self.ddrqn_target.set_weights(self.ddrqn.get_weights())
