from copy import deepcopy
from collections import deque

import gymnasium as gym
import our_gym_environments

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import *
from mushroom_rl.utils.frames import LazyFrames, preprocess_frame


class MaxAndSkipGymnasiumWrapper(gym.Wrapper):
    def __init__(self, env, skip, max_pooling=True):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape,
                                    dtype=np.uint8)
        self._skip = skip
        self._max_pooling = max_pooling

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0.
        for i in range(self._skip):
            obs, reward, absorbing, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if absorbing:
                break
        if self._max_pooling:
            frame = self._obs_buffer.max(axis=0)
        else:
            frame = self._obs_buffer.mean(axis=0)

        return frame, total_reward, absorbing, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class AtariGymnasiumWrapper(Environment):
    """
    The Atari environment as presented in:
    "Human-level control through deep reinforcement learning". Mnih et. al..
    2015.

    """

    def __init__(self, env, width=84, height=84, ends_at_life=False,
                 max_pooling=True, history_length=4, max_no_op_actions=30, max_steps_before_reset=10000):
        """
        Constructor.

        Args:
            name (str): id name of the Atari game in Gym;
            width (int, 84): width of the screen;
            height (int, 84): height of the screen;
            ends_at_life (bool, False): whether the episode ends when a life is
               lost or not;
            max_pooling (bool, True): whether to do max-pooling or
                average-pooling of the last two frames when using NoFrameskip;
            history_length (int, 4): number of frames to form a state;
            max_no_op_actions (int, 30): maximum number of no-op action to
                execute at the beginning of an episode.

        """
        # MPD creation
        if 'NoFrameskip' in env.spec.id:
            #self.env = MaxAndSkipGymnasium(gym.make(name), history_length, max_pooling)
            pass
        else:
            self.env = env

        # MDP parameters
        self._img_size = (width, height)
        self._episode_ends_at_life = ends_at_life
        #TODO manejar las vidas como un atributo propio de mi ambiente
        self._max_lives = self.env.unwrapped.max_lives
        self._lives = self._max_lives
        self._force_fire = None
        self._real_reset = True
        self._max_no_op_actions = max_no_op_actions
        self._history_length = history_length
        self._current_no_op = None

        # TODO ver si esto tambien lo pongo en la clase o si es una propiedad de los env tipo ALE
        # assert self.env.unwrapped.get_action_meanings()[0] == 'NOOP'

        # MDP properties
        action_space = Discrete(self.env.action_space.n)
        observation_space = Box(
            low=0., high=255., shape=(history_length, self._img_size[1], self._img_size[0]))
        #horizon = np.inf  # the gym time limit is used.
        horizon = max_steps_before_reset  # OurGym env max steps per episodes.
        gamma = .99
        dt = 1 / 60

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if self._real_reset:
            # TODO aqui self.env.reset()[0] es por el estilo de gymnasium de los metodos reset y step
            self._state = preprocess_frame(self.env.reset()[0], self._img_size)
            self._state = deque([deepcopy(
                self._state) for _ in range(self._history_length)],
                maxlen=self._history_length
            )
            self._lives = self._max_lives

        #TODO Esto se puede quitar creo
        #self._force_fire = self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
        self._force_fire = False

        #TODO Esto se puede quitar ya que yo no necesito la accion no_op
        self._current_no_op = np.random.randint(self._max_no_op_actions + 1)

        return LazyFrames(list(self._state), self._history_length)

    def step(self, action):
        action = action[0]

        # Force FIRE action to start episodes in games with lives
        # We dont need to force any fire action in our environment to reset
        #if self._force_fire:
        #    obs, _, _, _, _ = self.env.env.step(1)
        #    self._force_fire = False

        # This has no sense to perform a number of no_op in our environment, so we set current_no_op to 0
        #while self._current_no_op > 0:
        #    obs, _, _, _, _ = self.env.env.step(0)
        #    self._current_no_op -= 1

        obs, reward, absorbing, truncated, info = self.env.step(action)
        #TODO tendria que implementar el truncated en mi ambiente
        self._real_reset = absorbing or truncated

        # If we lose a live
        if info['lives'] != self._lives:
            if self._episode_ends_at_life:
                absorbing = True
            self._lives = info['lives']
            self._force_fire = self.env.unwrapped.get_action_meanings()[
                                   1] == 'FIRE'

        self._state.append(preprocess_frame(obs, self._img_size))

        return LazyFrames(list(self._state),
                          self._history_length), reward, absorbing, truncated, info

    def render(self, record=False):
        #self.env.render(mode='human')
        self.env.render()

        if record:
            return self.env.render(mode='rgb_array')
        else:
            return None

    def stop(self):
        self.env.close()
        self._real_reset = True

    def set_episode_end(self, ends_at_life):
        """
        Setter.

        Args:
            ends_at_life (bool): whether the episode ends when a life is
                lost or not.

        """
        self._episode_ends_at_life = ends_at_life