import atexit
import functools
import sys
import threading
import traceback
import gym
import numpy as np
from PIL import Image


class Robotics:
    LOCK = threading.Lock()

    def __init__(self, name, size=(64, 64)):
        import gym
        with self.LOCK:
            self._env = gym.make(name, reward_type='dense')  # TODO: Reward type dense or sparse
            self._size = size

    @property
    def observation_space(self):
        spaces = {}
        for key in self._env.observation_space:
             spaces[key] = gym.spaces.Box(
                 -np.inf, np.inf, self._env.observation_space[key].shape, dtype=np.float32)
        spaces['observation'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space
    
    @property
    def metadata(self):
        return self._env.metadata
        
    @property
    def reward_range(self):
        return self._env.reward_range

    def step(self, action):
        time_step = self._env.step(action)
        obs = time_step[0]
        obs['observation'] = self.render()
        reward = time_step[1] or 0
        done = time_step[2]
        info = time_step[3]
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = time_step
        obs['observation'] = self.render()
        return obs

    def render(self, *args, **kwargs):
        image = self._env.render(mode='rgb_array', width=128, height=128)  # fetch reach env
        image = Image.fromarray(image).crop((28, 22, 28+self._size[0], 22+self._size[1]))  # fetch reach env
        # image = self._env.render(mode='rgb_array', width=90, height=90)  # hand reach env
        # image = Image.fromarray(image).crop((30, 15, 30+self._size[0], 15+self._size[1]))  # fetch reach env
        # image = np.array(Image.fromarray(image).resize(self._size, Image.BILINEAR))
        # image = np.clip(image, 0, 255).astype(np.uint8)
        self.save(np.array(image))  # Uncomment this line if you want to save an image locally
        return np.array(image)

    def combine(self, image, goal_image):
        pass

    def save(self, array):
        Image.fromarray(array).save("image.png", "PNG")


class Collect:

    def __init__(self, env, callbacks=None, precision=16):
        self._env = env
        self._callbacks = callbacks or ()
        self._precision = precision
        self._episode = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {k: self._convert(v) for k, v in obs.items()}
        transition = obs.copy()
        transition['action'] = action
        transition['reward'] = reward
        transition['discount'] = info.get('discount', np.array(1 - float(done)))
        self._episode.append(transition)
        if done:
            episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
            episode = {k: self._convert(v) for k, v in episode.items()}
            info['episode'] = episode
            for callback in self._callbacks:
                callback(episode)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        transition = obs.copy()
        transition['action'] = np.zeros(self._env.action_space.shape)
        transition['reward'] = 0.0
        transition['discount'] = 1.0
        self._episode = [transition]
        return obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)


class TimeLimit:

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if 'discount' not in info:
                info['discount'] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class ActionRepeat:

    def __init__(self, env, amount):
        self._env = env
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class NormalizeActions:

    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class ObsDict:

    def __init__(self, env, key='obs'):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = {self._key: self._env.observation_space}
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {self._key: np.array(obs)}
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = {self._key: np.array(obs)}
        return obs


class OneHotAction:

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = (self._env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        return space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step(index)

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs:

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert 'reward' not in spaces
        spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs['reward'] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs['reward'] = 0.0
        return obs


class Async:
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _EXCEPTION = 4
    _CLOSE = 5

    def __init__(self, ctor, strategy='process'):
        self._strategy = strategy
        if strategy == 'none':
            self._env = ctor()
        elif strategy == 'thread':
            import multiprocessing.dummy as mp
        elif strategy == 'process':
            import multiprocessing as mp
        else:
            raise NotImplementedError(strategy)
        if strategy != 'none':
            self._conn, conn = mp.Pipe()
            self._process = mp.Process(target=self._worker, args=(ctor, conn))
            atexit.register(self.close)
            self._process.start()
        self._obs_space = None
        self._action_space = None

    @property
    def observation_space(self):
        if not self._obs_space:
            self._obs_space = self.__getattr__('observation_space')
        return self._obs_space

    @property
    def action_space(self):
        if not self._action_space:
            self._action_space = self.__getattr__('action_space')
        return self._action_space

    def __getattr__(self, name):
        if self._strategy == 'none':
            return getattr(self._env, name)
        self._conn.send((self._ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        blocking = kwargs.pop('blocking', True)
        if self._strategy == 'none':
            return functools.partial(getattr(self._env, name), *args, **kwargs)
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        promise = self._receive
        return promise() if blocking else promise

    def close(self):
        if self._strategy == 'none':
            try:
                self._env.close()
            except AttributeError:
                pass
            return
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join()

    def step(self, action, blocking=True):
        return self.call('step', action, blocking=blocking)

    def reset(self, blocking=True):
        return self.call('reset', blocking=blocking)

    def _receive(self):
        try:
            message, payload = self._conn.recv()
        except ConnectionResetError:
            raise RuntimeError('Environment worker crashed.')
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError(f'Received message of unexpected type {message}')

    def _worker(self, ctor, conn):
        try:
            env = ctor()
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    assert payload is None
                    break
                raise KeyError(f'Received message of unknown type {message}')
        except Exception:
            stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            print(f'Error in environment process: {stacktrace}')
            conn.send((self._EXCEPTION, stacktrace))
        conn.close()
