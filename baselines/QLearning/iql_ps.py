"""
End-to-End JAX Implementation of Multi-Agent Independent Q-Learning with Parameters Sharing

Notice:
- Agents are controlled by a single RNN (parameters sharing).
- Experience replay is a simple buffer with uniform sampling.
- Uses Double Q-Learning with a target agent network (hard-updated).
- Loss is the 1-step TD error.
- Adam optimizer is used instead (not RMSPROP as in pymarl).
- The environment is reset at the end of each episode.
- Assumes all agents are homogeneous (same observation-action spaces).
- Assumes every agent has an independent reward.
- At the moment, agents_ids and last_action features are not included in the agents' observations.

The implementation closely follows the original Pymarl: https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py
"""

import jax
import jax.numpy as jnp
import jax.experimental.checkify as checkify
import numpy as np

import optax
import chex
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from smax import make
from .utils import RolloutManager
from .buffers import uniform_replay
from .buffers.uniform import UniformReplayBufferState

from typing import NamedTuple
from functools import partial



class Transition(NamedTuple):
    obs: dict
    actions: dict
    rewards: dict
    dones: dict

class UniformBuffer:
    # Uniform Buffer replay buffer aggregating transitions from parallel envs
    # based on dejax: https://github.com/hr0nix/dejax/tree/main
    def __init__(self, parallel_envs:int=10, batch_size:int=32, max_size:int=5000):
        self.batch_size = batch_size
        self.buffer = uniform_replay(max_size=max_size)
        self.parallel_envs = parallel_envs
        self.sample = checkify.checkify(self.sample)
        
    def reset(self, transition_sample: Transition) -> UniformReplayBufferState:
        zero_transition = jax.tree_util.tree_map(jnp.zeros_like, transition_sample)
        return self.buffer.init_fn(zero_transition)
    
    @partial(jax.jit, static_argnums=0)
    def add(self, buffer_state: UniformReplayBufferState, transition: Transition) -> UniformReplayBufferState:
        def add_to_buffer(i, buffer_state):
            # assumes the transition is coming from jax.lax so the batch is on dimension 1
            return self.buffer.add_fn(buffer_state, jax.tree_util.tree_map(lambda x: x[:, i], transition))
        # need to use for and not vmap because you can't add multiple transitions on the same buffer in parallel
        return jax.lax.fori_loop(0, self.parallel_envs, add_to_buffer, buffer_state)
    
    @partial(jax.jit, static_argnums=0)
    def sample(self, buffer_state: UniformReplayBufferState, key: chex.PRNGKey) -> Transition:
        return self.buffer.sample_fn(buffer_state, key, self.batch_size)

class EpsilonGreedy:

    def __init__(self, start_e: float, end_e: float, duration: int):
        self.start_e  = start_e
        self.end_e    = end_e
        self.duration = duration
        self.slope    = (end_e - start_e) / duration
        
    @partial(jax.jit, static_argnums=0)
    def get_epsilon(self, t: int):
        e = self.slope*t + self.start_e
        return jnp.clip(e, self.end_e)
    
    @partial(jax.jit, static_argnums=0)
    def choose_actions(self, q_vals: dict, t: int, rng: chex.PRNGKey):
        
        def explore(q, eps, key):
            key_a, key_e   = jax.random.split(key, 2) # a key for sampling random actions and one for picking
            greedy_actions = jnp.argmax(q, axis=-1) # get the greedy actions 
            random_actions = jax.random.randint(key_a, shape=greedy_actions.shape, minval=0, maxval=q.shape[-1]) # sample random actions
            pick_random    = jax.random.uniform(key_e, greedy_actions.shape)<eps # pick which actions should be random
            chosed_actions = jnp.where(pick_random, random_actions, greedy_actions)
            return chosed_actions
        
        eps = self.get_epsilon(t)
        keys = dict(zip(q_vals.keys(), jax.random.split(rng, len(q_vals)))) # get a key for each agent
        choosed_actions = jax.tree_map(lambda q, k: explore(q, eps, k), q_vals, keys)
        return choosed_actions


class ScannedRNN(nn.Module):
    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[-1], *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell()(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell.initialize_carry(
            jax.random.PRNGKey(0), (*batch_size,), hidden_size
        )


class AgentRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)

        return hidden, q_vals
    
    @partial(jax.jit, static_argnums=0)
    def homogeneous_pass(self, params, hidden_state, obs, dones):
        """
        - concatenate agents and parallel envs to process them in one batch
        - assumes all agents are homogenous (same obs and action shapes)
        - assumes the first dimension is the time step
        - assumes the other dimensions except the last one can be considered as batches
        - returns a dictionary of q_vals indexed by the agent names
        """
        agents, flatten_agents_obs = zip(*obs.items())
        original_shape = flatten_agents_obs[0].shape # assumes obs shape is the same for all agents
        batched_input = (
            jnp.concatenate(flatten_agents_obs, axis=1), # (time_step, n_agents*n_envs, obs_size)
            jnp.concatenate([dones[agent] for agent in agents], axis=1), # ensure to not pass other keys (like __all__)
        )
        hidden_state, q_vals = self.apply(params, hidden_state, batched_input)
        q_vals = q_vals.reshape(original_shape[0], len(agents), *original_shape[1:-1], -1) # (time_steps, n_agents, n_envs, action_dim)
        q_vals = {a:q_vals[:,i] for i,a in enumerate(agents)}
        return hidden_state, q_vals



def make_train(config):
    
    def train(rng):
        
        env, env_params = make(config['ENV_NAME'], num_agents=config["NUM_AGENTS"])
        config["NUM_STEPS"] = config.get("NUM_STEPS", env_params.max_steps)
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        batched_env = RolloutManager(env, batch_size=config["NUM_ENVS"])
        init_obs, env_state = batched_env.batch_reset(_rng)
        init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

        # INIT BUFFER
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3) # use a dummy rng here
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {agent: batched_env.batch_sample(key_a[i], agent) for i, agent in enumerate(env.agents)}
            obs, env_state, rewards, dones, infos = batched_env.batch_step(key_s, env_state, actions)
            transition = Transition(obs, actions, rewards, dones)
            return env_state, transition
        _, sample_traj = jax.lax.scan(
            _env_sample_step, env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree_map(lambda x: x[:, 0], sample_traj) # remove the NUM_ENV dim
        buffer = UniformBuffer(parallel_envs=config["NUM_ENVS"], batch_size=config["BUFFER_BATCH_SIZE"], max_size=config["BUFFER_SIZE"])
        buffer_state = buffer.reset(sample_traj_unbatched)

        # INIT NETWORK
        agent = AgentRNN(action_dim=env.action_space('agent_0').n, hidden_dim=config['AGENT_HIDDEN_DIM'])
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, 1, *env.observation_space('agent_0').shape)), # (time_step, batch_size, obs_size)
            jnp.zeros((1, 1)) # (time_step, batch size)
        )
        init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], 1) # (batch_size, hidden_dim)
        network_params = agent.init(_rng, init_hs, init_x)
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=agent.apply,
            params=network_params,
            tx=tx,
        )
        # target network params
        target_agent_params = jax.tree_map(lambda x: jnp.copy(x), train_state.params)

        # INIT EXPLORATION STRATEGY
        explorer = EpsilonGreedy(
            start_e=config["EPSILON_START"],
            end_e=config["EPSILON_FINISH"],
            duration=config["EPSILON_ANNEAL_TIME"]
        )


        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, target_agent_params, env_state, buffer_state, time_state, init_obs, init_dones, rng = runner_state


            # EPISODE STEP
            def _env_step(step_state, unused):

                params, env_state, last_obs, last_dones, hstate, rng, t = step_state

                # prepare rngs for actions and step
                rng, key_a, key_s = jax.random.split(rng, 3)

                # SELECT ACTION
                # add a dummy time_step dimension to the agent input
                obs_   = jax.tree_map(lambda x: x[np.newaxis, :], last_obs)
                dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
                # get the q_values from the agent netwoek
                hstate, q_vals = agent.homogeneous_pass(params, hstate, obs_, dones_)
                # remove the dummy time_step dimension
                q_vals = jax.tree_map(lambda x: x.squeeze(0), q_vals)
                # explore with epsilon greedy_exploration
                actions = explorer.choose_actions(q_vals, t, key_a)

                # STEP ENV
                obs, env_state, rewards, dones, infos = batched_env.batch_step(key_s, env_state, actions)
                transition = Transition(last_obs, actions, rewards, dones)

                step_state = (params, env_state, obs, dones, hstate, rng, t+1)
                return step_state, transition


            # prepare the step state and collect the episode trajectory
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["NUM_ENVS"])

            step_state = (
                train_state.params,
                env_state,
                init_obs,
                init_dones,
                hstate, 
                _rng,
                time_state['timesteps'] # t is needed to compute epsilon
            )

            step_state, traj_batch = jax.lax.scan(
                _env_step, step_state, None, config["NUM_STEPS"]
            )

            # BUFFER UPDATE: save the collected trajectory in the buffer
            buffer_state = buffer.add(buffer_state, traj_batch)

            # LEARN PHASE
            def q_of_action(q, u):
                """index the q_values with action indices"""
                q_u = jnp.take_along_axis(q, jnp.expand_dims(u, axis=-1), axis=-1)
                return jnp.squeeze(q_u, axis=-1)

            def compute_target(target_q_val, q_val, reward, done):
                """compute the 1-step Q-Learning target"""
                greedy_actions = jnp.argmax(q_val, axis=-1)
                target_max_qvals = q_of_action(target_q_val, greedy_actions)
                target = reward[:-1] + config["GAMMA"]*(1-done[:-1])*target_max_qvals[1:]
                return target

            def _loss_fn(params, target_agent_params, init_hs, learn_traj):

                _, q_vals = agent.homogeneous_pass(params, init_hs, learn_traj.obs, learn_traj.dones)
                _, target_q_vals = agent.homogeneous_pass(target_agent_params, init_hs, learn_traj.obs, learn_traj.dones)

                # get the q_vals of the taken actions (with exploration) for each agent
                chosen_action_qvals = jax.tree_map(
                    lambda q, u: q_of_action(q, u)[:-1], # avoid last timestep
                    q_vals,
                    learn_traj.actions
                )

                # get the target for each agent (assumes every agent has a reward)
                targets = jax.tree_map(
                    compute_target,
                    target_q_vals,
                    q_vals,
                    {agent:learn_traj.rewards[agent] for agent in env.agents}, # rewards and agents could contain additional keys
                    {agent:learn_traj.dones[agent] for agent in env.agents}
                )

                # compute a single l2 loss for all the agents in one pass (parameter sharing)
                chosen_action_qvals = jnp.concatenate(list(chosen_action_qvals.values()))
                targets = jnp.concatenate(list(targets.values()))
                loss = jnp.mean((chosen_action_qvals - jax.lax.stop_gradient(targets))**2)

                return loss


            # sample a batched trajectory from the buffer and set the time step dim in first axis
            rng, _rng = jax.random.split(rng)
            _, learn_traj = buffer.sample(buffer_state, _rng) # (batch_size, max_time_steps, ...)
            learn_traj = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), learn_traj) # (max_time_steps, batch_size, ...)
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["BUFFER_BATCH_SIZE"]) 

            # compute loss and optimize grad
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
            loss, grads = grad_fn(train_state.params, target_agent_params, init_hs, learn_traj)
            train_state = train_state.apply_gradients(grads=grads)


            # UPDATE THE VARIABLES AND RETURN
            # reset the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = batched_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

            # update the states
            time_state['timesteps'] = step_state[-1]
            time_state['updates']   = time_state['updates'] + 1

            # update the target network if necessary
            target_agent_params = jax.lax.cond(
                time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0,
                lambda _: jax.tree_map(lambda x: jnp.copy(x), train_state.params),
                lambda _: target_agent_params,
                operand=None
            )

            # update the returning metrics
            metrics = {
                'timesteps': time_state['timesteps']*config['NUM_ENVS'],
                'updates' : time_state['updates'],
                'loss': loss,
                'rewards': jax.tree_util.tree_map(lambda x: jnp.sum(x), traj_batch.rewards)
            }

            if config.get("DEBUG"):

                def callback(info):
                    print(
                        f"""
                        Update {info['updates']}:
                        \t n_timesteps: {info['updates']*config['NUM_ENVS']}
                        \t avg_reward: {info['rewards']}
                        \t loss: {info['loss']}
                        """
                    )

                jax.debug.callback(callback, metrics)

            runner_state = (train_state, target_agent_params, env_state, buffer_state, time_state, init_obs, init_dones, rng)

            return runner_state, metrics
        
        # train
        time_state = {
            'timesteps':jnp.array(0),
            'updates':  jnp.array(0)
        }
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, target_agent_params, env_state, buffer_state, time_state, init_obs, init_dones, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {'runner_state':runner_state, 'metrics':metrics}
    
    return train


if __name__ == "__main__":
    config = {
        "NUM_ENVS":1,
        "NUM_AGENTS":3,
        "BUFFER_SIZE":5000,
        "BUFFER_BATCH_SIZE":32,
        "TOTAL_TIMESTEPS":2e+6,
        "AGENT_HIDDEN_DIM":64,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.05,
        "EPSILON_ANNEAL_TIME": 100000,
        "AGENT_HIDDEN_DIM": 64,
        "MAX_GRAD_NORM": 10,
        "TARGET_UPDATE_INTERVAL": 200, 
        "LR": 0.0005,
        "GAMMA": 0.99,
        "ENV_NAME": "MPE_simple_spread_v2",
        "DEBUG": False,
    }

    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))

    import time
    t0 = time.time()
    out = train_jit(rng)
    print(f"time: {time.time() - t0:.2f} s")