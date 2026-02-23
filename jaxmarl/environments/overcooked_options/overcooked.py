from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax import lax
from jaxmarl.environments.overcooked.common import (
    COLOR_TO_INDEX,
    DIR_TO_VEC,
    OBJECT_INDEX_TO_VEC,
    OBJECT_TO_INDEX,
)
from jaxmarl.environments.overcooked.overcooked import Overcooked, Actions, POT_EMPTY_STATUS, POT_READY_STATUS, POT_FULL_STATUS


@struct.dataclass
class State:
    agent_pos: chex.Array
    agent_dir: chex.Array
    agent_dir_idx: chex.Array
    agent_inv: chex.Array
    prev_onion_drops: chex.Array
    prev_plate_drops: chex.Array
    prev_onion_pickups: chex.Array
    prev_plate_pickups: chex.Array
    goal_pos: chex.Array
    pot_pos: chex.Array
    wall_map: chex.Array
    maze_map: chex.Array
    time: int
    terminal: bool

class OvercookedOptions(Overcooked):
    """Overcooked for Options training"""

    def reset(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], State]:
        obs, state = super().reset(key)
        state = State(
            agent_pos=state.agent_pos,
            agent_dir=state.agent_dir,
            agent_dir_idx=state.agent_dir_idx,
            agent_inv=state.agent_inv,
            prev_onion_drops=jnp.zeros((self.num_agents,), dtype=jnp.bool_),
            prev_plate_drops=jnp.zeros((self.num_agents,), dtype=jnp.bool_),
            prev_onion_pickups=jnp.zeros((self.num_agents,), dtype=jnp.bool_),
            prev_plate_pickups=jnp.zeros((self.num_agents,), dtype=jnp.bool_),
            goal_pos=state.goal_pos,
            pot_pos=state.pot_pos,
            wall_map=state.wall_map.astype(jnp.bool_),
            maze_map=state.maze_map,
            time=0,
            terminal=False,
        )
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        option_id: int = 0,
        single_agent: bool = True,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""
        acts = self.action_set.take(indices=jnp.array([actions["agent_0"], actions["agent_1"]]))

        state, reward, shaped_rewards, successful_delivery = self.step_agents(key, state, acts, option_id, single_agent)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)
        rewards = {"agent_0": reward[0], "agent_1": reward[1]}
        shaped_rewards = {"agent_0": shaped_rewards[0], "agent_1": shaped_rewards[1]}
        dones = {"agent_0": done, "agent_1": done, "__all__": done}

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {"shaped_reward": shaped_rewards,
             "successful_delivery": successful_delivery},
        )

    def step_agents(
        self,
        key: chex.PRNGKey,
        state: State,
        action: chex.Array,
        option_id: int = 0,
        single_agent: bool = True,
    ) -> Tuple[State, float]:
        # Update agent position (forward action)
        is_move_action = jnp.logical_and(action != Actions.stay, action != Actions.interact)
        is_move_action_transposed = jnp.expand_dims(is_move_action, 0).transpose()  # Necessary to broadcast correctly

        fwd_pos = jnp.minimum(
            jnp.maximum(
                state.agent_pos
                + is_move_action_transposed * DIR_TO_VEC[jnp.minimum(action, 3)]
                + ~is_move_action_transposed * state.agent_dir,
                0,
            ),
            jnp.array((self.width - 1, self.height - 1), dtype=jnp.uint32),
        )

        # Can't go past wall or goal
        def _wall_or_goal(fwd_position, wall_map, goal_pos):
            fwd_wall = wall_map.at[fwd_position[1], fwd_position[0]].get()
            goal_collision = lambda pos, goal: jnp.logical_and(pos[0] == goal[0], pos[1] == goal[1])
            fwd_goal = jax.vmap(goal_collision, in_axes=(None, 0))(fwd_position, goal_pos)
            # fwd_goal = jnp.logical_and(fwd_position[0] == goal_pos[0], fwd_position[1] == goal_pos[1])
            fwd_goal = jnp.any(fwd_goal)
            return fwd_wall, fwd_goal

        fwd_pos_has_wall, fwd_pos_has_goal = jax.vmap(_wall_or_goal, in_axes=(0, None, None))(
            fwd_pos, state.wall_map, state.goal_pos
        )

        fwd_pos_blocked = jnp.logical_or(fwd_pos_has_wall, fwd_pos_has_goal).reshape((self.num_agents, 1))

        bounced = jnp.logical_or(fwd_pos_blocked, ~is_move_action_transposed)

        # Agents can't overlap
        # Hardcoded for 2 agents (call them Alice and Bob)
        agent_pos_prev = jnp.array(state.agent_pos)
        fwd_pos = (bounced * state.agent_pos + (~bounced) * fwd_pos).astype(jnp.uint32)
        collision = jnp.all(fwd_pos[0] == fwd_pos[1])

        # No collision = No movement. This matches original Overcooked env.
        alice_pos = jnp.where(
            collision,
            state.agent_pos[0],  # collision and Bob bounced
            fwd_pos[0],
        )
        bob_pos = jnp.where(
            collision,
            state.agent_pos[1],  # collision and Alice bounced
            fwd_pos[1],
        )

        # Prevent swapping places (i.e. passing through each other)
        swap_places = jnp.logical_and(
            jnp.all(fwd_pos[0] == state.agent_pos[1]),
            jnp.all(fwd_pos[1] == state.agent_pos[0]),
        )
        alice_pos = jnp.where(~collision * swap_places, state.agent_pos[0], alice_pos)
        bob_pos = jnp.where(~collision * swap_places, state.agent_pos[1], bob_pos)

        fwd_pos = fwd_pos.at[0].set(alice_pos)
        fwd_pos = fwd_pos.at[1].set(bob_pos)
        agent_pos = fwd_pos.astype(jnp.uint32)

        # Update agent direction
        agent_dir_idx = ~is_move_action * state.agent_dir_idx + is_move_action * action
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        # Handle interacts. Agent 1 first, agent 2 second, no collision handling.
        # This matches the original Overcooked
        fwd_pos = state.agent_pos + state.agent_dir
        maze_map = state.maze_map
        is_interact_action = action == Actions.interact

        # Compute the effect of interact first, then apply it if needed
        candidate_maze_map, alice_inv, alice_reward, alice_shaped_reward, alice_delivery, alice_pickup, alice_drop = self.process_interact(
            maze_map, state.wall_map, fwd_pos, state.agent_inv, 0
        )
        alice_interact = is_interact_action[0]
        bob_interact = is_interact_action[1]

        maze_map = jax.lax.select(alice_interact, candidate_maze_map, maze_map)
        alice_inv = jax.lax.select(alice_interact, alice_inv, state.agent_inv[0])
        alice_reward = jax.lax.select(alice_interact, alice_reward, 0.0)
        alice_shaped_reward = jax.lax.select(alice_interact, alice_shaped_reward, 0.0)

        candidate_maze_map, bob_inv, bob_reward, bob_shaped_reward, bob_delivery, bob_pickup, bob_drop = self.process_interact(
            maze_map, state.wall_map, fwd_pos, state.agent_inv, 1
        )
        maze_map = jax.lax.select(bob_interact, candidate_maze_map, maze_map)
        bob_inv = jax.lax.select(bob_interact, bob_inv, state.agent_inv[1])
        bob_reward = jax.lax.select(bob_interact, bob_reward, 0.0)
        bob_shaped_reward = jax.lax.select(bob_interact, bob_shaped_reward, 0.0)

        agent_inv = jnp.array([alice_inv, bob_inv])

        # Update agent component in maze_map
        def _get_agent_updates(agent_dir_idx, agent_pos, agent_pos_prev, agent_idx):
            agent = jnp.array(
                [OBJECT_TO_INDEX["agent"], COLOR_TO_INDEX["red"] + agent_idx * 2, agent_dir_idx], dtype=jnp.uint8
            )
            agent_x_prev, agent_y_prev = agent_pos_prev
            agent_x, agent_y = agent_pos
            return agent_x, agent_y, agent_x_prev, agent_y_prev, agent

        vec_update = jax.vmap(_get_agent_updates, in_axes=(0, 0, 0, 0))
        agent_x, agent_y, agent_x_prev, agent_y_prev, agent_vec = vec_update(
            agent_dir_idx, agent_pos, agent_pos_prev, jnp.arange(self.num_agents)
        )
        empty = jnp.array([OBJECT_TO_INDEX["empty"], 0, 0], dtype=jnp.uint8)

        # Compute padding, added automatically by map maker function
        height = self.obs_shape[1]
        padding = (state.maze_map.shape[0] - height) // 2

        maze_map = maze_map.at[padding + agent_y_prev, padding + agent_x_prev, :].set(empty)
        maze_map = maze_map.at[padding + agent_y, padding + agent_x, :].set(agent_vec)

        # Update pot cooking status
        def _cook_pots(pot):
            pot_status = pot[-1]
            is_cooking = jnp.array(pot_status <= POT_FULL_STATUS)
            not_done = jnp.array(pot_status > POT_READY_STATUS)
            pot_status = (
                is_cooking * not_done * (pot_status - 1) + (~is_cooking) * pot_status
            )  # defaults to zero if done
            return pot.at[-1].set(pot_status)

        pot_x = state.pot_pos[:, 0]
        pot_y = state.pot_pos[:, 1]
        pots = maze_map.at[padding + pot_y, padding + pot_x].get()
        pots = jax.vmap(_cook_pots, in_axes=0)(pots)
        maze_map = maze_map.at[padding + pot_y, padding + pot_x, :].set(pots)

        picks = jnp.array([alice_pickup * alice_interact, bob_pickup * bob_interact], dtype=bool)
        drops = jnp.array([alice_drop * alice_interact, bob_drop * bob_interact], dtype=bool)
        old_invs = state.agent_inv
        new_invs  = agent_inv

        def single_agent_option_reward_fn(agent_pos, successful_pickup, successful_drop, object_in_inv, new_object_in_inv, prev_onion_pickup, prev_plate_pickup, prev_onion_drop, prev_plate_drop, fwd_pos, option_id):
            def pickup_onion_reward():
                reward = jnp.array(successful_pickup * (new_object_in_inv == OBJECT_TO_INDEX["onion"]), dtype=float) * (1-prev_onion_pickup)
                valid_table_condition = (fwd_pos[0] > 0) & (fwd_pos[0] < self.width - 1) * (fwd_pos[1] > 0) & (fwd_pos[1] < self.height - 1)
                reward += valid_table_condition * reward * 0.5  # Bonus for picking up onion while facing valid table location to encourage picking up onions near the table 
                # Penalize to enusre termination action is taken after picking up onion
                reward += jnp.array(successful_drop * (object_in_inv == OBJECT_TO_INDEX["onion"]), dtype=float) * -2.0
                return reward

            def pickup_plate_reward():
                reward = jnp.array(successful_pickup * (new_object_in_inv == OBJECT_TO_INDEX["plate"]), dtype=float) * (1-prev_plate_pickup)
                valid_table_condition = (fwd_pos[0] > 0) & (fwd_pos[0] < self.width - 1) * (fwd_pos[1] > 0) & (fwd_pos[1] < self.height - 1)
                reward += valid_table_condition * reward * 0.5  # Bonus for picking up plate while facing valid table location to encourage picking up plates near the table
                # Penalize to enusre termination action is taken after picking up plate
                reward += jnp.array(successful_drop * (object_in_inv == OBJECT_TO_INDEX["plate"]), dtype=float) * -2.0
                return reward

            def place_onion_on_table_reward():
                valid_table_condition = (fwd_pos[0] > 0) & (fwd_pos[0] < self.width - 1) * (fwd_pos[1] > 0) & (fwd_pos[1] < self.height - 1)

                pick_first_onion_bonus =(new_object_in_inv == OBJECT_TO_INDEX["onion"]) * ~prev_onion_pickup * ~valid_table_condition * 1
                onion_on_table_reward = successful_drop * (object_in_inv == OBJECT_TO_INDEX["onion"]) * valid_table_condition * (1-prev_onion_drop) * 2.0
                pick_penalty_after_onion_drop = successful_pickup * (new_object_in_inv == OBJECT_TO_INDEX["onion"]) * prev_onion_drop * - 1
                drops_onion_outside_table_penalty = successful_drop * (object_in_inv == OBJECT_TO_INDEX["onion"]) * ~valid_table_condition * -1.0
                picks_onion_from_table_penalty = successful_pickup * (new_object_in_inv == OBJECT_TO_INDEX["onion"]) * valid_table_condition * -3.0
                bottom_row_penalty = 0 # (agent_pos[1] != self.height - 2) * -0.1
                reward = pick_first_onion_bonus + onion_on_table_reward + pick_penalty_after_onion_drop + drops_onion_outside_table_penalty + picks_onion_from_table_penalty + bottom_row_penalty
                reward -= prev_onion_drop * 0.025
                return reward.astype(float)

            def place_plate_on_table_reward():
                valid_table_condition = (fwd_pos[0] > 0) & (fwd_pos[0] < self.width - 1) * (fwd_pos[1] > 0) & (fwd_pos[1] < self.height - 1)

                pick_first_plate_bonus =(new_object_in_inv == OBJECT_TO_INDEX["plate"]) * ~prev_plate_pickup * ~valid_table_condition * 1
                plate_on_table_reward = successful_drop * (object_in_inv == OBJECT_TO_INDEX["plate"]) * valid_table_condition * (1-prev_plate_drop) * 2.0
                pick_penalty_after_plate_drop = successful_pickup * (new_object_in_inv == OBJECT_TO_INDEX["plate"]) * prev_plate_drop * - 1
                drops_plate_outside_table_penalty = successful_drop * (object_in_inv == OBJECT_TO_INDEX["plate"]) * ~valid_table_condition * -1.0
                picks_plate_from_table_penalty = successful_pickup * (new_object_in_inv == OBJECT_TO_INDEX["plate"]) * valid_table_condition * -3.0
                bottom_row_penalty = 0 # (agent_pos[1] != self.height - 2) * -0.1
                reward = pick_first_plate_bonus + plate_on_table_reward + pick_penalty_after_plate_drop + drops_plate_outside_table_penalty + picks_plate_from_table_penalty + bottom_row_penalty
                reward -= prev_plate_drop * 0.025
                return reward.astype(float)

            reward = jax.lax.switch(
                option_id,
                [
                    pickup_onion_reward,
                    pickup_plate_reward,
                    place_onion_on_table_reward,
                    place_plate_on_table_reward,
                ],
            )
            return reward

        def multi_agent_option_reward_fn(
            picks, drops, old_invs, new_invs, prev_onion_pickups, prev_onion_drops, prev_plate_pickups, prev_plate_drops, option_id
        ):
            def all_pickup_onion_reward():
                onion_pick_reward =  jnp.any(picks * (new_invs == OBJECT_TO_INDEX["onion"]))
                penalty_for_dropping_onion = jnp.any(drops * (old_invs == OBJECT_TO_INDEX["onion"])) * -2.0
                no_onion_carried_penalty = jnp.any(new_invs != OBJECT_TO_INDEX["onion"]) * -0.025
                reward = jnp.zeros(2) + onion_pick_reward + penalty_for_dropping_onion + no_onion_carried_penalty
                return reward

            def all_pickup_plate_reward():
                plate_pick_reward =  jnp.any(picks * (new_invs == OBJECT_TO_INDEX["plate"]))
                penalty_for_dropping_plate = jnp.any(drops * (old_invs == OBJECT_TO_INDEX["plate"])) * -2.0
                no_plate_carried_penalty = jnp.any(new_invs != OBJECT_TO_INDEX["plate"]) * -0.025
                reward = jnp.zeros(2) + plate_pick_reward + penalty_for_dropping_plate + no_plate_carried_penalty
                return reward

            def different_item_pickup_reward():
                # Reward first pickup of different items by two agents. This encourages the agents to coordinate and pick up different items needed for the recipe, instead of both going for the same item
                onion_pick_reward = jnp.any(picks * (new_invs == OBJECT_TO_INDEX["onion"]) * (picks * (1 - prev_onion_pickups))) * (1 - jnp.any(prev_onion_pickups))
                plate_pick_reward = jnp.any(picks * (new_invs == OBJECT_TO_INDEX["plate"]) * (picks * (1 - prev_plate_pickups))) * (1 - jnp.any(prev_plate_pickups))
                reward = jnp.zeros(2) + onion_pick_reward + plate_pick_reward
                # Small penaly for dropping
                reward -= jnp.any(drops) * 0.25
                return reward

            def onoin_passing_reward(receiving_agent, giving_agent):
                # Table location condition
                valid_table_condition = (
                    ((fwd_pos[:, 0] > 0) & (fwd_pos[:, 0] < self.width - 1)) *
                    ((fwd_pos[:, 1] > 0) & (fwd_pos[:, 1] < self.height - 1))
                )
                receive_onion_on_table_reward = picks[receiving_agent] * (new_invs[receiving_agent] == OBJECT_TO_INDEX["onion"]) * ~prev_onion_pickups[receiving_agent] * valid_table_condition[receiving_agent] * 2.0
                receive_onion_outside_table_penalty = picks[receiving_agent] * ~valid_table_condition[receiving_agent] * -1.0
                drop_onion_penalty = drops[receiving_agent] * (old_invs[receiving_agent] == OBJECT_TO_INDEX["onion"]) * -1.0
                receiver_outside_of_top_row_penalty = 0 # (state.agent_pos[receiving_agent][1] != 1) * -0.1
                receiver_reward = receive_onion_on_table_reward + receive_onion_outside_table_penalty + drop_onion_penalty + receiver_outside_of_top_row_penalty

                giver_pick_first_onion_bonus =(new_invs[giving_agent] == OBJECT_TO_INDEX["onion"]) * ~prev_onion_pickups[giving_agent] * ~valid_table_condition[giving_agent] * 0.5
                give_onion_on_table_reward = drops[giving_agent] * (old_invs[giving_agent] == OBJECT_TO_INDEX["onion"]) * valid_table_condition[giving_agent] * (1-prev_onion_drops[giving_agent]) * 1.0
                giver_pick_penalty_after_onion_drop = picks[giving_agent] * (new_invs[giving_agent] == OBJECT_TO_INDEX["onion"]) * prev_onion_drops[giving_agent] * - 1
                giver_drops_onion_outside_table_penalty = drops[giving_agent] * (old_invs[giving_agent] == OBJECT_TO_INDEX["onion"]) * ~valid_table_condition[giving_agent] * -1.0
                giver_picks_onion_from_table_penalty = picks[giving_agent] * (new_invs[giving_agent] == OBJECT_TO_INDEX["onion"]) * valid_table_condition[giving_agent] * -1.0
                giver_bottom_row_penalty = 0 #(state.agent_pos[giving_agent][1] != self.height - 2) * -0.1
                giver_reward = giver_pick_first_onion_bonus + give_onion_on_table_reward + giver_pick_penalty_after_onion_drop + giver_drops_onion_outside_table_penalty + giver_picks_onion_from_table_penalty + giver_bottom_row_penalty

                # vertical_alignment_bonus = (state.agent_pos[receiving_agent][0] == state.agent_pos[giving_agent][0]) * 0.0125
                horizontal_alignment = (state.agent_pos[receiving_agent][0] == state.agent_pos[giving_agent][0]) * 0.0125

                reward = jnp.zeros(2)
                reward = reward.at[receiving_agent].set(receiver_reward + horizontal_alignment)
                reward = reward.at[giving_agent].set(giver_reward + receive_onion_on_table_reward * 0.5)
                reward = reward - prev_onion_pickups[receiving_agent] * 0.05

                return reward

            def plate_passing_reward(receiving_agent, giving_agent):
                # Table location condition
                valid_table_condition = (
                    ((fwd_pos[:, 0] > 0) & (fwd_pos[:, 0] < self.width - 1)) *
                    ((fwd_pos[:, 1] > 0) & (fwd_pos[:, 1] < self.height - 1))
                )
                receive_plate_on_table_reward = picks[receiving_agent] * (new_invs[receiving_agent] == OBJECT_TO_INDEX["plate"]) * ~prev_plate_pickups[receiving_agent] * valid_table_condition[receiving_agent] * 2.0
                receive_plate_outside_table_penalty = picks[receiving_agent] * ~valid_table_condition[receiving_agent] * -1.0
                drop_plate_penalty = drops[receiving_agent] * (old_invs[receiving_agent] == OBJECT_TO_INDEX["plate"]) * -1.0
                receiver_reward = receive_plate_on_table_reward + receive_plate_outside_table_penalty + drop_plate_penalty

                giver_pick_first_plate_bonus =(new_invs[giving_agent] == OBJECT_TO_INDEX["plate"]) * ~prev_plate_pickups[giving_agent] * ~valid_table_condition[giving_agent] * 0.5
                give_plate_on_table_reward = drops[giving_agent] * (old_invs[giving_agent] == OBJECT_TO_INDEX["plate"]) * valid_table_condition[giving_agent] * (1-prev_plate_drops[giving_agent]) * 1.0
                giver_pick_penalty_after_plate_drop = picks[giving_agent] * (new_invs[giving_agent] == OBJECT_TO_INDEX["plate"]) * prev_plate_drops[giving_agent] * - 1
                giver_drops_plate_outside_table_penalty = drops[giving_agent] * (old_invs[giving_agent] == OBJECT_TO_INDEX["plate"]) * ~valid_table_condition[giving_agent] * -1.0
                giver_picks_plate_from_table_penalty = picks[giving_agent] * (new_invs[giving_agent] == OBJECT_TO_INDEX["plate"]) * valid_table_condition[giving_agent] * -1.0
                giver_reward = giver_pick_first_plate_bonus + give_plate_on_table_reward + giver_pick_penalty_after_plate_drop + giver_drops_plate_outside_table_penalty + giver_picks_plate_from_table_penalty

                # vertical_alignment_bonus = (state.agent_pos[receiving_agent][0] == state.agent_pos[giving_agent][0]) * 0.0125
                horizontal_alignment = (state.agent_pos[receiving_agent][0] == state.agent_pos[giving_agent][0]) * 0.0125

                reward = jnp.zeros(2)
                reward = reward.at[receiving_agent].set(receiver_reward + horizontal_alignment)
                reward = reward.at[giving_agent].set(giver_reward + receive_plate_on_table_reward * 0.5)
                reward = reward - prev_plate_pickups[receiving_agent] * 0.05

                return reward


            reward = jax.lax.switch(
                option_id,
                [
                    all_pickup_onion_reward,
                    all_pickup_plate_reward,
                    different_item_pickup_reward,
                    lambda: onoin_passing_reward(0, 1), # Alice receives onion from Bob
                    lambda: onoin_passing_reward(1, 0), # Bob receives onion from Alice
                    lambda: plate_passing_reward(0, 1), # Alice receives plate from Bob
                    lambda: plate_passing_reward(1, 0), # Bob receives plate from Alice
                ],
            )
            return reward

        distributed_single_agent_reward_fn = jax.vmap(single_agent_option_reward_fn, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None))
        reward = jax.lax.cond(
            single_agent,
            lambda _: distributed_single_agent_reward_fn(state.agent_pos, picks, drops, old_invs, new_invs, state.prev_onion_pickups, state.prev_plate_pickups, state.prev_onion_drops, state.prev_plate_drops, fwd_pos, option_id),
            lambda _: multi_agent_option_reward_fn(
                picks, drops, old_invs, new_invs, state.prev_onion_pickups, state.prev_onion_drops, state.prev_plate_pickups, state.prev_plate_drops, option_id),
            operand=None,
        )

        prev_onion_pickups = state.prev_onion_pickups | (new_invs == OBJECT_TO_INDEX["onion"])
        prev_onion_drops   = state.prev_onion_drops   | (drops & (old_invs == OBJECT_TO_INDEX["onion"]))
        prev_plate_pickups = state.prev_plate_pickups | (new_invs == OBJECT_TO_INDEX["plate"])
        prev_plate_drops   = state.prev_plate_drops   | (drops & (old_invs == OBJECT_TO_INDEX["plate"]))

        successful_delivery = alice_shaped_reward + bob_shaped_reward

        return (
            state.replace(
                agent_pos=agent_pos,
                agent_dir_idx=agent_dir_idx,
                agent_dir=agent_dir,
                agent_inv=agent_inv,
                prev_onion_pickups=prev_onion_pickups,
                prev_onion_drops=prev_onion_drops,
                prev_plate_pickups=prev_plate_pickups,
                prev_plate_drops=prev_plate_drops,
                maze_map=maze_map,
                terminal=False,
            ),
            reward,
            (alice_shaped_reward, bob_shaped_reward),
            successful_delivery,
        )


    def process_interact(
        self,
        maze_map: chex.Array,
        wall_map: chex.Array,
        fwd_pos_all: chex.Array,
        inventory_all: chex.Array,
        player_idx: int,
    ):
        """Assume agent took interact actions. Result depends on what agent is facing and what it is holding."""

        fwd_pos = fwd_pos_all[player_idx]
        inventory = inventory_all[player_idx]

        height = self.obs_shape[1]
        padding = (maze_map.shape[0] - height) // 2

        # Get object in front of agent (on the "table")
        maze_object_on_table = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0]].get()
        object_on_table = maze_object_on_table[0]  # Simple index

        # Booleans depending on what the object is
        object_is_pile = jnp.logical_or(
            object_on_table == OBJECT_TO_INDEX["plate_pile"], object_on_table == OBJECT_TO_INDEX["onion_pile"]
        )
        object_is_pot = jnp.array(object_on_table == OBJECT_TO_INDEX["pot"])
        object_is_goal = jnp.array(object_on_table == OBJECT_TO_INDEX["goal"])
        object_is_agent = jnp.array(object_on_table == OBJECT_TO_INDEX["agent"])
        object_is_pickable = jnp.logical_or(
            jnp.logical_or(object_on_table == OBJECT_TO_INDEX["plate"], object_on_table == OBJECT_TO_INDEX["onion"]),
            object_on_table == OBJECT_TO_INDEX["dish"],
        )
        # Whether the object in front is counter space that the agent can drop on.
        is_table = jnp.logical_and(wall_map.at[fwd_pos[1], fwd_pos[0]].get(), ~object_is_pot)

        table_is_empty = jnp.logical_or(
            object_on_table == OBJECT_TO_INDEX["wall"], object_on_table == OBJECT_TO_INDEX["empty"]
        )

        # Pot status (used if the object is a pot)
        pot_status = maze_object_on_table[-1]

        # Get inventory object, and related booleans
        inv_is_empty = jnp.array(inventory == OBJECT_TO_INDEX["empty"])
        object_in_inv = inventory
        holding_onion = jnp.array(object_in_inv == OBJECT_TO_INDEX["onion"])
        holding_plate = jnp.array(object_in_inv == OBJECT_TO_INDEX["plate"])
        holding_dish = jnp.array(object_in_inv == OBJECT_TO_INDEX["dish"])

        # Interactions with pot. 3 cases: add onion if missing, collect soup if ready, do nothing otherwise
        case_1 = (pot_status > POT_FULL_STATUS) * holding_onion * object_is_pot
        case_2 = (pot_status == POT_READY_STATUS) * holding_plate * object_is_pot
        case_3 = (pot_status > POT_READY_STATUS) * (pot_status <= POT_FULL_STATUS) * object_is_pot
        else_case = ~case_1 * ~case_2 * ~case_3

        # Update pot status and object in inventory
        new_pot_status = (
            case_1 * (pot_status - 1) + case_2 * POT_EMPTY_STATUS + case_3 * pot_status + else_case * pot_status
        )
        new_object_in_inv = (
            case_1 * OBJECT_TO_INDEX["empty"]
            + case_2 * OBJECT_TO_INDEX["dish"]
            + case_3 * object_in_inv
            + else_case * object_in_inv
        )

        # Interactions with onion/plate piles and objects on counter
        # Pickup if: table, not empty, room in inv & object is not something unpickable (e.g. pot or goal)
        successful_pickup = (
            is_table * ~table_is_empty * inv_is_empty * jnp.logical_or(object_is_pile, object_is_pickable)
        )
        successful_drop = is_table * table_is_empty * ~inv_is_empty
        successful_delivery = is_table * object_is_goal * holding_dish
        no_effect = jnp.logical_and(jnp.logical_and(~successful_pickup, ~successful_drop), ~successful_delivery)

        # Update object on table
        new_object_on_table = (
            no_effect * object_on_table
            + successful_delivery * object_on_table
            + successful_pickup * object_is_pile * object_on_table
            + successful_pickup * object_is_pickable * OBJECT_TO_INDEX["wall"]
            + successful_drop * object_in_inv
        )

        # Update object in inventory
        new_object_in_inv = (
            no_effect * new_object_in_inv
            + successful_delivery * OBJECT_TO_INDEX["empty"]
            + successful_pickup * object_is_pickable * object_on_table
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["plate_pile"]) * OBJECT_TO_INDEX["plate"]
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["onion_pile"]) * OBJECT_TO_INDEX["onion"]
            + successful_drop * OBJECT_TO_INDEX["empty"]
        )

        inventory = new_object_in_inv

        # Apply changes to maze
        new_maze_object_on_table = (
            object_is_pot * OBJECT_INDEX_TO_VEC[new_object_on_table].at[-1].set(new_pot_status)
            + ~object_is_pot * ~object_is_agent * OBJECT_INDEX_TO_VEC[new_object_on_table]
            + object_is_agent * maze_object_on_table
        )

        maze_map = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0], :].set(new_maze_object_on_table)
        return maze_map, inventory, 0.0, 0.0, successful_delivery, successful_pickup, successful_drop
