# Dungeons and Dragons RL Bot

Welcome to dungeons and data. This repository contains implementation of the following RL Algorithms:

* Tabular Q-Learning
* Q-Learning (Linear function approximation)
* Deep Q-Learning
* Double Deep Q-Learning
* Dueling Double Deep Q-Learning
* Proximal Policy Optimization

## Dungeons and Dragons Primer:

This section provides a brief overview of Dungeons & Dragons rules and mechanics:

* In D&D, all characters have a certain amount of `hit_points` (health)
* The goal of each agent (i.e character) is to reduce the `hit_points` of all enemies to zero
* The game takes place in rounds in which each character gets a turn within these rounds
* Within a character's turn, they may take the following actions:
    * Attack: Attacks have two distinct steps:
        1. The character rolls a 20 sided `hit_dice` in order to see if they successfully hit their opponent. 
        2. If the roll is above their enemy's `armor_class`, a `damage_dice` is rolled and the amount is subtracted from their enemies total `hit_points` (Depending on the attack, the `damage_dice` can be a 6 sided dice, 12 sided dice, etc.)
        3. Generally, each character may only attack once per turn
    * Move: A character can move up to a certain amount, usually 30 ft, during their turn. 
* Once all characters have finished their turn within a round, all movement and used attacks are refunded to all characters

## Scenario description
### Environment
    * Currently, the agents operate in a [50ft x 50ft room]
#### Characters:
    * Leotris:
        * `hit_points`: 25
        * `armor_class`: 16
        * `shoot_arrow` attack: 
            * 60 ft range
            * `hit_bonus`: +5
            * `damage_dice`: 1d12
            * `damage_bonus`: +3
     * Strahd:
        * `hit_points`: 200
        * `armor_class`: 16
        * `vampire_bite` attack:
            * 5 ft range
            * `hit_bonus`: +10
            * `damage_dice`: 3d12
            * `damage_bonus`: +10

### Combat Handler (This documentation is a work in progress):

1. Combat is [initialized]()
2. A [round is executed]()
    1. While an [EndTurn]() action has not been sampled from the creature:
        1. The current state is observed (`current_state`) and saved
        2. An [action is sampled]() from the creature based off of it's strategy and saved (ex: [PPO]())
        3. An [action is used]()
        4. The resulting state is observed (`next_state`) and saved
        5. A [reward is determined]() and saved
3. Once a round has finished execution, all characters [update their strategies]() from saved states, actions, and rewards
4. If combat is not over, return to step 2
5. Once combat is over and a winner is determined, allow all creatures to updated based off of the entire combat sequence (trajectory that has reached a terminal state)

### Results:
#### Random Actions:
* Both agents taking random actions, this is the win % of `Leotris`

#### PPO:
* With `Strahd` taking random actions and `Leotris` following a proximal policy optimization strategy, this is the win % of `Leotris`:
![PPO results]()