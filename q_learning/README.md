

```python
from snake import Snake as SnakeGame, KeyboardUser as SnakeKeyboardUser
from time import sleep
from IPython.display import clear_output
from graph_tools import get_unique_cycles, reward_per_turn
from functools import partial
import numpy as np
import pandas as pd
import random
```

# Q-Learning

Question number one: what does the "Q" stand for? Answer: "Quality". The premise is that we have an agent in an environment where the state is known and where the environment provides the agents with "rewards" for it's actions. The key idea behind Q-learning is to learn a mapping $Q(s_t, a_t)$ from states $s_t$ and actions $a_t$ to expected an reward, which is referred to as the quality of doing that action in that state. Given an event that gives the agent reward, we define an update rule for the $Q$ function:

$$
{\displaystyle Q(s_{t},a_{t})\leftarrow (1-\alpha )\cdot \underbrace {Q(s_{t},a_{t})} _{\rm {old~value}}+\underbrace {\alpha } _{\rm {learning~rate}}\cdot \overbrace {{\bigg (}\underbrace {r_{t}} _{\rm {reward}}+\underbrace {\gamma } _{\rm {discount~factor}}\cdot \underbrace {V^*(s_t)} _{\rm {estimate~of~optimal~future~value}}{\bigg )}} ^{\rm {learned~value}}}
$$


## Simple State Traversal

This "game" lays bare the fundamental nature of Q-learning. There are $n$ states $S = \{s_1, \dots, s_n\}$ that the agent can be in. For each state the agent can choose to transition to a select number of other states as defined. A reward function $R : T \rightarrow \mathbb{R}$ is defined for all legal transistions $T \subseteq S^2$.

For this game we will encode the Q-function as a matrix real $Q$, where $Q[i, j]$ represents the quality being in state $s_i$ and taking the action of transitioning to state $s_j$. In this example the states and transistions are described by the following graph:

<img src="./state_game.png" width="300px" />

The nodes represent five states; $A$, $B$, $C$, $D$ and $E$, and the edges represent the reward for transitioning between states. The agent will be given 100 transistions to maximise it's total reward. Notice how the structure of the game gives agents different options for generating reward, with a purposeful local maxima at state $C$.

Therefore we should expect agents with a low emphasis on future reward (i.e. low $\gamma$) to assign the highest quality on continually transitioning from $C$ to $C$. On the other hand, a more intelligent agent should notice that continually moving along the path $A\rightarrow E\rightarrow D\rightarrow B\rightarrow A\rightarrow\dots$ provides a net reward of 50 per turn (compared to 40 per turn).

### Optimal Play:

Given how simple this game is, we can brute force through all possible strategies to find the best one. To do this we use a depth-first search algorithm to find all the cycles in state space and then for each cycle we measure the expected reward gained per turn for constant traversal.


```python
graph = { 'A' : {'E': 120},
          'B' : {'A': -20, 'C': 0, 'D': -100},
          'C' : {'B': -20, 'C': 40, 'D': -100},
          'D' : {'B': 120, 'E': 0},
          'E' : {'D': -20} }

cycles = get_unique_cycles(graph)
rewards = map(partial(reward_per_turn, graph), cycles)
pd.DataFrame(list(zip(cycles, rewards)), 
             columns = ['Cycle', 'Reward Per Turn'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cycle</th>
      <th>Reward Per Turn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[A, E, D, B, A]</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[B, C, D, B]</td>
      <td>6.666667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[B, C, B]</td>
      <td>-10.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[B, D, B]</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[D, E, D]</td>
      <td>-10.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[C, C]</td>
      <td>40.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Immediate Reward Focused Q-Learning Agent:

The following code implements the game and agent. The agent maintains and updates the $Q$ matrix, which it uses to make decisions by mutiplying $Q$ values with values sampled from a uniform probability distribution and picking moving to the next possible state with the highest such value. Therefore, the expected choice of state transition is always the one with the highest $Q$ value, however, exploration is encouraged as there is a possibility to pick routes that are believed to be less optimal.

Here we refer to our agent as "immediate reward focused" to annotate the way we're defining the estimate of optimal future reward for this agent.

$$
V^*(s_t) = \max _{a}Q(s_{t+1},a)
$$

Which is to say; given the agents current understanding of how reward is given (encoded in $Q$), we will increase the quality of $s_t$ proportionally to the maximum immediate reward potential gained by the move that was just made.


```python
class StateTraversalGame:
    
    def _convert_graph_labels_to_indices(self, graph):
        state_to_idx = { s : i for i, s in enumerate(graph.keys()) }
        cvt_options = lambda options : { state_to_idx[end] : reward 
                                        for end, reward in options.items() }
        self.graph = { state_to_idx[start] : cvt_options(options)
                       for start, options in graph.items() }
        self.idx_to_state = dict(enumerate(graph.keys()))
    
    def __init__(self, graph):
        self._convert_graph_labels_to_indices(graph)
        self.states = list(self.graph.keys())
        self.num_states = len(self.states)
        self.state = random.choice(self.states)
        
    def get_possible_actions(self, from_state = None):
        return list(self.graph[from_state or self.state].keys())
        
    def reward_for_action(self, end_state):
        return self.graph[self.state][end_state]
    
    def go_to(self, next_state):
        self.state = next_state

class QLearningAgent:
    
    def __init__(self, 
                 game,
                 discount_factor = 0.5,
                 learning_rate = 0.1):
        
        self.game = game
        self.q_matrix = np.zeros((game.num_states, game.num_states))
        self.lr = learning_rate
        self.df = discount_factor
        self.lifetime_reward = 0
        
        self.record = True
        self.q_history = []
        self.reward_history = []
        self.state_history = [self.game.state]
        
    def update_q_matrix(self, next_state, reward):
        actions = self.game.get_possible_actions(from_state = next_state)
        future_value = self.q_matrix[next_state, actions].max()
        curr_q = self.q_matrix[self.game.state, next_state]
        self.q_matrix[self.game.state, next_state] = \
            (1 - self.lr) * curr_q + self.lr * (reward + self.df * future_value)
    
    def get_next_state(self):
        possible_actions = self.game.get_possible_actions()
        exploration_factor = np.random.rand(len(possible_actions))
        q_values = self.q_matrix[self.game.state, possible_actions]
        next_state = possible_actions[(q_values * exploration_factor).argmax()]
        return next_state
    
    def turn(self):
        next_state = self.get_next_state()
        reward = self.game.reward_for_action(next_state)
        self.update_q_matrix(next_state, reward)
        self.game.go_to(next_state)
        self.lifetime_reward += reward
        
        if self.record:
            self.reward_history += [reward]
            self.q_history += [self.q_matrix.copy()]
            self.state_history += [self.game.state]
    
    def play(self, num_turns = 100):
        for turn in range(num_turns):
            self.turn()
```

### Evaluation:

With this implementation we can set up a game and run an agent for 100 timesteps. The following code does that and we can see some relevant information about the game session. It's quite clear that the agent has fallen for the local-optima trap at C, which is hardly surprising given the agent's update rule for $Q$ never consider's the 


```python
game = StateTraversalGame(graph)
agent = QLearningAgent(game)
agent.play()
```


```python
keys = list(game.idx_to_state.values())
print('Agent\'s Path:', [game.idx_to_state[s] for s in agent.state_history])
print('Lifetime Reward:', agent.lifetime_reward)
print('Final Q Matrix:')
pd.DataFrame(agent.q_matrix, 
             columns = keys, 
             index = keys)
```

    Agent's Path: ['A', 'E', 'D', 'B', 'A', 'E', 'D', 'B', 'D', 'B', 'C', 'B', 'C', 'D', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
    Lifetime Reward: 3840
    Final Q Matrix:
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>D</th>
      <th>E</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>22.7</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-2.0</td>
      <td>0.000</td>
      <td>-8.860</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.0</td>
      <td>41.268</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.0</td>
      <td>0.000</td>
      <td>-3.200</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.0</td>
      <td>-2.000</td>
      <td>-8.374</td>
      <td>0.0</td>
      <td>78.977657</td>
    </tr>
  </tbody>
</table>
</div>



## Snake

Snake is a game where an agent has four controls that steer the head of a "snake" in four directions. The snake is composed of a set of continguous cells in two-dimensions. On each timestep the head cell moves in agent-defined direction and the last cell of the tail disapears unless the head cell touches a "berry". This is termed as eating the berry to increase the snake's length; the ultimate goal of the game is to make the snake as long as possible. The game ends when the head of the snake collides with the body. If the head of the snake comes up against the walls of the environment, it wraps around to the opposite edge.

<img src="./snake_screenshot.png", width=300px />


```python
snake = SnakeGame()
user = SnakeKeyboardUser()

try:
    while not snake.is_game_over():
        user.control(snake)
        snake.update()
        sleep(0.1)
        
        print('Score: %d' % snake.score())
        clear_output(wait = True)
except: pass
finally:
    user.close_game()
print('Game Over. Score: %d' % snake.score())
```

    Game Over. Score: 0
    
