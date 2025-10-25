# Q Learning Algorithm

## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.

## Q LEARNING ALGORITHM
→ Initialize Q-table and hyperparameters.<br>
→ Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.<br>
→ After training, derive the optimal policy from the Q-table.<br>
→ Implement the Monte Carlo method to estimate state values.<br>
→ Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.<br>

## Q LEARNING FUNCTION
#### Name: SABARI S
#### Register Number: 212222240085
```python
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilon = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        while not done:
          action = select_action(state, Q, epsilon[e])
          next_state, reward, done, _ = env.step(action)
          td_target = reward + gamma * Q[next_state].max() * (not done)
          td_error = td_target - Q[state][action]
          Q[state][action] = Q[state][action] + alphas[e] * td_error
          state = next_state

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```





## OUTPUT:
### Optimal State Value Functions:
<img width="343" height="234" alt="image" src="https://github.com/user-attachments/assets/1ec568fd-98d5-4d0a-8c02-58076f044bd0" />


### Optimal Action Value Functions:
<img width="768" height="540" alt="image" src="https://github.com/user-attachments/assets/cd43552d-3881-42ca-83d2-8b9263f3e600" />

<img width="1027" height="44" alt="image" src="https://github.com/user-attachments/assets/32c6612b-35b2-4f8a-a20f-d7a5118ca585" />


### State value functions of Monte Carlo method:
<img width="1436" height="653" alt="image" src="https://github.com/user-attachments/assets/d9ebf618-903a-47c3-880a-3b1ba1039a23" />



### State value functions of Qlearning method:
<img width="1450" height="655" alt="image" src="https://github.com/user-attachments/assets/e9a85490-b001-485f-a586-8b4a7127b4ae" />



## RESULT:
Thus, Q-Learning outperformed Monte Carlo in finding the optimal policy and state values for the RL problem.
