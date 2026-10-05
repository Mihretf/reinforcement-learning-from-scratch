Professional/research side

Define an MDP formally.
Design the state and action spaces.
Define the reward function.
Implement the RL algorithm from scratch.
Study exploration vs. exploitation.
Measure learning across episodes.
Run controlled experiments.
Analyze failure cases and limitations.



# Visual/game side
Actually watch the agent interact with the environment.
See random exploration at the beginning.
Watch its behavior improve.
Visualize the learned policy.
Make the final result understandable immediately.

For example, the project could be framed as:

Learning to Navigate: A Reinforcement Learning Agent in a Grid-World Environment


# Overview

Not necessarily. **For Project 4, we don't need a dataset like Projects 1–3.** That's actually one of the interesting differences with reinforcement learning.

Our data is **generated through interaction with the environment**.

### Project 4 steps

1. **Choose the environment**

   * We'll use a small **grid-world game**.
   * Agent moves around a grid and tries to reach a goal while avoiding obstacles.

2. **Define the RL problem**

   * **States:** where the agent is
   * **Actions:** up/down/left/right
   * **Rewards:** goal, movement, obstacles, etc.
   * **Episode:** one attempt from start → goal/failure

3. **Build the environment from scratch**

   * No RL library.
   * Just Python + NumPy, with a simple visualization.

4. **Create a baseline**

   * Random agent.
   * This tells us what happens without learning.

5. **Implement Q-learning from scratch**

   * The agent learns a Q-table from its experience.

6. **Train the agent**

   * Run many episodes:
     `state → action → reward → next state → update Q`

7. **Evaluate learning**

   * Success rate
   * Steps needed to reach the goal
   * Episode reward
   * Learning curve

8. **Run actual experiments**

   * Exploration (`epsilon`)
   * Reward design
   * Environment difficulty
   * Possibly unseen obstacle layouts

9. **Visualize the learned behavior**

   * Watch the agent go from random wandering → learning → reliably navigating.

10. **Research conclusion**

* What did the agent actually learn?
* What affected learning?
* Where did it fail?
* Did it generalize?

So the overall progression is:

**Project 1:** dataset → learn prediction
**Project 2:** dataset → discover structure
**Project 3:** dataset → create learning signal
**Project 4:** **environment → interact → receive rewards → learn behavior**

And honestly, **not choosing a dataset is part of what makes Project 4 different.**

For now, our **Step 1 is choosing/designing the grid-world environment**.



-----

Can an agent learn an effective navigation policy through trial-and-error using rewards?


--- terminal outputs
Barney@MERCY MINGW64 ~/AI-projects/project-4-reinforcement-learning
$ [200~python -m scripts.random_agent~^C

Barney@MERCY MINGW64 ~/AI-projects/project-4-reinforcement-learning
$ python -m scripts.random_agent
Random Agent Baseline
---------------------
Episodes: 1000
Success rate: 10.00%
Average reward: -134.24
Average steps: 97.27

Barney@MERCY MINGW64 ~/AI-projects/project-4-reinforcement-learning
$ python -m scripts.train_agent
Episode 100/1000 | Reward: -13.0 | Steps: 29 | Success rate: 53.00% | Epsilon: 0.606
Episode 200/1000 | Reward: -7.0 | Steps: 24 | Success rate: 100.00% | Epsilon: 0.367
Episode 300/1000 | Reward: -11.0 | Steps: 29 | Success rate: 100.00% | Epsilon: 0.222
Episode 400/1000 | Reward: 5.0 | Steps: 16 | Success rate:100.00% | Epsilon: 0.135
Episode 500/1000 | Reward: 7.0 | Steps: 14 | Success rate:100.00% | Epsilon: 0.082
Episode 600/1000 | Reward: 7.0 | Steps: 14 | Success rate:100.00% | Epsilon: 0.050
Episode 700/1000 | Reward: 7.0 | Steps: 14 | Success rate:100.00% | Epsilon: 0.050
Episode 800/1000 | Reward: 7.0 | Steps: 14 | Success rate:100.00% | Epsilon: 0.050
Episode 900/1000 | Reward: 7.0 | Steps: 14 | Success rate:100.00% | Epsilon: 0.050
Episode 1000/1000 | Reward: 7.0 | Steps: 14 | Success rate: 100.00% | Epsilon: 0.050

Q-table saved to experiments/q_table.npy
Reward plot saved to plots/q_learning_rewards.png
Success-rate plot saved to plots/q_learning_success_rate.png

Barney@MERCY MINGW64 ~/AI-projects/project-4-reinforcement-learning
$ python -m scripts.exploration_experiment

Running: High Exploration
Initial epsilon: 1.0
Final success rate: 100.00%
Final average reward: 6.36
Final average steps: 14.47

Running: Moderate Exploration
Initial epsilon: 0.5
Final success rate: 100.00%
Final average reward: 6.02
Final average steps: 14.75

Running: Low Exploration
Initial epsilon: 0.1
Final success rate: 100.00%
Final average reward: 5.68
Final average steps: 15.09

Experiment complete.
Saved:
plots/exploration_success_rate.png
plots/exploration_reward.png
plots/exploration_steps.png

Barney@MERCY MINGW64 ~/AI-projects/project-4-reinforcement-learning
$ 


----

Barney@MERCY MINGW64 ~/AI-projects/project-4-reinforcement-learning
$ python -m scripts.exploration_experiment

Running: High Exploration
Initial epsilon: 1.0
Final success rate: 100.00%
Final average reward: 6.02
Final average steps: 14.77

Running: Moderate Exploration
Initial epsilon: 0.5
Final success rate: 100.00%
Final average reward: 5.82
Final average steps: 14.87

Running: Low Exploration
Initial epsilon: 0.1
Final success rate: 100.00%
Final average reward: 5.92
Final average steps: 14.93

Raw experiment results saved to experiments/exploration_results.pkl

Experiment complete.
Saved:
  experiments/exploration_results.pkl
  plots/exploration_success_rate.png
  plots/exploration_reward.png
  plots/exploration_steps.png

Barney@MERCY MINGW64 ~/AI-projects/project-4-reinforcement-learning
$ python -m scripts.analyze_exploration
EXPLORATION CONVERGENCE ANALYSIS
============================================================
Definition of convergence: 90% success over a 100-episode rolling window.

High Exploration
Reached 90% success at episode 140

Moderate Exploration
Reached 90% success at episode 108

Low Exploration
Reached 90% success at episode 104

============================================================
Analysis complete.

Barney@MERCY MINGW64 ~/AI-projects/project-4-reinforcement-learning
$ 

What the experiment actually shows
Strategy	90% success reached	Final success	Final avg. reward	Final avg. steps
Low exploration	104	100%	5.92	14.93
Moderate exploration	108	100%	5.82	14.87
High exploration	140	100%	6.02	14.77
The interesting part

Low exploration learned fastest.

It reached our predefined 90%-success threshold at episode 104, compared with:

Moderate → 108
High → 140

So higher exploration slowed down early learning in this particular environment.

But look at the final performance: high exploration actually finished slightly more efficiently:

High: 14.77 steps
Moderate: 14.87
Low: 14.93

That's a much more interesting result than simply saying "less exploration is better."

Our conclusion should therefore be:

Lower initial exploration enabled faster convergence in the GridWorld environment, while higher exploration produced slightly more efficient final policies.

And importantly, we should phrase it as specific to this environment and experimental setup, not as a universal RL rule.

# STEP 1
So the main idea we were trying to learn was can an agent learn an effective naviagation policy through trail-and error using rewards? 
So when it gets a correct step we give it a higher reward, if it gets a wrong one, we give it a less reward or sometimes penalize them as well. 


So, in terms of seeing this project we decidied to do a game. This thing was chosen because this GridWorld gives us things that are super useful to our project: 

1. We can see where our agent is, what action it can take, what the obstacles are, what the goal is, what rewards it recieves, what it learns and finally what path it finally choose to get to the goal. It is really important for seeing or visualizing the things. 

We can literally watch it try things, get rewards, update the q values, trys better results, learns a route and reach the goal

The game is the enviroment we have developed to see what happens as the model learns. 



---- So the grid we created:
There is a point at which the agent will start then there will be a goal that it will need to go to and acheive, now there are obstacles between them. The agent can move, up, dowm, left and rgiht. The job is simply reach the goal while accumulating as much reward as possible. 


 # STEP 2

What makes this reinforcement learning?
- We don't tell the agent the correct path. The agent does not know which actions are good so it has to discover these things through experience. 

So while like taking steps if it the correct one, the reward will be given more and it knows that that is the correct path, if it not correct it gets less reward and hence it knows that it is taking the incorrect path. So it will learn the idea that some actions form some states tend to lead to better outcomes. 


-- In this specifc project we had a reward function such as this:
- Normal movement we get -1 rewards
- Hit wall/ obstacle we get -2 rewards
- Reach goal  we get +20 


Why are we punishing for the simple movement, because we don't want to just reach the goal, we want to reach the goal efficiently. So the agent will prefer shorter routes. 

We made the obstacles -2 because hitting the obstalces is worse than normal movements. 


Environment	The 8×8 grid with obstacles and the goal
Agent	The learner navigating the grid
State	The agent's current position, e.g. (2, 4)
Action	UP, DOWN, LEFT, RIGHT
Reward	-1 for normal movement, -2 for hitting an obstacle/wall, +20 for reaching the goal
Policy	The agent's strategy: given this state, which action should I take?
Episode	One attempt from the start until reaching the goal or hitting the step limit
Value	How good it is to be in a particular state

----

# STEP 3 Q-Learning, how the agent actually learns
- The bog question is how does the agent go from knowing nothing to knowing which path to take> This where q learning comes

- The basic idea for every state there is an action combination to it, that number is what we call the q-value. It sort of tells us how good is it to take this action when I'm in this state?

so basically what happens is on the q table it is going to be filled with a memory of experiences means from state 0, (0,0) going to the right might have 5.9, left 0.2 up 0.2 down 3.1 we will have an entire table filled like this 

State       UP     DOWN    LEFT    RIGHT
------------------------------------------------
(0,0)       0.2     3.1     0.2     5.8
(0,1)       0.4     2.9     0.7     6.3
(0,2)       0.3     3.5     0.5     6.9


So now we can emerge policies from the Q-table from this, so taking the state 0,0 we know that going to the right has the hightest value, so I'll go to the RIGHT. 

This is the update we implement for q-learning update 
Q(s,a)←Q(s,a)+α[target−Q(s,a)]

- The new estimate = old estimate + small correction. And the correction is based on target - old estimate. 

So, the agent basically keeps a little memory of what it has learned. So for every position on the grid it keeps track how useful each possible action seems to be. This is what our q-table represents. It is basically a collection of experience about which actions seem useful from which positions. 

But we have a problem. Imagine the agent has only tried going right from a particular position. It might think, "Right is good!" But maybe it has never tried going down. How would it know whether dowm might actually be better? This is why we have exploration. Sometimes our agent deliberatley tries something random instead of doing what it currently thinks is best. It might migth make a stupid move, hit an obstacle, and learn from that, or it might accidentally discover a much better route. 

Starting at the beginning, we make the agent explore a lot, because it knows nothing. As training continues however, it gradually explores less and starts relying more on what it has learned. So it will say, "okay, I've figured this place out. I'll mostly follow what I know works."

# STEP 4

What happened during training?
 So we are training over 1000 attempsts. So the first attempts, our agent knows basically nothing, so it wanders around a lot. So the early success rate was only around 53%.

 After some attempts, the agent is starting to figure things out. Then for this particular project around episode 200, the success rate becomes 100%. This does not mean the agent became perfect at everything, it just means tht under this particular setup, it had learned a route to that was reliable enough to reach the goal in every recent episode being measured. 

# GRAPH MEANING
1. Reward graph: this basically answers how good was this attempt? Is the agent actually getting better with experience? (How good was your behavior)

2. Success-rate graph: this asks how often is the agent actually reaching the goal, So when our graph climbs toward 1.0 the agent becomes more relibale. (Can you actually solve the task)

3. Steps graph: this means how many moves does the agent need before the episode ends. It will decrease in number when it has found a more direct route. (How efficiently did you solve it?)

# Step 5
Our experiment exploration vs exploitation
- Once we had confirmed that the agent could learn the grid, we had another question:

How does the amount of exploration affect how the agent learns?

Exploration means the agent sometimes tries random actions instead of following what it currently thinks is best. 
Exploitation means it trusts what it has already learned. 

## Section 5 — Our Experiment: Exploration vs. Exploitation

Once we had confirmed that the agent could learn the grid, we had another question:

> **How does the amount of exploration affect how the agent learns?**

Remember, exploration means the agent sometimes tries random actions instead of following what it currently thinks is best. Exploitation means it trusts what it has already learned.

This creates a natural tension. If the agent explores too much, it keeps trying things instead of using its knowledge. But if it explores too little, it might settle on a mediocre strategy before discovering a better one.

So we created three versions of the same Q-learning experiment. The only thing we changed was **how much exploration the agent started with**. Everything else stayed the same: same grid, same obstacles, same rewards, same learning algorithm, same training length, and the same exploration-decay rule.

That is important because if we changed several things at once, we wouldn't know what caused any difference we observed.

### What happened?

All three agents eventually learned to solve the environment reliably.

But they got there at slightly different speeds.

The low-exploration agent reached our chosen **90% success threshold** earlier than the moderate and high-exploration agents. That makes sense: because it spent less time randomly experimenting, it started taking advantage of useful things it had already discovered sooner.

The high-exploration agent took longer to settle into reliable behavior. It was still spending more time trying different things.

However, by the end of training, the high-exploration agent had slightly better average efficiency in our particular experiment. So there isn't a simple story of **“less exploration is always better.”**

The actual lesson is more interesting:

> **Exploration affects the way an agent learns, and the best balance depends on the environment and the learning problem.**

For our small, fixed GridWorld, the lower-exploration agent converged faster, while the higher-exploration agent ended with slightly more efficient behavior.

And that's exactly why we ran the experiment instead of just assuming what would happen.

---

# Section 6 — What Are the Limitations?

Now we need to be honest about what our project actually demonstrates.

Our agent learned our **specific GridWorld** very well. But that doesn't mean it has learned navigation in general.

The biggest limitation is that the environment is fixed. The obstacles always appear in the same places, the start is always the same, and the goal is always the same.

So after enough training, the agent has become very familiar with this particular problem.

If we suddenly changed the obstacles, the agent wouldn't necessarily know what to do. Its Q-table was built from experience in the original environment.

That's also why your earlier idea of creating **new grids with different obstacle layouts** is interesting. That would test something different:

> **Can the agent use what it learned to handle a new environment?**

That would be a generalization experiment.

We haven't tested that yet, so we shouldn't claim that our current agent can generalize.

---

### Another limitation: the Q-table

Our Q-table works beautifully because our environment is tiny.

We only have a small number of possible positions and actions.

Imagine we changed the problem to a huge map, or a game where the state contains an image, or hundreds of variables.

We couldn't realistically make a simple table containing every possible state and action.

That's one of the reasons more advanced reinforcement-learning methods use **neural networks** instead of storing everything in a table.

But that's not a weakness of the project itself. We deliberately chose a small environment because our goal was to understand the **core mechanics of RL and Q-learning from scratch**.

---

### And finally, our experiment is only one environment

Our exploration experiment tells us what happened **in our GridWorld under our particular settings**.

It doesn't tell us that:

> “Low exploration is always better.”

or:

> “High exploration is always better.”

It tells us what happened when we changed exploration **in this particular problem**.

That's an important research habit:

> **Don't generalize beyond what your experiment actually tested.**

---

## So where are we now?

At this point, you understand the whole chain:

**We created a problem**
→ an agent has to navigate a grid.

**We defined the MDP**
→ states, actions, transitions, rewards, goal.

**We gave the agent a way to learn**
→ Q-learning.

**We let it interact repeatedly**
→ 1,000 training episodes.

**We watched whether it improved**
→ reward, success, steps.

**We compared it against a random agent**
→ to show what learning actually added.

**We changed exploration**
→ to investigate how exploration affects learning.

**We considered the limitations**
→ the agent learned this environment, but we haven't established generalization to new environments.

That's essentially the **entire intellectual structure of the project**.

The final thing I'd have you learn is **Section 7: putting the whole project into your own words** — basically, *“If someone asks me what I built, what exactly do I say?”* That's the last understanding section before documentation.
