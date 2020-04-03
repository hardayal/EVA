# EVA Phase2_S9 Assignment
***

 > # Step 1: Class ReplayBuffer

Explanation : 

Init Function

1. Initialize the storage to store the transitions
2. Set the Replay Buffer size to 1e6 to store Replay Memory
3. set the pointer to the zeroth position of the replay memory buffer

Add Function

1. Add the new transition to the max size of the replay buffer if not full. iF full, the new transition to be add to the first position of the replay buffer

Sample Function

1. To return random set of transitions. Total number of transitions will be equal to the batch size.
2. Each transition to contain current state, Current action, Reward, Next state and done.
3. Done is 1 if the episode is complete. Done is 0 if the episode is not complete.

 

![My Image](https://github.com/hardayal/EVA/blob/master/Phase2_S9/images/Step1.png)


> # Step 2: Class Actor

Explanation : 

It takes state as an Input and prediction max action.
Max action is calculated using multiplying tanh function on predefined set of actions so to get a continouse action space.
The action network is defined by taking states as input. state includes all the physical parameters of a robot.
we are defning only one network for both actor model and actor target.

![My Image](https://github.com/hardayal/EVA/blob/master/Phase2_S9/images/Step2.png)

> # Step 3: Class Critic

Explanation : 

Critic class takes max action predicted by the actors and new state as the input to predict Q values.
State and action are concatenated in Pytorch and passed as inpt to the networks.
The weights of both the networks will be different.
Any one of the critics can be used for back progation of values

![My Image](https://github.com/hardayal/EVA/blob/master/Phase2_S9/images/Step3.png)

> # Step 4 and 5: 

T3D Class : 
Initialize actor and target models and targets. Initialize both actor to same weights and all critics to same weight

- Step4
Sample a batch of transitions according to batch size from replay buffer. All the transitions are taken by the actor model

- Step 5
Use the actor target to predict next action by passing next state as input from the replay buffer

![My Image](https://github.com/hardayal/EVA/blob/master/Phase2_S9/images/Step4andStep5.png) 

> # Step 6:

We add Gaussian noise to this next action a' and we clamp it in a range of values supported by the environment

> # Step 7 and 8:

Takes the next action and next state and calculate Q values.
here the next state and next action are passed on to two critic targets so that the we can take modest of the Q values

![My Image](https://github.com/hardayal/EVA/blob/master/Phase2_S9/images/Step7andStep8.png) 

> # Step 9:

In pytorch the computational graph target Q needs to keep going until an episode is over. Only when the episode is over the computation graph needs to be detached.
Qt = r + gamma * min(Qt1, Qt2) bold text

![My Image](https://github.com/hardayal/EVA/blob/master/Phase2_S9/images/Step9.png) 

> # Step 10:

Two critic models take (s, a) and return two Q-Vales

![My Image](https://github.com/hardayal/EVA/blob/master/Phase2_S9/images/Step10.png) 

> # Step 11:

Compute critic loss for both critic model 1 and critic model 2 using MSE error against Min(TargetQ1, Target Q2)

![My Image](https://github.com/hardayal/EVA/blob/master/Phase2_S9/images/Step11.png) 

> # Step 12:

Backpropagate this critic loss and update the parameters of two Critic models

![My Image](https://github.com/hardayal/EVA/blob/master/Phase2_S9/images/Step12.png) 

> # Step 13:

Once every two iterations, we update our Actor model by performing gradient ASCENT on the output of the first Critic model

![My Image](https://github.com/hardayal/EVA/blob/master/Phase2_S9/images/Step13.png) 

> # Step 14:

We update our Actor Target by Polyak Averaging once in every two iterations. This is to ensure there is a momentum maintained before updating weights for the actor target. There should be significant evidence before updating.

![My Image](https://github.com/hardayal/EVA/blob/master/Phase2_S9/images/Step14.png)

> # Step 15:

We update our Critic Targets by Polyak Averaging once in every two iterations. This is to ensure there is a momentum maintained before updating weights for the critic targets. There should be significant evidence before updating.

![My Image](https://github.com/hardayal/EVA/blob/master/Phase2_S9/images/Step15.png) 
