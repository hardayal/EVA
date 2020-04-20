# EVA Phase2_S10 Assignment
***

# Twin Delayed DDPG (TD3) Implementation on Custom Environment

### Created whole environment using GYM templets


<a rel="Link to the video">![Car Movement](https://github.com/hardayal/EVA/blob/master/Phase2_S10/images/car_movement.jpg)</a>


## Implementation steps:

> ### Step 1: Defining Replay memory:

![alt ALGO](https://i.imgur.com/l6IoD3h.png)

> ### Step 2: Defining the Model architecture:

![alt actor critic](https://i.imgur.com/TI8naMe.png)

We define a function to update actor loss named as Q1. So the entire architecture looks like:

![alt full_arch](https://i.imgur.com/40dicZM.png)


> ## Training Process

![alt training](https://i.imgur.com/XchSvHL.png)

> ### Step 4: Sampling Transitions

![alt traning1](https://i.imgur.com/Nd5IdSl.png)


> ### Step 5: Actor Target Predicts Next Action 

![alt traning2](https://i.imgur.com/YN9fWkf.png)

> ### Step 6: Noise regularization on the predicted next action a'


```python
        noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
        noise = noise.clamp(-noise_clip, noice_clip)
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
```



> ### Step 7: Q Value Estimation by Critic Targets

Predict Q values from both Critic target and take the min value

```python
          target_Q1, target_Q2 = self.critic_target.forward(next_state, next_action)

          #Keep the minimum of these two Q-Values
          target_Q = torch.min(target_Q1, target_Q2)
```

> ### Step 8: Target value Computation

We use the target_Q computed in last code block


![alt training 4](https://i.imgur.com/1D9SRsQ.png)



```python
		  target_Q = reward + ((1-done) * discount * target_Q).detach()
```


> ### Step 9: Q value Estimation by Critic Models

Two critic models take (s, a) and return two Q-Values

![Alt training5](https://i.imgur.com/oa129cc.png)

```python
		 current_Q1, current_Q2 = self.critic.forward(state, action)
```


> ### Step 10: Compute the Critic loss 

We compute the critic loss using the Q-values returned from the Critic model networks.

![alt training6](https://i.imgur.com/hmhAElA.png)

```python
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
```



> ### Step 11: Update Critic Models

Backpropagate using Critic Loss and update the parameters of two Critic models.

![alt backprop](https://i.imgur.com/MtNQqjV.png)



```python
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
```



> ### Step 12: Update Actor Model

![alt training8](https://i.imgur.com/KV9YnPx.png)


```python
      if it % policy_freq == 0:
        actor_loss = -(self.critic.Q1(state, self.actor(state)).mean())
        self.actor_optimizer.grad_zero()
        actor_loss.backward()
        self.actor_optimizer.step()
```

> ### Step 13: Update Actor Target

This way our target comes closer to our model. 

![alt training9](https://i.imgur.com/akToYxM.png)

```python
        for param, target_param in zip(self.actor.parameters(), \
                                       self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

```





> ### Step 14: Update Critic Target 

We soft update our critic target network along with our Actor Target using Polyak averaging.

![alt training final](https://i.imgur.com/fvX9eZK.png)

```python
        for param, target_param in zip(self.critic.parameters(),\
                                       self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
```

