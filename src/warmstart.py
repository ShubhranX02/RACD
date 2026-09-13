import torch
import numpy as np


def pretrain_policy_toward_retrieved(model, examples, epochs=200, lr=1e-3):
    """
    Behavior-cloning warm start: directly trains the PPO policy network's
    action outputs toward a set of (target_spec, retrieved_action) examples
    via supervised regression, BEFORE any RL training happens. This biases
    the network itself toward known-good answers from the first RL step
    onward -- model.policy is a real PyTorch nn.Module, so a standard
    supervised training loop against it is a legitimate, direct technique,
    not a workaround.
    """
    if not examples:
        return model  # cold start: nothing to pretrain toward

    obs_batch = torch.tensor(
        np.array([[ex['target_peaking_db']] for ex in examples]), dtype=torch.float32
    )
    action_batch = torch.tensor(
        np.array([ex['action'] for ex in examples]), dtype=torch.float32
    )

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    for epoch in range(epochs):
        distribution = model.policy.get_distribution(obs_batch)
        predicted_actions = distribution.distribution.mean
        loss = torch.nn.functional.mse_loss(predicted_actions, action_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
