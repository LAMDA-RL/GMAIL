import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from code.algos.policy_base import IRLPolicy
from code.algos.gail import GAIL, Discriminator as DiscriminatorGAIL


class Discriminator(DiscriminatorGAIL):
    def __init__(self, state_shape, out_features=1, units=(32, 32),
                 enable_sn=False, output_activation="relu", name="Discriminator"):
        nn.Module.__init__(self)
        in_features = state_shape[0]
        self.fc1 = nn.Linear(in_features=in_features, out_features=units[0], bias=True)
        self.fc2 = nn.Linear(in_features=units[0], out_features=units[1], bias=True)
        self.fc3 = nn.Linear(in_features=units[1], out_features=out_features, bias=True)
        if output_activation == "relu":
            self.net = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.fc3, nn.Sigmoid())
        elif output_activation == "tanh":
            self.net = nn.Sequential(self.fc1, nn.Tanh(), self.fc2, nn.Tanh(), self.fc3, nn.Sigmoid())
        # self.net = nn.Sequential(self.fc1, nn.ReLU(), self.fc3, nn.Sigmoid())

class GAIfO(GAIL):
    """
    Generative Adversarial Imitation from Observation (GAIfO) Agent: https://arxiv.org/abs/1807.06158

    Command Line Args:

        * ``--n-warmup`` (int): Number of warmup steps before training. The default is ``1e4``.
        * ``--batch-size`` (int): Batch size of training. The default is ``32``.
        * ``--gpu`` (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        * ``--memory-capacity`` (int): Replay Buffer size. The default is ``1e4``.
        * ``--enable-sn``: Enable Spectral Normalization
    """
    def __init__(
            self,
            state_shape,
            units=[32, 32],
            lr=0.001,
            lr_decay=1.,
            enable_sn=False,
            use_gp=False,
            H=1,
            beta=1.0,
            gp_weight=10,
            n_training=1,
            name="GAIfO",
            **kwargs):
        """
        Initialize GAIfO

        Args:
            state_shape (iterable of int):
            action_dim (int):
            units (iterable of int): The default is ``[32, 32]``
            lr (float): Learning rate. The default is ``0.001``
            enable_sn (bool): Whether enable Spectral Normalization. The defailt is ``False``
            name (str): The default is ``"GAIL"``
        """
        IRLPolicy.__init__(self, name=name, n_training=n_training, **kwargs)
        self.name = name
        self.disc = Discriminator(
            state_shape=state_shape, out_features=1, units=units, 
            output_activation="relu", enable_sn=enable_sn)
        self.optimizer = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1e5, gamma=lr_decay)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[3e5, 6e5, 9e5], gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
        self.H = H
        self.beta = beta
        self.gp_weight = gp_weight
        self.use_gp = use_gp
        
    def train(self, agent_states, agent_next_states,
              expert_states, expert_next_states, total_steps, writer=None, **kwargs):
        """
        Train GAIfO

        Args:
            agent_states
            agent_next_states
            expert_states
            expected_next_states
        """
        agent_states = torch.tensor(agent_states).type(torch.float32).to(self.device)
        agent_next_states = torch.tensor(agent_next_states).type(torch.float32).to(self.device)
        expert_states = torch.tensor(expert_states).type(torch.float32).to(self.device)
        expert_next_states = torch.tensor(expert_next_states).type(torch.float32).to(self.device)
        loss, accuracy, js_divergence = self._train_body(
            agent_states, agent_next_states, expert_states, expert_next_states)
        
        if writer is not None:
            writer.add_scalar(self.policy_name + "/DiscriminatorLoss", loss, total_steps)
            writer.add_scalar(self.policy_name + "/Accuracy", accuracy, total_steps)
            writer.add_scalar(self.policy_name + "/JSdivergence", js_divergence, total_steps)
            
        return dict(
            GAIfOLoss=loss.item(), 
            DiscriminatorAcc=accuracy.item(),
            JSDivergence=js_divergence.item()
        )

    def _train_body(self, agent_states, agent_next_states, expert_states, expert_next_states):
        epsilon = 1e-8
        fake_logits = self.disc.forward(agent_next_states)
        real_logits = self.disc.forward(expert_next_states)
        loss = -(torch.mean(torch.log(real_logits + epsilon)) +
                torch.mean(torch.log(1. - fake_logits + epsilon)))
        
        if self.use_gp:
            if len(expert_next_states.shape) == 3:
                expert_next_states = expert_next_states[:, 0, :]
            gradient_penalty = self._gradient_penalty(expert_next_states, agent_next_states)
            loss = loss + gradient_penalty
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        accuracy = (torch.mean((real_logits >= 0.5).type(torch.float32)) / 2. +
                    torch.mean((fake_logits < 0.5).type(torch.float32)) / 2.)
        js_divergence = self._compute_js_divergence(
            fake_logits, real_logits)
        return loss, accuracy, js_divergence
    
    def _gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data).to(self.device)
        interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)
        
        # Calculate probability of interpolated examples
        prob_interpolated = self.disc.forward(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
    
    def inference(self, states, actions, next_states):
        """
        Infer Reward with GAIfO

        Args:
            states
            actions
            next_states

        Returns:
            tf.Tensor: Reward
        """
        assert states.shape == next_states.shape
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
            next_states = np.expand_dims(next_states, axis=0)
        inputs = next_states
        return self._inference_body(inputs.astype(np.float32))
