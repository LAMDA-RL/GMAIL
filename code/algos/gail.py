import numpy as np
from torch import nn
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from code.algos.policy_base import IRLPolicy


class Discriminator(nn.Module):
    def __init__(self, state_shape, action_dim, out_features=1, units=(32, 32),
                 enable_sn=False, output_activation="sigmoid", name="Discriminator"):
        super().__init__()
        in_features = state_shape[0] + action_dim
        self.fc1 = nn.Linear(in_features=in_features, out_features=units[0], bias=True)
        self.fc2 = nn.Linear(in_features=units[0], out_features=units[1], bias=True)
        self.fc3 = nn.Linear(in_features=units[1], out_features=out_features, bias=True)
        self.net = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.fc3, nn.Sigmoid())
        # self.net = nn.Sequential(self.fc1, nn.ReLU(), self.fc3, nn.Sigmoid())
        
    def forward(self, inputs):
        return self.net(inputs)
    
    def compute_reward(self, inputs):
        return torch.log(self.forward(inputs) + 1e-8)


class GAIL(IRLPolicy):
    """
    Generative Adversarial Imitation Learning (GAIL) Agent: https://arxiv.org/abs/1606.03476

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
            action_dim,
            units=[32, 32],
            lr=0.001,
            enable_sn=False,
            use_gp=False,
            gp_weight=10,
            name="GAIL",
            **kwargs):
        """
        Initialize GAIL

        Args:
            state_shape (iterable of int):
            action_dim (int):
            units (iterable of int): The default is ``[32, 32]``
            lr (float): Learning rate. The default is ``0.001``
            enable_sn (bool): Whether enable Spectral Normalization. The defailt is ``False``
            name (str): The default is ``"GAIL"``
        """
        super().__init__(name=name, n_training=1, **kwargs)
        self.name = name
        self.disc = Discriminator(
            state_shape=state_shape, action_dim=action_dim, out_features=1,
            units=units, enable_sn=enable_sn)
        self.optimizer = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(0.5, 0.999))
        self.gp_weight = gp_weight
        self.use_gp = use_gp
        
    def train(self, agent_states, agent_acts,
              expert_states, expert_acts, total_steps=0, writer=None, **kwargs):
        """
        Train GAIL

        Args:
            agent_states
            agent_acts
            expert_states
            expected_acts
        """
        agent_states = torch.tensor(agent_states).type(torch.float32).to(self.device)
        agent_acts = torch.tensor(agent_acts).type(torch.float32).to(self.device)
        expert_states = torch.tensor(expert_states).type(torch.float32).to(self.device)
        expert_acts = torch.tensor(expert_acts).type(torch.float32).to(self.device)
        loss, accuracy, js_divergence = self._train_body(
            agent_states, agent_acts, expert_states, expert_acts)
        
        if writer is not None:
            writer.add_scalar(self.policy_name + "/DiscriminatorLoss", loss, total_steps)
            writer.add_scalar(self.policy_name + "/Accuracy", accuracy, total_steps)
            writer.add_scalar(self.policy_name + "/JSdivergence", js_divergence, total_steps)
            
        return dict(
            GAILLoss=loss.item(), 
            DiscriminatorAcc=accuracy.item(),
            JSDivergence=js_divergence.item()
        )

    def _compute_js_divergence(self, fake_logits, real_logits):
        m = (fake_logits + real_logits) / 2.
        return torch.mean((
            fake_logits * torch.log(fake_logits / m + 1e-8) + real_logits * torch.log(real_logits / m + 1e-8)) / 2.)

    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts):
        epsilon = 1e-8
        real_logits = self.disc.forward(torch.cat((expert_states, expert_acts), axis=1))
        fake_logits = self.disc.forward(torch.cat((agent_states, agent_acts), axis=1))
        loss = -(torch.mean(torch.log(real_logits + epsilon)) +
                 torch.mean(torch.log(1. - fake_logits + epsilon)))
        
        if self.use_gp:
            real_data = torch.cat((expert_states, expert_acts), axis=1)
            fake_data = torch.cat((agent_states, agent_acts), axis=1)
            gradient_penalty = self._gradient_penalty(real_data, fake_data)
            loss = loss + gradient_penalty

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
        Infer Reward with GAIL

        Args:
            states
            actions
            next_states

        Returns:
            tf.Tensor: Reward
        """
        if states.ndim == actions.ndim == 1:
            states = np.expand_dims(states, axis=0)
            actions = np.expand_dims(actions, axis=0)
        inputs = np.concatenate((states, actions), axis=1)
        return self._inference_body(inputs.astype(np.float32))

    def _inference_body(self, inputs):
        inputs = torch.tensor(inputs).to(self.device)
        return self.disc.compute_reward(inputs)

    @staticmethod
    def get_argument(parser=None):
        parser = IRLPolicy.get_argument(parser)
        parser.add_argument('--enable-sn', action='store_true')
        return parser

