# https://jmtomczak.github.io/blog/18/18_fm.html
# MIT License

# Copyright (c) 2021 Jakub M. Tomczak

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# https://github.com/jmtomczak/intro_dgm/blob/main/LICENSE

import numpy as np
import torch
from torch import nn

# ruff: noqa: FBT003, N803, SIM108, SIM114, RET503


class FlowMatching(nn.Module):
    def __init__(self, sigma, D, T, prob_path="icfm"):
        super().__init__()

        self.vnet = nn.Linear(D, D)

        self.time_embedding = nn.Sequential(nn.Linear(1, D), nn.Tanh())

        # other params
        self.D = D

        self.T = T

        self.sigma = sigma

        assert prob_path in ["icfm", "fm"], (
            f"Error: The probability path could be either Independent CFM (icfm) or Lipman's Flow Matching (fm) but {prob_path} was provided."
        )
        self.prob_path = prob_path

        self.PI = torch.from_numpy(np.asarray(np.pi))

    def log_p_base(self, x, reduction="sum", dim=1):
        log_p = -0.5 * torch.log(2.0 * self.PI) - 0.5 * x**2.0
        if reduction == "mean":
            return torch.mean(log_p, dim)
        elif reduction == "sum":
            return torch.sum(log_p, dim)
        else:
            return log_p

    def sample_base(self, x_1):
        # Gaussian base distribution
        return torch.randn_like(x_1)

    def sample_p_t(self, x_0, x_1, t):
        if self.prob_path == "icfm":
            mu_t = (1.0 - t) * x_0 + t * x_1
            sigma_t = self.sigma
        elif self.prob_path == "fm":
            mu_t = t * x_1
            sigma_t = t * self.sigma - t + 1.0

        x = mu_t + sigma_t * torch.randn_like(x_1)

        return x

    def conditional_vector_field(self, x, x_0, x_1, t):
        if self.prob_path == "icfm":
            u_t = x_1 - x_0
        elif self.prob_path == "fm":
            u_t = (x_1 - (1.0 - self.sigma) * x) / (1.0 - (1.0 - self.sigma) * t)

        return u_t

    def forward(self, x_1, reduction="mean"):
        # =====Flow Matching
        # =====
        # z ~ q(z), e.g., q(z) = q(x_0) q(x_1), q(x_0) = base, q(x_1) = empirical
        # t ~ Uniform(0, 1)
        x_0 = self.sample_base(x_1)  # sample from the base distribution (e.g., Normal(0,I))
        t = torch.rand(size=(x_1.shape[0], 1))

        # =====
        # sample from p(x|z)
        x = self.sample_p_t(x_0, x_1, t)  # sample independent rv

        # =====
        # invert interpolation, i.e., calculate vector field v(x,t)
        t_embd = self.time_embedding(t)
        v = self.vnet(x + t_embd)

        # =====
        # conditional vector field
        u_t = self.conditional_vector_field(x, x_0, x_1, t)

        # =====LOSS: Flow Matching
        FM_loss = torch.pow(v - u_t, 2).mean(-1)

        # Final LOSS
        if reduction == "sum":
            loss = FM_loss.sum()
        else:
            loss = FM_loss.mean()

        return loss

    def sample(self, batch_size=64):
        # Euler method
        # sample x_0 first
        x_t = self.sample_base(torch.empty(batch_size, self.D))

        # then go step-by-step to x_1 (data)
        ts = torch.linspace(0.0, 1.0, self.T)
        delta_t = ts[1] - ts[0]

        for t in ts[1:]:
            t_embedding = self.time_embedding(torch.Tensor([t]))
            x_t = x_t + self.vnet(x_t + t_embedding) * delta_t

        x_final = torch.tanh(x_t)
        return x_final

    def log_prob(self, x_1, reduction="mean"):
        # backward Euler (see Appendix C in Lipman's paper)
        ts = torch.linspace(1.0, 0.0, self.T)
        delta_t = ts[1] - ts[0]

        for t in ts:
            if t == 1.0:
                x_t = x_1 * 1.0
                f_t = 0.0
            else:
                # Calculate phi_t
                t_embedding = self.time_embedding(torch.Tensor([t]))
                x_t = x_t - self.vnet(x_t + t_embedding) * delta_t

                # Calculate f_t
                # approximate the divergence using the Hutchinson trace estimator and the autograd
                self.vnet.eval()  # set the vector field net to evaluation

                x = torch.FloatTensor(
                    x_t.data
                )  # copy the original data (it doesn't require grads!)
                x.requires_grad = True

                e = torch.randn_like(x)  # epsilon ~ Normal(0, I)

                e_grad = torch.autograd.grad(self.vnet(x).sum(), x, create_graph=True)[0]
                e_grad_e = e_grad * e
                f_t = e_grad_e.view(x.shape[0], -1).sum(dim=1)

                self.vnet.eval()  # set the vector field net to train again

        log_p_1 = self.log_p_base(x_t, reduction="sum") - f_t

        if reduction == "mean":
            return log_p_1.mean()
        elif reduction == "sum":
            return log_p_1.sum()

    def sample_with_log_prob(self, batch_size=64):
        device = next(self.parameters()).device
        # Sample from base distribution at t=0 and compute its log probability
        x_t = self.sample_base(torch.empty(batch_size, self.D, device=device))
        logp = self.log_p_base(x_t, reduction="sum")  # (batch_size,) tensor

        ts = torch.linspace(0.0, 1.0, self.T, device=device, dtype=x_t.dtype)
        delta_t = ts[1] - ts[0]

        for t in ts[1:]:
            # t embedding
            t_emb = self.time_embedding(t.unsqueeze(0).unsqueeze(0))  # shape: (1, D)
            # vnet is evaluated at (x_t+t_emb)
            point = x_t + t_emb
            v = self.vnet(point)
            # Euler update
            x_t = x_t + v * delta_t

            # Approximate divergence using Hutchinson estimator (vnet evaluation point is same as point)
            point_detached = point.detach().clone().requires_grad_(True)
            v_temp = self.vnet(point_detached)
            e = torch.randn_like(point_detached)
            grad_v = torch.autograd.grad(v_temp.sum(), point_detached, create_graph=True)[0]
            divergence = (grad_v * e).view(batch_size, -1).sum(dim=1)
            # Density decays by -div(v) according to ODE
            logp = logp - divergence * delta_t

        # Scaling by tanh and its Jacobian correction
        x_final = torch.tanh(x_t)
        jacobian = torch.log(1 - x_final**2 + 1e-6)
        logp = logp - jacobian.sum(dim=1)

        return x_final, logp


if __name__ == "__main__":
    flow_matching = FlowMatching(sigma=0.1, D=256, T=20, prob_path="icfm")

    x, log_p = flow_matching.sample_with_log_prob(batch_size=64)
    print(x.shape, log_p.shape)

    optimizer = torch.optim.Adam(flow_matching.parameters(), lr=1e-3)

    for i in range(100):
        optimizer.zero_grad()
        x, log_p = flow_matching.sample_with_log_prob(batch_size=64)
        loss = -log_p.mean()
        loss.backward()
        optimizer.step()
        print(f"{i}\t{loss.item()}")
