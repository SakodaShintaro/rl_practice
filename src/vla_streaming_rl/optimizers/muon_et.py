"""Muon optimizer with eligibility traces for streaming RL.

Combines Muon's Newton-Schulz orthogonalization with eligibility traces
for TD(λ)-style credit assignment.
For 2D+ parameters, updates are orthogonalized via Newton-Schulz iteration.
For 1D parameters (biases etc.), a plain scaled update is applied.
"""

import torch


def _zeropower_via_newtonschulz5(G, steps):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    original_dtype = G.dtype
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(dtype=original_dtype)


class MuonET(torch.optim.Optimizer):
    """Muon with eligibility traces for streaming RL.

    For each 2D+ parameter, the update direction is:
      1. Eligibility trace: e = γλ * e + ∇
      2. Effective gradient: g = delta * e
      3. Muon momentum: m = β * m + (1 - β) * g
      4. Nesterov blend: u = (1 - β) * g + β * m  (or u = m if nesterov=False)
      5. Newton-Schulz orthogonalization of u
      6. θ ← θ - lr * orth(u)

    1D parameters use a plain update without orthogonalization.
    """

    def __init__(
        self,
        params,
        lr,
        gamma,
        et_lambda,
        weight_decay=0.0,
        momentum=0.95,
        ns_steps=5,
        nesterov=True,
    ):
        defaults = dict(
            lr=lr,
            gamma=gamma,
            et_lambda=et_lambda,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    def step(self, delta, reset=False):
        for group in self.param_groups:
            beta = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                e = state["eligibility_trace"]
                m = state["momentum_buffer"]

                # eligibility trace update: e = γλ * e + ∇
                e.mul_(group["gamma"] * group["et_lambda"]).add_(p.grad, alpha=1.0)

                # effective gradient: g = delta * e
                g = delta * e

                # Muon momentum: m = β * m + (1 - β) * g
                m.lerp_(g, 1.0 - beta)

                # nesterov blend or plain momentum
                if group["nesterov"]:
                    update = g.lerp_(m, beta)
                else:
                    update = m.clone()

                p.data.mul_(1.0 - group["lr"] * group["weight_decay"])

                if update.ndim >= 2:
                    update_2d = update.view(update.size(0), -1)
                    update_orth = _zeropower_via_newtonschulz5(update_2d, steps=group["ns_steps"])
                    scale = max(1, update_orth.size(-2) / update_orth.size(-1)) ** 0.5
                    p.data.add_(update_orth.view(p.shape) * scale, alpha=-group["lr"])
                else:
                    p.data.add_(update, alpha=-group["lr"])

                if reset:
                    e.zero_()
