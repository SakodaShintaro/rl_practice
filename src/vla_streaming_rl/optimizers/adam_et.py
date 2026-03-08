"""Adam optimizer with eligibility traces for streaming RL.

Combines the full Adam optimizer (first and second moment estimates)
with eligibility traces for TD(λ)-style credit assignment.
No overshooting bound is applied.
"""

import torch


class AdamET(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, gamma=0.99, et_lambda=0.8, beta1=0.9, beta2=0.999, eps=1e-8
    ):
        defaults = dict(lr=lr, gamma=gamma, et_lambda=et_lambda, beta1=beta1, beta2=beta2, eps=eps)
        super().__init__(params, defaults)

    def step(self, delta, reset=False):
        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]

            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["step"] = 0
                if p.grad is None:
                    continue

                state["step"] += 1
                t = state["step"]
                e = state["eligibility_trace"]
                m = state["m"]
                v = state["v"]

                # eligibility trace update: e = γλ * e + ∇(-v(s))
                e.mul_(group["gamma"] * group["et_lambda"]).add_(p.grad, alpha=1.0)

                # raw update direction
                update = delta * e

                # Adam first moment: m = β1 * m + (1 - β1) * update
                m.mul_(beta1).add_(update, alpha=1.0 - beta1)

                # Adam second moment: v = β2 * v + (1 - β2) * update²
                v.mul_(beta2).addcmul_(update, update, value=1.0 - beta2)

                # bias correction
                m_hat = m / (1.0 - beta1**t)
                v_hat = v / (1.0 - beta2**t)

                # parameter update: θ ← θ - lr * m̂ / (√v̂ + ε)
                p.data.addcdiv_(m_hat, (v_hat.sqrt() + eps), value=-group["lr"])

                if reset:
                    e.zero_()
