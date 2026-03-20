"""SGD optimizer with eligibility traces for streaming RL.

Combines SGD (with optional momentum) with eligibility traces
for TD(λ)-style credit assignment. No overshooting bound is applied.
"""

import torch


class SGDET(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, gamma=0.99, et_lambda=0.8, momentum=0.0):
        defaults = dict(lr=lr, gamma=gamma, et_lambda=et_lambda, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, delta, reset=False):
        for group in self.param_groups:
            momentum = group["momentum"]

            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(p.data)
                if p.grad is None:
                    continue

                e = state["eligibility_trace"]

                # eligibility trace update: e = γλ * e + ∇(-v(s))
                e.mul_(group["gamma"] * group["et_lambda"]).add_(p.grad, alpha=1.0)

                # raw update direction
                update = delta * e

                if momentum > 0:
                    buf = state["momentum_buffer"]
                    # momentum: buf = μ * buf + update
                    buf.mul_(momentum).add_(update)
                    # parameter update: θ ← θ - lr * buf
                    p.data.add_(buf, alpha=-group["lr"])
                else:
                    # parameter update: θ ← θ - lr * δ * e
                    p.data.add_(update, alpha=-group["lr"])

                if reset:
                    e.zero_()
