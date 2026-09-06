# E1 Cloud Attribution Ablation

## Question

Does the ABI tensor help because it is correctly paired with the task outcome, or is the improvement just ordinary fine-tuning?

## Setup

The cloud backbone was Qwen3-8B on one RTX 4090. The prompt was neutral and did not reveal the answer. A sensor value selected one of four answer tokens. The external module wrote an orthogonal class-direction tensor into the tap-layer hidden state. The LoRA was applied to `q_proj` and `v_proj` from layer 16 upward.

Four arms used the same token budget and number of optimization steps:

- **Real**: correct sensor-tensor/target pairing.
- **Permuted**: sensor-tensor/target pairing destroyed, marginal distributions preserved.
- **Self-distill**: no tensor; trained on the backbone's own output.
- **Trajectory-only**: no tensor; output-level supervision only.

## Result

Held-out sensor-conditioned next-token accuracy over three seeds:

| Arm | Mean | Std | Runs |
|---|---:|---:|---|
| Real | **1.0000** | 0.0000 | 1.0000, 1.0000, 1.0000 |
| Permuted | 0.5365 | 0.2565 | 0.1875, 0.6250, 0.7969 |
| Self-distill | 0.0000 | 0.0000 | 0.0000, 0.0000, 0.0000 |
| Trajectory-only | 0.2292 | 0.0483 | 0.1875, 0.2969, 0.2031 |

Chance was 0.2500.

## Interpretation

The Real minus Permuted gap was **0.4635**, and the Real minus Trajectory-only gap was **0.7708**. This passes the primary E1 criterion: the paired ABI tensor carried task-relevant information beyond the marginal distributions and beyond output-level supervision.

The Permuted arm's high variance is expected: it sometimes learns the permutation residual, but has no stable causal pairing. The Self-distill arm's zero confirms that merely fine-tuning the backbone does not solve the sensor-only task.

This is an attribution experiment with a deliberately compressed class-coded tensor. It supports the E1 mechanism; it does not yet claim natural robot-scale data, cross-embodiment transfer, or long-horizon stability.
