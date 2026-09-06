# E1 Cloud Attribution Ablation

## Status

The first cloud run is now treated as **E1-v1 (lookup-channel audit)**. It established that the tap-layer tensor can carry class information, but it did **not** establish external computation. E1-v2 adds the decisive `sensor_permuted` and `raw_onehot` controls.

## E1-v1 result

The cloud model was Qwen3-8B on one RTX 4090. The prompt was neutral. A sensor value selected one of four answer tokens, and the module wrote a class-direction tensor at tap layer 16.

Module-active and module-removed accuracy were both logged:

| Arm | Module active | Module removed |
|---|---:|---:|
| Real | **1.0000** | **0.2396** |
| Permuted | 0.5365 | 0.2813 |
| Self-distill | 0.0000 | 0.0000 |
| Trajectory-only | 0.2292 | 0.2292 |

Chance was `0.2500`. Therefore, Real did **not** bypass the tensor: removing it dropped from `1.0000` to chance. The tensor channel is real.

However, the module was a lookup table:

```python
classes = torch.clamp((sensor_values * N_CLASSES).long(), max=N_CLASSES - 1)
delta = self.vector[classes]
```

So E1-v1 proves **transmission/encoding**, not computation or capability injection. The self-distill `0.0000` was also an artifact of training the model toward a token outside the four-answer set; that arm was removed.

The old `permuted` arm permuted labels while keeping the tensor truthful. That measured whether the model trusts tensor over conflicting labels. The decisive control is to permute the tensor input while keeping labels correct.

## E1-v2 controls

- `real`: correct tensor and correct labels.
- `sensor_permuted`: tensor is encoded from the wrong sensor value, labels remain correct.
- `raw_onehot`: a fixed one-hot projection at the same tensor norm, without the external module's class basis. If this also reaches Real-level accuracy, the module is a no-op in the information chain.
- `trajectory_only`: output-level labels only.

The default run now uses five seeds. The decision separately reports:

1. whether the correctly paired tensor beats the incorrectly paired tensor;
2. whether Real depends on the module, using `no_module_accuracy`;
3. whether Real beats the raw one-hot projection, which would be evidence of external computation rather than mere bandwidth.

Raw JSON is in `cloud/e1_cloud_results.json`; the executable is `cloud/cloud_host_e1.py`.
