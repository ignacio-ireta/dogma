# Batch normalization forward (training and inference)

Implement `batch_norm_forward(x, gamma, beta, running_mean, running_var, *, training, momentum=0.1, eps=1e-5)`
for a batch of `N` rows with `C` features. In training mode, normalize each feature with the batch
statistics and update the running statistics. In inference mode, normalize with the stored running
statistics and leave them unchanged. Return the output and the two updated running statistics.

## Contract

- `x`: floating-point tensor, shape `(N, C)`, with `N >= 1` and `C >= 1`. Must not be modified.
- `gamma`, `beta`, `running_mean`, `running_var`: tensors of shape `(C,)`. Must not be modified.
- `training`: `bool`.
- `momentum`: float in `[0, 1]`. It is the weight on the **new** batch statistic.
- `eps`: float `> 0`, added to the variance inside the square root. PyTorch also rejects `eps <= 0` in training mode.
- Training mode, per feature over the `N` rows:
  - normalize with the **biased** variance (divide by `N`);
  - update `running_var` with the **unbiased** variance (divide by `N - 1`);
  - update `running_mean` and `running_var` with `new = (1 - momentum) * old + momentum * batch_statistic`.
- Inference mode: normalize with `running_mean` and `running_var`. The returned running statistics have the same values as the inputs.
- Returns: `(y, new_running_mean, new_running_var)`. `y` has shape `(N, C)` and the dtype of `x`. The running statistics have shape `(C,)` and the dtype of the input running statistics.
- Raises `ValueError` when: `x` is not 2-D or not floating-point; `N == 0`; any `(C,)` argument has a different length from `C`; `training` is true and `N < 2`; `momentum` is outside `[0, 1]`; `eps <= 0`.
- Must not mutate inputs. PyTorch's own batch norm updates running buffers in place. Your function must return new tensors instead.

## Examples

The examples write tensors as nested lists for brevity. The arguments are `torch` tensors, for example `torch.tensor([[1.0, 2.0], [3.0, 6.0]])`.

```python
x = [[1.0, 2.0], [3.0, 6.0]]  # N=2, C=2
gamma, beta = [1.0, 1.0], [0.0, 0.0]
running_mean, running_var = [0.0, 0.0], [1.0, 1.0]
batch_norm_forward(
    x, gamma, beta, running_mean, running_var, training=True, momentum=0.1, eps=1e-12
)
# y ~= [[-1.0, -1.0], [1.0, 1.0]]
# new running_mean = [0.2, 0.4]
# new running_var  = [1.1, 1.7]     (unbiased batch variance is [2.0, 8.0])
```

```python
x = [[1.0, 2.0], [3.0, 6.0]]
gamma, beta = [2.0, 1.0], [0.0, 1.0]
running_mean, running_var = [0.5, 1.0], [4.0, 9.0]
batch_norm_forward(x, gamma, beta, running_mean, running_var, training=False, eps=1e-12)
# y ~= [[0.5, 1.3333], [2.5, 2.6667]]
# new running_mean = [0.5, 1.0], new running_var = [4.0, 9.0]   (unchanged)
```

## Constraints

- Allowed: Python standard library and PyTorch tensor operations, such as `mean`, `var`, `sqrt`, `clone`.
- Not allowed: `torch.nn` (this covers `torch.nn.functional`, `nn` modules and `from torch import nn`), and any library routine that performs the normalization or the running update as a whole: `torch.batch_norm`, `torch.native_batch_norm`, `torch._native_batch_norm_legit`, `torch._native_batch_norm_legit_no_training`, `torch._batch_norm_impl_index`, `torch.batch_norm_stats`, `torch.batch_norm_elemt`, `torch.batch_norm_update_stats`, `torch.batch_norm_gather_stats`, `torch.batch_norm_gather_stats_with_counts`, `torch.cudnn_batch_norm`, `torch.miopen_batch_norm`, `torch.instance_norm`, `torch.group_norm`, and `torch.ops`. Write the normalization from its definition.

## Done looks like

The two public examples and the single-row training rejection pass. The hidden tests compare both modes against PyTorch's batch norm on random batches: outputs and running statistics, default and non-default momentum, several eps values, and batch shapes where rows and features differ. They also cover per-feature normalization, the unbiased running variance, eval mode leaving statistics unchanged, large common offsets, float32 dtype, invalid arguments, and no in-place mutation.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
