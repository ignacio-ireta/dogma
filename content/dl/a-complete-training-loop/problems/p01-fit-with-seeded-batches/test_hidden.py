from __future__ import annotations

import copy
import math

import pytest
import torch
from submission import fit

ATOL = 1e-9  # matches meta.yaml tolerance
RTOL = 1e-9  # matches meta.yaml tolerance


def checked_fit(model, X, y, Xv, yv, **kw):
    try:
        result = fit(model, X, y, Xv, yv, **kw)
    except ValueError:
        raise AssertionError(
            "valid training run raised ValueError: check the argument rules"
        ) from None
    except Exception as exc:
        raise AssertionError(
            f"valid training run raised {type(exc).__name__}: honor the documented contract"
        ) from None
    if not (isinstance(result, tuple) and len(result) == 2):
        raise AssertionError("return (train_losses, val_losses)")
    train, val = result
    if not (isinstance(train, list) and isinstance(val, list)):
        raise AssertionError("return two lists, not arrays or tensors")
    if len(train) != kw["epochs"] or len(val) != kw["epochs"]:
        raise AssertionError("return one loss per epoch: list length must equal epochs")
    return train, val


def mlp(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(2, 5), torch.nn.Tanh(), torch.nn.Linear(5, 1)
    ).double()


def dataset(seed: int, n: int, m: int):
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 2, generator=gen, dtype=torch.float64)
    y = torch.sin(X[:, :1]) + 0.3 * X[:, 1:]
    Xv = torch.randn(m, 2, generator=gen, dtype=torch.float64)
    yv = torch.sin(Xv[:, :1]) + 0.3 * Xv[:, 1:]
    return X, y, Xv, yv


def reference_fit(model, X, y, Xv, yv, *, epochs, batch_size, lr, seed):
    """Independent loop: hand-written SGD and the same shuffle, no optimizer and no loaders."""
    model = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    train_hist, val_hist = [], []
    best_val, best_state = math.inf, None
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        model.train()
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            loss = ((model(X[idx]) - y[idx]) ** 2).mean()
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p.sub_(lr * p.grad)
            total += loss.item() * len(idx)
        train_hist.append(total / n)
        model.eval()
        with torch.no_grad():
            v = ((model(Xv) - yv) ** 2).mean().item()
        val_hist.append(v)
        if v < best_val:
            best_val, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return train_hist, val_hist, model


def close_lists(got, want, message):
    if len(got) != len(want) or not all(
        abs(a - b) <= ATOL + RTOL * abs(b) for a, b in zip(got, want, strict=True)
    ):
        raise AssertionError(message)


def test_losses_and_final_parameters_match_independent_reference_with_partial_last_batch():
    X, y, Xv, yv = dataset(0, n=10, m=7)
    base = mlp(1)
    kw = {"epochs": 4, "batch_size": 4, "lr": 0.05, "seed": 3}
    want_train, want_val, want_model = reference_fit(base, X, y, Xv, yv, **kw)
    got_model = copy.deepcopy(base)
    got_train, got_val = checked_fit(got_model, X, y, Xv, yv, **kw)
    close_lists(
        got_train,
        want_train,
        "training losses differ from the reference loop: check the "
        "batch weighting, the shuffle and the order of zero/backward/step",
    )
    close_lists(
        got_val,
        want_val,
        "validation losses differ from the reference: check eval mode "
        "and the timing of validation within the epoch",
    )
    for got, want in zip(got_model.parameters(), want_model.parameters(), strict=True):
        assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), (
            "final parameters differ from the best-epoch reference: restore the best state"
        )


def test_last_partial_batch_is_weighted_by_its_size_not_counted_as_a_full_batch():
    X, y, Xv, yv = dataset(1, n=5, m=3)
    base = mlp(2)
    # lr is tiny but positive (0 is invalid): the parameters barely move, so only weighting matters.
    kw = {"epochs": 1, "batch_size": 4, "lr": 1e-300, "seed": 4}
    train, _ = checked_fit(copy.deepcopy(base), X, y, Xv, yv, **kw)
    with torch.no_grad():
        full = ((base(X) - y) ** 2).mean().item()
    assert abs(train[0] - full) <= ATOL + RTOL * abs(full), (
        "with lr=0 the epoch loss must equal the loss over all examples, with the partial batch "
        "weighted by its size"
    )


def test_same_seed_reproduces_the_run_bit_for_bit():
    X, y, Xv, yv = dataset(2, n=9, m=5)
    first = checked_fit(mlp(3), X, y, Xv, yv, epochs=3, batch_size=2, lr=0.1, seed=11)
    second = checked_fit(mlp(3), X, y, Xv, yv, epochs=3, batch_size=2, lr=0.1, seed=11)
    assert first == second, "the same seed and initial parameters must give identical losses"


def test_shuffle_ignores_the_global_random_state():
    X, y, Xv, yv = dataset(3, n=8, m=4)
    torch.manual_seed(100)
    first = checked_fit(mlp(4), X, y, Xv, yv, epochs=2, batch_size=3, lr=0.1, seed=7)
    torch.manual_seed(999)
    torch.rand(50)  # disturb the global stream before the second run
    second = checked_fit(mlp(4), X, y, Xv, yv, epochs=2, batch_size=3, lr=0.1, seed=7)
    assert first == second, (
        "the shuffle must come from the seeded generator, not from the global random state"
    )


def test_different_seeds_give_different_shuffle_orders():
    X, y, Xv, yv = dataset(4, n=12, m=4)
    a = checked_fit(mlp(5), X, y, Xv, yv, epochs=3, batch_size=3, lr=0.2, seed=0)
    b = checked_fit(mlp(5), X, y, Xv, yv, epochs=3, batch_size=3, lr=0.2, seed=1)
    assert a[0] != b[0], "different seeds should change the shuffle and therefore the losses"


def test_validation_is_evaluated_in_eval_mode_so_dropout_does_not_leak_into_val_loss():
    # Dropout(p=1) zeroes every activation in train mode, so training loss is mean(y^2) and
    # nothing learns. In eval mode the layer is an identity, so validation sees the linear model.
    torch.manual_seed(5)
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Dropout(p=1.0)).double()
    X, y, Xv, yv = dataset(5, n=6, m=4)
    with torch.no_grad():
        eval_val = ((model[0](Xv) - yv) ** 2).mean().item()
    train, val = checked_fit(model, X, y, Xv, yv, epochs=3, batch_size=4, lr=0.1, seed=6)
    expected_train = ((y**2).mean()).item()
    assert all(abs(t - expected_train) <= ATOL + RTOL * expected_train for t in train), (
        "training must run in train mode, where Dropout(p=1) outputs zeros"
    )
    assert all(abs(v - eval_val) <= ATOL + RTOL * eval_val for v in val), (
        "validation must run in eval mode: dropout must be off when measuring validation loss"
    )


def test_best_epoch_is_restored_even_when_validation_loss_rises_afterwards():
    X, y, Xv, yv = dataset(6, n=6, m=20)  # few training points, many validation points
    base = mlp(7)
    kw = {"epochs": 12, "batch_size": 2, "lr": 0.5, "seed": 8}
    _want_train, want_val, want_model = reference_fit(base, X, y, Xv, yv, **kw)
    assert min(want_val) < want_val[-1], "fixture must make validation rise after its best epoch"
    got_model = copy.deepcopy(base)
    _got_train, got_val = checked_fit(got_model, X, y, Xv, yv, **kw)
    with torch.no_grad():
        restored = ((got_model(Xv) - yv) ** 2).mean().item()
    assert abs(restored - min(got_val)) <= ATOL + RTOL * min(got_val), (
        "the model must hold the best-validation parameters, not the last epoch's"
    )
    for got, want in zip(got_model.parameters(), want_model.parameters(), strict=True):
        assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), (
            "restored parameters are not the best epoch's: keep the state with lowest val loss"
        )


class EvalIgnoresWeights(torch.nn.Module):
    """Trains like a linear model but returns a constant in eval mode.

    Validation ties exactly across epochs while the weights keep changing, so only the tie rule
    can decide which checkpoint is kept."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 1).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin(x)
        if self.training:
            return out
        return torch.full_like(out, 0.25).detach()


def test_tied_validation_losses_keep_the_earlier_epoch():
    X, y, Xv, yv = dataset(7, n=6, m=3)
    torch.manual_seed(9)
    base = EvalIgnoresWeights()
    kw = {"epochs": 3, "batch_size": 2, "lr": 0.3, "seed": 10}
    _, val = checked_fit(copy.deepcopy(base), X, y, Xv, yv, **kw)
    assert val[0] == val[1] == val[2], "fixture must tie validation losses exactly"
    model = copy.deepcopy(base)
    checked_fit(model, X, y, Xv, yv, **kw)
    _, _, first_epoch_only = reference_fit(base, X, y, Xv, yv, **{**kw, "epochs": 1})
    for got, want in zip(model.parameters(), first_epoch_only.parameters(), strict=True):
        assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), (
            "with tied validation losses the earlier epoch's state must be kept"
        )


def test_batch_size_larger_than_dataset_equals_one_full_batch():
    X, y, Xv, yv = dataset(8, n=5, m=3)
    base = mlp(10)
    big = checked_fit(copy.deepcopy(base), X, y, Xv, yv, epochs=2, batch_size=50, lr=0.1, seed=12)
    full = checked_fit(copy.deepcopy(base), X, y, Xv, yv, epochs=2, batch_size=5, lr=0.1, seed=12)
    assert big == full, "a batch size above n must behave exactly like one batch of all examples"


def test_inputs_are_not_mutated():
    X, y, Xv, yv = dataset(9, n=7, m=4)
    before = [t.clone() for t in (X, y, Xv, yv)]
    checked_fit(mlp(11), X, y, Xv, yv, epochs=2, batch_size=3, lr=0.1, seed=13)
    for name, now, old in zip(("X", "y", "X_val", "y_val"), (X, y, Xv, yv), before, strict=True):
        assert torch.equal(now, old), f"{name} was modified: do not write into the data"


@pytest.mark.parametrize(
    "bad",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"batch_size": True},
        {"lr": 0.0},
        {"lr": float("nan")},
        {"seed": -1},
    ],
    ids=["zero-epochs", "zero-batch", "bool-batch", "zero-lr", "nan-lr", "negative-seed"],
)
def test_invalid_hyperparameters_raise_value_error(bad):
    X, y, Xv, yv = dataset(10, n=4, m=2)
    kw = {"epochs": 1, "batch_size": 2, "lr": 0.1, "seed": 0}
    kw.update(bad)
    try:
        fit(mlp(12), X, y, Xv, yv, **kw)
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(
            f"invalid setting must raise ValueError, not {type(exc).__name__}"
        ) from None
    raise AssertionError("invalid hyperparameter was accepted: check the documented range")


def test_mismatched_rows_between_inputs_and_targets_raise_value_error():
    X, y, Xv, yv = dataset(11, n=4, m=2)
    for args in ((X, y[:3], Xv, yv), (X, y, Xv, yv[:1])):
        try:
            fit(mlp(13), *args, epochs=1, batch_size=2, lr=0.1, seed=0)
        except ValueError:
            continue
        except Exception as exc:
            raise AssertionError(
                f"mismatched rows must raise ValueError, not {type(exc).__name__}"
            ) from None
        raise AssertionError("inputs and targets with different row counts were accepted")


class RecordsGradMode(torch.nn.Module):
    """Linear model that records whether autograd was enabled on every eval-mode forward."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 1).double()
        self.eval_grad_flags: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            self.eval_grad_flags.append(torch.is_grad_enabled())
        return self.lin(x)


def test_validation_runs_without_gradients_enabled():
    X, y, Xv, yv = dataset(12, n=6, m=4)
    model = RecordsGradMode()
    checked_fit(model, X, y, Xv, yv, epochs=2, batch_size=3, lr=0.1, seed=15)
    assert model.eval_grad_flags, "validation must call the model in eval mode"
    assert not any(model.eval_grad_flags), (
        "validation ran with gradients enabled: wrap the validation pass in torch.no_grad()"
    )


def test_single_row_training_set_is_valid():
    X, y, Xv, yv = dataset(13, n=1, m=1)
    train, val = checked_fit(mlp(14), X, y, Xv, yv, epochs=2, batch_size=1, lr=0.1, seed=0)
    assert len(train) == 2 and len(val) == 2, (
        "a training set of one row is valid (n >= 1): the run must return one loss per epoch"
    )


def test_training_loss_averages_over_every_target_column():
    # Two target columns: the loss is the mean over all squared errors, not column 0 alone.
    # One full batch, so the epoch's train loss is the loss before the one step (initial weights).
    torch.manual_seed(5)
    base = torch.nn.Linear(2, 2).double()
    gen = torch.Generator().manual_seed(5)
    X = torch.randn(6, 2, generator=gen, dtype=torch.float64)
    y = torch.randn(6, 2, generator=gen, dtype=torch.float64)
    with torch.no_grad():
        full = torch.nn.functional.mse_loss(base(X), y).item()
    train, _ = checked_fit(copy.deepcopy(base), X, y, X, y, epochs=1, batch_size=6, lr=0.1, seed=0)
    assert abs(train[0] - full) <= ATOL + RTOL * abs(full), (
        "the training loss must average the squared error over every target column"
    )
