import torch
from src.models.util_models import EarlyStopping


def test_EarlyStopper():
    tolerance = 15
    EarlyStopper = EarlyStopping(tolerance=tolerance)
    A = torch.arange(0, 100, 1).flip(dims=(0,))
    for loss in A:
        EarlyStopper(loss.item())
        if EarlyStopper.early_stop:
            break
    assert EarlyStopper.early_stop == False

    def run_test(expect, A, tolerance, finish=False):
        EarlyStopper = EarlyStopping(tolerance=tolerance)
        for i, loss in enumerate(A):
            EarlyStopper(loss.item())
            if EarlyStopper.early_stop:
                assert i == expect
                break
        if not finish:
            assert EarlyStopper.early_stop == False

    A = torch.tensor([15] * 20)
    run_test(15, A, tolerance, finish=True)
    A = torch.tensor(([1] * 100), dtype=torch.float32)
    A[0:67] -= torch.tensor([(0.0001) * x for x in range(67)], dtype=torch.float32)
    A[67:] = 0.5
    run_test(67 + 15, A, tolerance, finish=True)

    A = torch.tensor([1] * 100)
    A[67:] = 0.5
    run_test(67 + 100, A, tolerance=100, finish=False)

    A = torch.tensor([1] * 100)
    A[14] = 0.5
    run_test(14 + 15, A, tolerance=15, finish=True)

    A = torch.tensor([1] * 100)
    A[0] = 0.5
    run_test(15, A, tolerance=15, finish=True)
