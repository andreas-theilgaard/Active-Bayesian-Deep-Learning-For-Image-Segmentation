import pytest
from src.models.train_active_loop import run_active
import pandas as pd
import numpy as np

model_params = {
    "in_ch": 1,
    "n_classes": 1,
    "bilinear_method": False,
    "momentum": 0.9,
    "enable_dropout": False,
    "dropout_prob": 0.5,
    "enable_pool_dropout": False,
    "pool_dropout_prob": 0.5,
}
dataset = "PhC-C2DH-U373"


@pytest.mark.skipif(1 == 1, reason="Too Computationally Expensive - Run Local !!!")
def test_active_run():
    seed = 261
    # # Random Test #
    AcquisitionFunction = "Random"
    model_method = "BatchNorm"
    torch_seeds = [17]
    run_active(
        model_params=model_params,
        dataset="PhC-C2DH-U373",
        epochs=3,
        start_size="2-Samples",
        model_method="BatchNorm",
        AcquisitionFunction="Random",
        torch_seeds=torch_seeds,
        seed=seed,
        testing=True,
    )
    out = pd.read_json(
        f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json"
    )
    assert np.sum(np.array(out["labels_added"][1]) == np.array([59, 134])) == 2
    assert np.sum(np.array(out["labels_added"][2]) == np.array([85, 111])) == 2

    # Deep Ensemble Test #
    AcquisitionFunction = "ShanonEntropy"
    model_method = "DeepEnsemble"
    torch_seeds = [17, 8, 42, 19, 5]
    run_active(
        model_params=model_params,
        dataset="PhC-C2DH-U373",
        epochs=3,
        start_size="2-Samples",
        model_method=model_method,
        AcquisitionFunction=AcquisitionFunction,
        torch_seeds=torch_seeds,
        seed=261,
        testing=True,
        loaders=False,  # Path to saved loaders if any
        timeLimit=23,  # In Hours
    )
    out = pd.read_json(
        f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json"
    )
    assert np.sum(np.array(out["labels_added"][1]) == np.array([84, 47])) == 2
    assert np.sum(np.array(out["labels_added"][2]) == np.array([131, 114])) == 2

    # MC Dropout Test #
    model_params["enable_pool_dropout"] = True
    AcquisitionFunction = "ShanonEntropy"
    model_method = "MCD"
    torch_seeds = [17]
    run_active(
        model_params=model_params,
        dataset="PhC-C2DH-U373",
        epochs=3,
        start_size="2-Samples",
        model_method=model_method,
        AcquisitionFunction=AcquisitionFunction,
        torch_seeds=torch_seeds,
        seed=seed,
        testing=True,
    )
    out = pd.read_json(
        f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json"
    )
    assert np.sum(np.array(out["labels_added"][1]) == np.array([47, 91])) == 2
    assert np.sum(np.array(out["labels_added"][2]) == np.array([131, 114])) == 2

    ## Laplace Test #
    model_params["enable_pool_dropout"] = True
    AcquisitionFunction = "ShanonEntropy"
    model_method = "Laplace"
    torch_seeds = [17]
    run_active(
        model_params=model_params,
        dataset="PhC-C2DH-U373",
        epochs=3,
        start_size="2-Samples",
        model_method=model_method,
        AcquisitionFunction=AcquisitionFunction,
        torch_seeds=torch_seeds,
        seed=seed,
        testing=True,
        device="cpu",
    )
    out = pd.read_json(
        f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json"
    )
    assert np.sum(np.array(out["labels_added"][1]) == np.array([47, 140])) == 2
    assert np.sum(np.array(out["labels_added"][2]) == np.array([131, 4])) == 2


@pytest.mark.skipif(1 == 1, reason="Too Computationally Expensive - Run Local !!!")
def test_active_checkpoint():
    model_params["enable_pool_dropout"] = False

    seed = 261
    # Random Test #
    AcquisitionFunction = "Random"
    model_method = "BatchNorm"
    torch_seeds = [17]
    run_active(
        model_params=model_params,
        dataset="PhC-C2DH-U373",
        epochs=3,
        start_size="2-Samples",
        model_method="BatchNorm",
        AcquisitionFunction="Random",
        torch_seeds=torch_seeds,
        seed=seed,
        testing=True,
        loaders=False,  # Path to saved loaders if any
        timeLimit=1 / 10000,
    )
    out = pd.read_json(
        f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json"
    )
    assert np.sum(np.array(out["labels_added"][1]) == np.array([59, 134])) == 2

    run_active(
        model_params=model_params,
        dataset="PhC-C2DH-U373",
        epochs=3,
        start_size="2-Samples",
        model_method=model_method,
        AcquisitionFunction=AcquisitionFunction,
        torch_seeds=torch_seeds,
        seed=261,
        testing=True,
        loaders=f"results/active_learning/{dataset}_saved_loaders_{model_method}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}.json",  # Path to saved loaders if any
        timeLimit=23,
        first_train=False,
    )
    out = pd.read_json(
        f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json"
    )
    assert np.sum(np.array(out["labels_added"][1]) == np.array([59, 134])) == 2
    assert np.sum(np.array(out["labels_added"][2]) == np.array([85, 111])) == 2
    # Random Test Done#

    # MC Dropout Test #
    model_params["enable_pool_dropout"] = True
    AcquisitionFunction = "ShanonEntropy"
    model_method = "MCD"
    torch_seeds = [17]
    run_active(
        model_params=model_params,
        dataset="PhC-C2DH-U373",
        epochs=3,
        start_size="2-Samples",
        model_method=model_method,
        AcquisitionFunction=AcquisitionFunction,
        torch_seeds=torch_seeds,
        seed=seed,
        testing=True,
        loaders=False,  # Path to saved loaders if any
        timeLimit=1 / 10000,
    )
    out = pd.read_json(
        f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json"
    )
    assert np.sum(np.array(out["labels_added"][1]) == np.array([47, 91])) == 2
    run_active(
        model_params=model_params,
        dataset="PhC-C2DH-U373",
        epochs=3,
        start_size="2-Samples",
        model_method=model_method,
        AcquisitionFunction=AcquisitionFunction,
        torch_seeds=torch_seeds,
        seed=seed,
        testing=True,
        loaders=False,  # Path to saved loaders if any
        timeLimit=1 / 10000,
    )
    run_active(
        model_params=model_params,
        dataset="PhC-C2DH-U373",
        epochs=3,
        start_size="2-Samples",
        model_method=model_method,
        AcquisitionFunction=AcquisitionFunction,
        torch_seeds=torch_seeds,
        seed=261,
        testing=True,
        loaders=f"results/active_learning/{dataset}_saved_loaders_{model_method}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}.json",  # Path to saved loaders if any
        timeLimit=23,
        first_train=False,
    )
    out = pd.read_json(
        f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json"
    )
    assert np.sum(np.array(out["labels_added"][1]) == np.array([47, 91])) == 2
    assert np.sum(np.array(out["labels_added"][2]) == np.array([131, 114])) == 2

    # MC Dropout Test Done#

    # Deep Ensemble Test #
    model_params["enable_pool_dropout"] = False

    seed = 261
    AcquisitionFunction = "ShanonEntropy"
    model_method = "DeepEnsemble"
    torch_seeds = [17, 8, 42, 19, 5]
    run_active(
        model_params=model_params,
        dataset="PhC-C2DH-U373",
        epochs=3,
        start_size="2-Samples",
        model_method=model_method,
        AcquisitionFunction=AcquisitionFunction,
        torch_seeds=torch_seeds,
        seed=261,
        testing=True,
        loaders=False,  # Path to saved loaders if any
        timeLimit=1 / 10000,
    )
    out = pd.read_json(
        f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json"
    )
    assert np.sum(np.array(out["labels_added"][1]) == np.array([84, 47])) == 2
    # Deep Ensemble Test Done#

    # Continue Training
    run_active(
        model_params=model_params,
        dataset="PhC-C2DH-U373",
        epochs=3,
        start_size="2-Samples",
        model_method=model_method,
        AcquisitionFunction=AcquisitionFunction,
        torch_seeds=torch_seeds,
        seed=261,
        testing=True,
        loaders=f"results/active_learning/{dataset}_saved_loaders_{model_method}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}.json",  # Path to saved loaders if any
        timeLimit=23,
        first_train=False,
    )
    out = pd.read_json(
        f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json"
    )
    assert np.sum(np.array(out["labels_added"][1]) == np.array([84, 47])) == 2
    assert np.sum(np.array(out["labels_added"][2]) == np.array([131, 114])) == 2

    ## Laplace Test #
    model_params["enable_pool_dropout"] = True
    AcquisitionFunction = "ShanonEntropy"
    model_method = "Laplace"
    torch_seeds = [17]
    run_active(
        model_params=model_params,
        dataset="PhC-C2DH-U373",
        epochs=3,
        start_size="2-Samples",
        model_method=model_method,
        AcquisitionFunction=AcquisitionFunction,
        torch_seeds=torch_seeds,
        seed=seed,
        testing=True,
        loaders=False,  # Path to saved loaders if any
        timeLimit=1 / 10000,
    )
    out = pd.read_json(
        f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json"
    )
    assert np.sum(np.array(out["labels_added"][1]) == np.array([47, 140])) == 2
    # Continue Traning
    run_active(
        model_params=model_params,
        dataset="PhC-C2DH-U373",
        epochs=3,
        start_size="2-Samples",
        model_method=model_method,
        AcquisitionFunction=AcquisitionFunction,
        torch_seeds=torch_seeds,
        seed=seed,
        testing=True,
        first_train=False,
        loaders=f"results/active_learning/{dataset}_saved_loaders_{model_method}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}.json",
        timeLimit=1 / 10000,
    )
    out = pd.read_json(
        f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json"
    )
    assert np.sum(np.array(out["labels_added"][1]) == np.array([47, 140])) == 2
    assert np.sum(np.array(out["labels_added"][2]) == np.array([131, 4])) == 2


if __name__ == "__main__":
    test_active_run()
    # print(f"Beginning checkpoint tests")
    # test_active_checkpoint()
