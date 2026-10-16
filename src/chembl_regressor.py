from __future__ import annotations

from collections.abc import Collection
from typing import Literal

import click
import joblib
import numpy as np
import pandas as pd
import seaborn as sb
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from molfeat.calc import FPCalculator
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

type FingerprintMethod = Literal["ecfp", "fcfp"]
type Smiles = str


def parse_smiles_into_fpts(
    smis: Collection[Smiles],
    method: FingerprintMethod,
    radius: int,
    fp_size: int,
) -> np.ndarray:
    fpts: np.ndarray  = np.zeros(shape=(len(smis), fp_size))
    fp_calc = FPCalculator(method, radius=radius, fpSize=fp_size)
    for ix, smi in enumerate(smis):
        fpts[ix] = fp_calc(smi)
    return fpts


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="ChEMBL .zip package"
)
@click.option("--plot", is_flag=True, help="display plots")
@click.option(
    "--output-model",
    default="regressor.joblib",
    type=click.Path(exists=False),
    help="output model filename",
)
@click.option(
    "--target-var",
    default="pChEMBL Value",
    type=click.Choice(["pChEMBL Value", "Standard Value"]),
    help="target variable",
)
@click.option("--random-seed", default=0)
@click.option("--test-size", default=0.2, help="fraction of external test size")
@click.option("--cross-val", default=10, help="number of cross validation folds")
@click.option("--drop-na", is_flag=True, help="drop NaN values")
@click.option("--n-jobs", default=-1, help="number of jobs")
@click.option(
    "--fingerprint",
    default="ecfp",
    type=click.Choice(["ecfp", "fcfp"]),
    help="fingerprint method"
)
@click.option("--fpt-radius", default=2, help="fingerprint radius")
@click.option("--fpt-n-bits", default=2048, help="fingerprint size")
@click.option(
    "--log10",
    is_flag=True,
    help="may be useful for standartize the target variable",
)
@click.option(
    "--scramble",
    is_flag=True,
    help="shuffle the input data before splitting",
)
def train(
    input,
    output_model,
    plot,
    random_seed,
    test_size,
    cross_val,
    drop_na,
    n_jobs,
    fingerprint,
    fpt_radius,
    fpt_n_bits,
    target_var,
    log10,
    scramble,
):
    """This program train a regressor based on a ChEMBL dataset."""
    data = pd.read_csv(
        input,
        na_values=["", "None"],
        sep=";",
        usecols=["Molecule ChEMBL ID", "Smiles", target_var],
    )

    if scramble:
        data = data.sample(frac=1.0, random_state=random_seed)
    
    y = data[target_var]
    y = y.rename("y")

    if log10:
        y = np.log10(y)

    if np.any(pd.isna(data)):
        if not drop_na:
            num_rows = np.sum(pd.isna(data), axis=0)
            click.secho(
                f"There are {num_rows} rows with NaN values into your dataset.",
                fg="red",
                err=True
            )
            return
        else:
            click.echo("Cleaning NaN values... ", nl=False)
            idx = ~(
                pd.isna(y)
                | np.any(pd.isnull(data), axis=1)
            )
            data = data[idx]
            y = y[idx]

            ratio = 100 * np.sum(idx) / len(idx)
            click.secho(f"About {ratio:.2f}% of data remained.", fg="yellow")

    click.secho(f"Min {target_var}: {np.min(y)}")
    click.secho(f"Max {target_var}: {np.max(y)}")

    if np.max(y) - np.min(y) < 3:
        click.secho(f"Short range of {target_var}.", fg="yellow")

    click.echo("Computing fingerprints... ", nl=False)
    smis = data.pop("Smiles")
    fpt_params = {
        'method': fingerprint,
        'radius': fpt_radius,
        'fp_size': fpt_n_bits
    }
    fpts = parse_smiles_into_fpts(
        smis=smis,
        **fpt_params
    )
    click.secho("Done.", fg="green")

    X_train, X_test, y_train, y_test = train_test_split(
        fpts, y, test_size=test_size, random_state=random_seed
    )

    click.secho(f"Train size: {len(X_train)}")
    click.secho(f"Test size: {len(X_test)}")

    scaler = StandardScaler()  # não precisa
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    verbose = -1
    models = {}
    for model in [
        LGBMRegressor(
            random_state=random_seed,
            n_jobs=n_jobs,
            verbose=verbose
        ),
        RandomForestRegressor(
            random_state=random_seed,
            n_jobs=n_jobs,
        ),
    ]:
        label = model.__class__.__name__
        click.echo("Fitting model ", nl=False)
        click.secho(f"{label}...", fg='black', bg='cyan', bold=True, nl=False)
        cv = GridSearchCV(model, {}, cv=cross_val, n_jobs=n_jobs)
        cv.fit(X_train_scaled, y_train)

        best_model = cv.best_estimator_

        y_pred = best_model.predict(X_test_scaled)
        corr = spearmanr(y_test, y_pred)
        click.secho(f" Spearman corr={corr[0]:.2f}. ", fg="green", bold=True, nl=False)
        click.secho("Done.", fg="green")

        models[label] = (best_model, corr[0])
    
    best_model = sorted(models.values(), key=lambda m: m[1], reverse=True)[0][0]
    joblib.dump((fpt_params, scaler, best_model), output_model)

    if plot:
        y_pred = best_model.predict(X_test_scaled)  # again

        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
        fig.suptitle(f'{best_model.__class__.__name__}\n(best regressor)')

        sb.residplot(x=y_test, y=y_pred, color="g", ax=ax1)
        ax1.set(label="Residuals", ylabel="Residuals", xlabel="Predicted Value")

        sb.regplot(x=y_test, y=y_pred, ax=ax2)
        ax2.set(label="Regression", xlabel="y test", ylabel="y pred")

        plt.tight_layout()
        plt.show()


@cli.command()
@click.option(
    "--input-smiles",
    type=click.Path(exists=True),
    help="Multi SMILES file "
)
@click.option(
    "--input-model",
    type=click.Path(exists=True),
    default="regressor.joblib",
    help="trained model file"
)
@click.option("--output-scores", default="scores.csv", type=click.Path())
def predict(
    input_smiles,
    input_model,
    output_scores
):
    """This program predict the pChEMBL value for a list of compounds."""
    click.echo("Loading model.")
    (fpt_params, scaler, model) = joblib.load(input_model)

    click.echo("Reading compounds.")
    data = pd.read_csv(input_smiles, sep=",", names=("Smiles", "Name"))

    click.echo("Computing results...")
    
    with open(output_scores, "w") as output_file:
        for idx, row in data.iterrows():
            try:
                fpt = parse_smiles_into_fpts(
                    smis=[row["Smiles"]],
                    **fpt_params
                )
                X = scaler.transform(fpt)
                y = model.predict(X)[0]
                print(row["Name"], round(y, 2), sep=",", file=output_file)
                output_file.flush()
            except Exception as e:
                click.echo(f"Error on {row['Name']}: {e}", err=True)

