ChEMBL Regressor
================

A program for biological activity prediction trained with ChEMBL assays.

<img width="668" height="342" alt="ChEMBL Regressor in Action" src="https://github.com/user-attachments/assets/7e761371-43a3-4437-bf3b-a2d0abb505f2" />

Install
-------

```bash
$ pip install https://github.com/pslacerda1/chembl-regressor/archive/refs/heads/main.zip
```

Usage
-----
```
$ chembl-regressor train --help
Usage: chembl-regressor train [OPTIONS]

  This program train a regressor based on a ChEMBL dataset.

Options:
  --input PATH                    ChEMBL .zip package.  [required]
  --plot                          Display plots.
  --output-model PATH             Output model filename.
  --target-var [pChEMBL Value|Standard Value]
                                  Target variable.
  --log10                         May be useful for standartize the target
                                  variable.
  --scramble                      Shuffle the input data before splitting.
  --random-seed INTEGER
  --test-size FLOAT               Fraction of external test size.
  --cross-val INTEGER             Number of cross validation folds.
  --drop-na                       Drop NaN values.
  --n-jobs INTEGER                Number of jobs.
  --fingerprint [ecfp|fcfp]       Fingerprint method.
  --fpt-radius INTEGER            Fingerprint radius.
  --fpt-n-bits INTEGER            Fingerprint size.
  --help                          Show this message and exit.
```

```
$ chembl-regressor predict --help
Usage: chembl-regressor predict [OPTIONS]

  This program predict the trained ChEMBL value for a list of compounds.

Options:
  --input-smiles PATH   Multi SMILES file.
  --input-model PATH    Trained model file.
  --output-scores PATH  CSV output file.
  --help                Show this message and exit.
```

Example
-------

```bash
$ chembl-regressor train \
    --input data/OXTR.zip \
    --n-jobs 7 \
    --test-size 0.25 \
    --drop-na \
    --target-var 'pChEMBL Value' \
    --output-model data/OXTR.joblib \
    --plot \
    --fingerprint fcfp
```

```bash
$ run chembl-regressor predict \
    --input-model data/OXTR.joblib \
    --input-smiles  data/sample.smi \
    --output-scores output.csv
```

```bash
$ cat output.csv
Aspirin,7.08
Paracetamol,6.87
Ibuprofen,7.11
Caffeine,6.97
Benzene,7.14
Ethanol,6.98
```
