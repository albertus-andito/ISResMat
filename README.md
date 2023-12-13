# ISResMat
A framework that matches the schemas of relational tables by fine-tuning
a pre-trained language model.

## Dependencies
If you use conda, you can create an environment with the following command:
```
conda env create -f environment.yml
```

## Benchmark
- You can download the benchmark dataset from [here](https://zenodo.org/records/10360876). 
This dataset has been modified from the benchmark provided by [Valentine](https://arxiv.org/pdf/2010.07386.pdf), 
so that in the fabricated dataset section, the matching column for table pairs with only one matching column is
random, rather than being the same column every time.
- After extracting the data, place it in the `data` folder of the project.
- Run the `run_benchmark.sh` script. (Refer to it and `isresmat.py` for parameter details.).
- The results will be in the `data/output`.
- Running `collect_ISResMat_results_into_csv.py` will gather the results 
into a CSV file in `data/output/summary_csv`.

## Custom Datasets
Modify the data location parameter in `run_pair.sh` to run with your own data.