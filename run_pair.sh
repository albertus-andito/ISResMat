#!/bin/bash


# --n-trn-cols: column frequency
# --batch-size: number of pairwise fragments in one update
# --frag-height and --frag-width: fragment size
# --col-name-prob: 0 is instance-based, in (0,1) is hybrid, >1 (e.g., 2) is schema-based
# --store-matches: be 1 to save the matching results in the JSON files

# --comment: marks a running, and be used as the folder name for the run results.
# --dataset-name: unique id of a table pair
# --orig-file-src, --orig-file-tgt: the location of the source table, target table
python -m isresmat \
  --n-trn-cols=200 \
  --batch-size=1 \
  --frag-height=6 \
  --frag-width=12 \
  --learning-rate=3e-5 \
  --col-name-prob=0 \
  --store-matches=1 \
  --comment=single_out_dir \
  --dataset-name=DeepMDatasets/itunes_amazon \
  --orig-file-src=data/sm-valentine-m1/DeepMDatasets/itunes_amazon/itunes_amazon_source.csv \
  --orig-file-tgt=data/sm-valentine-m1/DeepMDatasets/itunes_amazon/itunes_amazon_target.csv \
