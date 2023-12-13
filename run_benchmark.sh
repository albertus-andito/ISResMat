#!/bin/bash

# --n-trn-cols: column frequency
# --batch-size: number of pairwise fragments in one update
# --frag-height and --frag-width: fragment size
# --col-name-prob: 0 is instance-based, in (0,1) is hybrid, >1 (e.g., 2) is schema-based
# --store-matches: whether to save the matching results in the JSON files (be mindful of storage space)

# --comment: marks a running, and be used as the folder name for the run results.
# --dataset-name: unique id of a table pair
# --orig-file-src, --orig-file-tgt, --orig-file-golden-matches: the location of the source table, target table
#     and the matching ground truth
run_isresmat() {
  python -m isresmat \
    --n-trn-cols=200 \
    --batch-size=1 \
    --frag-height=6 \
    --frag-width=12 \
    --learning-rate=3e-5 \
    --col-name-prob=0 \
    --store-matches=0 \
    --comment=inst-001 \
    --dataset-name=$data_name \
    --orig-file-src=$src_orig_file \
    --orig-file-tgt=$tgt_orig_file \
    --orig-file-golden-matches=$golden_mappings
}

split_path() {
  local path=$1
  local IFS=/
  read -ra parts <<<"$path"
  echo "${parts[@]}"
}

extract_file_paths() {
  local subsubdir="$1"
  for file in $subsubdir/*; do
    if [[ $file = *source.csv ]]; then
      src_orig_file=$file
    elif [[ $file = *target.csv ]]; then
      tgt_orig_file=$file
    elif [[ $file = *mapping.json ]]; then
      golden_mappings=$file
    fi
  done
}

data_dir="data/sm-valentine-m1"
count=0
for data in $data_dir/*; do
  if [[ $data = */Wikidata || $data = */miller2 || $data = */assays || $data = */prospect ]]; then
    for subdir in $data/*; do
      if [[ $subdir = * ]]; then
        for pair_dir in $subdir/*; do
          if [[ $pair_dir = * ]]; then

            extract_file_paths "$pair_dir"
            path_parts=($(split_path "$pair_dir"))
            data_name=${path_parts[-3]}/${path_parts[-2]}/${path_parts[-1]}

            #            echo $src_orig_file
            #            echo $tgt_orig_file
            #            echo $golden_mappings
            #            echo $data_name
            #            break

            count=$((count + 1))
            echo "$data_name Dataset Counter: $count"
            run_isresmat
          fi
        done
      fi
    done
  elif [[ $data = */DeepMDatasets ]]; then
    for pair_dir in $data/*; do
      extract_file_paths "$pair_dir"
      path_parts=($(split_path "$pair_dir"))
      data_name=${path_parts[-2]}/${path_parts[-1]}
      #        echo $src_orig_file
      #        echo $tgt_orig_file
      #        echo $golden_mappings
      #        echo $data_name
      #        break

      count=$((count + 1))
      echo "$data_name Dataset Counter: $count"
      run_isresmat
    done
  fi
done
