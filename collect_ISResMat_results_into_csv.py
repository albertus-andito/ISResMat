import json
import os

import pandas as pd
from tqdm import tqdm

# Location of the directory storing the running results.
directory_path = 'data/output/inst-001'


n_files_contained_in_dir = 551
# all the running results in the current dir
total_csv_path = os.path.join(os.path.dirname(directory_path), 'summary_csv', directory_path.split('/')[-1],
                              'total.csv')
# the overall metric (specially the recall@ground truth and the micro average version of the recall@ground truth)
# for each setting of the parameters in the current dir
mean_filepath = os.path.join(os.path.dirname(directory_path), 'summary_csv', directory_path.split('/')[-1], 'means.csv')

if os.path.dirname(total_csv_path):
    os.makedirs(os.path.dirname(total_csv_path), exist_ok=True)

if os.path.dirname(mean_filepath):
    os.makedirs(os.path.dirname(mean_filepath), exist_ok=True)


def json2dict(json_file, ignore_keys):
    """
        maximal two levels
    """
    with open(json_file, 'r') as jf:
        data = json.load(jf)
    data_in_json = {}
    for key, value in data.items():
        if key not in ignore_keys:
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    data_in_json[sub_key] = sub_value
            else:
                data_in_json[key] = value
    return data_in_json

compact_ls = ['src_center_sh_ls', 'src_agent_sh_ls', 'tgt_center_sh_ls', 'tgt_agent_sh_ls']

###################################
# get all the json files contained in the dir, I do this first
# so that when there new field added into the newly generated
# json files, I do not have to manually modify the code in this script.
all_json_files = []
all_fields = []
# put the name of the keys that do not want to be saved in this list
ignore_keys = ['matches']
for root, _, files in os.walk(directory_path):
    if root.split('/')[-1].startswith('dsf'):
        assert len(files) == n_files_contained_in_dir, root
    all_json_files += [os.path.join(root, f) for f in files if f.endswith('.json')]
# Loop through all your JSON files and extract field names
print('prepare all the possible fields')
for json_file in tqdm(all_json_files):
    json_dc = json2dict(json_file, ignore_keys)
    for k in json_dc.keys():
        if k not in all_fields:
            all_fields.append(k)

print(all_fields)
###################################

# all the running results of all the dataset under all the settings
total_data_list = []
# the average performance of different settings
diff_setting_avg_list = []

###################################
# all the result json file for a parameter setting
all_json_files = []
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.json'):
            all_json_files.append(os.path.join(root, file))

assert len(all_json_files) == n_files_contained_in_dir

data_list = []
for json_filename in all_json_files:
    json_dc = json2dict(json_filename, ignore_keys)
    # get a field value that is list, which is normally one of the metrics, and each
    # value in the list is the corresponding metric value at a checkpoint in the middle
    # of the training
    foo_ls = json_dc.get('recall_at_sizeof_ground_truth')
    # for field in all_fields:
    #     fv = json_dc.get(field, 'NaN')
    #     if fv != 'NaN' and isinstance(fv, list):
    #         if not isinstance(fv[0], str) and not fv[0].startswith('comment='):
    #             foo_ls = fv
    #             break

    # build the training records row by row, and each row corresponds to a checkpoints
    # at the middle of the training
    ntc_value = -1
    for i, _ in enumerate(foo_ls):
        row = {}
        for field in all_fields:
            fv = json_dc.get(field, 'NaN')
            if fv == 'NaN':
                row[field] = 'NaN'
            elif not isinstance(fv, list):
                row[field] = fv
            elif isinstance(fv, list):
                if isinstance(fv[0], str) and fv[0].startswith('comment='):
                    pass
                    # # double check
                    # comment = f'comment={run_dir_name}'
                    # assert comment == fv[0]
                    #
                    # # update the comment (run_id / run_name) to correspond to the
                    # # middle checking points
                    # match = re.search(r'ntc(\d+\.\d+|\d+)_', fv[0])
                    # ntc_value = float(match.group(1))
                    # assert ntc_value != -1
                    # # especially for the static dataset experiment
                    # scale_factor_match = re.search(r'dsf(\d+\.\d+|\d+)_', fv[0])
                    # scale_factor_value = float(scale_factor_match.group(1))
                    # ntc_value = round(ntc_value * scale_factor_value)
                    # loc_ntc_value = int((ntc_value / len(foo_ls)) * (i + 1))
                    #
                    # run_name = re.sub(r'ntc\d+', f'ntc{loc_ntc_value}', fv[0])
                    # row['run_name'] = run_name.replace('comment=', '')
                elif field not in compact_ls:
                    row[field] = fv[i]
                else:
                    row[field] = fv
            else:
                assert False
        data_list.append(row)
total_data_list += data_list

df = pd.DataFrame(data_list)

####################################################################
# compute the micro recall@ground truth
fab_df = df[df['dataname'].str.contains('assays|miller2|prospect')]
fab_mrg = fab_df['n_generated_matches'].sum() / fab_df['n_expected_matches'].sum()

man_df = df[df['dataname'].str.contains('DeepM|Wikidata')]
man_mrg = man_df['n_generated_matches'].sum() / man_df['n_expected_matches'].sum()


mrg = df['n_generated_matches'].sum() / df['n_expected_matches'].sum()

print('fabricated dataset mrg:', fab_mrg)
print('human-curated dataset mrg:', man_mrg)
print('overall dataset mrg:', mrg)


####################################################################

# for i in range(len(foo_ls)):
#     loc_ntc_value = int((ntc_value / len(foo_ls)) * (i + 1))
#     run_name = re.sub(r'ntc\d+', f'ntc{loc_ntc_value}', f'{run_dir_name}')
#     subdf = df[df['run_name'] == run_name]
#     diff_setting_avg_list.append(
#         {
#             'run_name': run_name,
#             'micro_avg_recall_at_sizeof_ground_truth': subdf['n_generated_matches'].sum() / subdf[
#                 'n_expected_matches'].sum(),
#             'avg_recall_at_sizeof_ground_truth': subdf['recall_at_sizeof_ground_truth'].mean(),
#             'avg_precision': subdf['precision'].mean(),
#             'avg_recall': subdf['recall'].mean(),
#             'avg_f1_score': subdf['f1_score'].mean(),
#         }
#     )
#     # print(run_name, subdf['recall_at_sizeof_ground_truth'].mean())
#     print(run_name, subdf['n_generated_matches'].sum() / subdf['n_expected_matches'].sum(), )
###################################

total_df = pd.DataFrame(total_data_list)
total_df.to_csv(total_csv_path, index=False)

# mean_df = pd.DataFrame(diff_setting_avg_list)
# mean_df.to_csv(mean_filepath, index=False)
