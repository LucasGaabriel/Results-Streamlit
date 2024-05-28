import pandas as pd

path = "results/ml/1nn/"
file_name = "1NN"

splits = ('biased_usual', 'biased_mirrored', 'unbiased')
datasets = ('cwru_12000', 'cwru_48000', 'ims', 'mfpt_48828', 'mfpt_97656', 'pu', 'uoc', 'xjtu')
metrics = ('test_f1_macro', 'test_balanced_accuracy')

data = []
for dataset in datasets:
    info_dataset = {}
    for metric in metrics:
        for split in splits:
            full_path = path + split + '/' + '' + dataset + '.csv'
            file = pd.read_csv(full_path)
            info_dataset[f"{metric}/{split}"] = f"{'{:.4f}'.format(file[metric].mean())} ({'{:.4f}'.format(file[metric].std())})"
    data.append(info_dataset.copy())

df = pd.DataFrame(data)
df.to_excel(file_name + '.xlsx', index=False)
