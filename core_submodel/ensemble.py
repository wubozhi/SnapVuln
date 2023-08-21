import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, average_precision_score

def deal_file(file):
    datas = pd.read_csv(file,usecols=['label','prob','predicted_label','file'])
    labels = [int(i) for i in datas['label']]
    predicted_labels = [int(i) for i in datas['predicted_label']]
    probs = [i for i in datas['prob']]
    files = [i.split("/")[-1] for i in datas['file']]
    samples = {'label':labels, 'prob':probs, 'predicted_label':predicted_labels, 'file':files}
    return samples

def compare_result(files):
    final = []
    file_samples = [deal_file(file) for file in files]
    all_files = []
    for file_sample in file_samples:
        all_files = all_files + file_sample['file']
    all_files = list(set(all_files))

    for file in all_files:
        labels = []
        predicteds = []
        for file_sample in file_samples:
            if file in file_sample['file']:
                index = file_sample['file'].index(file)
                label = file_sample['label'][index]
                predicted = file_sample['prob'][index]
                labels.append(label)
                predicteds.append(predicted)

        label = max(labels,key=labels.count)
        predicted = sum(predicteds)/len(predicteds)
        
        sample = {'label':label, 'prob':predicted, 'file':file}
        final.append(sample)
    return final


def evaluate_preds(final):
    labels = [result['label'] for result in final]
    probs = [result['prob'] for result in final]

    predicted_labels = []
    for i in probs:
        if i > 0.5:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    acc = accuracy_score(labels, predicted_labels)
    reports = classification_report(labels, predicted_labels, target_names=['0', '1'], output_dict=True)
    pre = reports['1']['precision']
    rec = reports['1']['recall']
    f1 = reports['1']['f1-score']
    metrics = {'Accuracy': acc, 'Precision': pre,
               'Recall': rec, 'F1': f1}
    # print(metrics)
    return metrics

def ensemble_result(files):
    final = compare_result(files)
    metrics = evaluate_preds(final)
    return metrics
