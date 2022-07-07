import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from datasets.bow_processors.abstract_processor import StreamingSparseDataset

# Suppress warnings from sklearn.metrics
warnings.filterwarnings('ignore')


class BagOfWordsRegressionEvaluator(object):
    def __init__(self, model, vectorizer, processor, args, split='dev'):
        self.args = args
        self.model = model
        self.processor = processor
        self.vectorizer = vectorizer

        if split == 'test':
            eval_examples = self.processor.get_test_examples(args.data_dir)
        else:
            eval_examples = self.processor.get_dev_examples(args.data_dir)

        self.eval_features = vectorizer.transform([x.text for x in eval_examples])
        self.eval_labels = [[float(x) for x in document.label] for document in eval_examples]

    def get_scores(self, silent=False):
        self.model.eval()
        eval_data = StreamingSparseDataset(self.eval_features, self.eval_labels)
        eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=self.args.batch_size)

        total_loss = 0
        nb_eval_steps = 0
        target_labels = list()
        predicted_labels = list()

        for features, labels in tqdm(eval_dataloader, desc="Evaluating", disable=silent):
            features = features.to(self.args.device)
            labels = labels.to(self.args.device)

            with torch.no_grad():
                logits = self.model(features)

            if self.args.n_gpu > 1:
                logits = logits.view(labels.size())

            predicted_labels.extend(logits.cpu().detach().numpy())
            target_labels.extend(labels.cpu().detach().numpy())
            loss = F.smooth_l1_loss(logits, labels)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            total_loss += loss.item()
            nb_eval_steps += 1

        avg_loss = total_loss / nb_eval_steps
        mean_squared_error = metrics.mean_squared_error(target_labels, predicted_labels)
        mean_absolute_error = metrics.mean_absolute_error(target_labels, predicted_labels)
        return [mean_squared_error , mean_absolute_error , avg_loss], ['mean_squared_error' , 'mean_absolute_error','avg_loss']
