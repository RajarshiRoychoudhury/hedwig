import imp
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.onnx

from common.evaluate import EvaluatorFactory
from common.train import TrainerFactory
from datasets.aapd import AAPD
from datasets.imdb import IMDB
from datasets.reuters import ReutersBOW
from datasets.twenty_news import TwentyNews
from datasets.ohsumed import OHSUMED
from datasets.r8 import R8
from datasets.r52 import R52
from datasets.trec6 import TREC6
from datasets.yelp2014 import Yelp2014
from datasets.ag_news import AGNews
from datasets.yahoo_answers import YahooAnswers
from datasets.yelp_review_polarity import YelpReviewPolarity
from datasets.imdb_torchtext import IMDBTorchtext
from datasets.sogou_news import SogouNews
from datasets.dbpedia import DBpedia
from datasets.shit_plos_classification import SHIT_PLOS_CLASSIFICATION
from datasets.shit_plos_regression import SHIT_PLOS_REGRESSION
from models.fasttext.args import get_args
from models.fasttext.model import FastText


class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].uniform_(-0.25, 0.25)
        return cls.cache[size_tup]


def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, batch_size, device, is_multilabel):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device)
    if hasattr(saved_model_evaluator, 'is_multilabel'):
        saved_model_evaluator.is_multilabel = is_multilabel

    scores, metric_names = saved_model_evaluator.get_scores()
    print('Evaluation metrics for', split_name)
    print(metric_names)
    print(scores)


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not args.cuda:
        args.gpu = -1

    if torch.cuda.is_available() and args.cuda:
        print('Note: You are using GPU for training')
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        args.gpu = torch.device('cuda:%d' % args.gpu)

    if torch.cuda.is_available() and not args.cuda:
        print('Warning: Using CPU for training')

    dataset_map = {
        'AAPD': AAPD,
        'IMDB': IMDB,
        'Yelp2014': Yelp2014,
        'AG_NEWS': AGNews,
        'DBpedia': DBpedia,
        'IMDB_torchtext': IMDBTorchtext,
        'SogouNews': SogouNews,
        'YahooAnswers': YahooAnswers,
        'YelpReviewPolarity': YelpReviewPolarity,
        'TwentyNews': TwentyNews,
        'OHSUMED': OHSUMED,
        'R8': R8,
        'R52': R52,
        'TREC6': TREC6,
        'SHIT_PLOS_CLASSIFICATION': SHIT_PLOS_CLASSIFICATION,
        'SHIT_PLOS_REGRESSION':  SHIT_PLOS_REGRESSION
    }

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')
    else:
        dataset_class = dataset_map[args.dataset]
        train_iter, dev_iter, test_iter = dataset_map[args.dataset].iters(args.data_dir, args.word_vectors_file,
                                                                          args.word_vectors_dir,
                                                                          batch_size=args.batch_size, device=args.gpu,
                                                                          unk_init=UnknownWordVecCache.unk)

    config = deepcopy(args)
    if args.regression:
        config.regression = True
    config.dataset = train_iter.dataset
    config.target_class = train_iter.dataset.NUM_CLASSES
    config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)

    print('Dataset:', args.dataset)
    print('No. of target classes:', train_iter.dataset.NUM_CLASSES)
    print('No. of train instances', len(train_iter.dataset))
    print('No. of dev instances', len(dev_iter.dataset))
    print('No. of test instances', len(test_iter.dataset))

    if args.resume_snapshot:
        if args.cuda:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
    else:
        model = FastText(config)
        if args.cuda:
            model.cuda()

    if not args.trained_model:
        save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME)
        os.makedirs(save_path, exist_ok=True)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)

    train_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, train_iter, args.batch_size, args.gpu)
    test_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, test_iter, args.batch_size, args.gpu)
    dev_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, dev_iter, args.batch_size, args.gpu)

    if hasattr(train_evaluator, 'is_multilabel'):
        train_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
    if hasattr(test_evaluator, 'is_multilabel'):
        test_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
    if hasattr(dev_evaluator, 'is_multilabel'):
        dev_evaluator.is_multilabel = dataset_class.IS_MULTILABEL

    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_every,
        'patience': args.patience,
        'model_outfile': args.save_path,
        'is_multilabel': dataset_class.IS_MULTILABEL,
        'regression': args.regression
    }

    trainer = TrainerFactory.get_trainer(args.dataset, model, None, train_iter, trainer_config, train_evaluator, test_evaluator, dev_evaluator)

    if not args.trained_model:
        trainer.train(args.epochs)
    else:
        if args.cuda:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage)

    # Calculate dev and test metrics
    if hasattr(trainer, 'snapshot_path'):
        model = torch.load(trainer.snapshot_path)

    evaluate_dataset('dev', dataset_map[args.dataset], model, None, dev_iter, args.batch_size,
                     is_multilabel=dataset_class.IS_MULTILABEL,
                     device=args.gpu)
    evaluate_dataset('test', dataset_map[args.dataset], model, None, test_iter, args.batch_size,
                     is_multilabel=dataset_class.IS_MULTILABEL,
                     device=args.gpu)
