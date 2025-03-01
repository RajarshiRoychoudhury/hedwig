import logging
import os
import random
from copy import deepcopy

import numpy as np
import torch

from common.evaluate import EvaluatorFactory
from common.train import TrainerFactory
from datasets.aapd import AAPD
from datasets.imdb import IMDB
from datasets.reuters import Reuters
from datasets.yelp2014 import Yelp2014
from datasets.ag_news import AGNews
from datasets.dbpedia import DBpedia
from datasets.imdb_torchtext import IMDBTorchtext
from datasets.sogou_news import SogouNews
from datasets.yahoo_answers import YahooAnswers
from datasets.yelp_review_polarity import YelpReviewPolarity
from datasets.twenty_news import TwentyNews
from datasets.ohsumed import OHSUMED
from datasets.r8 import R8
from datasets.r52 import R52
from datasets.trec6 import TREC6
from datasets.shit_plos_classification import SHIT_PLOS_CLASSIFICATION
from datasets.shit_plos_regression import SHIT_PLOS_REGRESSION
from models.reg_lstm.args import get_args
from models.reg_lstm.model import RegLSTM


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


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


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
    logger = get_logger()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('Note: You are using GPU for training')
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        print('Warning: Using CPU for training')

    dataset_map = {
        'Reuters': Reuters,
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
        iters = dataset_class.iters(args.data_dir,
                                    args.word_vectors_file,
                                    args.word_vectors_dir,
                                    batch_size=args.batch_size,
                                    device=args.gpu,
                                    unk_init=UnknownWordVecCache.unk)

        # Some datasets (e.g. AG_NEWS) only have train and test splits
        if len(iters) == 2:
            train_iter, test_iter = iters
            dev_iter = test_iter
        else:
            train_iter, dev_iter, test_iter = iters

    config = deepcopy(args)
    config.dataset = train_iter.dataset
    config.target_class = train_iter.dataset.NUM_CLASSES
    config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)

    print('Dataset:', args.dataset)
    print('No. of target classes:', train_iter.dataset.NUM_CLASSES)
    print('No. of train instances', len(train_iter.dataset))
    print('No. of dev instances', len(dev_iter.dataset) if dev_iter else 0)
    print('No. of test instances', len(test_iter.dataset))

    if args.resume_snapshot:
        if args.cuda:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
    else:
        model = RegLSTM(config)
        if args.cuda:
            model.cuda()

    if not args.trained_model:
        save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME)
        os.makedirs(save_path, exist_ok=True)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)

    train_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, train_iter, args.batch_size, args.gpu)
    test_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, test_iter, args.batch_size, args.gpu)
    dev_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, dev_iter, args.batch_size, args.gpu)

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
        'logger': logger,
        'is_multilabel': dataset_class.IS_MULTILABEL,
        'regression': args.regression
    }

    trainer = TrainerFactory.get_trainer(args.dataset, model, None, train_iter, trainer_config, train_evaluator, test_evaluator, dev_evaluator)

    if not args.trained_model:
        trainer.train(args.epochs)
        model = torch.load(trainer.snapshot_path)
    else:
        if args.cuda:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage)

    if model.beta_ema > 0:
        old_params = model.get_params()
        model.load_ema_params()

    # Calculate dev and test metrics
    if dev_iter:
        evaluate_dataset('dev', dataset_class, model, None, dev_iter, args.batch_size,
                         is_multilabel=dataset_class.IS_MULTILABEL,
                         device=args.gpu)

    evaluate_dataset('test', dataset_class, model, None, test_iter, args.batch_size,
                     is_multilabel=dataset_class.IS_MULTILABEL,
                     device=args.gpu)

    if model.beta_ema > 0:
        model.load_params(old_params)
