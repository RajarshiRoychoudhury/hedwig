import random
import time

import numpy as np
import torch
from transformers import AdamW, BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

from common.constants import *
from common.evaluators.bert_evaluator import BertEvaluator
from common.evaluators.bert_regression_evaluator import BertRegressionEvaluator
from common.trainers.bert_trainer import BertTrainer
from datasets.bert_processors.aapd_processor import AAPDProcessor
from datasets.bert_processors.agnews_processor import AGNewsProcessor
from datasets.bert_processors.imdb_processor import IMDBProcessor
from datasets.bert_processors.reuters_processor import ReutersProcessor
from datasets.bert_processors.sogou_processor import SogouProcessor
from datasets.bert_processors.sst_processor import SST2Processor
from datasets.bert_processors.yelp2014_processor import Yelp2014Processor
from datasets.bert_processors.shit_plos_classtification_processor import SHIT_PLOS_CLASSIFICATION_Processor
from datasets.bert_processors.shit_plos_regression_processor import SHIT_PLOS_REGRESSION_Processor
from models.bert.args import get_args
from models.bert.model import Model, getModelAndTokenizer


def evaluate_split(model, processor, tokenizer, args, split='dev'):
    if not args.regression:
        evaluator = BertEvaluator(model, processor, tokenizer, args, split)
        accuracy, precision, recall, f1, avg_loss = evaluator.get_scores(silent=True)[0]
        print('\n' + LOG_HEADER)
        print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss))
    else:
        evaluator = BertRegressionEvaluator(model, processor, tokenizer, args, split)
        mse, mle, avg_loss = evaluator.get_scores(silent=True)[0]
        print('\n' + LOG_REGRESSION_HEADER)
        print(LOG_REGRESSION_TEMPLATE.format(split.upper(), mse, mle, avg_loss))


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    print('Device:', str(device).upper())
    print('Number of GPUs:', n_gpu)
    print('FP16:', args.fp16)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    dataset_map = {
        'SST-2': SST2Processor,
        'Reuters': ReutersProcessor,
        'IMDB': IMDBProcessor,
        'AAPD': AAPDProcessor,
        'AGNews': AGNewsProcessor,
        'Yelp2014': Yelp2014Processor,
        'Sogou': SogouProcessor,
        'SHIT_PLOS_CLASSIFICATION': SHIT_PLOS_CLASSIFICATION_Processor,
        'SHIT_PLOS_REGRESSION':  SHIT_PLOS_REGRESSION_Processor
    }

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    if args.regression:
        args.num_labels = 1
    else:
        args.num_labels = dataset_map[args.dataset].NUM_CLASSES
    args.is_multilabel = dataset_map[args.dataset].IS_MULTILABEL

    if not args.trained_model:
        save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME)
        os.makedirs(save_path, exist_ok=True)

    args.is_hierarchical = False
    processor = dataset_map[args.dataset]()
    pretrained_vocab_path = PRETRAINED_VOCAB_ARCHIVE_MAP[args.model]
    tokenizer = BertTokenizer.from_pretrained(pretrained_vocab_path)

    train_examples = None
    num_train_optimization_steps = None
    if not args.trained_model:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs

    pretrained_model_path = args.model if os.path.isfile(args.model) else PRETRAINED_MODEL_ARCHIVE_MAP[args.model]
    _, model = getModelAndTokenizer(pretrained_model_path, num_labels=args.num_labels, regression=args.regression)
    # for param in model.parameters():
    #         param.requires_grad = False

    if args.fp16:
        model.half()
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    if not args.trained_model:
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install NVIDIA Apex for FP16 training")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.lr,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=0.01, correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_train_optimization_steps,
                                             num_warmup_steps=args.warmup_proportion * num_train_optimization_steps)

        trainer = BertTrainer(model, optimizer, processor, scheduler, tokenizer, args)
        trainer.train()
        model = torch.load(trainer.snapshot_path)

    else:
        _, model = getModelAndTokenizer(pretrained_model_path, num_labels=args.num_labels, regression=args.regression)
        model_ = torch.load(args.trained_model, map_location=lambda storage, loc: storage)
        state = {}
        for key in model_.state_dict().keys():
            new_key = key.replace("module.", "")
            state[new_key] = model_.state_dict()[key]
        model.load_state_dict(state)
        model = model.to(device)

    evaluate_split(model, processor, tokenizer, args, split='dev')
    evaluate_split(model, processor, tokenizer, args, split='test')

