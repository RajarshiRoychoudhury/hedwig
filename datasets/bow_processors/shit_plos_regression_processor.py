import os

from datasets.bow_processors.abstract_processor import BagOfWordsProcessor, InputExample


class SHIT_PLOS_REGRESSION_Processor(BagOfWordsProcessor):
    NAME = 'SHIT_PLOS_REGRESSION'
    NUM_CLASSES = 1
    VOCAB_SIZE = 66192
    IS_MULTILABEL = True

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir,'SHIT_PLOS_REGRESSION', 'train.tsv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'SHIT_PLOS_REGRESSION', 'dev.tsv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'SHIT_PLOS_REGRESSION', 'test.tsv')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = '%s-%s' % (set_type, i)
            examples.append(InputExample(guid=guid, text=line[0], label=line[1]))
        return examples