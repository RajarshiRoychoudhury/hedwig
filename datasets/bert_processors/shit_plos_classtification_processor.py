import os

from datasets.bert_processors.abstract_processor import BertProcessor, InputExample


class SHIT_PLOS_CLASSIFICATION_Processor(BertProcessor):
    NAME = 'SHIT_PLOS_CLASSIFICATION'
    NUM_CLASSES = 3
    IS_MULTILABEL = False

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir,'SHIT_PLOS_CLASSIFICATION', 'train.tsv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'SHIT_PLOS_CLASSIFICATION', 'dev.tsv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'SHIT_PLOS_CLASSIFICATION', 'test.tsv')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
