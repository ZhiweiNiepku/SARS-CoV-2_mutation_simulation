# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""
import numpy as np
from mindspore.mindrecord import FileWriter
import os
import argparse
import collections
import glob
import time
import gc

cv_schema_json = {"input_ids": {"type": "int32", "shape": [-1]}, "input_mask": {"type": "int32", "shape": [-1]},
                  "masked_lm_positions": {"type": "int32", "shape": [-1]},
                  "masked_lm_ids": {"type": "int32", "shape": [-1]},
                  "masked_lm_weights": {"type": "float32", "shape": [-1]}}

map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}


def numpy_mask_tokens(inputs, mlm_probability=0.80, mask_token_id=9, token_length=5):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    labels = np.copy(inputs)
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = np.full(labels.shape, mlm_probability)
    #     for value in range(5):
    probability_matrix[inputs < 5] = 0
    # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
    masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)
    masked_lm_positions = np.where(masked_indices == True)[0]

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
    inputs[indices_replaced] = mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced)
    random_words = np.random.randint(
        low=0, high=token_length, size=np.count_nonzero(indices_random), dtype=np.int64
    )
    inputs[indices_random] = random_words
    masked_lm_ids = labels[masked_lm_positions]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, masked_lm_positions, masked_lm_ids


def get_mask(seq, seq_length, max_lm, mask_percent):
    seq = [map_dict['[CLS]']] + [map_dict[x] for x in seq] + [map_dict['[SEP]']]
    seq = np.array(seq)
    input_id, masked_lm_position, masked_lm_id = numpy_mask_tokens(inputs=seq, mlm_probability=mask_percent,
                                                                   mask_token_id=map_dict['[MASK]'],
                                                                   token_length=len(map_dict) - 1)
    input_mask = np.pad(np.ones(input_id.shape, dtype=np.int32), (0, seq_length - len(seq)),
                        constant_values=map_dict['[PAD]'])
    input_id = np.pad(input_id.astype(np.int32), (0, seq_length - len(seq)),
                      constant_values=map_dict['[PAD]'])
    masked_lm_id = masked_lm_id[:max_lm]
    masked_lm_position = masked_lm_position[:max_lm]
    masked_lm_weight = np.pad(np.ones(masked_lm_id.shape, dtype=np.float32), (0, max_lm - len(masked_lm_id)),
                              constant_values=map_dict['[PAD]'])
    masked_lm_position = np.pad(masked_lm_position.astype(np.int32), (0, max_lm - len(masked_lm_position)),
                                constant_values=map_dict['[PAD]'])
    masked_lm_id = np.pad(masked_lm_id.astype(np.int32), (0, max_lm - len(masked_lm_id)),
                          constant_values=map_dict['[PAD]'])
    return input_id, input_mask, masked_lm_position, masked_lm_id, masked_lm_weight


def process(data_file, seq_length, max_ml, mask_percent):
    actual_length = FLAGS["max_seq_length"] - 2
    data = []
    f = open(data_file, "r")
    for line in f:
        line = line.strip().upper()
        seqs = [line[i:i + actual_length] for i in range(0, len(line), actual_length)]
        for seq in seqs:
            item = get_mask(seq,
                            seq_length=seq_length,
                            max_lm=max_ml,
                            mask_percent=mask_percent)
            features = collections.OrderedDict()
            features["input_ids"] = item[0]
            features["input_mask"] = item[1]
            features["masked_lm_positions"] = item[2]
            features["masked_lm_ids"] = item[3]
            features["masked_lm_weights"] = item[4]
            data.append(features)
            del features
    f.close()
    md_name = os.path.join(save_dir, os.path.basename(data_file) + '.mindrecord')
    print(">>>>>>>>>>>>>>>>>save data:", md_name)
    writer = FileWriter(file_name=md_name, shard_num=1)
    writer.add_schema(cv_schema_json, "train_schema")
    writer.write_raw_data(data)
    writer.commit()
    del writer
    del data
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Total data processing")
    parser.add_argument('--data_url',
                        type=str,
                        required=True,
                        default=None,
                        help="data dir")
    parser.add_argument('--save_dir',
                        type=str,
                        default="mindrecord data dir",
                        help="save dir")

    args = parser.parse_args()
    FLAGS = {}

    FLAGS["max_seq_length"] = 256

    FLAGS["max_predictions_per_seq"] = 38

    FLAGS["random_seed"] = 10000

    FLAGS["dupe_factor"] = 1

    FLAGS["masked_lm_prob"] = 0.15

    FLAGS["short_seq_prob"] = 0.1
    np.random.seed(FLAGS["random_seed"])
    save_dir = args.save_dir
    data_path = args.data_url
    os.makedirs(data_path, exist_ok=True)
    files = glob.glob(data_path + "/*.txt")
    os.makedirs(save_dir, exist_ok=True)
    for file in files:
        process(data_file=file, seq_length=FLAGS["max_seq_length"], max_ml=FLAGS["max_predictions_per_seq"],
                mask_percent=FLAGS["masked_lm_prob"])
