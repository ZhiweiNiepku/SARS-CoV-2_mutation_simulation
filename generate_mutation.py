# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Bert evaluation script.
"""
from datetime import datetime
import mindspore.communication.management as D
from mindspore.context import ParallelMode
import mindspore as ms
from src.model_utils.device_adapter import get_device_id, get_device_num
from mindspore.ops import operations as P
import os
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset as ds
from mindspore.common import dtype as mstype
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model_utils.config import config as cfg, bert_net_cfg
from src.bert_for_pre_training import BertPreTraining
import numpy as np

map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}


def numpy_mask_tokens(inputs, mask_token_id=4):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    labels = np.copy(inputs)
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = probility_mutation.copy()
    # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
    masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)
    masked_lm_positions = np.where(masked_indices == True)[0]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    inputs[masked_lm_positions] = mask_token_id
    return inputs, masked_lm_positions


class SarsDataset:

    def __init__(self, num_samples, rbd_seq, seq_length, max_ml, mask_percent):
        self.max_len = seq_length
        self.num_samples = num_samples
        self.mask_percent = mask_percent
        self.max_lm = max_ml
        self.rbd_seq = rbd_seq
        self.rbd_id = np.array([map_dict['[CLS]']] + [map_dict[x] for x in self.rbd_seq] + [map_dict['[SEP]']])
        self.count = 0

    def __getitem__(self, index):
        seq = self.rbd_id.copy()
        input_id, masked_lm_position = numpy_mask_tokens(inputs=seq, mask_token_id=map_dict['[MASK]'])
        input_id = np.pad(seq.astype(np.int32), (0, self.max_len - len(seq)), constant_values=map_dict['[PAD]'])
        input_mask = np.pad(np.ones(seq.shape, dtype=np.int32), (0, self.max_len - len(seq)),
                            constant_values=map_dict['[PAD]'])
        masked_lm_position = masked_lm_position[:self.max_lm]
        masked_lm_position = np.pad(masked_lm_position.astype(np.int32), (0, self.max_lm - len(masked_lm_position)),
                                    constant_values=map_dict['[PAD]'])
        return input_id, input_mask, masked_lm_position

    def __len__(self):
        return self.num_samples

    def __del__(self):
        del self.rbd_id


def create_txt_dataset(num_samples, rbd_seq, device_num=1, rank=0, batch_size=2):
    """
    Create dataset

    Inputs:
        batch_size: batch size
        data_path: path of your MindRecord files
        device_num: total device number
        rank: current rank id
        drop: whether drop remainder
        eod_reset: whether enable position reset and attention mask reset
        column_name: the column name of the mindrecord file. Default is input_ids
        epoch: The repeat times of the dataset
    Returns:
        dataset_restore: the dataset for training or evaluating
    """
    dataset_generator = SarsDataset(num_samples=num_samples, rbd_seq=rbd_seq, seq_length=max_len, max_ml=max_lm,
                                    mask_percent=0.3)
    data_set = ds.GeneratorDataset(dataset_generator,
                                   column_names=["input_ids",
                                                 "input_mask",
                                                 "masked_lm_positions"],
                                   shuffle=False,
                                   num_shards=device_num,
                                   shard_id=rank)

    data_set = data_set.batch(batch_size, drop_remainder=True)
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_positions")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    return data_set


def bulid_model():
    '''
    Predict function
    '''
    net = BertPreTraining(bert_net_cfg, is_training=False, use_one_hot_embeddings=False)
    net.set_train(False)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    return net


if __name__ == "__main__":
    now = datetime.now()
    dir_name = now.strftime("%d-%m-%Y")
    cfg.device_id = get_device_id()
    cfg.device_num = get_device_num()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=cfg.device_id)
    ms.context.set_context(variable_memory_max_size="30GB")
    D.init()
    device_num = cfg.device_num
    rank = cfg.device_id % device_num
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                      device_num=device_num)

    # hyperparameters for generation
    print_steps = 1000
    Number = device_num * int(cfg.generate_number)  # total generated number
    mutation_save_path = "result"
    os.makedirs(mutation_save_path, exist_ok=True)
    # load mutation probability distribution
    probility_mutation = np.load(
        "probility_mutation_203.npy")  # with a start token probility=0 and  with a end token probility=0
    id2tag = {val: key for key, val in map_dict.items()}
    rbd_sequence_dict = {
        "Wild_type": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST",
        "BA2": "NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST"
    }

    max_len = 203  # max length of sequence, including a start token and a ending token, RBD length=201.
    max_lm = 5  # max mutation amino acid number
    topk = cfg.topk  # Randomly pick one from topk default=10
    batch_size = cfg.batch_size
    base_rbd_seq = rbd_sequence_dict[cfg.rbd_name]   # template sequence for generation
    rbd_id = np.array([map_dict['[CLS]']] + [map_dict[x] for x in base_rbd_seq] + [map_dict['[SEP]']])
    # load model
    ckpt_path = cfg.load_checkpoint_path
    net = bulid_model()
    dataset = create_txt_dataset(num_samples=Number, rbd_seq=base_rbd_seq, device_num=device_num, rank=rank,
                                 batch_size=batch_size)
    output_seq = []
    for i, inputs in enumerate(dataset):
        out = net(*inputs)
        _, indices = P.TopK(sorted=True)(out, topk)
        indices = indices.asnumpy()
        for _ in range(20):
            # Randomly pick one from topk
            index_ran = np.random.randint(0, topk, size=(indices.shape[0]))
            predict_id = indices[range(indices.shape[0]), index_ran]
            predict_id_batch = predict_id.reshape(batch_size, -1)
            masked_position = inputs[-1].asnumpy()
            rbd_id_batch = np.repeat([rbd_id], batch_size, axis=0)
            # Avoid a X mutations
            predict_id_batch = predict_id_batch.reshape(-1)
            masked_position = masked_position.reshape(-1)
            x_index = np.where(predict_id_batch == map_dict['X'])
            masked_position[x_index] = 0
            # The original sequence is replaced with the mutated amino acid
            a = np.repeat([np.arange(batch_size)], max_lm, axis=0).transpose()
            rbd_id_batch[a.reshape(-1), masked_position] = predict_id_batch
            mutation = rbd_id_batch.copy()[:, 1:-1]
            for i in range(mutation.shape[0]):
                muta = mutation[i]
                sequence = "".join([id2tag[AA] for AA in list(muta)])
                output_seq.append(sequence)
    file_name = os.path.join(mutation_save_path, "mutation_base_on_{}_rank_{}.txt".format(base_rbd_seq, rank))
    f = open(file_name, "w")
    for seq in output_seq:
        f.write(seq + "\n")
    f.close()
