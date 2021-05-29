import json
from tqdm import tqdm
import os
import numpy as np
from transformers import BertTokenizer, AdamW, BertTokenizerFast
import torch
from model import ObjectModel, SubjectModel

GPU_NUM = 0

device = torch.device(f'cuda:{GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')

vocab = {}
with open('bert/vocab.txt', encoding='utf_8')as file:
    for l in file.readlines():
        vocab[len(vocab)] = l.strip()


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [[s, p, o],[s, p, o]]}
    """
    with open(filename, encoding='utf-8') as f:
        json_list = json.load(f)
    return json_list


# 加载数据集
train_data = load_data('data/train.json')
valid_data = load_data('data/dev.json')

tokenizer = BertTokenizerFast.from_pretrained('bert')

with open('data/schemas.json', encoding='utf-8') as f:
    json_list = json.load(f)
    id2predicate = json_list[0]
    predicate2id = json_list[1]


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def sequence_padding(inputs, length=None, padding=0, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        if mode == 'post':
            pad_width[0] = (0, length - len(x))
        elif mode == 'pre':
            pad_width[0] = (length - len(x), 0)
        else:
            raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)


def data_generator(data, batch_size=3):
    batch_input_ids, batch_attention_mask = [], []
    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
    texts = []
    for i, d in enumerate(data):
        text = d['text']

        texts.append(text)
        encoding = tokenizer(text=text)
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        # 整理三元组 {s: [(o, p)]}
        spoes = {}
        for s, p, o in d['spo_list']:
            # cls x x x sep
            s_encoding = tokenizer(text=s).input_ids[1:-1]
            o_encoding = tokenizer(text=o).input_ids[1:-1]
            # 找对应的s与o的起始位置
            s_idx = search(s_encoding, input_ids)
            o_idx = search(o_encoding, input_ids)

            p = predicate2id[p]

            if s_idx != -1 and o_idx != -1:
                s = (s_idx, s_idx + len(s_encoding) - 1)
                o = (o_idx, o_idx + len(o_encoding) - 1, p)
                if s not in spoes:
                    spoes[s] = []
                spoes[s].append(o)
        if spoes:
            # subject标签
            subject_labels = np.zeros((len(input_ids), 2))
            for s in spoes:
                # 注意要+1，因为有cls符号
                subject_labels[s[0], 0] = 1
                subject_labels[s[1], 1] = 1
            # 一个s对应多个o时，随机选一个subject
            start, end = np.array(list(spoes.keys())).T
            start = np.random.choice(start)
            # end = np.random.choice(end[end >= start])
            end = end[end >= start][0]
            subject_ids = (start, end)
            # 对应的object标签
            object_labels = np.zeros((len(input_ids), len(predicate2id), 2))
            for o in spoes.get(subject_ids, []):
                object_labels[o[0], o[2], 0] = 1
                object_labels[o[1], o[2], 1] = 1
            # 构建batch
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_subject_labels.append(subject_labels)
            batch_subject_ids.append(subject_ids)
            batch_object_labels.append(object_labels)
            if len(batch_subject_labels) == batch_size or i == len(data) - 1:
                batch_input_ids = sequence_padding(batch_input_ids)
                batch_attention_mask = sequence_padding(batch_attention_mask)
                batch_subject_labels = sequence_padding(batch_subject_labels)
                batch_subject_ids = np.array(batch_subject_ids)
                batch_object_labels = sequence_padding(batch_object_labels)
                yield [
                          torch.from_numpy(batch_input_ids).long(), torch.from_numpy(batch_attention_mask).long(),
                          torch.from_numpy(batch_subject_labels), torch.from_numpy(batch_subject_ids),
                          torch.from_numpy(batch_object_labels)
                      ]
                batch_input_ids, batch_attention_mask = [], []
                batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []


if os.path.exists('graph_model.bin'):
    print('load model')
    model = torch.load('graph_model.bin').to(device)
    subject_model = model.encoder
else:
    subject_model = SubjectModel.from_pretrained('./bert')
    subject_model.to(device)

    model = ObjectModel(subject_model)
    model.to(device)

train_loader = data_generator(train_data, batch_size=8)

optim = AdamW(model.parameters(), lr=5e-5)
loss_func = torch.nn.BCELoss()

model.train()


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """

    def __init__(self, spo):
        self.spox = (
            spo[0],
            spo[1],
            spo[2],
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def train_func():
    train_loss = 0
    pbar = tqdm(train_loader)
    for step, batch in enumerate(pbar):
        optim.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        subject_labels = batch[2].to(device)
        subject_ids = batch[3].to(device)
        object_labels = batch[4].to(device)
        subject_out, object_out = model(input_ids, subject_ids.float(), attention_mask)
        subject_out = subject_out * attention_mask.unsqueeze(-1)
        object_out = object_out * attention_mask.unsqueeze(-1).unsqueeze(-1)

        subject_loss = loss_func(subject_out, subject_labels.float())
        object_loss = loss_func(object_out, object_labels.float())

        # subject_loss = torch.mean(subject_loss, dim=2)
        # subject_loss = torch.sum(subject_loss * attention_mask) / torch.sum(attention_mask)

        loss = subject_loss + object_loss

        train_loss += loss.item()
        loss.backward()
        optim.step()

        pbar.update()
        pbar.set_description(f'train loss:{loss.item()}')

        if step % 1000 == 0 and step != 0:
            torch.save(model, 'graph_model.bin')
            with torch.no_grad():
                # texts = ['如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈',
                #          '茶树茶网蝽，Stephanitis chinensis Drake，属半翅目网蝽科冠网椿属的一种昆虫',
                #          '爱德华·尼科·埃尔南迪斯（1986-），是一位身高只有70公分哥伦比亚男子，体重10公斤，只比随身行李高一些，2010年获吉尼斯世界纪录正式认证，成为全球当今最矮的成年男人']
                X, Y, Z = 1e-10, 1e-10, 1e-10
                pbar = tqdm()
                for data in valid_data[0:100]:
                    spo = []
                    # for text in texts:
                    text = data['text']
                    spo_ori = data['spo_list']
                    en = tokenizer(text=text, return_tensors='pt')
                    _, subject_preds = subject_model(en.input_ids.to(device), en.attention_mask.to(device))
                    # !!!
                    subject_preds = subject_preds.cpu().data.numpy()
                    start = np.where(subject_preds[0, :, 0] > 0.6)[0]
                    end = np.where(subject_preds[0, :, 1] > 0.5)[0]

                    subjects = []
                    for i in start:
                        j = end[end >= i]
                        if len(j) > 0:
                            j = j[0]
                            subjects.append((i, j))
                    # print(subjects)
                    if subjects:
                        for s in subjects:
                            index = en.input_ids.cpu().data.numpy().squeeze(0)[s[0]:s[1] + 1]
                            subject = ''.join([vocab[i] for i in index])
                            # print(subject)

                            _, object_preds = model(en.input_ids.to(device),
                                                    torch.from_numpy(np.array([s])).float().to(device),
                                                    en.attention_mask.to(device))
                            object_preds = object_preds.cpu().data.numpy()
                            for object_pred in object_preds:
                                start = np.where(object_pred[:, :, 0] > 0.2)
                                end = np.where(object_pred[:, :, 1] > 0.2)
                                for _start, predicate1 in zip(*start):
                                    for _end, predicate2 in zip(*end):
                                        if _start <= _end and predicate1 == predicate2:
                                            index = en.input_ids.cpu().data.numpy().squeeze(0)[_start:_end + 1]
                                            object = ''.join([vocab[i] for i in index])
                                            predicate = id2predicate[str(predicate1)]
                                            # print(object, '\t', predicate)
                                            spo.append([subject, predicate, object])
                    print(spo)
                    # 预测结果
                    R = set([SPO(_spo) for _spo in spo])
                    # 真实结果
                    T = set([SPO(_spo) for _spo in spo_ori])
                    # R = set(spo_ori)
                    # T = set(spo)
                    # 交集
                    X += len(R & T)
                    Y += len(R)
                    Z += len(T)
                    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
                    pbar.update()
                    pbar.set_description(
                        'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
                    )
                pbar.close()
                print('f1:', f1, 'precision:', precision, 'recall:', recall)


for epoch in range(100):
    print('************start train************')
    # 训练
    train_func()
    # min_loss = float('inf')
    # dev_loss = dev_func()
    # if min_loss > dev_loss:
    #     min_loss = dev_loss
    #     torch.save(model,'model.p')



