from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch


class GraphModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # self.layer_norm = nn.LayerNorm([16, 768])
        self.dense = nn.Linear(config.hidden_size, 2)
        # self.layer_norm = nn.LayerNorm()
        self.dense_subject_position = nn.Linear(2, config.hidden_size)
        self.dense_object = nn.Linear(config.hidden_size, 2)

    def forward(self,
                input_ids,
                subject_position,
                attention_mask=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        # output = self.layer_norm(output)
        output = self.dense(output[0])
        subject_out = torch.sigmoid(output)

        subject_position = self.dense_subject_position(subject_position)
        object_out = subject_out + subject_position

        return subject_out, object_out


class SubjectModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, 2)

    def forward(self,
                input_ids,
                attention_mask=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        subject_out = self.dense(output[0])
        subject_out = torch.sigmoid(subject_out)

        return output[0], subject_out


class ObjectModel(nn.Module):
    def __init__(self, subject_model):
        super().__init__()
        self.encoder = subject_model
        self.dense_subject_position = nn.Linear(2, 768)
        self.dense_object = nn.Linear(768, 49 * 2)

    def forward(self,
                input_ids,
                subject_position,
                attention_mask=None):
        output, subject_out = self.encoder(input_ids, attention_mask)

        subject_position = self.dense_subject_position(subject_position).unsqueeze(1)
        object_out = output + subject_position
        object_out = self.dense_object(object_out)
        object_out = torch.reshape(object_out, (object_out.shape[0], object_out.shape[1], 49, 2))
        object_out = torch.sigmoid(object_out)
        object_out = torch.pow(object_out, 4)
        return subject_out, object_out
