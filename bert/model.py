import torch
from transformers import AutoModel, BertConfig


class PhoBertFineTune(torch.nn.Module):
    config = BertConfig.from_pretrained('vinai/phobert-base', output_hidden_states=True)

    def __init__(self):
        super(PhoBertFineTune, self).__init__()
        self.l1 = AutoModel.from_pretrained("vinai/phobert-base", config=self.config)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0)
        self.classifier = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
