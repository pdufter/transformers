#from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from transformers.modeling_bert import BertModel
from transformers.modeling_utils import BertPreTrainedModel
#from typers.nnTyper import MLP
# from vis.overrides_bert_modelling import BertModel as BertModel_
import torch


class LinearPredictor(BertPreTrainedModel):
    def __init__(self, bert_config, o_dim=2):
        super(LinearPredictor, self).__init__(bert_config)
        self.bert = BertModel(bert_config)
        self.linear = torch.nn.Linear(768, o_dim)
        self.dropout = torch.nn.Dropout(0.1)
        self.apply(self.init_bert_weights)
    
    def forward(self):
        raise NotImplementedError


class BinarySentClsPredictor(LinearPredictor):
    def __init__(self, bert_config):
        super(BinarySentClsPredictor, self).__init__(bert_config, o_dim=2)
    
    def forward(self, inp_ids, inp_type_ids, inp_masks):
        # bert_layers, sent_pool_out, all_attentions for BertModel_
        bert_layers, sent_pool_out = self.bert(
            input_ids=inp_ids,
            token_type_ids=inp_type_ids,
            attention_mask=inp_masks,
            output_all_encoded_layers=False)
        sent_pool_out = self.dropout(sent_pool_out)
        sent_pool_out = self.linear(sent_pool_out)
        return sent_pool_out


class POSTagger(LinearPredictor):
    def __init__(self, bert_config, o_dim):
        super(POSTagger, self).__init__(bert_config, o_dim)
    
    def forward(self, inp_ids, inp_mask):
        bert_out, _ = self.bert(
            input_ids=inp_ids,
            token_type_ids=None,
            attention_mask=inp_mask,
            output_all_encoded_layers=False)
        bert_out = self.dropout(bert_out)
        bert_out = self.linear(bert_out)
        return bert_out


class WeightedPOSTagger(LinearPredictor):
    def __init__(self, bert_config, o_dim):
        super(WeightedPOSTagger, self).__init__(bert_config, o_dim)
        # i just initialize this with uniform for better interpretation
        self.layer_weights = torch.nn.Parameter(
            torch.tensor([1./12] * 12)) 
    
    def forward(self, inp_ids, inp_mask):
        bert_out, _ = self.bert(
            input_ids=inp_ids,
            token_type_ids=None,
            attention_mask=inp_mask,
            output_all_encoded_layers=True)
        bert_out = torch.stack(bert_out, dim=-1)
        weighted_out = torch.matmul(bert_out, self.layer_weights)
        out = self.linear(weighted_out)
        return out, bert_out