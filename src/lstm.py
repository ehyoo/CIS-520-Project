import torch 
import torch.nn as nn 
import numpy as np
import torch.optim as optim
from allennlp.data.tokenizers import Token
from typing import *
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, ArrayField 
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from overrides import overrides
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.training.metrics import F1Measure, BooleanAccuracy, CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from nltk.tokenize import word_tokenize
from allennlp.nn import util as nn_util
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from torch.utils.data import random_split
from scipy.special import expit # the sigmoid function
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from allennlp.data.iterators import BasicIterator
from allennlp.common.params import Params
from allennlp.data.token_indexers import SingleIdTokenIndexer
import time
import math 
class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
        
config = Config(
    testing=True,
    seed=1,
    batch_size=128,
    lr=1e-5, #learning rate
    epochs=50,
    hidden_sz=64, 
    max_seq_len=100, # necessary to limit memory usage
    max_vocab_size=100000,
)

torch.manual_seed(1)
class lstmDatasetReader(DatasetReader):
    """
    DatasetReader for tagging data, one sentence per line, like
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
        max_seq_len: Optional[int]=config.max_seq_len) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers
        self.max_seq_len = max_seq_len

    def text_to_instance(self, tokens: List[Token], labels: int = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}
        fields["labels"] = ArrayField(np.array([labels]))

        return Instance(fields)
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                label = line.split('\t')[0]
                if label == "label":
                    continue
                label = int(label)
                sentence = line.split('\t')[1]
                if(len(sentence) == 0):
                    continue
                yield self.text_to_instance([Token(word) for word in word_tokenize(sentence)], label)

class LSTM_Model(Model): 
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 out_size: int=1) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=self.encoder.get_output_dim(),
                                          out_features=1)                             
        self.f1 = F1Measure(positive_label = 1) #index
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([4.21])) #negative/positive samples

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                labels: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        prob = expit(tag_logits.detach().cpu().numpy())
        zero_prob = 1 - prob
        all_pred = np.append(zero_prob, prob, 1)
        predictions_prob = torch.from_numpy(all_pred)
        self.f1(predictions_prob, labels) # wants index of the measure 
        output["loss"] = self.loss(tag_logits, labels)
        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"f1": self.f1.get_metric(reset=reset)[2],}


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

token_indexer = SingleIdTokenIndexer()
reader = lstmDatasetReader(token_indexers={"tokens": token_indexer})
full_dataset = reader.read("pol_train_semibalanced.csv")
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, validation_dataset = random_split(full_dataset, [train_size, test_size])
test_dataset = reader.read("pol_test_semibalanced.csv")

vocab = Vocabulary.from_instances(train_dataset + validation_dataset , max_vocab_size=config.max_vocab_size)
token_embedder = Embedding.from_params(vocab=vocab, params=Params({'pretrained_file':'glove.twitter.27B.50d.txt',
                                         'embedding_dim' :50})                         )
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedder})



#Iterrator: batching data + preparing it for input
from allennlp.data.iterators import BucketIterator

iterator = BucketIterator(batch_size=config.batch_size, 
                          sorting_keys=[("tokens", "num_tokens")],max_instances_in_memory=512)
iterator.index_with(vocab)

lstm = PytorchSeq2VecWrapper(nn.LSTM(word_embeddings.get_output_dim(), config.hidden_sz,bidirectional=True, batch_first=True, dropout=.25))

model = LSTM_Model(word_embeddings, lstm, 2)
optimizer = optim.SGD(model.parameters(), lr =config.lr)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  validation_metric="+f1",
                  patience=10,
                  num_epochs=config.epochs,
                  cuda_device= -1,)
start = time.time()
trainer.train()
end = time.time()
print(time_since(end))
# iterate over the dataset without changing its order
seq_iterator = BasicIterator(config.batch_size, max_instances_in_memory = 512)
seq_iterator.index_with(vocab)

predictor = Predictor(model, seq_iterator)
test_preds, labels = predictor.predict(test_dataset) 
accuracy_test = accuracy_score(test_preds,labels) 
f1_test = f1_score(test_preds,labels)
precision_test = precision_score(test_preds,labels)
recall_test = recall_score(test_preds,labels)
print("Accuracy score: {:.4f}, F1 Score: {:.4f}, Precision: {:.4f}, Recall: {:.4f} ".format(accuracy_test, f1_test, precision_test, recall_test))
save_file = "model_v6.th"
with open(save_file, 'wb') as f:
    torch.save(model.state_dict(), f)
