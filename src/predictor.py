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
from sklearn.metrics import roc_curve, auc
from scipy.special import expit # the sigmoid function
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from allennlp.data.iterators import BasicIterator
from allennlp.common.params import Params
from allennlp.data.token_indexers import SingleIdTokenIndexer
import matplotlib.pyplot as plt
import time
import math 
#Predictor: Given a saved model, we instanciate the model and classify testing data, calculate accuracy, 
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
    batch_size=64, #TODO: CHANGE, overfitting, inc batch size
    lr=1e-6,
    epochs=25,
    hidden_sz=64,
    max_seq_len=100, # necessary to limit memory usage
    max_vocab_size=100000,
)
sentences = []
torch.manual_seed(1)
training_f1 = []
#Model
class rnnDatasetReader(DatasetReader):
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
        fields["labels"] = ArrayField(np.array([labels])) #skip indexing?

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
                sentence = line.split('\t')[9].rstrip() + " <PAD> " + sentence 
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
                                          out_features=1) #out size: 2 x 2                                
        self.f1 = F1Measure(positive_label = 1) #index 
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([4.21]))

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

def tonp(tsr): return tsr.detach().cpu().numpy()

class Predictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device
        
    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch) # m x 1 : score for classifying? 
        return expit(out_dict["tag_logits"]) 
    
    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        labels = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                labels.append(batch["labels"])
                preds.append(self._extract_data(batch))
        return np.concatenate(preds, axis=0), np.concatenate(labels, axis=0)


token_indexer = SingleIdTokenIndexer()
reader = rnnDatasetReader(token_indexers={"tokens": token_indexer})
test_dataset = reader.read("/home/dkeren/Documents/Spring2019/CIS520/project/pol_test_semibalanced.csv")

vocab = Vocabulary.from_files("/tmp/vocabulary") #preloaded vocab, required to do lazy computations

token_embedder = Embedding.from_params(vocab=vocab, params=Params({'pretrained_file':'/home/dkeren/Documents/Spring2019/CIS520/project/glove.twitter.27B.50d.txt',
                                           'embedding_dim' :50})
                            )
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedder})

lstm = PytorchSeq2VecWrapper(nn.LSTM(word_embeddings.get_output_dim(),config.hidden_sz,bidirectional=True, batch_first=True))

save_file = "model_v12.th" ## models saved by lstm training
model2 = LSTM_Model(word_embeddings, lstm, 2)
with open(save_file, 'rb') as f:
    model2.load_state_dict(torch.load(f))

# iterate over the dataset without changing its order
seq_iterator = BasicIterator(config.batch_size)
seq_iterator.index_with(vocab)
predictor = Predictor(model2, seq_iterator)
prob, labels= predictor.predict(test_dataset) 
test_preds = 1*(prob > .525) #optimal threshold

#Evaluation
accuracy_test = accuracy_score(test_preds,labels) 
f1_test = f1_score(test_preds,labels)
precision_test = precision_score(test_preds,labels)
recall_test = recall_score(test_preds,labels)
matrix = confusion_matrix(labels, test_preds)
print(matrix)
print("Accuracy score: {:.4f}, F1 Score: {:.4f}, Precision: {:.4f}, Recall: {:.4f} ".format(accuracy_test, f1_test, precision_test, recall_test))
fpr,tpr,_ = roc_curve(labels,prob) 
roc_auc = auc(fpr, tpr)
print(fpr, tpr)
np.save("fpr", fpr)
np.save("tpr",tpr) 
plt.figure()
lw = 2

plt.plot(fpr, tpr, color='darkorange',
        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LSTM ROC curve')
plt.legend(loc="lower right")
plt.show()

