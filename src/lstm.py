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
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.training.metrics import F1Measure, BooleanAccuracy, CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from nltk.tokenize import word_tokenize
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.nn import util as nn_util
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from scipy.special import expit # the sigmoid function
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from allennlp.data.iterators import BasicIterator
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
    batch_size=128, #TODO: CHANGE
    lr=1e-5,
    epochs=100,
    hidden_sz=64,
    max_seq_len=100, # necessary to limit memory usage
)

USE_GPU = torch.cuda.is_available()
torch.manual_seed(1)
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
        self.loss = nn.BCEWithLogitsLoss()

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
        predictions_prob = torch.from_numpy(all_pred) #np.round(expit(class_logits.detach().cpu().numpy())))
        self.f1(predictions_prob, labels) # wants index of the measure 
        output["loss"] = self.loss(tag_logits, labels)
        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"f1": self.f1.get_metric(reset=reset)[2]}




def tonp(tsr): return tsr.detach().cpu().numpy()

class Predictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device
        
    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch) # m x 1 : score for classifying? 
        return np.round(expit(out_dict["tag_logits"]))
    
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

token_indexer = ELMoTokenCharactersIndexer()
reader = rnnDatasetReader(token_indexers={"tokens": token_indexer})
train_dataset = reader.read("../data/samples/pol_train_cleaned.csv")
# test_dataset = reader.read("../data/samples/pol_test_cleaned.csv")


# Token embedding 
options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

vocab = Vocabulary()
elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})




#Iterrator: batching data + preparing it for input
from allennlp.data.iterators import BucketIterator

iterator = BucketIterator(batch_size=config.batch_size, 
                          sorting_keys=[("tokens", "num_tokens")])
iterator.index_with(vocab)


batch = next(iter(iterator(train_dataset)))
tokens = batch["tokens"] #batch number 16 sentences x max num of words  x 
labels = batch
mask = get_text_field_mask(tokens)
lstm = PytorchSeq2VecWrapper(nn.LSTM(word_embeddings.get_output_dim(), config.hidden_sz,bidirectional=True, batch_first=True))

model = LSTM_Model(word_embeddings, lstm, 2)
optimizer = optim.SGD(model.parameters(), lr =config.lr)
embeddings = model.word_embeddings(tokens)

state = model.encoder(embeddings, mask)
class_logits = model.hidden2tag(state)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_metric="+f1",
                  patience=10,
                  num_epochs=config.epochs,
                  cuda_device= 0 if USE_GPU else -1,)
print("Training...")
trainer.train()
print("Done")


# iterate over the dataset without changing its order
seq_iterator = BasicIterator(config.batch_size)
seq_iterator.index_with(vocab)

predictor = Predictor(model, seq_iterator)
# test_preds, labels = predictor.predict(test_dataset) 
#labels = test_dataset["labels"].detach().cu().numpy()
accuracy_test = accuracy_score(test_preds,labels) 
f1_test = f1_score(test_preds,labels)
precision_test = precision_score(test_preds,labels)
recall_test = recall_score(test_preds,labels)
print("Accuracy score: {:.4f}, F1 Score: {:.4f}, Precision: {:.4f}, Recall: {:.4f} ".format(accuracy_test, f1_test, precision_test, recall_test))
save_file = "first_model.th"
with open(save_file, 'wb') as f:
    torch.save(model.state_dict(), f)
