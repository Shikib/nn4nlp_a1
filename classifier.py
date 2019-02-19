import copy
import gensim
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter

random.seed(42)
torch.cuda.manual_seed(42)

def load_data():
  """
  Load in train, dev and test data as a list of samples.

  Each sample should contain ([list of tokens], label).
  """
  def _load(filepath):
    return [(l.split('|||')[1].strip().lower().split(), l.split('|||')[0].strip())
            for l in open(filepath).readlines()]

  def _numberize(data, in_vocab, out_vocab):
    return [([in_vocab.get(w, 0) for w in inp], out_vocab.get(out)) 
            for inp,out in data]

  train = _load('data/topicclass_train.txt')
  valid = _load('data/topicclass_valid.txt')
  test = _load('data/topicclass_test.txt')

  all_inp_words = [w for inp,_ in train for w in inp]
  in_vocab = ["<UNK>", "<PAD>"] + [w for w,_ in Counter(all_inp_words).most_common(10000)]
  in_vocab = {e:i for i,e in enumerate(in_vocab)}
  out_vocab = {w:i for i,w in enumerate(set([out for _,out in train]))}

  train = _numberize(train, in_vocab, out_vocab)
  valid = _numberize(valid, in_vocab, out_vocab)
  test = _numberize(test, in_vocab, out_vocab)
  return train, valid, test, in_vocab, out_vocab

def load_w2v(vocab):
  """
  Load in word2vec weights for the vocabulary words.
  """
  w2v_model = gensim.models.KeyedVectors.load_word2vec_format('w2v.bin.gz', binary=True)
  return [w2v_model.word_vec(w) if w in w2v_model.vocab else [0]*300 for w in vocab]

class Classifier(nn.Module):
  def __init__(self, 
               vocab_size=10001, 
               output_size=0, 
               embed_size=300, 
               filter_windows=[3,4,5], 
               feature_size=100,
               hidden_size=300,
               w2v=None):
    super(Classifier, self).__init__()

    # Model architecture
    self.embed = nn.Embedding(vocab_size, embed_size)
    # Load word2vec embeddings
    if w2v is not None:
      #self.embed = nn.Embedding.from_pretrained(torch.cuda.FloatTensor(w2v))
      self.embed_w2v = nn.Embedding.from_pretrained(torch.cuda.FloatTensor(w2v))
      # Freeze one set of embeddings
      #self.embed_w2v.weight.requires_grad = True
      self.double_embed = True
    else:
      self.double_embed = False
     

    self.bidirectional = True
    self.convs = [nn.Conv1d(in_channels=(2 if self.bidirectional else 1) *hidden_size, out_channels=feature_size, kernel_size=kernel)
                  for kernel in filter_windows]
    self.convs = nn.ModuleList(self.convs)
    self.lstm = nn.LSTM(embed_size * (2 if self.double_embed else 1), hidden_size, bidirectional=self.bidirectional)
    self.dropout = nn.Dropout(p=0.5)
    self.out = nn.Linear((2 if self.bidirectional else 1) * hidden_size + len(filter_windows) * feature_size, output_size) 

    # Model training
    self.criterion = nn.CrossEntropyLoss()
    self.optim = torch.optim.Adam(params=self.parameters(), lr=0.001)

  def forward(self, x):
    embedded = self.embed(x)
    if self.double_embed:
      embedded2 = self.embed_w2v(x)
      lstm_embedded = torch.cat((embedded, embedded2), dim=2)
      embedded = torch.cat((embedded, embedded2), dim=1)
    else:
      lstm_embedded = embedded
    
    # CNN pass
    #embedded = embedded.permute(0,2,1) # bsz x embed_size x nwords 
    #feature_vecs = []
    #for conv in self.convs:
    #  feature_vec = conv(embedded)                  # bsz x num_filters x nwords
    #  feature_vec,_ = feature_vec.max(dim=2)        # bsz x num_filters
    #  feature_vecs.append(feature_vec)

    #feature_vec = torch.cat(feature_vecs, dim=1)     # bsz x num_filter_sizes*num_filters
    #feature_vec = F.relu(feature_vec)

    # LSTM Pass 
    input_lens = [e.item() for e in x.ne(self.pad_ind).sum(dim=1)]
    sort_idx = sorted(range(len(input_lens)), key=lambda i: -input_lens[i])
    lstm_embedded = lstm_embedded[sort_idx].permute(1,0,2)
    packed = nn.utils.rnn.pack_padded_sequence(
      lstm_embedded, 
      sorted(input_lens, reverse=True))
    
    outputs, enc_last = self.lstm(packed)
    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

    unsort_idx = sorted(range(len(sort_idx)), key=lambda i: sort_idx[i])
    enc_last = enc_last[0][:,unsort_idx]
    outputs = outputs[:,unsort_idx]

    #outputs = torch.cat((outputs, lstm_embedded), dim=2)
    self.dropout(outputs)

    feature_vecs = []
    for conv in self.convs:
      feature_vec = conv(outputs.permute(1,2,0))                  # bsz x num_filters x nwords
      feature_vec,_ = feature_vec.max(dim=2)        # bsz x num_filters
      feature_vecs.append(feature_vec)

    feature_vec = torch.cat(feature_vecs, dim=1)     # bsz x num_filter_sizes*num_filters
    feature_vec = F.relu(feature_vec)

    #feature_vec = torch.cat((feature_vec, torch.cat((enc_last[0], enc_last[1]), dim=1)), dim=1)
    feature_vec = self.dropout(feature_vec)
    return self.out(feature_vec)

  def prep(self, batch, ignore_y=False):
    """
    Given a batch, return an x and y tensor.
    """
    inputs = [sent for sent,_ in batch]

    # Pad inputs
    self.pad_ind = 1
    max_len = max([len(s) for s in inputs])
    inputs = [s+[self.pad_ind]*(max_len-len(s)) for s in inputs]

    x = torch.cuda.LongTensor(inputs)
    if ignore_y:
      return x
  
    outs = [out for _,out in batch]
    y = torch.cuda.LongTensor(outs)
 
    return x,y

  def train_batch(self, batch):
    # Zero gradients
    self.optim.zero_grad() 

    # Prep tensors
    x,y = self.prep(batch)

    # Forward pass
    proba = self.forward(x)
   
    # Backward pass
    loss = self.criterion(proba, y)
    loss.backward()
    nn.utils.clip_grad_norm_(self.parameters(), 10.0)

    # Optimizer
    self.optim.step()
    
    return loss.item()
   
  def predict(self, batch, return_loss=True): 
    if return_loss:
      x,y = self.prep(batch)
      proba = self.forward(x)
      loss = self.criterion(proba, y)
      pred = proba.max(dim=1)[1]
      return (pred==y).sum().item(), loss.item()
    else:
      x = self.prep(batch, ignore_y=True)
      proba = self.forward(x)
      pred = proba.max(dim=1)[1]
      return [e.item() for e in pred]

def predict_ensemble(models, batch):
  proba = None
  for m in models:
    x = m.prep(batch, ignore_y=True)
    if proba is None:
      proba = m.forward(x)
    else:
      proba += m.forward(x)
  return [e.item() for e in proba.max(dim=1)[1]]

def train(model, epochs=3, batch_size=32, print_every=500):
  best_acc = 0
  best_model = None
  for epoch in range(epochs):
    random.shuffle(train_data)
    cum_loss = 0
    for i in range(1+len(train_data)//batch_size):
      if i > 0 and i % print_every == 0:
        print("Epoch: ", epoch, "Batch: ", i, "Loss: ", cum_loss/i)

      cum_loss += model.train_batch(train_data[i*batch_size:(i+1)*batch_size])

    acc = evaluate(model, batch_size=50)
    if acc > best_acc:
      best_acc = acc
      best_model = copy.deepcopy(model)

  return best_model

def evaluate(model, batch_size=50):
  model.eval()
  total_loss = 0
  total_batches = 0
  total_correct = 0
  total_samples = 0
  for i in range(1+len(dev_data)//batch_size):
    correct,loss = model.predict(dev_data[i*batch_size:(i+1)*batch_size])
    total_loss += loss
    total_batches += 1
    total_correct += correct
    total_samples += len(dev_data[i*batch_size:(i+1)*batch_size])

  model.train()
  print("Validation Loss: ", total_loss/total_batches, 
        "Validation Acc: ", total_correct/total_samples)
  return total_correct/total_samples

def evaluate_ensemble(models, data, batch_size=50):
  [m.eval() for m in models]
  total_correct = 0
  total_samples = 0
  all_preds = []
  for i in range(1+len(data)//batch_size):
    pred = predict_ensemble(models, data[i*batch_size:(i+1)*batch_size])
    gt = [e[1] for e in data[i*batch_size:(i+1)*batch_size]]
    all_preds += pred
    total_correct += sum(p==g for p,g in zip(pred,gt))
    total_samples += len(data[i*batch_size:(i+1)*batch_size])

  [m.train() for m in models]
  print("Validation Acc: ", total_correct/total_samples)
  return all_preds

def augment(data):
  for i in range(len(data)*10):
    sample = random.choice(data)
    j = random.randint(0,len(sample[0])-1)
    if j > len(sample[0]) / 2:
      data.append((sample[0][:j], sample[1]))
    else:
      data.append((sample[0][j:], sample[1]))

if __name__ == '__main__':
  train_data, dev_data, test_data, vocab, out_vocab = load_data()
  #augment(train_data)
  w2v = load_w2v(vocab)
  models = [ Classifier(vocab_size=len(vocab), output_size=len(out_vocab), w2v=w2v).cuda() for i in range(1) ]
  best_models = []
  for model in models:
    best_models.append(train(model, epochs=3, batch_size=50))

  out_words = sorted(out_vocab.keys(), key=out_vocab.get)
  dev_preds = [out_words[e] for e in evaluate_ensemble(best_models, dev_data)]
  #open('dev_preds.txt', 'w+').writelines([l+'\n' for l in dev_preds])
  test_preds = [out_words[e] for e in evaluate_ensemble(best_models, test_data)]
  #open('test_preds.txt', 'w+').writelines([l+'\n' for l in test_preds])
  import pdb; pdb.set_trace()
