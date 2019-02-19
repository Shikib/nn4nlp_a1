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
      self.embed = nn.Embedding.from_pretrained(torch.cuda.FloatTensor(w2v))
      self.embed_w2v = nn.Embedding.from_pretrained(torch.cuda.FloatTensor(w2v))
      # Freeze one set of embeddings
      self.embed_w2v.weight.requires_grad = False
      self.double_embed = False

    self.bidirectional = True
    self.convs = [nn.Conv1d(in_channels=(2 if self.bidirectional else 1) *hidden_size, out_channels=feature_size, kernel_size=kernel)
                  for kernel in filter_windows]
    self.convs = nn.ModuleList(self.convs)
    self.lstm = nn.LSTM(embed_size * (2 if self.double_embed else 1), hidden_size, bidirectional=self.bidirectional)
    self.dropout = nn.Dropout(p=0.5)
    self.out = nn.Linear((2 if self.bidirectional else 1) * hidden_size + embed_size + len(filter_windows) * feature_size, output_size) 

    # Model training
    self.criterion = nn.CrossEntropyLoss()
    self.optim = torch.optim.Adam(params=self.parameters(), lr=0.001)

  def forward(self, x, return_hid=False, ratio=0.0):
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

    feature_vec = torch.cat((lstm_embedded.mean(dim=0), feature_vec, torch.cat((enc_last[0], enc_last[1]), dim=1)), dim=1)
    feature_vec = self.dropout(feature_vec)
    if return_hid:
      return feature_vec

    weights = F.softmax(feature_vec.unsqueeze(0).bmm(self.train_hids.permute(1,0).unsqueeze(0)), dim=2)
    pred = weights.permute(1,0,2).bmm(model.train_ans.repeat((weights.size(1),1,1))).squeeze(1)
    return ratio*pred + (1-ratio)*self.out(feature_vec)

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

  def train_batch(self, batch, ratio=0):
    # Zero gradients
    self.optim.zero_grad() 

    # Prep tensors
    x,y = self.prep(batch)

    # Forward pass
    proba = self.forward(x, ratio=ratio)
   
    # Backward pass
    loss = self.criterion(proba, y)
    loss.backward()
    nn.utils.clip_grad_norm_(self.parameters(), 10.0)

    # Optimizer
    self.optim.step()
    
    return loss.item()
   
  def pred_train(self, print_every=500, batch_size=100):
    self.train_hids = []
    self.train_ans = []
    for i in range(1+len(train_data)//batch_size):
      if i > 0 and i % print_every == 0:
        print("Forward Batch: ", i)

      x,y = self.prep(train_data[i*batch_size:(i+1)*batch_size])
      self.train_hids.append(self.forward(x, return_hid=True).cpu().detach())
      self.train_ans.append(y.cpu())

    self.train_hids = torch.cat(self.train_hids, dim=0)[:50000].cuda()
    self.train_ans = torch.cat(self.train_ans, dim=0)[:50000].cuda()
    self.train_ans = torch.eye(self.train_ans.max(0)[0]+1).cuda()[self.train_ans]
      
  def predict(self, batch):
    x,y = self.prep(batch)
    hid = self.forward(x, return_hid=True).detach()
    weights = F.softmax(hid.unsqueeze(0).bmm(self.train_hids.permute(1,0).unsqueeze(0)), dim=2)
    pred = weights.permute(1,0,2).bmm(model.train_ans.repeat((weights.size(1),1,1))).squeeze(1).max(dim=1)[1]
    return (pred==y).sum().item()

def train(model, epochs=10, batch_size=50, print_every=500):
  for epoch in range(epochs):
    random.shuffle(train_data)
    cum_loss = 0
    model.pred_train()
    for i in range(1+len(train_data)//batch_size):
      if i > 0 and i % print_every == 0:
        print("Epoch: ", epoch, "Batch: ", i, "Loss: ", cum_loss/i)

      cum_loss += model.train_batch(train_data[i*batch_size:(i+1)*batch_size], ratio=epoch/5 if epoch < 5 else 1.0)

    evaluate(model, batch_size=50)

def evaluate(model, batch_size=50):
  model.eval()
  model.pred_train()
  total_correct = 0
  total_samples = 0
  for i in range(1+len(dev_data)//batch_size):
    correct = model.predict(dev_data[i*batch_size:(i+1)*batch_size])
    total_correct += correct
    total_samples += len(dev_data[i*batch_size:(i+1)*batch_size])

  model.train()
  print("Validation Acc: ", total_correct/total_samples)

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
  model = Classifier(vocab_size=len(vocab), output_size=len(out_vocab), w2v=w2v).cuda()
  train(model, epochs=20, batch_size=50)
