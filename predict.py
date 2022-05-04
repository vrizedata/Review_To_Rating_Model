from clean_data import clean_text
import torch
import torch.nn.functional as F
import numpy as np

def prediction_preprocess(text,pred_vocab,max_seq_length=410):
    text = clean_text(text)
    text_numeric = pred_vocab.generate_numeric_tokens(text)
    if max_seq_length is not None:
          text_numeric = pred_vocab.pad_sequence(text_numeric,max_seq_length)
    seq = torch.Tensor(text_numeric).long()
    return seq

def model_predict(text, pred_vocab, model):
    seq = prediction_preprocess(text,pred_vocab,max_seq_length=410)
    seq = torch.unsqueeze(seq,axis=0)
    
    h = model.init_hidden(1)
    #h = tuple([each.data.to(device) for each in h])
    h = tuple([each.data for each in h])
    # = seq.to(device)
    out,h_ = model(seq,h)
    out = F.softmax(out)
    return out.detach().cpu().numpy()
    

def predict_label(text, pred_vocab, model):
      pred_prob = model_predict(text, pred_vocab, model)
      pred_label = np.argmax(pred_prob[0])
      return pred_label