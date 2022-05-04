import streamlit as st
import torch
import torch.nn as nn
import spacy
import json
from predict import prediction_preprocess, model_predict, predict_label


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_size,dropout=0.2,no_of_lstm_layers=1,bidirectional=False):
        super(SentimentModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.no_of_lstm_layers = no_of_lstm_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=no_of_lstm_layers, batch_first=True, dropout=dropout,bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_dim, 32)
        self.linear2 = nn.Linear(32,output_size)
    def forward(self,x,hidden_in):
        batch_size = x.shape[0]
        embedding_out = self.embedding(x)  
        output, (hidden_out, cell_state) = self.lstm(embedding_out,hidden_in)
        # print(output.shape,hidden_out.shape,cell_state.shape)
        # we will adjust the shape because bidirectional lstmcan give outputs which are in concatenated manner
        out_flatten = output.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.linear1(out_flatten)
        out = self.linear2(out)
        # print(out.shape)
        
        # out = out.view(batch_size,-1,self.output_size) #batch_size x -1 x output_size
        out = out[:,-1,:]
        return out,(hidden_out,cell_state)
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        if self.bidirectional:
            h0 = torch.zeros((self.no_of_lstm_layers*2,batch_size,self.hidden_dim)).to(device)
            c0 = torch.zeros((self.no_of_lstm_layers*2,batch_size,self.hidden_dim)).to(device)
        else:
            h0 = torch.zeros((self.no_of_lstm_layers,batch_size,self.hidden_dim)).to(device)
            c0 = torch.zeros((self.no_of_lstm_layers,batch_size,self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden


class VocabPred:
    def __init__(self,vocab):
        self.itos = vocab
        self.tokenizer = spacy.load("en_core_web_sm")

    def generate_numeric_tokens(self,text):
        tokenized_out = [token.text for token in self.tokenizer(text)]
        out = [self.itos[token] if token in self.itos.keys() else self.itos["<UNK>"] for token in tokenized_out]
        return out

    def pad_sequence(self,numeric_list,max_seq_length):
        """
        padding and truncation
        """
        if len(numeric_list) < max_seq_length:
            no_zeros = max_seq_length - len(numeric_list)
            numeric_list = numeric_list + [self.itos['<PAD>'] for i in range(no_zeros)]
        else:
            numeric_list = numeric_list[:max_seq_length]
        return numeric_list

st.title("Review To Rating Prediction Model")

st.caption('Model is trained on a corpus of Artificial christmas tree reviews. Model takes review as an input and predicts rating.')


model_dir_path = "model_weights"

#loading vocab
with open(model_dir_path+'/vocab.json','r') as f:
    vocab = json.load(f)
pred_vocab = VocabPred(vocab)

#loading model
VOCAB_SIZE = vocab.__len__()
EMBED_DIM = 100
HIDDEN_DIM = 256
OUTPUT_SIZE = 5
NO_OF_LAYERS = 2
DROP_OUT = 0.2
BIDIRECTIONAL = False
PATH = model_dir_path+'/model.pt'
model = SentimentModel(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM , output_size = OUTPUT_SIZE,
                       dropout=DROP_OUT,no_of_lstm_layers=NO_OF_LAYERS, bidirectional=BIDIRECTIONAL)
# Load
device = torch.device('cpu')
model.load_state_dict(torch.load(PATH, map_location=device))




# res = predict_label("lights are not working", pred_vocab, model)
# print("Predicted label: ",res)

input_review = st.text_input('Input your review here:') 
if input_review!="":
    predicted_rating = predict_label(input_review, pred_vocab, model)
    st.write("Predicted rating:  ",predicted_rating)