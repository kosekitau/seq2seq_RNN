# -*- coding: utf-8 -*-
!apt install aptitude swig

!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y

!pip install mecab-python3

!git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git

!echo yes | mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -a

import subprocess

cmd='echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
path = (subprocess.Popen(cmd, stdout=subprocess.PIPE,
                           shell=True).communicate()[0]).decode('utf-8')
mecab = MeCab.Tagger("-d {0} -Owakati".format(path))
#mecab = MeCab.Tagger("-Owakati")

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import MeCab
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
hidden_size = 256

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

mecab = MeCab.Tagger("-d {0} -Owakati".format(path))

#テキストの前処理
def ja_normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub('\r', '', s)
    s = re.sub('\n', '', s)
    s = re.sub(' ', ' ', s)
    s = mecab.parse(s)[:-2] #分かち書き
    return s #sentence

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def en_normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def preprocessing(ss):
  return [ja_normalizeString(ss.split('\t')[0]), en_normalizeString(ss.split('\t')[1])]

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # lang1, lang2は言語の洗濯(lang1が英語、lang2がフランス)
    lines = open('drive/My Drive/dataset/nitiei_kihon/kyodai_ja2en.txt', encoding='utf-8').read().strip().split('\n')

    # l = 'eng ¥t french'
    # lに対してユニコードや正規化の処理
    #pairs = [[i1, o1], [i2, o2],....]
    pairs = [preprocessing(l) for l in lines]

    # 言語の翻訳の順番を変えるだけ
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        #初期化
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 20 #文章の最大単語数(句読点を含む)

#上記2つの条件を満たすデータのみを抽出
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    #テキストファイルを読んで行に分割し、行をペアに分割する
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    #データの選別
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    #単語の辞書を作る
    for pair in pairs:
        input_lang.addSentence(pair[0]) #英語の1文
        output_lang.addSentence(pair[1]) #フランス語の一文
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('ja', 'en')
print(len(pairs)) #shape->[10599, 2]
print(pairs[0])
print(random.choice(pairs))

"""
#入力データを反転させる
for i in range(len(pairs)):
  pairs[i][0] = ' '.join(pairs[i][0].split(' ')[::-1])
"""

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # input_size：辞書の単語種類数、 hidden_size：分散表現の次元数
        #self.embedding = nn.Embedding.from_pretrained(embeddings=text_embedding_vectors, freeze=True)
        #self.emb_dim = text_embedding_vectors.shape[1]
        self.embedding = nn.Embedding(input_size, hidden_size)

        #層を増やす実験
        #self.gru1 = nn.GRU(text_embedding_vectors.shape[1], hidden_size)
        #self.lstm = nn.LSTM(text_embedding_vectors.shape[1],hidden_size) #[200, 256]
        self.lstm = nn.LSTM(hidden_size, hidden_size) #[200, 256]

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        #output, hidden2 = self.gru2(output, hidden2)
        #出力と隠れ状態
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))
        #return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        #self.embedding = nn.Embedding.from_pretrained(embeddings=text_embedding_vectors, freeze=True)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        #self.gru1 = nn.GRU(self.hidden_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        #input:[1], hidden:[1,1,256], encoder_outputs:[10,256]
        embedded = self.embedding(input).view(1, 1, -1) #[1, 1, 256]
        embedded = self.dropout(embedded)

        #加法注意によりattention_weightを求める。
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)

        #weightとエンコーダのアウトプットを用いる
        #bmmは各バッチごとに内積を出せる
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1) #[1, 512]
        output = self.attn_combine(output).unsqueeze(0) #[1, 1, 256]

        output = F.relu(output) #[1, 1, 256]
        output, hidden = self.lstm(output, hidden) #[1, 1, 256], ([1, 1, 256], [1, 1, 256])

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))
        #return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):
    #各文を単語indexのならびに
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1) #[[2], [25], [9]....]の形に


def tensorsFromPair(pair):
    #pairは各文(0がフランス)
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    #input_tensorは[その文の単語数, 1]のテンソル、つまり1文のデータ
    for ei in range(input_length):
        #1文字のindexと隠れ状態を入れていく
        #input_tensor[ei]:[1], encoder_hidden:[1,1,256]
        encoder_output, encoder_hidden, = encoder(
            input_tensor[ei], encoder_hidden)
        #encoder_output:[1,1,256], encoder_hidden:[1,1,256]
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            #decoder_input:[1], decoder_hidden:[1,1,256], encoder_outputs:[10,256]
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            #decoder_output:[1,2803], decoder_hidden:[1,1,256], decoder_attention:[1,10]
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        #########
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        #一文の損失をだす
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        #5000回で表示
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every #5000会のlossの平均
            ppl = math.exp(print_loss_avg) #pplをだす
            print_loss_total = 0
            print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg, ppl))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<eos>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        #pair = pairs[i]
        pair = random.choice(pairs)
        print('入力：', pair[0])
        print('正解', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('応答：', output_sentence)
        print('')

output_lang.n_words

"""
encoder1 = EncoderRNN(input_lang.n_words, hidden_size, ).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
"""

trainIters(encoder1, attn_decoder1, 10000, print_every=5000)

evaluateRandomly(encoder1, attn_decoder1)

trainIters(encoder1, attn_decoder1, 50000, print_every=5000)

evaluateRandomly(encoder1, attn_decoder1)

trainIters(encoder1, attn_decoder1, 10000, print_every=5000)

evaluateRandomly(encoder1, attn_decoder1)

import pickle

model_path = 'drive/My Drive/model/ja2en_encoder0519.pth'
torch.save(encoder1.to('cpu').state_dict(), model_path)
model_path = 'drive/My Drive/model/ja2en_decoder0519.pth'
torch.save(attn_decoder1.to('cpu').state_dict(), model_path)

with open('drive/My Drive/model/ja2en_input_word2index0519.pkl', 'wb') as f:
    pickle.dump(input_lang.word2index, f)
with open('drive/My Drive/model/ja2en_input_index2word0519.pkl', 'wb') as f:
    pickle.dump(input_lang.index2word, f)
with open('drive/My Drive/model/ja2en_output_index2word0519.pkl', 'wb') as f:
    pickle.dump(output_lang.index2word, f)
with open('drive/My Drive/model/ja2en_output_word2index0519.pkl', 'wb') as f:
    pickle.dump(output_lang.word2index, f)
