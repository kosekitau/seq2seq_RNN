# -*- coding: utf-8 -*-
!apt install aptitude swig

!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y

!pip install mecab-python3

!git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git

!echo yes | mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -a

import MeCab
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

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import gensim

hidden_size = 256
model_dir = 'drive/My Drive/embedding/japanese_wiki/entity_vector.model.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_dir, binary=True)
embedding_vector = torch.from_numpy(model.vectors)

SOS_token = 0 #start of string
EOS_token = 1
MAX_LENGTH = 30

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {} #単語の頻度
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # 辞書に登録してある単語の種類

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

def normalizeString(s):
    s = re.sub('\r', '', s)
    s = re.sub('\n', '', s)
    s = re.sub(' ', ' ', s)
    s = mecab.parse(s)[:-2] #分かち書き
    return s #sentence

#pairsの作成と、inputLang、outputLangの初期化
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # lang1, lang2は言語の洗濯(lang1が英語、lang2がフランス)
    lines = open('drive/My Drive/dataset/taiwahatan/0516_30token.txt', encoding='utf-8').\
        read().strip().split('\n')

    # l = 'eng ¥t french'
    # lに対してユニコードや正規化の処理
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

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

def prepareData(lang1, lang2, reverse=False):
    #テキストファイルを読んで行に分割し、行をペアに分割する
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    #単語の辞書を作る
    for pair in pairs:
        input_lang.addSentence(pair[0]) #入力の1文
        output_lang.addSentence(pair[1]) #出力の一文
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(len(pairs)) #shape->[10599, 2]
print(random.choice(pairs))

#入力文を反転させる
for i in range(len(pairs)):
  pairs[i][0] = ' '.join(pairs[i][0].split(' ')[::-1])

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, text_embedding_vectors, dropout_p):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # input_size：辞書の単語種類数、 hidden_size：分散表現の次元数
        self.embedding = nn.Embedding.from_pretrained(
        embeddings=text_embedding_vectors, freeze=True) # [*, 200]
        #self.lstm = nn.GRU(text_embedding_vectors.shape[1], hidden_size) #[200, 256]
        self.lstm = nn.LSTM(text_embedding_vectors.shape[1], hidden_size, dropout=dropout_p) #[200, 256]

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1) #[1, 1, 256]
        output = embedded
        output, hidden = self.lstm(output, hidden) #[1, 1, 256], [1, 1, 256]
        #出力と隠れ状態
        #隠れ状態を出力する
        return output, hidden

    #隠れ状態0の初期化
    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))
        #return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        #self.lstm = nn.GRU(hidden_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    #hidedenはencoderの隠れ状態
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        #return torch.zeros(1, 1, self.hidden_size, device=device)
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, text_embedding_vectors, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size #RNNの重みの次元数
        self.embedding = nn.Embedding.from_pretrained(embeddings=text_embedding_vectors, freeze=True)
        self.output_size = output_size #単語数
        self.dropout_p = dropout_p
        self.max_length = max_length #最大単語数
        self.dropout = nn.Dropout(self.dropout_p)
        #self.lstm = nn.GRU(text_embedding_vectors.shape[1], self.hidden_size)
        self.lstm = nn.LSTM(text_embedding_vectors.shape[1], self.hidden_size, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        #input:[1], hidden:[1,1,256], encoder_outputs:[10,256]
        embedded = self.embedding(input).view(1, 1, -1) # [1,1,256]
        embedded = self.dropout(embedded)

        query, hidden = self.lstm(embedded, hidden) # [1,1,256], ([1,1,256], [1,1,256])

        #内積注意でAttention Weightを求める
        attn_weights = torch.matmul(query[0], encoder_outputs.transpose(1, 0)) # [1,30]
        attn_weights = F.softmax(attn_weights, -1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0)) # [1,1,256]

        output = torch.cat((query[0], attn_applied[0]), 1)
        output = F.log_softmax(self.out(output), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        #return torch.zeros(1, 1, self.hidden_size, device=device)
        #隠れ状態と記憶セルの初期化
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))

def indexesFromSentence(lang, sentence):
    #各文を単語indexのならびに
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1) #[[2], [25], [9]....]の形に


def tensorsFromPair(pair):
    #pairは各文(0が入力)
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(2)]
training_pairs

#teaching_forceは学習時のエンコーダーに生成されたtokenではなく、
#教師データを入力すること
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
        #input_tensor[ei]:[1], encoder_hidden:([1,1,256], [1,1,256])
        encoder_output, encoder_hidden, = encoder(
            input_tensor[ei], encoder_hidden)
        #encoder_output:[1,1,256], encoder_hidden:[1,1,256]
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    #エンコーダーの隠れ状態と記憶セル
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

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        #########
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

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
        pair = random.choice(pairs)
        print('入力：', pair[0])
        print('正解', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('応答：', output_sentence)
        print('')

output_lang.n_words

encoder1 = EncoderRNN(input_lang.n_words, hidden_size, embedding_vector, dropout_p=0.5).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size,  embedding_vector, output_lang.n_words, dropout_p=0.5).to(device)

#trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

evaluateRandomly(encoder1, attn_decoder1)
