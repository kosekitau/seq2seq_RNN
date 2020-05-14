from os import listdir, path
import json
import collections
#対話破綻コーパスから入力応答ペアのテキストデータを生成する


#記事ファイルをダウンロードしたディレクトリから取得する関数を定義する。
#ダウンロードしたフォルダの名前をjsonにかえ、同じディレクトリに配置する
def corpus_files():
    dirs = [path.join('./json', x)
            for x in listdir('./json')]
    docs = [path.join(x, y)
            for x in dirs for y in listdir(x) ]
    return docs

#パスの中にある文章を読んで取得
def read_document(path):
  doc = ""
  with open(path, 'r') as f:
    doc = json.load(f)
    f.close()
  return doc

def corpus_to_sentences(corpus):
  docs = [read_document(x) for x in corpus]#パス取得してパスごとの文章取得してリスト
  return docs

corpus = corpus_files()#jsonファイルへのパスリストを生成
sentences = corpus_to_sentences(corpus)#一文ずつリストに保存
data = []
length = []

for i in range(len(sentences)):
  for j in range(len(sentences[i]["turns"])-1):
    #入力と応答でタブ区切り
    length.append(len(sentences[i]["turns"][j]["utterance"]))
    data.append(sentences[i]["turns"][j]["utterance"] + '\t' + sentences[i]["turns"][j+1]["utterance"])

print('length: ', len(data))
c = collections.Counter(length)

MAX_LENGTH = 36 #最大単語数(句読点を含む)

#上記2つの条件を満たすデータのみを抽出
def filterPair(p):
    return len(p.split('\t')[0]) < MAX_LENGTH and \
        len(p.split('\t')[1]) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
    #return [pair for pair in pairs if filterPair(pair)]
data = filterPairs(data)
print('filter length', len(data))
#テキストファイルに出力
with open('unk.txt', 'w') as f:
    f.write('\n'.join(data))
