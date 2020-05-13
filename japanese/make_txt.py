from os import listdir, path
import json

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

for i in range(len(sentences)):
  for j in range(len(sentences[i]["turns"])-1):
    #入力と応答でタブ区切り
    data.append(sentences[i]["turns"][j]["utterance"] + '\t' + sentences[i]["turns"][j+1]["utterance"])

print('length: ', len(data))
#テキストファイルに出力
with open('unk.txt', 'w') as f:
    f.write('\n'.join(data))
