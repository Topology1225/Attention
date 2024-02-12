"""
英日翻訳モデルの学習を行うデータセットの定義
"""
import torch
import torchtext
import pandas as pd

from janome.tokenizer import Tokenizer
import spacy
from collections import Counter
import torchtext.transforms as T

class Dataset(torch.utils.data.Dataset):
    def __init__(self, device=torch.device("cpu")):
        # define dataset
        self.df = pd.read_excel("./JEC_basic_sentence_v1-3/JEC_basic_sentence_v1-3.xls", header = None) 
        
        # define tokenizer function
        self.j_tokenizer_function = Tokenizer()
        self.e_tokenizer_function = spacy.load('en_core_web_sm')
        
        # preprocess
        ## tokenに変換
        japanese_tokenized_df = self.df.iloc[:, 1].apply(self.j_tokenizer)
        english_tokenized_df  = self.df.iloc[:, 2].apply(self.e_tokenizer)
        
        # token数のカウント
        j_list = []
        for i in japanese_tokenized_df:
            j_list += i
        j_counter = Counter()
        j_counter.update(j_list)
        japanese_vocabrary = torchtext.vocab.vocab(
            j_counter, specials=(
                ['<unk>', '<pad>', '<bos>', '<eos>']
            )
        )
        japanese_vocabrary.set_default_index(
            japanese_vocabrary['<unk>']
        )
        
        e_list = []
        for i in english_tokenized_df:
            e_list += i
        
        e_counter = Counter()
        e_counter.update(e_list)
        english_vocabrary = torchtext.vocab.vocab(
            e_counter,
            specials=(['<unk>', '<pad>', '<bos>', '<eos>'])
        )
        english_vocabrary.set_default_index(
            english_vocabrary['<unk>']
        )
        self.japanese_vocabrary = japanese_vocabrary
        self.english_vocabrary  = english_vocabrary
        
        self.japanese_tokenized_df = japanese_tokenized_df
        self.english_tokenized_df  = english_tokenized_df
        
        # define transforms
        self.japanese_text_transform = T.Sequential(
            T.VocabTransform(japanese_vocabrary),
            T.Truncate(14),
            T.AddToken(token=japanese_vocabrary['<bos>'], begin=True),
            T.AddToken(token=japanese_vocabrary['<eos>'], begin=False),
            T.ToTensor(),
            T.PadTransform(
                14 + 2, japanese_vocabrary['<pad>']
            )
        )
        self.english_text_transform = T.Sequential(
            T.VocabTransform(english_vocabrary),
            T.Truncate(14),
            T.AddToken(token=english_vocabrary['<bos>'], begin=True),
            T.AddToken(token=english_vocabrary['<eos>'], begin=False),
            T.ToTensor(),
            T.PadTransform(14 + 2, english_vocabrary['<pad>'])
        )
        
        self.device = device

    def __getitem__(self, index):
        """
        collate_fnでまとめてtransformした方が良い
        入力する単語数を同じにできるはず
        tokenizerはひとまとめにできるのか？
        """
        # Implement your logic to retrieve an item from the dataset
        japanese = self.japanese_tokenized_df[index]
        engilsh  = self.english_tokenized_df[index]
        
        transformed_japanese = self.japanese_text_transform(japanese)
        transformed_english  = self.english_text_transform(engilsh)
        
        # maskもここで定義したい
        mask = torch.nn.Transformer.generate_square_subsequent_mask(14 + 1)
        data = \
        {
            "japanese_token": transformed_english.to(self.device),
            "decoder_input": transformed_english[:-1].to(self.device),
            "decoder_output": transformed_english[1:].to(self.device),
            "mask": mask.to(self.device)
        }
        return data 
    
    def __len__(self):
        # Implement your logic to return the length of the dataset
        return len(self.df)
    
    def j_tokenizer(self, text):
        return [
            token for token in self.j_tokenizer_function.tokenize(text, wakati=True)
        ]
    
    def e_tokenizer(self, text):
        return [
            token.text for token in self.e_tokenizer_function.tokenizer(text)
        ]
        
    def get_vocab_size(self):
        return {"japanese": len(self.japanese_vocabrary), "english": len(self.english_vocabrary)}



def main():
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)
    print(dataset.get_vocab_size())
    # {'japanese': 6446, 'english': 6072}
    # for data in dataloader:
    #     print(data)
    
    
if __name__=="__main__":
    main()
    
