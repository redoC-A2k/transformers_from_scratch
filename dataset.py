import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype = torch.int64)
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index) -> any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # padding
        src_num_pad = int(self.seq_len) - len(src_tokens) - 2 # 2 for [SOS] and [EOS]
        tgt_num_pad = int(self.seq_len) - len(tgt_tokens) - 1 # While training we add only the SOS token to the decoder side and not the EOS token

        if src_num_pad < 0 or tgt_num_pad < 0:
            raise ValueError("Sequence length is too short")
        
        # Add SOS and EOS token to the source text
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(src_tokens),
            self.eos_token,
            torch.tensor([self.pad_token] * src_num_pad, dtype=torch.int64)
        ])

        # Add SOS token to the target text
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(tgt_tokens),
            torch.tensor([self.pad_token] * tgt_num_pad, dtype=torch.int64)
        ])

        # Add EOS token to the target text ( what we expect from decoder)
        label = torch.cat([ # or target ie y
            torch.tensor(tgt_tokens),
            self.eos_token,
            torch.tensor([self.pad_token] * tgt_num_pad, dtype=torch.int64)
        ])

        # print(f"{encoder_input.size(0)} {self.seq_len}")
        assert encoder_input.size(0) == int(self.seq_len)
        assert decoder_input.size(0) == int(self.seq_len)
        assert label.size(0) == int(self.seq_len)

        return {
            "encoder_input":encoder_input,
            "decoder_input":decoder_input,
            "encoder_mask":(encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1,1,seq_len) & (1,seq_len,seq_len)
            "label":label,
            "src_text":src_text,
            "tgt_text":tgt_text
        }
    
def casual_mask(size):
    mask = torch.triu(torch.ones((1 , size, size)), diagonal=1)
    return mask == 0





