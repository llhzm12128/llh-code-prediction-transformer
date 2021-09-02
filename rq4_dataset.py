import torch
import utils
import json

class Dataset(torch.utils.data.Dataset):
    def __init__(self, fp):
        super().__init__()
        self.fp = fp
        self._line_pos_dp = list(utils.line_positions(fp))
    
    def __len__(self):
        return len(self._line_pos_dp)
    
    def __getitem__(self, idx):
        line_pos = self._line_pos_dp[idx]
        with open(self.fp) as f:
            f.seek(line_pos)
            dp_line = f.readline().strip()
        return json.loads(dp_line)

    @staticmethod
    def collate(seqs, pad_idx=None):
        max_len = max(len(seq[0]) for seq in seqs)
        max_len = max(max_len, 2)
        input_seqs = []
        target_seqs = []
        extended = []

        for i,  (seq, ext) in enumerate(seqs):
            padding = [pad_idx] * (max_len - len(seq))
            input_seqs.append(seq[:-1] + padding)
            target_seqs.append(seq[1:] + padding)
            extended.append(ext)

        return {
            "input_seq": torch.tensor(input_seqs),
            "target_seq": torch.tensor(target_seqs),
            "extended": torch.tensor(extended)
        }