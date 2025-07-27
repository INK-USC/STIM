from typing import List, Dict
from scipy.stats import spearmanr
from mem_metrics.mem import MemCalculator

class LocalCalculator(MemCalculator):
    def __init__(
        self,
        model_input,
        model_output,
        is_correct,
        model,
        tokenizer_olmo,
        task_type,
        original_str=None,
        step=None,
        k=20,
        index='v4_dolma-v1_7_llama'
    ):
        """
        Params:
            model_input: prompt given to the model
            model_output: model's output for the test case
            is_correct: whether model correctly answer the given question
            model: Current language model
            tokenizer_olmo: OLMo tokenizer
            task_type: The task for memorization calculation
            step: Model's wrong reasoning step
            k: Numbers of alternative tokens to choose for frequency searching
        """
        super().__init__(
            model_input=model_input,
            model_output=model_output,
            is_correct=is_correct,
            model=model,
            tokenizer_olmo=tokenizer_olmo,
            k=k,
            task_type=task_type,
            original_str=original_str,
            index=index,
            step=step
        )

    def get_fre(self, token_alternative_ls: List[Dict]) -> List[Dict]:
        """
        For each alternative token, get the frequency of "prefix' + token", where prefix' is the longest prefix with non-zero frequency for current token
        Output element in the list
            token: str
            start: int
            end: int
            previous: str
            longest_nonzero_seq: str
            fre: int
            alternative_tokens: List[Dict], keys are 'al_token', 'prob', 'al_seq', 'fre'
        """
        if len(token_alternative_ls) == 0:
            return []
        token_alternative_fre_ls = []
        for ele in token_alternative_ls:
            start = ele["start"]
            token_all = self.tokenizer_olmo.encode(self.model_input + self.model_output[:start])
            token_all.append(self.tokenizer_olmo.convert_tokens_to_ids(ele["token"]))
            # Find the longest prefix with non-zero frequency
            seq_start = len(token_all)-1
            seq_dy = self.tokenizer_olmo.decode(token_all[seq_start:])
            seq = seq_dy
            seq_fre = self.infinigram_count(seq, 'single')
            while seq_fre > 0 and seq_start > 0:
                seq_start -= 1
                seq_dy = self.tokenizer_olmo.decode(token_all[seq_start:])
                seq_fre = self.infinigram_count(seq_dy, 'single')
                if seq_fre > 0:
                    seq = seq_dy
                    ele['fre'] = seq_fre
            token_id_seq = self.tokenizer_olmo.encode(seq)
            ele["longest_nonzero_seq"] = seq
            if 'fre' not in ele:
                ele['fre'] = 0

            # Get the frequency for each alternative tokens
            alternative_tokens = ele["alternative_tokens"]
            for j, at in enumerate(alternative_tokens):
                al_token_id = self.tokenizer_olmo.convert_tokens_to_ids(at["al_token"])
                token_id_seq[-1] = al_token_id
                al_seq = self.tokenizer_olmo.decode(token_id_seq)
                fre = self.infinigram_count(al_seq, 'single')
                alternative_tokens[j]["al_seq"] = al_seq
                alternative_tokens[j]["fre"] = fre
            ele["alternative_tokens"] = alternative_tokens
            token_alternative_fre_ls.append(ele)
        return token_alternative_fre_ls
    
    def cal_score(self, token_alternative_fre_ls: List[Dict]) -> List[Dict]:
        """
        Get the local memorization score for each token

        Input element in the list
            token: str
            start: int
            end: int
            longest_nonzero_seq: str
            fre: int
            alternative_tokens: List[Dict], keys are 'al_token', 'prob', 'al_seq', 'fre'

        Output element in the list
            token: str
            start: int
            end: int
            longest_nonzero_seq: str
            fre: int
            alternative_tokens: List[Dict], keys are 'al_token', 'prob', 'al_seq', 'fre'
            corr: float
        """
        if len(token_alternative_fre_ls) == 0:
            return []
        for i, ele in enumerate(token_alternative_fre_ls):
            fre_ls, prob_ls = [], []
            alternative_tokens = ele["alternative_tokens"]
            # Select top-k probability
            alternative_tokens = sorted(alternative_tokens, key=lambda x: x['prob'], reverse=True)[:self.k]
            for at in alternative_tokens:
                fre_ls.append(at["fre"])
                prob_ls.append(at["prob"])

            # Calculate the spearman rank coefficient
            corr, _ = spearmanr(fre_ls, prob_ls)
            token_alternative_fre_ls[i]['corr'] = corr
        token_alternative_fre_ls = sorted(token_alternative_fre_ls, key=lambda x: x['corr'], reverse=True)
        return token_alternative_fre_ls