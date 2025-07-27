from scipy.stats import spearmanr
from mem_metrics.mem import MemCalculator
from tqdm import tqdm

class LongCalculator(MemCalculator):
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

    def get_fre(self, token_alternative_ls):
        """
        For each alternative token and each selected word in the input (non-stop words), get the frequency of "input_unigram + token"
        Output element in the list
            token: str
            start: int
            end: int
            previous: str
            alternative_tokens: List[Dict], keys are 'al_token', 'prob', 'fre' ('fre' is List[Dict])
            top_attr: List[Dict], keys are 'word', 'start', 'end', 'score'
        """
        if len(token_alternative_ls) == 0:
            return []
        token_alternative_fre_ls = []
        for ele in tqdm(token_alternative_ls, total=len(token_alternative_ls)):
            # We use the unigrams which have high attribution score to selected tokens
            unigram_ls = [t["word"] for t in ele["top_attr"]]
            # Only select top k alternative tokens
            alternative_tokens = sorted(ele["alternative_tokens"], key=lambda x: x["prob"], reverse=True)[:self.k]
            ele["alternative_tokens"] = alternative_tokens
            for i, at in enumerate(alternative_tokens):
                if 'fre' in ele["alternative_tokens"][i] and ele["alternative_tokens"][i]['fre'] is not None and ele["alternative_tokens"][i]['fre'] != []:
                    continue
                fre_ls = []
                at_word = self.tokenizer_olmo.decode(self.tokenizer_olmo.convert_tokens_to_ids(at["al_token"]))
                for unigram in unigram_ls:
                    # Search the co-occurrence of at_word and unigram
                    fre = self.infinigram_count(query=f"{unigram} AND {at_word}", count_type='comb')
                    fre_ls.append({"unigram": unigram, "at_word": at_word, "count": fre})
                ele["alternative_tokens"][i]['fre'] = fre_ls
            token_alternative_fre_ls.append(ele)
            
        return token_alternative_fre_ls
    
    def cal_score(self, token_alternative_fre_ls):
        """
        Get the memorization score for the given case
        Add the key 'corr' into the element in token_alternative_fre_ls
        """
        if len(token_alternative_fre_ls) == 0:
            return []
        for idx, ele in enumerate(token_alternative_fre_ls):
            fre_ls, prob_ls = [], []
            for at in ele["alternative_tokens"]:
                fre_unigrams = [at['fre'][i]['count'] for i in range(len(at['fre']))]
                if len(fre_unigrams) > 0 and None not in fre_unigrams:
                    fre_ls.append(sum(fre_unigrams) / len(fre_unigrams))
                else:
                    fre_ls.append(0)
                prob_ls.append(at["prob"])

            # Calculate the spearman rank coefficient
            corr, _ = spearmanr(fre_ls, prob_ls)
            token_alternative_fre_ls[idx]['corr'] = corr
        token_alternative_fre_ls = sorted(token_alternative_fre_ls, key=lambda x: x['corr'], reverse=True)
        return token_alternative_fre_ls