from typing import List, Dict
from scipy.stats import spearmanr
from mem_metrics.mem import MemCalculator
from tqdm import tqdm

class MidCalculator(MemCalculator):
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

    def is_elicit(self, cur_prefix, token) -> bool:
        """
        Test whether current prefix can elicit the token
        Params:
            cur_prefix: current prefix to elicit the output_ngram
            token: current token
        """
        message = [cur_prefix]
        inputs = self.tokenizer_olmo(message, return_tensors='pt', return_token_type_ids=False)
        inputs = inputs.to(self.model.device)
        response = self.model.generate(**inputs, max_new_tokens=1, do_sample=False)
        decoded_answer = self.tokenizer_olmo.batch_decode(response, skip_special_tokens=True)[0][len(cur_prefix):]
        if self.tokenizer_olmo.tokenize(decoded_answer) == [token]:
            return True
        else:
            return False

    def get_shortest_prefix(self, token_candidate_olmo: List[Dict]) -> List[Dict]:
        """
        Get the shortest prefix which can elicit the specific token in the token_candidate_olmo

        Input element's format in the list
            token: str
            start: int
            end: int
            previous: str
            alternative_tokens: List[Dict], keys are "al_token", "prob" in each dict

        Add "prefix" and "is_elicit" to the original dict
        """
        for i in range(len(token_candidate_olmo)):
            token = token_candidate_olmo[i]["token"]
            start = token_candidate_olmo[i]["start"]

            # Find the shortest prefix
            token_previous_all = self.tokenizer_olmo.encode(self.model_input + self.model_output[:start])
            for j in range(len(token_previous_all)-1, -1, -1):
                cur_prefix = self.tokenizer_olmo.decode(token_previous_all[j:])
                if self.is_elicit(cur_prefix, token):
                    token_candidate_olmo[i]["prefix"] = cur_prefix
                    token_candidate_olmo[i]["is_elicit"] = True
                    break
                if j == 0:
                    token_candidate_olmo[i]["prefix"] = cur_prefix
                    token_candidate_olmo[i]["is_elicit"] = False
                    break
        return token_candidate_olmo

    def get_fre(self, token_alternative_ls) -> List[Dict]:
        """
        For each alternative token and each selected word in the input (non-stop words), get the frequency of "input_unigram + token"
        Output element in the list
            token: str
            start: int
            end: int
            previous: str
            prefix: str
            input_str: str
            is_elicit: bool
            alternative_tokens: List[Dict], keys are 'al_token', 'prob', 'fre' ('fre' is List[Dict])
            top_attr: List[Dict], keys are 'word', 'start', 'end', 'score'
        """
        if len(token_alternative_ls) == 0:
            return []
        token_alternative_fre_ls = []
        for ele in tqdm(token_alternative_ls, total=len(token_alternative_ls)):
            # We use the unigrams which have high attribution score to selected tokens
            if ele["top_attr"] is not None and len(ele["top_attr"]) > 0:
                unigram_ls = [t["word"] for t in ele["top_attr"]]
            else:
                # If no words are selected, we use the original input_str
                unigram_ls = [ele["input_str"].split('<|assistant|>')[-1].strip()]
            alternative_tokens = ele["alternative_tokens"]
            for i, at in enumerate(alternative_tokens):
                fre_ls = []
                at_word = self.tokenizer_olmo.decode(self.tokenizer_olmo.convert_tokens_to_ids(at["al_token"]))
                for unigram in unigram_ls:
                    # Search the co-occurrence of at_word and unigram
                    fre = self.infinigram_count(query=f"{unigram} AND {at_word}", count_type='comb')
                    fre_ls.append({"unigram": unigram, "at_word": at_word, "count": fre})
                ele["alternative_tokens"][i]['fre'] = fre_ls
            if "saliency" in ele:
                del ele["saliency"]
            if "token_set" in ele:
                del ele["token_set"]
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