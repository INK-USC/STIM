import torch
import requests
from typing import List, Dict
from mem_metrics.utils import find_all_substring_indices, is_stop_words

class MemCalculator():
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
        index='v4_dolma-v1_7_llama',
        max_clause_freq=500000,
        max_diff_tokens=1000,
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
        self.model_input = model_input
        self.model_output = model_output
        self.is_correct = is_correct
        self.model = model
        self.tokenizer_olmo = tokenizer_olmo
        self.k = k
        self.task_type = task_type
        self.original_str = original_str
        self.index = index
        self.step = step
        self.max_clause_freq = max_clause_freq
        self.max_diff_tokens = max_diff_tokens

    def get_token_id(self) -> List[Dict]:
        """
        Get the start and end position of the token in the original string
        """
        # Mark the start idx and end idx of the candidate tokens in model's output string
        token_model_output_olmo = self.tokenizer_olmo.tokenize(self.model_output)
        token_model_output_olmo_mapping = self.tokenizer_olmo(self.model_output, return_offsets_mapping=True)['offset_mapping']
        token_candidate_olmo = []
        start_step, end_step = find_all_substring_indices(self.model_output, self.step)[0]
        for token, start_end in zip(token_model_output_olmo, token_model_output_olmo_mapping):
            token_word = self.tokenizer_olmo.decode(self.tokenizer_olmo.convert_tokens_to_ids(token)).replace(' ', '')
            if start_end[0] >= start_step-1 and start_end[1] <= end_step and not is_stop_words(token_word) and self.task_type != "cap":
                token_candidate_olmo.append({"token": token, "start": start_end[0], "end": start_end[1], "previous": self.model_output[:start_end[0]]})
            elif self.task_type == 'cap' and start_end[0] >= start_step-1 and start_end[1] <= end_step and (not is_stop_words(token_word) or token_word.lower() in self.original_str):
                token_candidate_olmo.append({"token": token, "start": start_end[0], "end": start_end[1], "previous": self.model_output[:start_end[0]]})
        return token_candidate_olmo
        
    def get_alternative_tokens(self, token_candidate_olmo) -> List[Dict]:
        """
        For each token, get the top-k alternative tokens
        Output element in the list
            token: str
            start: int
            end: int
            alternative_tokens: List[Dict]
        """
        token_alternative_ls = []
        for ele in token_candidate_olmo:
            start = ele["start"]
            previous_all = self.model_input + self.model_output[:start]
            inputs = self.tokenizer_olmo(previous_all, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = outputs.logits.softmax(dim=-1)
            probs = probs[0, -1, :] # shape: (vocab_size, )
            token_id = self.tokenizer_olmo.convert_tokens_to_ids(ele['token'])
            # Find the maximum k index
            al_topk_indices = torch.topk(probs, self.k).indices.tolist()
            if token_id not in al_topk_indices:
                al_topk_indices.append(token_id)
            alternative_tokens = []
            # Record each candidate sequence with its score
            for al_idx in al_topk_indices:
                candidate = self.tokenizer_olmo.convert_ids_to_tokens(al_idx)
                score = probs[al_idx]
                alternative_tokens.append({"al_token": candidate, "prob": score.item()})

            ele["alternative_tokens"] = alternative_tokens
            token_alternative_ls.append(ele)
        
        return token_alternative_ls

    def infinigram_count(self, query, count_type):
        """
        Using infinigram API to get the document frequency
            query: Query for infinigram searching, which should follow the infinigram's format
            count_type: Should be 'single' or 'comb'
        """
        if count_type not in ['single', 'comb']:
            raise ValueError(f"Invalid count_type! It should be single or comb, but gets {count_type}")
        payload = {
            'index': self.index,
            'query_type': 'count',
            'query': query,
        }
        if count_type == 'comb':
            payload['max_clause_freq'] = self.max_clause_freq
            payload['max_diff_tokens'] = self.max_diff_tokens
        num, flag = 0, 0
        # Count for 100 time to reduce http error probability
        while num < 100:
            try:
                count = requests.post('https://api.infini-gram.io/', json=payload).json()['count']
                flag = 1
                break
            except:
                num += 1
                continue
        count = count if flag == 1 else None
        return count