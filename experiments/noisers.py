import json
import random

def findall(haystack, needle):
    first_ids = []
    i = haystack.find(needle)
    while i != -1:
        first_ids.append(i)
        i = haystack.find(needle, i+1)
    return first_ids


class Noiser():
    def __init__(self, deletions_cnt=None, new2olds=None, edit_total_proba=None, char_prior=None):
        self.deletions_cnt = deletions_cnt
        self.new2olds = new2olds
        self.edit_total_proba = edit_total_proba
        self.char_prior = char_prior
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(
                {
                    'deletions_cnt': self.deletions_cnt, 
                    'new2olds': self.new2olds, 
                    'edit_total_proba': self.edit_total_proba, 
                    'char_prior': self.char_prior,
                }, 
                f, 
                ensure_ascii=False, indent=1
            )
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            deletions_cnt=data['deletions_cnt'],
            new2olds=data['new2olds'],
            edit_total_proba=data['edit_total_proba'],
            char_prior=data['char_prior'],
        )
    
    def add_noise(
        self, text, min_edits=1, max_edits=10, p_exit=None, p_insert=0.18, temp=0.3, p_randchar=0.05, edit_rate=0.02,
        eps=1e-10,
    ):
        # todo: smooth all probabilities
        if p_exit is None:
            p_exit = 1 / (max(2, len(text)) * edit_rate)
        # replacing a random character (including deletions)
        candidates = [text[i:i+n] for n in range(1, 4) for i in range(len(text)-n+1)]
        cand_weights = [self.edit_total_proba.get(c, 0) ** temp + eps for c in candidates]
        n_edits = 0
        for i in range(max_edits):
            if random.random() < p_randchar:
                rep = random.choices(
                    list(self.char_prior.keys()), weights=[c + eps for c in self.char_prior.values()]
                )[0]
                idx = random.randint(0, len(text)-1)
                text = text[:idx] + rep + text[idx + 1:]
            elif random.random() < p_insert:
                idx = random.randint(0, len(text))
                insertion = random.choices(
                    list(self.deletions_cnt.keys()), 
                    weights=[c + eps for c in self.deletions_cnt.values()]
                )[0]
                text = text[:idx] + insertion +  text[idx:]
                n_edits += 1
            else:
                choice = random.choices(candidates, weights=cand_weights)[0]
                if choice not in text:
                    continue
                idx = random.choice(findall(text, choice))
                rd = self.new2olds.get(choice, self.char_prior)
                replacement = random.choices(
                    list(rd.keys()), 
                    weights=[c + eps for c in rd.values()]
                )[0]
                text = text[:idx] + replacement +  text[idx + len(choice):]
                n_edits += 1
            if n_edits >= min_edits and random.random() < p_exit:
                break
        return text

    
def add_simple_noise(text, chars, edit_rate=0.05, p_del=0.2, p_add=0.2, p_cap=0.2):
    result = []
    for c in text:
        if random.random() < edit_rate:
            r = random.random()
            if r < p_del:
                continue
            elif r < p_del + p_add:
                result.append(random.choice(chars))
                result.append(c)
            elif r < p_del + p_add + p_cap:
                if c.islower():
                    result.append(c.upper())
                else:
                    result.append(c.lower())
            else:
                result.append(random.choice(chars))
        else:
            result.append(c)
    return ''.join(result)
