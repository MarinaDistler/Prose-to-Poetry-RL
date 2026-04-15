from collections import defaultdict
import nltk
from subprocess import check_output, DEVNULL

from rhymetagger import RhymeTagger

from util import filter_lines

nltk.download('punkt_tab')
scheme_map_dict = {
    "ABAB": [(0, 2), (1, 3)],
    "AABB": [(0, 1), (2, 3)],
    "ABBA": [(0, 3), (1, 2)],
}

class MyRhymeTagger(RhymeTagger): # переписываю чтобы доставать вероятности а не только рифмы
    def _detect_rhymes(self, ngram=True, update_train_set=True):        
        '''
        Count rhyme-scores for pairs of lines that are within a 
        specified window
        --------------------------------------------------------------
        :ngram             = [boolean] whether to take into account n-grams
        :update_train_set  = [boolean] whether to update train set or to return
                             list of rhymes
            '''
        rhymes_detected = defaultdict(set)
        rhyme_scores = {}
        ngram_scores = {}
        
        # Iterarate over lines in dataset
        for i,l in enumerate(self.data):           

            # Skip if no word at all in i-line
            if not self.data[i][0]:
                continue            
                                        
            # Iterate forward over lines that are in specified window
            for j in range(i+1, i+self.window+1):
                
                # Skip if 
                # (1) end of dataset was reached OR
                # (2) end of poem was reached OR
                # (3) end of stanza was reached and inter-stanza rhymes are forbidden
                # (4) both words are the same and same-rhymes are forbidden
                if (
                    j > len(self.data) - 1 or
                    l[1] != self.data[j][1] or
                    ( self.stanza_limit and l[2] != self.data[j][2] ) 
                ):
                    continue

                if ( not self.same_words and l[0] == self.data[j][0] ):
                    rhyme_scores[(i, j)] = 0.
                    continue
                
                # Skip if no word at all in j-line
                if not self.data[j][0]:
                    continue            
                
                # Get rhyme score based on components 
                ipa_score = self._rhyme_score(l[0], self.data[j][0])
                rhyme_scores[(i, j)] = ipa_score

                # If score is high enough
                if ipa_score > self.prob_ipa_min:

                    # Add j to i-line and i to j-line 
                    rhymes_detected[i].add(j)
                    rhymes_detected[j].add(i)

                    # Annotate distant rhymes
                    for k in rhymes_detected[i]:
                        if k != j:
                            rhymes_detected[k].add(j)
                            rhymes_detected[j].add(k)
                            
            # If ngrams should be used and no rhymes were found for i-line,
            # iterate over window once again and perform ngram-based recognition
            if not ngram:
                continue
            if i in rhymes_detected:
                continue
            for j in range(i+1, i+self.window+1):
                if (
                    j > len(self.data) - 1 or
                    l[1] != self.data[j][1] or
                    ( self.stanza_limit and l[2] != self.data[j][2] ) 
                ):
                    continue

                if ( not self.same_words and l[0] == self.data[j][0] ):
                    ngram_scores[(i, j)] = 0.
                    continue

                if (j in rhymes_detected):
                    ngram_scores[(i, j)] = rhyme_scores[(i, j)]
                    continue

                # Skip if no word at all in j-line
                if not self.data[j][0]:
                    continue            
            
                ngram_score = self._ngram_score(l[0], self.data[j][0])
                ngram_scores[(i, j)] = ngram_score

                if ngram_score > self.prob_ngram_min:
                    rhymes_detected[i].add(j)
                    rhymes_detected[j].add(i)

        # Update train set if required (training)
        if update_train_set:
            for i in rhymes_detected:
                for j in rhymes_detected[i]:
                    if i > j:
                        continue
                    self._add_to_train_set(self.data[i][0], self.data[j][0])

        # Otherwise format output and return                    
        else:
            output = self.output(rhymes_detected)
            return output, rhyme_scores, ngram_scores

    def _transcription(self, text):
        '''
        Transcribe a text to IPA using eSpeak
        --------------------------------------------------------------
        :text  = [string] in specified language
        '''
        ipa = check_output(
            ["espeak", "-q", "--ipa=1", "-v", self.lang, text],
            stderr=DEVNULL
        ).decode("utf-8").strip()
        return ipa

            

rt = MyRhymeTagger()
rt.load_model(model='ru')  # Загрузка русской модели рифм


def check_rhyme_scheme(lines, scheme="ABAB"):
    rhymes, _, _ = rt.tag(lines, output_format=1)

    scheme_map = []
    for position in range(len(scheme)):
        scheme_map.append([])
        for i in range(len(scheme)):
            if i != position and scheme[i] == scheme[position]:
                scheme_map[position].append(i)

    correct_rhymes = 0
    for i, rhyme_group in enumerate(rhymes):
        scheme_group = scheme_map[i % len(scheme_map)]
        correct_rhymes += len(set(rhyme_group) & set(scheme_group))

    total_possible = len(lines)
    return correct_rhymes / total_possible if total_possible > 0 else 0.


def get_rhyme_score(lines, scheme="ABAB", alpha=0.):
    pos_pairs = scheme_map_dict[scheme]
    pos_score, pos_count = 0., 0
    neg_score, neg_count = 0., 0
    for i in range(0, len(lines), len(scheme)):
        _, rhyme_scores, ngram_score = rt.tag(lines[i:i+len(scheme)], output_format=1)

        for pair, score in rhyme_scores.items():
            if pair in pos_pairs:
                pos_score += score
                pos_count += 1
            else:
                neg_score += score
                neg_count += 1

    pos_mean = pos_score / pos_count if pos_count > 0 else 0.
    neg_mean = neg_score / neg_count if neg_count > 0 else 0.
    score = pos_mean - alpha * neg_mean
    score = (score + alpha) / (1 + alpha)
    return score

def make_rhyme_reward(coef, alpha):
    def rhyme_reward(completions, rhyme_scheme=None, **kwargs):
        rewards = []
        
        for text, scheme in zip(completions, rhyme_scheme):
            lines = text.split('\n')
            f_lines = filter_lines(lines)
            
            score = get_rhyme_score(f_lines, scheme, alpha=alpha)
            rewards.append(coef * score)
        
        return rewards
    
    return rhyme_reward