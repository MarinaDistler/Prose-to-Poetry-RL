import os
import sys
import torch.nn.functional as F
import numpy as np

from poetry.phonetic import Accents
from generative_poetry.metre_classifier import ErrorsTable, MetreClassifier, \
                PatternAnalyzer, StressPredictorAdapter, Markup

from .rhyme_metric import scheme_map_dict
from util import filter_lines


proj_dir = os.path.join('external_code', 'verslibre')
tmp_dir = os.path.join(proj_dir, 'tmp')
data_dir = os.path.join(proj_dir, 'data')
models_dir = os.path.join(proj_dir, 'models')

accents = Accents()
accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
accents.after_loading(os.path.join(tmp_dir, 'stress_model'))

stress_predictor = StressPredictorAdapter(accents)


metre_patterns = {
    "iambos": 'us',
    "choreios": 'su',
    "daktylos": 'suu',
    "amphibrachys": 'usu',
    "anapaistos":  'uus',
}


def find_errors_all_meters(markup):
    num_lines = len(markup.lines)
    errors_table = ErrorsTable(num_lines)
    for l, line in enumerate(markup.lines):
        for metre_name, metre_pattern in MetreClassifier.metres.items():
            line_syllables_count = sum([len(word.syllables) for word in line.words])

            # Строчки длиной больше border_syllables_count слогов не обрабатываем.
            if line_syllables_count > MetreClassifier.border_syllables_count or line_syllables_count == 0:
                continue
            error_border = 7
            if metre_name == "dolnik2" or metre_name == "dolnik3":
                error_border = 3
            if metre_name == "taktovik2" or metre_name == "taktovik3":
                error_border = 2
            pattern, strong_errors, weak_errors, analysis_errored = \
                PatternAnalyzer.count_errors(MetreClassifier.metres[metre_name],
                                             MetreClassifier._MetreClassifier__get_line_pattern(line),
                                             error_border)
            #print(MetreClassifier._MetreClassifier__get_line_pattern(line))
            if analysis_errored or len(pattern) == 0:
                errors_table.add_record(metre_name, l, strong_errors, weak_errors, pattern, True)
                continue
            #corrections = MetreClassifier._MetreClassifier__get_line_pattern_matching_corrections(line, l, pattern)[0]
            #accentuation_errors = len(corrections)
            #strong_errors += accentuation_errors
            errors_table.add_record(metre_name, l, strong_errors, weak_errors, pattern)
    return errors_table

def find_errors_meter_(markup, metre_name):
    num_lines = len(markup.lines)
    errors_table = ErrorsTable(num_lines)
    for l, line in enumerate(markup.lines):
        line_syllables_count = sum([len(word.syllables) for word in line.words])

        # Строчки длиной больше border_syllables_count слогов не обрабатываем.
        if line_syllables_count > MetreClassifier.border_syllables_count or line_syllables_count == 0:
            continue
        error_border = 7
        if metre_name == "dolnik2" or metre_name == "dolnik3":
            error_border = 3
        if metre_name == "taktovik2" or metre_name == "taktovik3":
            error_border = 2
        pattern, strong_errors, weak_errors, analysis_errored = \
            PatternAnalyzer.count_errors(MetreClassifier.metres[metre_name],
                                            MetreClassifier._MetreClassifier__get_line_pattern(line),
                                            error_border)
        #print(MetreClassifier._MetreClassifier__get_line_pattern(line))
        if analysis_errored or len(pattern) == 0:
            errors_table.add_record(metre_name, l, strong_errors, weak_errors, pattern, True)
            continue
        #corrections = MetreClassifier._MetreClassifier__get_line_pattern_matching_corrections(line, l, pattern)[0]
        #accentuation_errors = len(corrections)
        #strong_errors += accentuation_errors
        errors_table.add_record(metre_name, l, strong_errors, weak_errors, pattern)
    return errors_table


def check_meter(lines, meter_name):
    score = 0.
    count_syllables = 0
    for line in lines:
        markup = Markup.process_text(line, stress_predictor)  
        errors = find_errors_meter_(markup, meter_name).data[meter_name]
        weak = errors[0].weak_errors
        strong = errors[0].strong_errors
        count_syllables_ = sum([len(word.syllables) for word in markup.lines[0].words])
        count_syllables += count_syllables_
        if count_syllables_ > 0:
            score += (strong + weak)
            if strong + weak > count_syllables:
                print('error')
                print(line)
    return score / float(count_syllables)


def check_meter_fast(lines, meter_name, rhyme_scheme):
    error_score = 0.
    count_sylables = 0
    markup = Markup.process_text('\n'.join(lines), stress_predictor) 
    target_pattern = metre_patterns[meter_name]
    line_sylables = [0] * len(markup.lines)
    for i, line in enumerate(markup.lines):
        index = 0
        for w, word in enumerate(line.words):
            if len(word.syllables) == 0:
                raise Exception("в слове нет слогов в тексте", line.text)
            elif len(word.syllables) == 1: # односложное слово
                index += 1
            else:
                for syllable in word.syllables:
                    target = target_pattern[index % len(target_pattern)]
                    if target == 's' and syllable.stress == -1: 
                        error_score += 0.3
                    elif target == 'u' and syllable.stress != -1: 
                        error_score += 1
                    index += 1
            count_sylables += len(word.syllables)
            line_sylables[i] += len(word.syllables)

    # длины строк по парам рифмы
    diffs = []
    for i, j in scheme_map_dict[rhyme_scheme]:
        if i < len(line_sylables) and j < len(line_sylables):
            diffs.append(abs(line_sylables[i] - line_sylables[j]))
    if len(diffs) == 0:
        line_len_score = 0.
    else:
        line_len_score = np.exp(-np.mean(diffs) / 2)

    if count_sylables == 0:
        return 0.
    return 0.8 * (1 - error_score / float(count_sylables)) + 0.2 * line_len_score

def get_meter(lines):
    markup = Markup.process_text(lines, stress_predictor)  
    errors = find_errors_all_meters(markup)
    return errors.get_best_metre()

def make_meter_reward(coef):
    def meter_reward(completions, meter=None, rhyme_scheme=None, **kwargs):
        rewards = []
        
        for text, meter, scheme in zip(completions, meter, rhyme_scheme):
            lines = text.split('\n')
            f_lines = filter_lines(lines)
            
            score = check_meter_fast(f_lines, meter, scheme)
            rewards.append(coef * score)
        
        return rewards
    
    return meter_reward