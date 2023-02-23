import numpy as np
import pandas as pd
from collections import Counter

# 임시 import
from feature import feature_selection

# This outcome is derived from df.y
# len(df.y.unique()) = 55
# len(df.description.unique()) = 16
# len(df.events.unique()) = 42
BATTER_NOOP_OUTCOME = '''
            foul ball called_strike swinging_strike blocked_ball foul_bunt 
            swinging_strike_blocked missed_bunt wild_pitch passed_ball pitchout 
            stolen_base_2b game_advisory foul_pitchout swinging_pitchout 
            intent_ball ejection pickoff_error_2b stolen_base_3b stolen_base_home other_out
            '''.split()
BATTER_OUT_OUTCOME = '''
            field_out strikeout grounded_into_double_play 
            sac_bunt foul_tip force_out field_error sac_fly 
            double_play fielders_choice fielders_choice_out
            strikeout_double_play bunt_foul_tip sac_fly_double_play 
            sac_bunt_double_play triple_play
            '''.split()
BATTER_ADVANCE_OUTCOME = ['walk', 'hit_by_pitch', 'intent_walk', 'catcher_interf']
BATTER_HIT_OUTCOME = ['single', 'double', 'triple', 'home_run']

BATTER_NOOP_PLIST = {k: 'noop' for k in BATTER_NOOP_OUTCOME}
BATTER_OUT_PLIST = {k: 'out' for k in BATTER_OUT_OUTCOME}
BATTER_ADVANCE_PLIST = {k: 'advance' for k in BATTER_ADVANCE_OUTCOME}
BATTER_HIT_PLIST = {k: 'hit' for k in BATTER_HIT_OUTCOME}

def merge_label_inplace(df):
    '''
    events와 description을 합쳐서 y라는 새로운 column을 만든다.
    '''
    df['y'] = np.where(df['events'].isna(), df['description'], df['events'])

def classify_label_inplace(df, separate_hit=1):
    '''
    학습에 필요한 레이블로 변환한다.

    separate_hit=1  -> {1루타, 2루타, 3루타, 홈런}  1 class
    separate_hit=2  -> 1루타, {2루타, 3루타, 홈런}  2 class
    separate_hit=3  -> 1루타, {2루타, 3루타}, 홈런  3 class
    separate_hit=4  -> 1루타, 2루타, 3루타, 홈런    4 class
    '''

    assert separate_hit in range(1, 4+5)

    plist = BATTER_HIT_PLIST

    if separate_hit == 1:
        plist = {k: 'hit' for k in plist}
    elif separate_hit == 2:
        plist['double'] = 'XBH'
        plist['triple'] = 'XBH'
        plist['home_run'] = 'XBH'
    elif separate_hit == 3:
        plist['double'] = 'XBH'
        plist['triple'] = 'XBH'
        plist['home_run'] = 'home_run'
    elif separate_hit == 4:
        plist = {k: k for k in plist}

    mapping = BATTER_NOOP_PLIST | BATTER_OUT_PLIST | BATTER_ADVANCE_PLIST | plist

    df['y_label'] = df['y'].map(mapping)

    breakpoint()


if __name__ == '__main__':
    df = pd.read_csv('data_csv/sorted-2015-to-2021(named,alpha)_v3.csv')
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    merge_label_inplace(df)
    classify_label_inplace(df)
    breakpoint()
    # df.drop(['events', 'description'], axis=1, inplace=True)
    # df.to_csv('data_csv/2021-06(named,alpha)_v2.csv', index=False)

'''
y의 종류
['field_out', 'foul', 'ball', 'strikeout', 'called_strike', 'swinging_strike', 
'grounded_into_double_play', 'single', 'blocked_ball', 'sac_bunt', 'walk', 'double', 
'foul_tip', 'foul_bunt', 'home_run', 'force_out', 'triple', 'caught_stealing_2b', 
'field_error', 'sac_fly', 'double_play', 'hit_by_pitch', 'swinging_strike_blocked', 
'caught_stealing_home', 'fielders_choice', 'missed_bunt', 'fielders_choice_out', 
'strikeout_double_play', 'other_out', 'bunt_foul_tip', 'pickoff_3b', 'catcher_interf', 
'sac_fly_double_play', 'caught_stealing_3b', 'wild_pitch', 'passed_ball', 'pitchout', 
'stolen_base_2b', 'pickoff_2b', 'pickoff_1b', 'sac_bunt_double_play', 'pickoff_caught_stealing_2b', 
'game_advisory', 'triple_play', 'foul_pitchout', 'pickoff_caught_stealing_3b', 
'pickoff_caught_stealing_home', 'runner_double_play', 'swinging_pitchout', 'stolen_base_home', 
'intent_walk', 'intent_ball', 'ejection', 'pickoff_error_2b', 'stolen_base_3b']

주자 out (10) : caught_stealing_2b caught_stealing_home pickoff_3b caught_stealing_3b pickoff_2b 
            pickoff_1b pickoff_caught_stealing_2b pickoff_caught_stealing_3b pickoff_caught_stealing_home runner_double_play
<< 2아웃에서만 기록되는거 같음

나머지 (21) : foul ball called_strike swinging_strike blocked_ball foul_bunt swinging_strike_blocked missed_bunt wild_pitch passed_ball pitchout stolen_base_2b
        game_advisory foul_pitchout swinging_pitchout intent_ball ejection pickoff_error_2b stolen_base_3b stolen_base_home other_out


타자 out (16) : field_out strikeout grounded_into_double_play sac_bunt foul_tip force_out field_error sac_fly double_play fielders_choice fielders_choice_out
            strikeout_double_play bunt_foul_tip sac_fly_double_play sac_bunt_double_play triple_play

타자 진루 (4) : walk hit_by_pitch intent_walk catcher_interf
타자 안타 (4) : single double home_run triple

{
    'ball': 1482062, 'foul': 816516, 'called_strike': 721205, 'field_out': 481258, 'swinging_strike': 308095, 'strikeout': 267419, 
    'single': 174331, 'blocked_ball': 101660, 'walk': 96497, 'double': 54198, 'home_run': 38585, 'force_out': 24401, 'foul_tip': 24216, 
    'grounded_into_double_play': 23417, 'hit_by_pitch': 12347, 'foul_bunt': 11732, 'swinging_strike_blocked': 9229, 'field_error': 8840, 
    'sac_fly': 7694, 'sac_bunt': 5757, 'triple': 5335, 'intent_ball': 4833, 'double_play': 2815, 'fielders_choice': 2451, 
    'missed_bunt': 2382, 'fielders_choice_out': 2021, 'intent_walk': 1923, 'caught_stealing_2b': 1317, 'strikeout_double_play': 939, 
    'pitchout': 702, 'other_out': 249, 'bunt_foul_tip': 224, 'catcher_interf': 196, 'sac_fly_double_play': 111, 'caught_stealing_3b': 88, 
    'pickoff_1b': 65, 'caught_stealing_home': 48, 'pickoff_2b': 47, 'wild_pitch': 35, 'triple_play': 31, 'pickoff_caught_stealing_2b': 12, 
    'pickoff_3b': 11, 'game_advisory': 10, 'sac_bunt_double_play': 9, 'pickoff_caught_stealing_3b': 9, 'passed_ball': 7, 'stolen_base_2b': 7, 
    'pickoff_caught_stealing_home': 7, 'swinging_pitchout': 5, 'foul_pitchout': 3, 'ejection': 3, 'runner_double_play': 2, 'stolen_base_home': 1, 
    'pickoff_error_2b': 1, 'stolen_base_3b': 1
}

전체 4695359
'''