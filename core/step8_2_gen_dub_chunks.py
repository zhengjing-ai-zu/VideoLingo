import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config_utils import load_key
from core.all_whisper_methods.audio_preprocess import get_audio_duration
from core.step8_1_gen_audio_task import time_diff_seconds
import datetime
import re
from core.all_tts_functions.estimate_duration import init_estimator, estimate_duration
from rich import print as rprint


"""
音频生成

step8_1_gen_audio_task.py：生成音频任务（选择 TTS 方案）。
step8_2_gen_dub_chunks.py：按字幕块生成配音音频。
step9_extract_refer_audio.py：提取参考音频（可能用于声音克隆）。
"""


INPUT_EXCEL = "output/audio/tts_tasks.xlsx"
OUTPUT_EXCEL = "output/audio/tts_tasks.xlsx"
SRC_SRT = "output/src.srt"
TRANS_SRT = "output/trans.srt"
MAX_MERGE_COUNT = 5
AUDIO_FILE = 'output/audio/raw.mp3'
ESTIMATOR = None

def calc_if_too_fast(est_dur, tol_dur, duration, tolerance):
    accept = load_key("speed_factor.accept") # Maximum acceptable speed factor
    if est_dur / accept > tol_dur:  # Even max speed factor cannot adapt
        return 2
    elif est_dur > tol_dur:  # Speed adjustment needed within acceptable range
        return 1
    elif est_dur < duration - tolerance:  # Speaking speed too slow
        return -1
    else:  # Normal speaking speed
        return 0

def merge_rows(df, start_idx, merge_count):
    """Merge multiple rows and calculate cumulative values"""
    merged = {
        'est_dur': df.iloc[start_idx]['est_dur'],
        'tol_dur': df.iloc[start_idx]['tol_dur'],
        'duration': df.iloc[start_idx]['duration']
    }
    
    while merge_count < MAX_MERGE_COUNT and (start_idx + merge_count) < len(df):
        next_row = df.iloc[start_idx + merge_count]
        merged['est_dur'] += next_row['est_dur']
        merged['tol_dur'] += next_row['tol_dur']
        merged['duration'] += next_row['duration']
        
        speed_flag = calc_if_too_fast(
            merged['est_dur'],
            merged['tol_dur'],
            merged['duration'],
            df.iloc[start_idx + merge_count]['tolerance']
        )
        
        if speed_flag <= 0 or merge_count == 2:
            df.at[start_idx + merge_count, 'cut_off'] = 1
            return merge_count + 1
        
        merge_count += 1
    
    # If no suitable merge point is found
    if merge_count >= MAX_MERGE_COUNT or (start_idx + merge_count) >= len(df):
        df.at[start_idx + merge_count - 1, 'cut_off'] = 1
    return merge_count

def analyze_subtitle_timing_and_speed(df):
    rprint("[🔍 Analyzing] Calculating subtitle timing and speed...")
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()
    TOLERANCE = load_key("tolerance")
    whole_dur = get_audio_duration(AUDIO_FILE)
    df['gap'] = 0.0  # Initialize gap column
    for i in range(len(df) - 1):
        current_end = datetime.datetime.strptime(df.loc[i, 'end_time'], '%H:%M:%S.%f').time()
        next_start = datetime.datetime.strptime(df.loc[i + 1, 'start_time'], '%H:%M:%S.%f').time()
        df.loc[i, 'gap'] = time_diff_seconds(current_end, next_start, datetime.date.today())
    
    # Set the gap for the last line
    last_end = datetime.datetime.strptime(df.iloc[-1]['end_time'], '%H:%M:%S.%f').time()
    last_end_seconds = (last_end.hour * 3600 + last_end.minute * 60 + 
                       last_end.second + last_end.microsecond / 1000000)
    df.iloc[-1, df.columns.get_loc('gap')] = whole_dur - last_end_seconds
    
    df['tolerance'] = df['gap'].apply(lambda x: TOLERANCE if x > TOLERANCE else x)
    df['tol_dur'] = df['duration'] + df['tolerance']
    df['est_dur'] = df.apply(lambda x: estimate_duration(x['text'], ESTIMATOR), axis=1)

    ## Calculate speed indicators
    accept = load_key("speed_factor.accept") # Maximum acceptable speed factor
    def calc_if_too_fast(row):
        est_dur = row['est_dur']
        tol_dur = row['tol_dur']
        duration = row['duration']
        tolerance = row['tolerance']
        
        if est_dur / accept > tol_dur:  # Even max speed factor cannot adapt
            return 2
        elif est_dur > tol_dur:  # Speed adjustment needed within acceptable range
            return 1
        elif est_dur < duration - tolerance:  # Speaking speed too slow
            return -1
        else:  # Normal speaking speed
            return 0
    
    df['if_too_fast'] = df.apply(calc_if_too_fast, axis=1)
    return df

def process_cutoffs(df):
    rprint("[✂️ Processing] Generating cutoff points...")
    df['cut_off'] = 0  # Initialize cut_off column
    df.loc[df['gap'] >= load_key("tolerance"), 'cut_off'] = 1  # Set to 1 when gap is greater than TOLERANCE
    idx = 0
    while idx < len(df):
        # Process marked split points
        if df.iloc[idx]['cut_off'] == 1:
            if df.iloc[idx]['if_too_fast'] == 2:
                rprint(f"[⚠️ Warning] Line {idx} is too fast and cannot be fixed by speed adjustment")
            idx += 1
            continue

        # Process the last line
        if idx + 1 >= len(df):
            df.at[idx, 'cut_off'] = 1
            break

        # Process normal or slow lines
        if df.iloc[idx]['if_too_fast'] <= 0:
            if df.iloc[idx + 1]['if_too_fast'] <= 0:
                df.at[idx, 'cut_off'] = 1
                idx += 1
            else:
                idx += merge_rows(df, idx, 1)
        # Process fast lines
        else:
            idx += merge_rows(df, idx, 1)
    
    return df

def gen_dub_chunks():
    rprint("[🎬 Starting] Generating dubbing chunks...")
    df = pd.read_excel(INPUT_EXCEL)
    
    rprint("[📊 Processing] Analyzing timing and speed...")
    df = analyze_subtitle_timing_and_speed(df)
    
    rprint("[✂️ Processing] Processing cutoffs...")
    df = process_cutoffs(df)

    rprint("[📝 Reading] Loading transcript files...")
    content = open(TRANS_SRT, "r", encoding="utf-8").read()
    ori_content = open(SRC_SRT, "r", encoding="utf-8").read()
    
    # Process subtitle content
    content_lines = []
    ori_content_lines = []
    
    # Process translated subtitles
    for block in content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 3:
            text = ' '.join(lines[2:])
            text = re.sub(r'\([^)]*\)|（[^）]*）', '', text).strip().replace('-', '')
            content_lines.append(text)
            
    # Process source subtitles (same structure)
    for block in ori_content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 3:
            text = ' '.join(lines[2:])
            text = re.sub(r'\([^)]*\)|（[^）]*）', '', text).strip().replace('-', '')
            ori_content_lines.append(text)

    # Match processing
    df['lines'] = None
    df['src_lines'] = None
    last_idx = 0

    def clean_text(text):
        """clean space and punctuation"""
        if not text or not isinstance(text, str):
            return ''
        return re.sub(r'[^\w\s]|[\s]', '', text)

    for idx, row in df.iterrows():
        target = clean_text(row['text'])
        matches = []
        current = ''
        match_indices = []  # Store indices for matching lines
        
        for i in range(last_idx, len(content_lines)):
            line = content_lines[i]
            cleaned_line = clean_text(line)
            current += cleaned_line
            matches.append(line)  # 存储原始文本
            match_indices.append(i)
            
            if current == target:
                df.at[idx, 'lines'] = matches
                df.at[idx, 'src_lines'] = [ori_content_lines[i] for i in match_indices]
                last_idx = i + 1
                break
        else:  # If no match is found
            rprint(f"[❌ Error] Matching failed at line {idx}:")
            rprint(f"Target: '{target}'")
            rprint(f"Current: '{current}'")
            raise ValueError("Matching failed")

    # Save results
    df.to_excel(OUTPUT_EXCEL, index=False)
    rprint("[✅ Complete] Matching completed successfully!")

if __name__ == "__main__":
    gen_dub_chunks()