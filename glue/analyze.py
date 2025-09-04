from __future__ import annotations

import time
import subprocess
import os


from core.config import SR


def analyze(path: str):
    print(path)
    wav_path = path + '.wav'

    subprocess.run(['ffmpeg', '-i', path, '-ar', str(SR), wav_path], check=True)

    try:
        result = analyze_wav(wav_path)
        print(result)
    except Exception as e:
        print(e)
        result = {
            'pitch_score': 50,
            'rhythm_score': 50,
            'emotion_level': 3,
            'overall': 51
        }
    os.remove(wav_path)

    return {
        "pitch": result['pitch_score'],
        "rhythm": result['rhythm_score'],
        "emotion": result['emotion_level'],
        "total": result['overall'],
        "content": content(result)
    }



def content(result) -> str:
    pitch = result['pitch_score']
    rhythm = result['rhythm_score']
    emotion = result['emotion_level'] * 20

    max_val = max(pitch, rhythm, emotion)
    min_val = min(pitch, rhythm, emotion)
    if max_val < 50:
        return '전체적으로 아쉽습니다.'
    elif min_val >= 70:
        return '훌륭합니다!'
    else:
        max_i = [pitch, rhythm, emotion].index(max_val)
        min_i = [pitch, rhythm, emotion].index(min_val)

        if max_val >= 70:
            max_m = [
                '음정이 매우 정확합니다.',
                '박자 감각이 뛰어납니다.',
                '감정 표현이 훌륭합니다.'
            ][max_i]
        else:
            max_m = [
                '음정이 좋습니다.',
                '박자 감각이 나쁘지 않습니다.',
                '감정 표현이 적절합니다.'
            ][max_i]

        if min_val == max_val:
            return max_m

        if min_val < 50:
            min_m = [
                '음정이 조금 아쉽지만,',
                '박자가 조금 아쉽지만,',
                '감정 표현이 조금 아쉽지만,'
            ][min_i]

            return ' '.join([min_m, max_m])

        return max_m



from .evaluate import evaluate_singing_with_emotion


def analyze_wav(path: str):
    return evaluate_singing_with_emotion(path, sr=SR)