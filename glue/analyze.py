import time

def analyze(path: str):
    print(path)
    time.sleep(5)
    return {
        "pitch": 85,
        "rhythm": 75,
        "emotion": 3,
        "total": 75,
        "content": "피치가 제법 정확합니다."
    }