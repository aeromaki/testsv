from __future__ import annotations  # 전방 참조 허용
from typing import Dict, List, Tuple, Union  # 반환 타입 유니온 등 사용
import numpy as np  # 수치 연산
import librosa  # 오디오·피처
import math  # 지수·로그

def evaluate_singing_with_emotion(
    wav_path: str,  # 입력 WAV 경로
    sr: int = 16000,  # 샘플레이트(감정·프로소디에 16kHz 권장)
    fmin_note: str = "C2",  # f0 탐색 하한
    fmax_note: str = "C6"  # f0 탐색 상한
) -> Dict[str, Union[float, int]]:
    # 음정(0~100), 박자(0~100), 감정(1~5 정수), 종합(0~100)을 반환하는 포화 방지 버전

    # ---------- 유틸 ----------
    def _clamp(x: float, lo: float, hi: float) -> float:
        # 값 제한
        return float(max(lo, min(hi, x)))

    def _hz_to_cents(x: np.ndarray) -> np.ndarray:
        # Hz를 A4=440 기준 cent로 변환
        return 1200.0 * np.log2(x / 440.0)

    def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
        # 중앙값 필터(잡음 완화, NaN 견고)
        if k <= 1 or x.size == 0:  # 짧은 윈도면
            return x  # 원본 반환
        pad: int = k // 2  # 패딩 길이
        xp: np.ndarray = np.pad(x, (pad, pad), mode="edge")  # 에지 패딩
        out: np.ndarray = np.empty_like(x)  # 출력 버퍼
        for i in range(x.size):  # 모든 위치
            out[i] = np.nanmedian(xp[i:i + k])  # 구간 중앙값
        return out  # 결과 반환

    def _softplus_margin(abs_err: np.ndarray, band: float, k: float) -> np.ndarray:
        # 허용 밴드 이후 초과를 소프트플러스(부드러운 마진)로 변환
        # band 이내도 0이 아닌 작은 벌점 → 만점 포화 방지
        x: np.ndarray = (abs_err - band) / (k + 1e-9)  # 스케일 정규화
        return np.log1p(np.exp(x)) * k  # softplus 역스케일

    def _shrink_to_range(q: float, lo: float, hi: float, alpha: float) -> float:
        # q∈[0,1]를 [lo,hi]로 변환하되 상단 포화 축소(α>1일수록 상단 수축)
        return float(lo + (hi - lo) * (q ** alpha))  # 거듭제곱 수축

    def _robust_z(x: np.ndarray) -> np.ndarray:
        # MAD 기반 Z-정규화(이상치 견고)
        m: float = float(np.median(x))  # 중앙값
        mad: float = float(np.median(np.abs(x - m)) + 1e-9)  # MAD
        return (x - m) / (1.4826 * mad)  # 표준화

    def _bell(x: float, mu: float, sigma: float) -> float:
        # 가우시안 종형 매핑
        return float(math.exp(-0.5 * ((x - mu) / (sigma + 1e-9)) ** 2))  # 0~1

    # ---------- 1) 로드 ----------
    y, _sr = librosa.load(wav_path, sr=sr, mono=True)  # 모노 로드
    if y.size < sr * 0.5:  # 0.5초 미만이면
        return {"pitch_score": 0.0, "rhythm_score": 0.0, "emotion_level": 1, "overall": 0.0}  # 조기 종료
    y = librosa.util.normalize(y)  # 레벨 정규화

    # ---------- 2) 공통 프레임·스펙트럼 ----------
    hop_length: int = 256  # 약 16ms 홉
    n_fft: int = 2048  # 약 128ms 창
    S: np.ndarray = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))  # STFT 크기
    rms: np.ndarray = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length, center=True)[0]  # RMS
    sc: np.ndarray = librosa.feature.spectral_centroid(S=S, sr=sr)[0]  # 센트로이드
    sflux: np.ndarray = librosa.onset.onset_strength(S=S)  # 플럭스
    times: np.ndarray = librosa.frames_to_time(np.arange(S.shape[1]), sr=_sr, hop_length=hop_length)  # 프레임 시각

    # ---------- 3) f0(pYIN) ----------
    fmin: float = float(librosa.note_to_hz(fmin_note))  # f0 하한
    fmax: float = float(librosa.note_to_hz(fmax_note))  # f0 상한
    f0, vflag, _ = librosa.pyin(  # pYIN 추정
        y=y, fmin=fmin, fmax=fmax, sr=_sr,
        hop_length=hop_length, frame_length=n_fft, fill_na=np.nan  # NaN 채움
    )  # pYIN 결과
    voiced_mask: np.ndarray = np.isfinite(f0) & vflag.astype(bool)  # 유효 발성
    if np.count_nonzero(voiced_mask) < 12:  # 최소 프레임 부족 시
        return {"pitch_score": 0.0, "rhythm_score": 0.0, "emotion_level": 1, "overall": 0.0}  # 종료
    f0v: np.ndarray = f0[voiced_mask]  # 유효 f0
    tv: np.ndarray = times[voiced_mask]  # 유효 시각
    cv: np.ndarray = _hz_to_cents(f0v)  # cent 스케일
    cv_smooth: np.ndarray = _median_filter(cv, 5)  # 중앙값 필터

    # ---------- 4) 노트 분절 ----------
    semi_round: np.ndarray = np.round(cv_smooth / 100.0)  # 반음 라운딩
    change_idx: np.ndarray = np.where(np.diff(semi_round) != 0)[0] + 1  # 상태 전이
    starts: np.ndarray = np.r_[0, change_idx]  # 시작 인덱스
    ends: np.ndarray = np.r_[change_idx, semi_round.size]  # 끝 인덱스(열림)
    min_dur: int = int(max(3, 0.08 / (hop_length / _sr)))  # ≥80ms 유지
    keep: np.ndarray = (ends - starts) >= min_dur  # 필터링
    starts, ends = starts[keep], ends[keep]  # 유효 구간
    if starts.size == 0:  # 노트 없음
        return {"pitch_score": 0.0, "rhythm_score": 0.0, "emotion_level": 1, "overall": 0.0}  # 종료

    # ---------- 5) 음정 오차(소프트 마진) ----------
    band: float = 20.0  # 허용 밴드(±20 cent로 약간 타이트)
    k_soft: float = 8.0  # 소프트플러스 스케일(클수록 완만)
    note_loss: List[float] = []  # 노트별 손실
    note_stab: List[float] = []  # 노트별 안정도
    for s, e in zip(starts, ends):  # 각 노트
        seg: np.ndarray = cv_smooth[s:e]  # cent 시계열
        if seg.size < 3:  # 너무 짧으면
            continue  # 스킵
        base: float = float(np.median(seg))  # 기준선(중앙값)
        resid: np.ndarray = seg - base  # 잔차
        abs_err: np.ndarray = np.abs(resid)  # 절대 오차
        loss_seg: np.ndarray = _softplus_margin(abs_err, band=band, k=k_soft)  # 소프트 마진 손실
        note_loss.append(float(np.median(loss_seg)))  # 노트 대표 손실
        dseg: np.ndarray = np.diff(seg)  # 프레임 변화
        note_stab.append(float(np.median(np.abs(dseg))))  # 안정도

    if len(note_loss) == 0:  # 손실 없음
        return {"pitch_score": 0.0, "rhythm_score": 0.0, "emotion_level": 1, "overall": 0.0}  # 종료

    pitch_err: float = float(np.median(note_loss))  # 대표 오차(cent 등가)
    pitch_stab: float = float(np.median(note_stab))  # 대표 안정도(cent/프레임)
    voiced_ratio: float = float(np.sum(voiced_mask) / len(f0))  # 유효 발성 비율

    # ---------- 6) 리듬(IOI + 정렬) ----------
    onsets: np.ndarray = librosa.onset.onset_detect(  # 온셋 검출
        y=y, sr=_sr, hop_length=hop_length, backtrack=True,
        pre_max=16, post_max=16, pre_avg=64, post_avg=64, delta=0.1, wait=1
    )  # 온셋 프레임
    onset_t: np.ndarray = librosa.frames_to_time(onsets, sr=_sr, hop_length=hop_length)  # 온셋 시각
    if onset_t.size >= 4:  # 충분한 온셋
        ioi: np.ndarray = np.diff(onset_t)  # 간격
        ioi = ioi[ioi > 0.06]  # 60ms 미만 제거
        if ioi.size >= 3:  # 충분한 IOI
            tactus: float = float(np.median(ioi))  # 대표 주기
            tc: float = float(np.mean(ioi) / (np.std(ioi) + 1e-9))  # 일관성(클수록 좋음)
            phase: np.ndarray = np.mod(onset_t, tactus) / (tactus + 1e-9)  # 위상
            off: np.ndarray = np.minimum(phase, 1.0 - phase)  # 오프셋 비
            align: float = float(1.0 - 2.0 * np.mean(off))  # 0~1 정렬
        else:  # IOI 부족
            tc, align = 1.5, 0.6  # 중립
    else:  # 온셋 부족
        tc, align = 1.5, 0.6  # 중립

    # ---------- 7) 감정(Intensity/Consistency/Dynamics) ----------
    # 7-1 Intensity: 에너지·피치 범위·비브라토 깊이(과포화 방지용 tanh)
    rms_std: float = float(np.std(rms))  # 에너지 표준편차
    rms_rng: float = float(np.percentile(rms, 95) - np.percentile(rms, 5))  # 에너지 범위
    win_v: int = max(3, int(0.30 / (hop_length / _sr)))  # ~300ms 평균
    ker_v: np.ndarray = np.ones(win_v, dtype=float) / float(win_v)  # 평균 커널
    trend: np.ndarray = np.convolve(cv, ker_v, mode="same")  # f0 트렌드
    vib_resid: np.ndarray = cv - trend  # 비브라토 잔차
    vib_depth: float = float(np.median(np.abs(vib_resid)))  # 비브라토 깊이(cent)
    f0_rng: float = float(np.nanpercentile(cv, 95) - np.nanpercentile(cv, 5))  # 피치 범위
    I_lin: float = (  # 선형 결합
        0.35 * (rms_std / 0.06) +  # 경험적 스케일
        0.25 * (rms_rng / 0.12) +  # 경험적 스케일
        0.25 * (f0_rng / 160.0) +  # 경험적 스케일
        0.15 * (vib_depth / 40.0)  # 경험적 스케일
    )  # 강도 선형 점수
    I: float = float(np.tanh(I_lin))  # tanh로 0~<1 포화(만점 방지)

    # 7-2 Consistency: 프레이즈별 A/V 중앙값 분산의 역수
    z_rms: np.ndarray = _robust_z(rms)  # RMS 표준화
    z_flux: np.ndarray = _robust_z(sflux)  # 플럭스 표준화
    z_sc: np.ndarray = _robust_z(sc)  # 센트로이드 표준화
    A_f: np.ndarray = 0.6 * z_rms + 0.4 * z_flux  # A 프레임 점수
    V_f: np.ndarray = -0.5 * z_sc - 0.2 * _robust_z(np.abs(np.diff(np.pad(sc, (1, 0), mode="edge"))))  # V 프레임 점수
    # 프레이즈 경계 선정(온셋 기반, 마지막 프레임 포함)
    bnds: np.ndarray = np.r_[onsets, S.shape[1] - 1] if onsets.size >= 1 else np.array([0, S.shape[1] - 1], dtype=int)  # 경계
    A_ph: List[float] = []  # 프레이즈 A
    V_ph: List[float] = []  # 프레이즈 V
    for i in range(len(bnds) - 1):  # 각 프레이즈
        s_idx: int = int(bnds[i])  # 시작
        e_idx: int = int(bnds[i + 1])  # 끝
        A_ph.append(float(np.median(A_f[s_idx:e_idx + 1])))  # A 중앙값
        V_ph.append(float(np.median(V_f[s_idx:e_idx + 1])))  # V 중앙값
    var_A: float = float(np.var(A_ph)) if len(A_ph) >= 2 else 0.2  # A 분산
    var_V: float = float(np.var(V_ph)) if len(V_ph) >= 2 else 0.2  # V 분산
    C: float = float(math.exp(- (var_A + var_V) / 2.0))  # 일관성(0~1, 과포화 완화)

    # 7-3 Dynamics: 프레이즈 수준 변동의 종형 매핑(상한 <1)
    pr: List[float] = []  # RMS 범위 리스트
    pf: List[float] = []  # f0 범위 리스트
    for i in range(len(bnds) - 1):  # 프레이즈 반복
        s_idx: int = int(bnds[i])  # 시작
        e_idx: int = int(bnds[i + 1])  # 끝
        r_seg: np.ndarray = rms[s_idx:e_idx + 1]  # RMS 구간
        pr.append(float(np.percentile(r_seg, 95) - np.percentile(r_seg, 5)))  # RMS 범위
        t_s: float = float(librosa.frames_to_time(s_idx, sr=_sr, hop_length=hop_length))  # 시작 시각
        t_e: float = float(librosa.frames_to_time(e_idx, sr=_sr, hop_length=hop_length))  # 끝 시각
        m: np.ndarray = (tv >= t_s) & (tv <= t_e)  # f0 마스크
        if np.any(m):  # 유효 f0가 있으면
            fseg: np.ndarray = cv[m]  # f0 구간
            pf.append(float(np.nanpercentile(fseg, 95) - np.nanpercentile(fseg, 5)))  # f0 범위
    cv_pr: float = float(np.std(pr) / (np.mean(pr) + 1e-9)) if len(pr) >= 2 else 0.3  # RMS 변동계수
    cv_pf: float = float(np.std(pf) / (np.mean(pf) + 1e-9)) if len(pf) >= 2 else 0.3  # f0 변동계수
    D: float = 0.6 * _bell(cv_pr, 0.35, 0.20) + 0.4 * _bell(cv_pf, 0.45, 0.25)  # 다이내믹스 0~<1

    # 7-4 Emotion 연속점수와 이산화(상단 수축 + 비균등 임계)
    emo_q: float = float(_clamp(0.5 * I + 0.3 * C + 0.2 * D, 0.0, 1.0))  # 0~1 연속
    emo_q_shrunk: float = float(emo_q ** 1.4)  # 상단 수축으로 만점 방지
    emotion_cont: float = _shrink_to_range(emo_q_shrunk, lo=30.0, hi=96.0, alpha=1.0)  # 30~96 맵
    # 비균등 임계(상단 엄격): [30, 48, 66, 82, 96]
    if emotion_cont < 48.0:  # 30~47.9
        emotion_level: int = 1  # 낮음
    elif emotion_cont < 66.0:  # 48~65.9
        emotion_level = 2  # 보통 이하
    elif emotion_cont < 82.0:  # 66~81.9
        emotion_level = 3  # 보통
    elif emotion_cont < 92.0:  # 82~91.9
        emotion_level = 4  # 좋음
    else:  # 92~96
        emotion_level = 5  # 매우 좋음

    # ---------- 8) 점수화(음정·박자, 상단 수축) ----------
    # 음정 품질 지표 → [40,98)으로 수축(프로도 쉽게 100 불가)
    q_err: float = float(math.exp(-pitch_err / 22.0))  # 오차 항
    q_stab: float = float(math.exp(-pitch_stab / 12.0))  # 안정 항
    q_cov: float = float(_clamp((voiced_ratio - 0.5) / 0.4, 0.0, 1.0))  # 유효 구간 커버리지
    q_mix: float = float(0.6 * q_err + 0.25 * q_stab + 0.15 * q_cov)  # 혼합 품질
    pitch_score: float = _shrink_to_range(q_mix, lo=40.0, hi=98.0, alpha=1.8)  # 상단 수축

    # 리듬 품질 지표 → [30,97) 수축
    tc_term: float = 1.0 - math.exp(-math.log1p(tc) / math.log(1.0 + 4.0))  # 0~1
    ra_term: float = _clamp(align, 0.0, 1.0)  # 0~1
    r_mix: float = float(0.7 * tc_term + 0.3 * ra_term)  # 혼합
    rhythm_score: float = _shrink_to_range(r_mix, lo=30.0, hi=97.0, alpha=1.6)  # 수축


    def fix(score: float, mean: float) -> float:
        z = (score - mean) / min(100 - mean, mean)
        return 50 + min(max(-50, 50 * z), 50)

    pitch_score = fix(pitch_score, 80)
    rhythm_score = fix(rhythm_score, 50)

    # ---------- 9) 종합 ----------
    overall: float = _clamp(0.7 * pitch_score + 0.2 * rhythm_score + 0.1 * emotion_cont, 0.0, 100.0)  # 가중 합

    # ---------- 10) 반환 ----------
    return {
        "pitch_score": float(round(pitch_score, 2)),  # 음정 점수
        "rhythm_score": float(round(rhythm_score, 2)),  # 박자 점수
        "emotion_level": int(emotion_level),  # 감정 레벨(1~5 정수)
        "overall": float(round(overall, 2))  # 종합 점수
    }  # 결과 딕셔너리