# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks, resample
from matplotlib import rc

# 한글 폰트 설정
rc('font', family='Malgun Gothic')

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# -------------------------------------------------
# (A) 공통 함수들
# -------------------------------------------------
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def resample_signal(data, orig_fs, target_fs):
    duration = len(data) / orig_fs
    new_len = int(np.round(duration * target_fs))
    return resample(data, new_len)

def pad_nan(arr, length):
    if len(arr) < length:
        return np.concatenate([arr, np.full(length - len(arr), np.nan)])
    else:
        return arr[:length]


def read_biopac_txt(txt_path):
    """
    Biopac txt 파일을 읽어서
    (ch1, ch2) 형태로 리턴하는 예시 함수.
    
    - 한 줄에 4개 float 값이 콤마로 구분되어 있다고 가정.
    - 여기서는 인덱스 0, 2 컬럼만 골라서 (ch1, ch2)로 사용.
    - 실제 txt 파일 구조(어느 컬럼이 압력, PCG인지)에 따라
      인덱스를 조정해야 함.
    """
    ch1_list = []
    ch2_list = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 끝에 있는 ',' 제거 (있는 경우)
            line = line.rstrip(',')

            vals = line.split(',')
            if len(vals) < 4:
                # 4개 미만이면 스킵
                continue

            # 예: 인덱스 0 -> ch1, 인덱스 2 -> ch2
            ch1_list.append(float(vals[0]))
            ch2_list.append(float(vals[2]))

    ch1 = np.array(ch1_list, dtype=np.float32)
    ch2 = np.array(ch2_list, dtype=np.float32)
    return ch1, ch2


# -------------------------------------------------
# (B) 세션 처리 함수
# -------------------------------------------------
def process_session(
    biopac_txt_path,   # biopac txt 경로
    eko_wav_paths,     # {'eko_PCG':~, 'eko_ECG':~, 'eko_ECG_1':~, 'eko_ECG_2':~}
    phone_session_id,  # "전화번호_b_세션번호"
    plot_debug=True,   # 디버그 플롯 여부
    plot_result=True,  # 최종 전체 플롯
    out_folder=None,   # CSV 저장 폴더 (None이면 저장 안 함)
    height_threshold=0.3
):
    """
    Biopac + Eko 파일 한 세션 처리 후,
    감압 구간(t_start~t_end)에 대해 '원본 Eko 신호'를
    NaN padding 방식으로 한 CSV에 저장.

    디버그 플롯에서 [y/n/qqq] 입력 처리:
      - y   : 그대로 진행
      - n   : threshold + (Biopac)피크 탐색 구간 수정 후 재시도
      - qqq : 해당 세션 스킵
    """

    biopac_basename = os.path.splitext(os.path.basename(biopac_txt_path))[0]

    # -----------------------
    # 1) Biopac 로드 (p3 + PCG)
    # -----------------------
    ch1, ch2 = read_biopac_txt(biopac_txt_path) 
    # 샘플링레이트(가정): 2000Hz
    biopac_fs = 2000  

    # ch1 -> p3, ch2 -> pcg_raw (예시)
    # 앞부분 500샘플 버림(필요시 조정)
    p3      = ch1[500:]
    pcg_raw = ch2[500:]

    biopac_time = np.arange(len(pcg_raw)) / biopac_fs
    # PCG 대역 필터
    biopac_pcg = bandpass_filter(pcg_raw, 50, 200, biopac_fs)

    # -----------------------
    # 2) Eko 로드
    # -----------------------
    eko_signals = []
    eko_fs_list = []
    eko_labels  = ["eko_PCG", "eko_ECG", "eko_ECG_1", "eko_ECG_2"]

    for lbl in eko_labels:
        wav_path = eko_wav_paths.get(lbl, None)
        if wav_path is None:
            eko_signals.append(None)
            eko_fs_list.append(None)
            continue

        fs, data_wav = wavfile.read(wav_path)
        data_wav = data_wav.astype(np.float32)

        # PCG면 20~200Hz 필터
        if "PCG" in lbl:
            data_wav = bandpass_filter(data_wav, 20, 200, fs)

        eko_signals.append(data_wav)
        eko_fs_list.append(fs)

    eko_pcg    = eko_signals[0]  # eko_PCG
    eko_fs_pcg = eko_fs_list[0]

    # Eko는 예전처럼 "첫 20초"로 피크 검출(또는 전체). 여기서는 20초로 가정
    if (eko_pcg is not None) and (eko_fs_pcg is not None):
        eko_time_pcg = np.arange(len(eko_pcg)) / eko_fs_pcg
        eko_20s_idx = int(20 * eko_fs_pcg)
        eko_time_20s = eko_time_pcg[:eko_20s_idx]
        eko_signal_20s = eko_pcg[:eko_20s_idx]
    else:
        eko_time_20s, eko_signal_20s = None, None

    # -----------------------
    # (X) 디버그용 플롯 + 사용자 입력
    # -----------------------
    # Biopac 피크 탐색 구간 (기본 0~20초)
    biopac_peak_start_s = 0
    biopac_peak_end_s   = 20

    while True:
        # ----- Biopac: 지정된 구간에서 PCG 추출 -----
        peak_start_idx = int(biopac_peak_start_s * biopac_fs)
        peak_end_idx   = int(biopac_peak_end_s * biopac_fs)
        if peak_start_idx < 0:
            peak_start_idx = 0
        if peak_end_idx > len(biopac_pcg):
            peak_end_idx = len(biopac_pcg)

        biopac_time_for_peak = biopac_time[peak_start_idx:peak_end_idx]
        biopac_signal_for_peak = biopac_pcg[peak_start_idx:peak_end_idx]

        # ----- Biopac 피크 검출 -----
        if len(biopac_signal_for_peak) > 0:
            biopac_peaks, _ = find_peaks(
                biopac_signal_for_peak,
                height=height_threshold * np.max(biopac_signal_for_peak),
                distance=200
            )
            if len(biopac_peaks) > 0:
                last_peak_idx_local = biopac_peaks[-1]  # 구간 내부 인덱스
                # biopac_time_for_peak[0]를 offset으로 해서
                biopac_first_segment_time = biopac_time_for_peak[last_peak_idx_local]
            else:
                biopac_first_segment_time = None
        else:
            biopac_peaks = []
            biopac_first_segment_time = None

        # ----- Eko 피크 검출 (기존 방식: 20초 한정) -----
        if (eko_signal_20s is not None) and (len(eko_signal_20s) > 0):
            eko_peaks, _ = find_peaks(
                eko_signal_20s,
                height=height_threshold * np.max(eko_signal_20s),
                distance=200
            )
            if len(eko_peaks) > 0:
                eko_last_peak_idx = eko_peaks[-1]
                eko_first_segment_time = eko_time_20s[eko_last_peak_idx]
            else:
                eko_first_segment_time = None
        else:
            eko_peaks = []
            eko_first_segment_time = None

        # ----- 디버그 플롯 -----
        if plot_debug:
            plt.figure(figsize=(15, 10))

            # (1) Biopac PCG: 사용자가 지정한 구간
            plt.subplot(2, 1, 1)
            plt.plot(biopac_time_for_peak, biopac_signal_for_peak, label='Biopac PCG (Filtered)', color='blue')
            if len(biopac_peaks) > 0:
                plt.scatter(
                    biopac_time_for_peak[biopac_peaks],
                    biopac_signal_for_peak[biopac_peaks],
                    color='cyan', s=100, label='Biopac Peaks'
                )
            if biopac_first_segment_time is not None:
                plt.axvline(x=biopac_first_segment_time, color='red', linestyle='--', label='1st Segment Point')
            plt.title(f'Biopac PCG [구간: {biopac_peak_start_s}~{biopac_peak_end_s}s, threshold={height_threshold}]\n{phone_session_id}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid()

            # (2) Eko PCG (첫 20초)
            plt.subplot(2, 1, 2)
            if eko_signal_20s is not None:
                plt.plot(eko_time_20s, eko_signal_20s, label='Eko PCG (Filtered)', color='orange')
                if len(eko_peaks) > 0:
                    plt.scatter(
                        eko_time_20s[eko_peaks],
                        eko_signal_20s[eko_peaks],
                        color='cyan', s=100, label='Eko Peaks'
                    )
                if eko_first_segment_time is not None:
                    plt.axvline(x=eko_first_segment_time, color='red', linestyle='--', label='1st Segment Point')
            else:
                plt.text(0.3, 0.5, "Eko PCG 없음", fontsize=14, color='red')

            plt.title(f'Eko PCG (First 20s) [threshold={height_threshold}]\n{phone_session_id}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.show()

        print(f"[디버그] Biopac 1st segmentation point = {biopac_first_segment_time}")
        print(f"[디버그] Eko    1st segmentation point = {eko_first_segment_time}")

        # ----- 사용자 입력 -----
        user_in = input("[y=진행 / n=구간+threshold 변경 / qqq=스킵] : ").strip().lower()
        if user_in == 'y':
            # 그대로 진행
            break

        elif user_in == 'n':
            # (1) threshold 변경
            new_th_str = input("새로운 height threshold 입력 (예: 0.2): ").strip()
            try:
                new_th = float(new_th_str)
                height_threshold = new_th
            except ValueError:
                print("잘못된 입력입니다. threshold 변경 실패. 기존 값 유지.")
            
            # (2) 구간 변경
            range_str = input(f"새 Biopac 피크 탐색 구간 입력 (시작 끝, 기본={biopac_peak_start_s} {biopac_peak_end_s}): ").strip()
            try:
                range_vals = range_str.split()
                if len(range_vals) == 2:
                    new_start_s = float(range_vals[0])
                    new_end_s   = float(range_vals[1])
                    if new_start_s < new_end_s:
                        biopac_peak_start_s = new_start_s
                        biopac_peak_end_s   = new_end_s
                    else:
                        print("구간 입력이 잘못됨 (start >= end). 기존 값 유지.")
                else:
                    print("두 개의 숫자를 입력해야 합니다. 기존 값 유지.")
            except ValueError:
                print("구간 입력이 잘못됨. 기존 값 유지.")

            print(f"[알림] threshold={height_threshold}, 구간={biopac_peak_start_s}~{biopac_peak_end_s} 로 재시도합니다.\n")
            # while문으로 돌아가 다시 피크검출+플롯

        elif user_in == 'qqq':
            print(f"[SKIP] {phone_session_id} 세션 건너뜁니다.")
            return "skip"

        else:
            print("잘못된 입력입니다. (y/n/qqq 중 하나) 다시 시도하세요.")


    # -------------------------------------------------
    # (C) ~ (E) : 감압 구간 탐색, CSV 저장, 최종 플롯
    # -------------------------------------------------
    # 여기부터는 디버그(피크검출)단계를 통과한 뒤 실제 처리

    # 1차 세그먼트(마지막 피크) 이전 구간 삭제
    if biopac_first_segment_time is not None:
        idx_biopac = int(np.round(biopac_first_segment_time * biopac_fs))
        if idx_biopac < len(p3):
            p3 = p3[idx_biopac:]
            biopac_pcg = biopac_pcg[idx_biopac:]
        biopac_time = np.arange(len(biopac_pcg)) / biopac_fs

    if (eko_pcg is not None) and (eko_fs_pcg is not None) and (eko_first_segment_time is not None):
        idx_eko = int(np.round(eko_first_segment_time * eko_fs_pcg))
        for i in range(len(eko_signals)):
            fs_i = eko_fs_list[i]
            sig_i = eko_signals[i]
            if (fs_i is None) or (sig_i is None):
                continue
            cut_index = int(np.round(eko_first_segment_time * fs_i))
            if cut_index < len(sig_i):
                eko_signals[i] = sig_i[cut_index:]
            else:
                eko_signals[i] = np.array([])

    # (C) 리샘플링(2000Hz) 후, 병합
    biopac_fs_target = 2000
    eko_signals_resampled = []
    for sig, fs in zip(eko_signals, eko_fs_list):
        if (sig is None) or (fs is None) or (len(sig) == 0):
            eko_signals_resampled.append(None)
        else:
            eko_signals_resampled.append(resample_signal(sig, fs, biopac_fs_target))

    valid_lengths = [len(biopac_pcg)] + [len(s) for s in eko_signals_resampled if s is not None]
    if len(valid_lengths) == 0:
        print("[오류] 유효 신호가 없음.")
        return "skip"
    max_len = max(valid_lengths)

    common_time = np.arange(max_len) / biopac_fs_target
    biopac_pcg_pad = pad_nan(biopac_pcg, max_len)
    p3_pad         = pad_nan(p3, max_len)

    df = pd.DataFrame({
        "Time (s)": common_time,
        "biopac_PCG": biopac_pcg_pad,
        "p3_voltage": p3_pad
    })

    eko_labels = ["eko_PCG", "eko_ECG", "eko_ECG_1", "eko_ECG_2"]
    for lbl, sig_rs in zip(eko_labels, eko_signals_resampled):
        if sig_rs is not None:
            df[lbl] = pad_nan(sig_rs, max_len)
        else:
            df[lbl] = np.full(max_len, np.nan)

    # (감압 구간) p3 >= 0.6, 연속 15초 유지
    threshold = 0.6
    min_len_samples = 20 * biopac_fs_target  # 15초

    p3_arr = df["p3_voltage"].values
    time_arr = df["Time (s)"].values
    p3_above = (p3_arr >= threshold)

    start_idx = None
    found_segment = None
    for i in range(len(p3_above)):
        if p3_above[i]:
            if start_idx is None:
                start_idx = i
            else:
                if (i - start_idx) >= min_len_samples:
                    j = i
                    while j < len(p3_above) and p3_above[j]:
                        j += 1
                    found_segment = (start_idx, j)
                    break
        else:
            start_idx = None

    if found_segment:
        seg_start, seg_end = found_segment
        t_start = time_arr[seg_start]
        t_end   = time_arr[seg_end - 1]
        print(f"[감압구간] {t_start:.2f}s ~ {t_end:.2f}s")
    else:
        t_start, t_end = None, None
        print("[감압구간] 없음")

    # (D) 감압 구간 Eko '원본' 신호를 CSV로 저장 (NaN 패딩)
    if out_folder and (t_start is not None) and (t_end is not None):
        os.makedirs(out_folder, exist_ok=True)

        raw_sigs  = {}
        for lbl, fs, sig in zip(eko_labels, eko_fs_list, eko_signals):
            if (sig is None) or (fs is None) or (len(sig) == 0):
                raw_sigs[lbl] = np.array([])
                continue

            start_idx_eko = int(np.round(t_start * fs))
            end_idx_eko   = int(np.round(t_end   * fs))
            if start_idx_eko >= len(sig):
                raw_sigs[lbl]  = np.array([])
                continue
            if end_idx_eko > len(sig):
                end_idx_eko = len(sig)

            seg_sig = sig[start_idx_eko:end_idx_eko]
            raw_sigs[lbl] = seg_sig

        lens = [len(raw_sigs[l]) for l in eko_labels]
        max_len_raw = max(lens) if lens else 0

        dict_for_df = {}
        for lbl in eko_labels:
            dict_for_df[lbl] = pad_nan(raw_sigs[lbl], max_len_raw)

        df_raw = pd.DataFrame(dict_for_df)
        csv_filename = f"{biopac_basename}_EkoRawSegment.csv"
        out_csv_path = os.path.join(out_folder, csv_filename)
        df_raw.to_csv(out_csv_path, index=False, encoding='utf-8-sig')
        print(f"[CSV 저장] 감압구간 원본(채널별 NaN패딩) → {out_csv_path}")

    # (E) 최종 플롯
    if plot_result:
        plt.figure(figsize=(20, 15))
        cols = [c for c in df.columns if c != "Time (s)"]

        for i, col in enumerate(cols, start=1):
            ax = plt.subplot(len(cols), 1, i)
            ax.plot(df["Time (s)"], df[col], label=col)
            if (t_start is not None) and (t_end is not None):
                ax.axvspan(t_start, t_end, color='red', alpha=0.2, label='감압구간')
            ax.set_title(col)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
            ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    return "ok"


# -------------------------------------------------
# (C) biopac 폴더와 Eko 폴더를 순회하며 반복 실행
# -------------------------------------------------
def main():
    # (1) 폴더 지정 (사용자 환경에 맞춰 수정)
    biopac_folder = r"C:\Users\user\Desktop\천안임상\b_reformatted\biopac"  # 예시
    eko_folder    = r"C:\Users\user\Desktop\천안임상\b_reformatted\eko"     # 예시
    out_folder    = r"C:\Users\user\Desktop\천안임상\결과저장폴더B"        # 예시

    # (2) biopac 폴더 내 .txt 파일 목록
    txt_files = [f for f in os.listdir(biopac_folder) if f.lower().endswith(".txt")]

    skip_list = []

    for txt_file in txt_files:
        # 예: 01024566284_b_1.txt → split → [01024566284, b, 1]
        txt_basename = os.path.splitext(txt_file)[0]
        parts = txt_basename.split('_')
        if len(parts) < 3:
            print(f"파일명 파싱 불가능: {txt_file}, 건너뜁니다.")
            continue

        phone_num   = parts[0]
        # 'b' 라벨
        session_num = parts[2]  # 예: "1"

        txt_path = os.path.join(biopac_folder, txt_file)

        # Eko 폴더명 예: phone_num_b_session_num
        phone_session_id = f"{phone_num}_b_{session_num}"
        eko_subfolder_path = os.path.join(eko_folder, phone_session_id)

        if not os.path.isdir(eko_subfolder_path):
            print(f"Eko 폴더 없음: {eko_subfolder_path}, 건너뜁니다.")
            continue

        # wav 파일(PCG, ECG, ECG_1, ECG_2) 파싱
        wav_files = [wf for wf in os.listdir(eko_subfolder_path) if wf.lower().endswith(".wav")]
        eko_wav_map = {"eko_PCG": None, "eko_ECG": None, "eko_ECG_1": None, "eko_ECG_2": None}
        for wf in wav_files:
            lwf = wf.lower()
            full_wf_path = os.path.join(eko_subfolder_path, wf)
            if "ecg_2" in lwf:
                eko_wav_map["eko_ECG_2"] = full_wf_path
            elif "ecg_1" in lwf:
                eko_wav_map["eko_ECG_1"] = full_wf_path
            elif "ecg" in lwf:
                eko_wav_map["eko_ECG"] = full_wf_path
            else:
                eko_wav_map["eko_PCG"] = full_wf_path

        print(f"\n=== 처리 시작: {txt_file} → Eko 폴더: {phone_session_id} ===")

        ret = process_session(
            biopac_txt_path=txt_path,
            eko_wav_paths=eko_wav_map,
            phone_session_id=phone_session_id,
            plot_debug=True,   # 디버그 플롯
            plot_result=True,  # 최종 신호 플롯
            out_folder=out_folder
        )

        if ret == "skip":
            skip_list.append(phone_session_id)
            continue

        print(f"=== 처리 완료: {phone_session_id} ===")

    if skip_list:
        skip_txt_path = os.path.join(out_folder, "skip_list.txt")
        with open(skip_txt_path, "w", encoding="utf-8") as f:
            for sk in skip_list:
                f.write(sk + "\n")
        print(f"\n[완료] 스킵된 세션 목록을 '{skip_txt_path}' 에 저장했습니다.")
    else:
        print("\n[완료] 스킵된 세션이 없습니다.")


if __name__ == "__main__":
    main()
