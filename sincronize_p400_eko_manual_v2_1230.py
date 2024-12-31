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


# -------------------------------------------------
# (B) 세션 처리 함수
#     - phone_session_id: "전화번호_a_세션번호" 형태(예: "01033378505_a_1")
# -------------------------------------------------
def process_session(
    pdt_path,         # p400 .pdt 경로
    eko_wav_paths,    # {'eko_PCG':~, 'eko_ECG':~, 'eko_ECG_1':~, 'eko_ECG_2':~}
    phone_session_id, # "전화번호_a_세션번호"
    plot_debug=True,  # 디버그 플롯 (피크 탐색용)
    plot_result=True, # 최종 전체 플롯
    out_folder=None,  # CSV 저장 폴더 (None이면 저장 안 함)
    height_threshold=0.3
):
    """
    p400 + Eko 파일 한 세션 처리 후,
    감압 구간(t_start~t_end)에 대해 '원본 Eko 신호'를
    NaN padding 방식으로 한 CSV에 저장.

    디버그 플롯에서 [y/n/qqq] 입력 처리:
      - y   : 그대로 진행
      - n   : threshold + (p400)피크 탐색 구간 수정 후 재시도
      - qqq : 해당 세션 스킵
    """

    pdt_basename = os.path.splitext(os.path.basename(pdt_path))[0]

    # 1) p400 파일 로드
    with open(pdt_path, 'rb') as f:
        pdt_data = f.read()
    header_size = 16
    data = pdt_data[header_size:]
    if len(data) % 16 != 0:
        data = data[: (len(data) - len(data) % 16)]

    signal_data = np.frombuffer(data, dtype=np.float32)

    # 여기서는 channel1=p3, channel2=PCG
    channel0 = signal_data[0::4]
    channel1 = signal_data[1::4]  # p3
    channel2 = signal_data[2::4]  # PCG
    channel3 = signal_data[3::4]

    p400_fs = 2000
    # 앞부분 500샘플 버림
    p3      = channel1[500:]
    pcg_raw = channel2[500:]

    p400_time = np.arange(len(pcg_raw)) / p400_fs
    p400_pcg  = bandpass_filter(pcg_raw, 50, 200, p400_fs)  # PCG 필터링

    # 2) Eko 파일 로드
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

    # Eko PCG
    eko_pcg    = eko_signals[0]
    eko_fs_pcg = eko_fs_list[0]
    # Eko 첫 20초(디버그용)
    if (eko_pcg is not None) and (eko_fs_pcg is not None):
        eko_time_pcg = np.arange(len(eko_pcg)) / eko_fs_pcg
        eko_20s_idx = int(20 * eko_fs_pcg)
        eko_time_20s = eko_time_pcg[:eko_20s_idx]
        eko_signal_20s = eko_pcg[:eko_20s_idx]
    else:
        eko_time_20s, eko_signal_20s = None, None

    # --------------------------
    # (X) 디버그 플롯 & 사용자 입력
    # --------------------------
    # p400 피크 탐색 구간(기본 0~20초)
    p400_peak_start_s = 0
    p400_peak_end_s   = 20

    while True:
        # 1) p400: 지정된 구간 슬라이싱
        peak_start_idx = int(p400_peak_start_s * p400_fs)
        peak_end_idx   = int(p400_peak_end_s * p400_fs)
        if peak_start_idx < 0:
            peak_start_idx = 0
        if peak_end_idx > len(p400_pcg):
            peak_end_idx = len(p400_pcg)

        p400_time_for_peak   = p400_time[peak_start_idx:peak_end_idx]
        p400_signal_for_peak = p400_pcg[peak_start_idx:peak_end_idx]

        # 2) p400 피크 찾기
        if len(p400_signal_for_peak) > 0:
            p400_peaks, _ = find_peaks(
                p400_signal_for_peak,
                height=height_threshold * np.max(p400_signal_for_peak),
                distance=200
            )
            if len(p400_peaks) > 0:
                last_peak_idx_local = p400_peaks[-1]
                p400_first_segment_time = p400_time_for_peak[last_peak_idx_local]
            else:
                p400_first_segment_time = None
        else:
            p400_peaks = []
            p400_first_segment_time = None

        # 3) Eko (첫 20초) 피크 찾기
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

        # 4) 디버그 플롯
        if plot_debug:
            plt.figure(figsize=(15, 10))

            # (1) p400 PCG (사용자 지정 구간)
            plt.subplot(2, 1, 1)
            plt.plot(p400_time_for_peak, p400_signal_for_peak,
                     label='p400 PCG (Filtered)', color='blue')
            if len(p400_peaks) > 0:
                plt.scatter(
                    p400_time_for_peak[p400_peaks],
                    p400_signal_for_peak[p400_peaks],
                    color='cyan', s=100, label='p400 Peaks'
                )
            if p400_first_segment_time is not None:
                plt.axvline(x=p400_first_segment_time, color='red', linestyle='--', label='1st Segment Point')
            plt.title(f"p400 PCG [구간: {p400_peak_start_s}~{p400_peak_end_s}s, threshold={height_threshold}]\n{phone_session_id}")
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

            plt.title(f"Eko PCG (First 20s) [threshold={height_threshold}]\n{phone_session_id}")
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.show()

        print(f"[디버그] p400 1st segmentation point = {p400_first_segment_time}")
        print(f"[디버그] Eko  1st segmentation point = {eko_first_segment_time}")

        # 5) 사용자 입력
        user_in = input("[y=진행 / n=구간+threshold 변경 / qqq=스킵] : ").strip().lower()
        if user_in == 'y':
            break
        elif user_in == 'n':
            # threshold 변경
            new_th_str = input("새로운 height threshold 입력 (예: 0.2): ").strip()
            try:
                new_th = float(new_th_str)
                height_threshold = new_th
            except ValueError:
                print("잘못된 입력: threshold 변경 실패. 기존 값 유지.")

            # 구간 변경
            range_str = input(f"새 p400 피크 탐색 구간 입력 (시작 끝, 기본={p400_peak_start_s} {p400_peak_end_s}): ").strip()
            try:
                parts_range = range_str.split()
                if len(parts_range) == 2:
                    new_start_s = float(parts_range[0])
                    new_end_s   = float(parts_range[1])
                    if new_start_s < new_end_s:
                        p400_peak_start_s = new_start_s
                        p400_peak_end_s   = new_end_s
                    else:
                        print("구간 입력이 잘못됨 (start >= end). 기존 값 유지.")
                else:
                    print("두 개의 숫자를 입력해야 함. 기존 값 유지.")
            except ValueError:
                print("구간 입력이 잘못됨. 기존 값 유지.")

            print(f"[알림] threshold={height_threshold}, 구간={p400_peak_start_s}~{p400_peak_end_s} 재시도...\n")

        elif user_in == 'qqq':
            print(f"[SKIP] {phone_session_id} 세션을 건너뜁니다.")
            return "skip"
        else:
            print("잘못된 입력입니다. (y/n/qqq) 중 하나를 입력하세요.")

    # --------------------------
    # (C) 이후 실제 감압 구간 탐색, CSV 저장, 최종 플롯
    # --------------------------
    # 1) p400 피크 지점 이전 구간 삭제
    if p400_first_segment_time is not None:
        idx_p400 = int(np.round(p400_first_segment_time * p400_fs))
        if idx_p400 < len(p3):
            p3       = p3[idx_p400:]
            p400_pcg = p400_pcg[idx_p400:]
        p400_time = np.arange(len(p400_pcg)) / p400_fs

    # eko도 동일 로직
    if (eko_fs_pcg is not None) and (eko_first_segment_time is not None):
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

    # 2) Eko 리샘플링(2000Hz) + DF 병합
    p400_fs_target = 2000
    eko_signals_resampled = []
    for sig, fs in zip(eko_signals, eko_fs_list):
        if (sig is None) or (fs is None) or (len(sig) == 0):
            eko_signals_resampled.append(None)
        else:
            eko_signals_resampled.append(resample_signal(sig, fs, p400_fs_target))

    valid_lengths = [len(p400_pcg)] + [len(s) for s in eko_signals_resampled if s is not None]
    if len(valid_lengths) == 0:
        print("[오류] 유효 신호가 없음. 세션 스킵.")
        return "skip"
    max_len = max(valid_lengths)

    common_time = np.arange(max_len) / p400_fs_target
    p400_pcg_pad = pad_nan(p400_pcg, max_len)
    p3_pad       = pad_nan(p3, max_len)

    df = pd.DataFrame({
        "Time (s)": common_time,
        "p400_PCG": p400_pcg_pad,
        "p3_voltage": p3_pad
    })

    for lbl, sig_rs in zip(eko_labels, eko_signals_resampled):
        if sig_rs is not None:
            df[lbl] = pad_nan(sig_rs, max_len)
        else:
            df[lbl] = np.full(max_len, np.nan)

    # 3) 감압 구간 찾기
    threshold = 0.6
    min_len_samples = 15 * p400_fs_target  # 15초

    p3_arr   = df["p3_voltage"].values
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

    # 4) 감압 구간 Eko '원본' 신호 CSV로 저장 (NaN 패딩)
    if out_folder and (t_start is not None) and (t_end is not None):
        os.makedirs(out_folder, exist_ok=True)
        raw_sigs = {}
        for lbl, fs, sig in zip(eko_labels, eko_fs_list, eko_signals):
            if (sig is None) or (fs is None) or (len(sig) == 0):
                raw_sigs[lbl] = np.array([])
                continue
            start_idx_eko = int(np.round(t_start * fs))
            end_idx_eko   = int(np.round(t_end   * fs))
            if start_idx_eko >= len(sig):
                raw_sigs[lbl] = np.array([])
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
        csv_filename = f"{pdt_basename}_EkoRawSegment.csv"
        out_csv_path = os.path.join(out_folder, csv_filename)
        df_raw.to_csv(out_csv_path, index=False, encoding='utf-8-sig')
        print(f"[CSV 저장] 감압구간 원본(채널별 NaN패딩) → {out_csv_path}")

    # 5) 최종 플롯
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
# (C) p400 폴더와 Eko 폴더를 순회하며 반복 실행
# -------------------------------------------------
def main():
    # (1) 폴더 지정 (사용자 환경에 맞춰 수정)
    p400_folder = r"C:\Users\user\Desktop\천안임상\a_reformatted\p400"
    eko_folder  = r"C:\Users\user\Desktop\천안임상\a_reformatted\eko"
    out_folder  = r"C:\Users\user\Desktop\천안임상\결과저장폴더213"

    # (2) p400 폴더 내 .pdt 파일 목록
    pdt_files = [f for f in os.listdir(p400_folder) if f.lower().endswith(".pdt")]

    skip_list = []

    for pdt_file in pdt_files:
        # 예: 01033378505_1211_1.pdt → split → [01033378505, 1211, 1]
        pdt_basename = os.path.splitext(pdt_file)[0]
        parts = pdt_basename.split('_')
        if len(parts) < 3:
            print(f"파일명 파싱 불가능: {pdt_file}, 건너뜁니다.")
            continue

        phone_num   = parts[0]
        # date_str = parts[1]  # 미사용
        session_num = parts[2]

        pdt_path = os.path.join(p400_folder, pdt_file)

        # Eko 폴더명: phone_num_a_session_num
        phone_session_id   = f"{phone_num}_a_{session_num}"
        eko_subfolder_name = phone_session_id
        eko_subfolder_path = os.path.join(eko_folder, eko_subfolder_name)

        if not os.path.isdir(eko_subfolder_path):
            print(f"Eko 폴더 없음: {eko_subfolder_path}, 건너뜁니다.")
            continue

        # Eko wav 파일 4개(PCG/ECG/ECG_1/ECG_2) 파싱
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

        print(f"\n=== 처리 시작: {pdt_file} → Eko 폴더: {phone_session_id} ===")

        ret = process_session(
            pdt_path=pdt_path,
            eko_wav_paths=eko_wav_map,
            phone_session_id=phone_session_id,
            plot_debug=True,   # 디버깅(피크 탐색) 플롯
            plot_result=True,  # 최종 플롯
            out_folder=out_folder
        )

        if ret == "skip":
            skip_list.append(phone_session_id)
            continue

        print(f"=== 처리 완료: {phone_session_id} ===")

    # (4) 스킵된 세션 목록 저장
    if skip_list:
        skip_txt_path = os.path.join(out_folder, "skip_list.txt")
        with open(skip_txt_path, "w", encoding="utf-8") as f:
            for sk in skip_list:
                f.write(sk + "\n")
        print(f"[완료] 스킵된 세션 목록을 '{skip_txt_path}'에 저장했습니다.")
    else:
        print("[완료] 스킵된 세션 없음.")


if __name__ == "__main__":
    main()
