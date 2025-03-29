import os
import tempfile
import uuid
import cv2
import numpy as np
import concurrent.futures
import gc
import streamlit as st

# 최대 워커 수를 64로 고정 (CPU 전용)
WORKERS = 64

st.title("비디오 슈퍼 해상도 (CPU 전용)")
st.write("CPU 전용 모드로 실행합니다. (동시 워커 수: 64)")

# 자동 색 보정 (입력 영상에 대해 자동 보정)
def auto_color_correction(frame):
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return corrected
    except Exception as e:
        st.write("auto_color_correction error:", e)
        return frame

# 비트 플레인 분해 (항상 8비트 고정)
def bit_plane_slicing(channel):
    return [(channel >> i) & 1 for i in range(8)]

def reconstruct_from_bit_planes(bit_planes, weights=None):
    if weights is None:
        weights = [2**i for i in range(8)]
    return sum((plane.astype(np.uint8) * weights[i]) for i, plane in enumerate(bit_planes))

# 표준 모드 업스케일 함수 (CPU 전용)
def advanced_upscale_block(block, scale_factor_w, scale_factor_h):
    try:
        channels = cv2.split(block)
        processed_channels = []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        for ch in channels:
            planes = bit_plane_slicing(ch)
            reconstructed = reconstruct_from_bit_planes(planes)
            enhanced = clahe.apply(reconstructed)
            processed_channels.append(enhanced)
        merged = cv2.merge(processed_channels)
    except Exception as e:
        st.write("Error during bit-plane processing:", e)
        merged = block

    try:
        filtered = cv2.bilateralFilter(merged, d=5, sigmaColor=75, sigmaSpace=75)
    except Exception as e:
        st.write("Error during bilateral filtering:", e)
        filtered = merged

    h, w = filtered.shape[:2]
    new_w = int(w * scale_factor_w)
    new_h = int(h * scale_factor_h)
    try:
        upscaled = cv2.resize(filtered, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        st.write("Error during resizing:", e)
        upscaled = filtered

    try:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)
    except Exception as e:
        st.write("Error during unsharp masking:", e)
        sharpened = upscaled
    return sharpened

# HDR 모드 업스케일 함수 (CPU 전용)
def advanced_upscale_block_hdr(block, scale_factor_w, scale_factor_h):
    try:
        block_float = block.astype(np.float32) / 255.0
    except Exception as e:
        st.write("Error converting block to float:", e)
        block_float = block / 255.0
    try:
        filtered = cv2.bilateralFilter(block_float, d=5, sigmaColor=75, sigmaSpace=75)
    except Exception as e:
        st.write("Error during HDR bilateral filtering:", e)
        filtered = block_float
    h, w = filtered.shape[:2]
    new_w = int(w * scale_factor_w)
    new_h = int(h * scale_factor_h)
    try:
        upscaled = cv2.resize(filtered, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        st.write("Error during HDR resizing:", e)
        upscaled = filtered
    try:
        blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
        sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
    except Exception as e:
        st.write("Error during HDR unsharp masking:", e)
        sharpened = upscaled
    return sharpened

# 프레임 처리 함수 (HDR/표준 모드 분기)
def process_frame(frame, scale_factor_w, scale_factor_h, block_size, hdr_mode=False):
    if hdr_mode:
        frame = auto_color_correction(frame)
        h, w = frame.shape[:2]
        out_h, out_w = int(h * scale_factor_h), int(w * scale_factor_w)
        output = np.zeros((out_h, out_w, 3), dtype=np.float32)
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = frame[i:min(i+block_size, h), j:min(j+block_size, w)]
                try:
                    upscaled_block = advanced_upscale_block_hdr(block, scale_factor_w, scale_factor_h)
                except Exception as e:
                    st.write(f"HDR processing error at block ({i},{j}):", e)
                    upscaled_block = cv2.resize(block.astype(np.float32)/255.0, 
                                                (int(block.shape[1]*scale_factor_w), int(block.shape[0]*scale_factor_h)),
                                                interpolation=cv2.INTER_CUBIC)
                di, dj = int(i * scale_factor_h), int(j * scale_factor_w)
                bh, bw = upscaled_block.shape[:2]
                output[di:di+bh, dj:dj+bw] = upscaled_block
        tonemap = cv2.createTonemapReinhard(gamma=1.0)
        try:
            hdr_result = tonemap.process(output)
            final_result = np.clip(hdr_result * 255, 0, 255).astype(np.uint8)
        except MemoryError as e:
            st.write("Tonemap processing failed due to memory error:", e)
            # 톤 매핑을 건너뛰고, 원본 output을 8비트로 변환
            final_result = np.clip(output * 255, 0, 255).astype(np.uint8)
        try:
            denoised = cv2.fastNlMeansDenoisingColored(final_result, None, h=10, hColor=10, 
                                                        templateWindowSize=7, searchWindowSize=21)
        except cv2.error as e:
            st.write("Denoising failed:", e)
            denoised = final_result
        anti_aliased = cv2.GaussianBlur(denoised, (3,3), sigmaX=0.5)
        return anti_aliased
    else:
        # 표준 모드 처리 (동일)
        h, w = frame.shape[:2]
        out_h, out_w = int(h * scale_factor_h), int(w * scale_factor_w)
        output = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = frame[i:min(i+block_size, h), j:min(j+block_size, w)]
                try:
                    upscaled_block = advanced_upscale_block(block, scale_factor_w, scale_factor_h)
                except Exception as e:
                    st.write(f"Standard processing error at block ({i},{j}):", e)
                    upscaled_block = cv2.resize(block, 
                                                (int(block.shape[1]*scale_factor_w), int(block.shape[0]*scale_factor_h)),
                                                interpolation=cv2.INTER_CUBIC)
                di, dj = int(i * scale_factor_h), int(j * scale_factor_w)
                bh, bw = upscaled_block.shape[:2]
                output[di:di+bh, dj:dj+bw] = upscaled_block
        try:
            denoised = cv2.fastNlMeansDenoisingColored(output, None, h=10, hColor=10, 
                                                        templateWindowSize=7, searchWindowSize=21)
        except cv2.error as e:
            st.write("Denoising failed:", e)
            denoised = output
        anti_aliased = cv2.GaussianBlur(denoised, (3,3), sigmaX=0.5)
        return anti_aliased

def process_frame_wrapper(args_tuple):
    idx, frame, scale_factor_w, scale_factor_h, block_size, hdr_mode = args_tuple
    processed = process_frame(frame, scale_factor_w, scale_factor_h, block_size, hdr_mode)
    return idx, processed

# 영상 처리 함수 (배치 단위로 처리하여 메모리 사용 최소화)
def process_video(input_path, output_path, quality, hdr_mode, block_size):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("입력 영상을 열 수 없습니다.")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    resolution_map = {"1080": (1920, 1080), "4k": (3840, 2160), "8k": (7680, 4320)}
    target_w, target_h = resolution_map.get(quality, (3840, 2160))
    scale_factor_w = target_w / orig_w if orig_w > 0 else 1.0
    scale_factor_h = target_h / orig_h if orig_h > 0 else 1.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    progress_bar = st.progress(0)
    progress_text = st.empty()
    processed_frames = 0
    batch_size = 100  # 한 번에 처리할 프레임 수
    frame_index = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        while True:
            batch_frames = []
            batch_indices = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
                batch_indices.append(frame_index)
                frame_index += 1
            if not batch_frames:
                break
            futures = {idx: executor.submit(process_frame, frame, scale_factor_w, scale_factor_h, block_size, hdr_mode)
                        for idx, frame in zip(batch_indices, batch_frames)}
            for idx in sorted(futures.keys()):
                result = futures[idx].result()
                out.write(result)
                processed_frames += 1
                progress_bar.progress(processed_frames / total_frames)
                progress_text.text(f"Processed {processed_frames} / {total_frames} frames")
                gc.collect()
    cap.release()
    out.release()
    return output_path

# --- Streamlit UI ---
st.header("비디오 슈퍼 해상도 (CPU 전용)")
uploaded_file = st.file_uploader("입력 비디오 파일을 선택하세요", type=["mp4", "avi", "mov", "mkv"])
quality = st.selectbox("화질 선택", options=["1080", "4k", "8k"], index=1)
hdr_mode = st.checkbox("HDR 모드 활성화")
block_size = st.number_input("블록 크기 (픽셀)", min_value=8, max_value=128, value=32, step=1)

if st.button("처리 시작"):
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        input_path = tfile.name
        output_path = os.path.join(tempfile.gettempdir(), f"output_{uuid.uuid4().hex}.mp4")
        st.info("영상 처리를 시작합니다. 잠시만 기다려 주세요...")
        result_path = process_video(input_path, output_path, quality, hdr_mode, block_size)
        if result_path:
            st.success("영상 처리가 완료되었습니다!")
            with open(result_path, "rb") as file:
                st.download_button("결과 다운로드", data=file, file_name="output.mp4", mime="video/mp4")
    else:
        st.error("먼저 비디오 파일을 업로드하세요.")