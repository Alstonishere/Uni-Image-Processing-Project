"""
Host Media Similarity Checker
Compares original and watermarked media using PSNR
to determine data loss due to watermarking
"""

import streamlit as st
import numpy as np
import cv2
import tempfile
import os


# ============== CONSTANTS ==============

MAX_PSNR = 60.0

PSNR_INTERPRETATION = """
**Interpretation:**
- PSNR > 40 dB: Excellent quality, minimal data loss
- PSNR 30-40 dB: Good quality, acceptable data loss
- PSNR 20-30 dB: Fair quality, noticeable data loss
- PSNR < 20 dB: Poor quality, significant data loss
"""


# ============== FUNCTIONS ==============

def calculate_psnr(original, extracted):
    """Calculate PSNR between two images"""
    if original.shape != extracted.shape:
        extracted = cv2.resize(extracted, (original.shape[1], original.shape[0]))
    
    original = original.astype(np.float64)
    extracted = extracted.astype(np.float64)
    
    mse = np.mean((original - extracted) ** 2)
    
    if mse == 0:
        return float('inf'), 100.0
    
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    psnr_percentage = min((psnr / MAX_PSNR) * 100, 100)
    
    return psnr, psnr_percentage


def calculate_video_psnr(original_path, watermarked_path, progress_bar=None):
    """Calculate average PSNR between two videos frame by frame"""
    cap_original = cv2.VideoCapture(original_path)
    cap_watermarked = cv2.VideoCapture(watermarked_path)
    
    if not cap_original.isOpened() or not cap_watermarked.isOpened():
        return None, None, 0, []
    
    total_frames = min(
        int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap_watermarked.get(cv2.CAP_PROP_FRAME_COUNT))
    )
    
    psnr_values = []
    frame_count = 0
    
    while True:
        ret1, frame1 = cap_original.read()
        ret2, frame2 = cap_watermarked.read()
        
        if not ret1 or not ret2:
            break
        
        psnr, _ = calculate_psnr(frame1, frame2)
        psnr_values.append(MAX_PSNR if psnr == float('inf') else psnr)
        
        frame_count += 1
        if progress_bar and total_frames > 0:
            progress_bar.progress(frame_count / total_frames)
    
    cap_original.release()
    cap_watermarked.release()
    
    if len(psnr_values) == 0:
        return None, None, 0, []
    
    avg_psnr = np.mean(psnr_values)
    avg_psnr_percentage = min((avg_psnr / MAX_PSNR) * 100, 100)
    
    return avg_psnr, avg_psnr_percentage, frame_count, psnr_values


def get_quality_label(psnr):
    """Get quality label based on PSNR value"""
    if psnr == float('inf') or psnr >= MAX_PSNR:
        return "Perfect Match"
    elif psnr > 40:
        return "Excellent"
    elif psnr > 30:
        return "Good"
    elif psnr > 20:
        return "Fair"
    else:
        return "Poor"


# ============== PAGE HEADER ==============

st.title("Host Media Similarity Checker")
st.write("Compare original and watermarked media to measure data loss from watermarking")


# ============== MODE SELECTION ==============

st.subheader("Select Comparison Mode")
mode = st.radio("Choose input type:", ["Image vs Image", "Video vs Video"], horizontal=True)


# ============== IMAGE COMPARISON ==============

if mode == "Image vs Image":
    st.subheader("Compare Two Images")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Host Image**")
        original_file = st.file_uploader("Upload original host image", type=['png', 'jpg', 'jpeg', 'bmp'])
    with col2:
        st.write("**Watermarked Host Image**")
        extracted_file = st.file_uploader("Upload watermarked host image", type=['png', 'jpg', 'jpeg', 'bmp'])
    
    if original_file and extracted_file:
        # Load images
        original_bytes = np.asarray(bytearray(original_file.read()), dtype=np.uint8)
        extracted_bytes = np.asarray(bytearray(extracted_file.read()), dtype=np.uint8)
        original_img = cv2.imdecode(original_bytes, cv2.IMREAD_COLOR)
        extracted_img = cv2.imdecode(extracted_bytes, cv2.IMREAD_COLOR)
        
        # Display images
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="Original Host Image", use_container_width=True)
        with col2:
            st.image(extracted_img, caption="Watermarked Host Image", use_container_width=True)
        
        # Calculate and display results
        psnr, psnr_percentage = calculate_psnr(original_img, extracted_img)
        
        st.divider()
        st.subheader("Similarity Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("PSNR", "∞ dB (Identical)" if psnr == float('inf') else f"{psnr:.2f} dB")
        with col2:
            st.metric("Similarity", f"{psnr_percentage:.2f}%")
        with col3:
            st.metric("Quality", get_quality_label(psnr))
        
        st.info(PSNR_INTERPRETATION)


# ============== VIDEO COMPARISON ==

if mode == "Video vs Video":
    st.subheader("Compare Two Videos")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Video**")
        original_video = st.file_uploader("Upload original video", type=['mp4', 'avi', 'mov', 'mkv'])
    with col2:
        st.write("**Watermarked Video**")
        watermarked_video = st.file_uploader("Upload watermarked video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if original_video and watermarked_video:
        # Save to temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_original:
            tmp_original.write(original_video.read())
            original_path = tmp_original.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_watermarked:
            tmp_watermarked.write(watermarked_video.read())
            watermarked_path = tmp_watermarked.name
        
        # Process videos
        st.divider()
        st.write("**Processing videos frame by frame...**")
        progress_bar = st.progress(0)
        
        avg_psnr, avg_psnr_percentage, frame_count, psnr_values = calculate_video_psnr(
            original_path, watermarked_path, progress_bar
        )
        
        # Clean up temp files
        os.unlink(original_path)
        os.unlink(watermarked_path)
        
        if avg_psnr is not None:
            # Display results
            st.divider()
            st.subheader("Similarity Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average PSNR", f"{avg_psnr:.2f} dB")
            with col2:
                st.metric("Similarity", f"{avg_psnr_percentage:.2f}%")
            with col3:
                st.metric("Quality", get_quality_label(avg_psnr))
            with col4:
                st.metric("Frames Analyzed", frame_count)
            
            # Frame statistics
            st.divider()
            st.subheader("Frame-by-Frame Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min PSNR", f"{min(psnr_values):.2f} dB")
            with col2:
                st.metric("Max PSNR", f"{max(psnr_values):.2f} dB")
            with col3:
                st.metric("Std Dev", f"{np.std(psnr_values):.2f} dB")
            
            st.line_chart(psnr_values, x_label="Frame", y_label="PSNR (dB)")
            
            st.info(PSNR_INTERPRETATION)
        else:
            st.error("Could not process the videos. Please ensure both videos are valid and have the same format.")


# ============== FOOTER ==============

st.divider()
st.caption("Host Media Similarity Checker - Measures data loss in image/video watermarking using PSNR")
