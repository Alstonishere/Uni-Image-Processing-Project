import streamlit as st
import login
import numpy as np
import cv2
from PIL import Image, ImageOps
import io

# --- DCT Algorithm Core ---

def preprocess_secret_exact(pil_image):
    """
    Resizes secret to 64x64 and returns flattened R,G,B arrays.
    """
    target_size = (64, 64)
    img = ImageOps.pad(pil_image, target_size, color='white')
    img_rgb = np.array(img.convert('RGB'))

    r, g, b = cv2.split(img_rgb)
    return r.flatten().astype(float), g.flatten().astype(float), b.flatten().astype(float)


def logistic_map(seed, length):
    """
    Scrambles block order for embedding.
    """
    np.random.seed(seed)
    indices = np.arange(length)
    np.random.shuffle(indices)
    return indices


def embed_8x8_dct(cover_channel, secret_flat, chaos_seed, Q=50.0):
    """
    Embeds secret pixels into 8x8 DCT blocks.
    """
    h, w = cover_channel.shape
    img_float = cover_channel.astype(np.float32)

    total_blocks = 4096
    indices = logistic_map(chaos_seed, total_blocks)

    block_idx = 0
    target_range = Q * 0.8

    for r in range(0, h, 8):
        for c in range(0, w, 8):
            if block_idx >= total_blocks:
                break

            pixel_idx = indices[block_idx]
            pixel_val = secret_flat[pixel_idx]

            block = img_float[r:r+8, c:c+8]
            dct_block = cv2.dct(block)

            val = dct_block[4, 4]
            scaled_pixel = (pixel_val / 255.0) * target_range
            centered_pixel = scaled_pixel - (target_range / 2.0)

            base = np.round(val / Q) * Q
            dct_block[4, 4] = base + centered_pixel

            img_float[r:r+8, c:c+8] = cv2.idct(dct_block)
            block_idx += 1

    return np.clip(img_float, 0, 255).astype(np.uint8)


def extract_8x8_dct(stego_channel, chaos_seed, Q=50.0):
    """
    Improved extraction with adaptive rescaling of coefficient offsets.
    """
    h, w = stego_channel.shape
    img_float = stego_channel.astype(np.float32)

    total_blocks = 4096
    indices = logistic_map(chaos_seed, total_blocks)

    decoded_center = np.zeros(total_blocks, dtype=np.float32)
    target_range = Q * 0.8

    block_idx = 0
    for r in range(0, h, 8):
        for c in range(0, w, 8):
            if block_idx >= total_blocks:
                break

            block = img_float[r:r+8, c:c+8]
            dct_block = cv2.dct(block)

            val = dct_block[4, 4]
            base = np.round(val / Q) * Q
            diff = val - base

            pixel_idx = indices[block_idx]
            decoded_center[pixel_idx] = diff
            block_idx += 1

    # Adaptive rescaling
    low = np.percentile(decoded_center, 1.0)
    high = np.percentile(decoded_center, 99.0)
    span = max(high - low, 1e-6)

    desired_low, desired_high = -0.4 * Q, +0.4 * Q
    scale = (desired_high - desired_low) / span
    shift = desired_low - low * scale

    adjusted = decoded_center * scale + shift

    scaled_pixel = adjusted + (target_range / 2.0)
    pixel_vals = (scaled_pixel / target_range) * 255.0
    pixel_vals = np.clip(pixel_vals, 0, 255)

    return pixel_vals.astype(np.uint8)


def embed_dct_rgb(cover_rgb, secret_pil, chaos_seed):
    """
    Embeds RGB secret into RGB cover using 8x8 DCT.
    """
    cover_rgb = cv2.resize(cover_rgb, (512, 512))
    r_cov, g_cov, b_cov = cv2.split(cover_rgb)
    r_sec, g_sec, b_sec = preprocess_secret_exact(secret_pil)

    r_stego = embed_8x8_dct(r_cov, r_sec, chaos_seed)
    g_stego = embed_8x8_dct(g_cov, g_sec, chaos_seed)
    b_stego = embed_8x8_dct(b_cov, b_sec, chaos_seed)

    return cv2.merge([r_stego, g_stego, b_stego])


def extract_dct_rgb(stego_rgb, chaos_seed):
    """
    Improved RGB extraction with gentler restoration filters.
    """
    stego_rgb = cv2.resize(stego_rgb, (512, 512))
    r_stego, g_stego, b_stego = cv2.split(stego_rgb)

    r_vals = extract_8x8_dct(r_stego, chaos_seed)
    g_vals = extract_8x8_dct(g_stego, chaos_seed)
    b_vals = extract_8x8_dct(b_stego, chaos_seed)

    r_img = r_vals.reshape((64, 64))
    g_img = g_vals.reshape((64, 64))
    b_img = b_vals.reshape((64, 64))
    raw_img = cv2.merge([r_img, g_img, b_img]).astype(np.uint8)

    # Restoration pipeline
    clean = cv2.bilateralFilter(raw_img, d=5, sigmaColor=20, sigmaSpace=10)

    lab = cv2.cvtColor(clean, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    clean = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    clean = cv2.medianBlur(clean, 3)
    final_view = cv2.resize(clean, (256, 256), interpolation=cv2.INTER_CUBIC)

    return final_view

def run(user=None):
    if user is None:
        user = st.session_state.get("user")

    st.info("ℹ️ **Algorithm:** 8x8 DCT High-Fidelity")
    st.write("Uses standard 8x8 blocks with Analog QIM for maximal color accuracy and robustness.")

    action = st.radio("Choose Action", ["Encrypt (Embed)", "Decrypt (Extract)"], horizontal=True, key="dct_act")

    if action == "Encrypt (Embed)":
        st.subheader("📉 Embed Data")
        col1, col2 = st.columns(2)
        with col1: cover_f = st.file_uploader("Cover Image (Color)", type=["jpg", "png"], key="dct_c")
        with col2: secret_f = st.file_uploader("Secret Image (Color)", type=["jpg", "png"], key="dct_s")

        receiver = st.text_input("Receiver Username")

        # Show uploaded image preview
        if cover_f or secret_f:
            st.divider()
            st.write("### Input Preview")
            c1, c2 = st.columns(2)
            with c1:
                if cover_f:
                    cover_pil = Image.open(cover_f).convert("RGB")
                    st.image(cover_pil, caption="Cover Image", use_column_width=True)
                    cover_f.seek(0)
            with c2:
                if secret_f:
                    secret_pil = Image.open(secret_f).convert("RGB")
                    st.image(secret_pil, caption="Secret Image", use_column_width=True)
                    secret_f.seek(0)

        if st.button("Embed Image"):
            if cover_f and secret_f and receiver:
                if login.get_user_key(receiver):
                    seed = int(login.get_user_key(receiver), 16) % 100000
                    cover_pil = Image.open(cover_f).convert("RGB")
                    secret_pil = Image.open(secret_f).convert("RGB")
                    cover_arr = np.array(cover_pil)
                    original_host = cover_arr.copy()
                    result_arr = embed_dct_rgb(cover_arr, secret_pil, seed)

                    st.success(f"Watermark embedded for {receiver}.")
                    st.write("### 2. Encryption Result")
                    c1, c2 = st.columns(2)
                    with c1: st.image(original_host, caption="Original Host Image", use_column_width=True)
                    with c2: st.image(result_arr, caption="Watermarked Host Image", use_column_width=True)

                    res_img = Image.fromarray(result_arr)
                    buf = io.BytesIO()
                    res_img.save(buf, format="PNG")
                    st.download_button("Download Watermarked Image", buf.getvalue(), "dct_high_fi.png", "image/png")
                else:
                    st.error("User not found.")
            else:
                st.warning("Please upload images.")

    else:
        st.subheader("🔓 Extract Data")
        stego_f = st.file_uploader("Upload Stego Image", key="dct_d")

        if st.button("Extract") and stego_f:
            my_key = login.get_user_key(user)
            seed = int(my_key, 16) % 100000
            stego_arr = np.array(Image.open(stego_f).convert("RGB"))

            try:
                extracted_arr = extract_dct_rgb(stego_arr, seed)
                
                # Store in session state to persist after download
                st.session_state['dwt_host_image'] = stego_arr.copy()
                st.session_state['dwt_secret_image'] = extracted_arr
                st.session_state['dwt_extraction_done'] = True
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Display images if extraction was done
        if st.session_state.get('dwt_extraction_done', False):
            host_image = st.session_state['dwt_host_image']
            extracted_arr = st.session_state['dwt_secret_image']

            st.write("### Decryption Result")
            c1, c2 = st.columns(2)
            with c1:
                st.image(host_image, caption="Extracted Host Image", use_column_width=True)
            with c2:
                st.image(extracted_arr, caption="Extracted Secret Image (Restored)", use_column_width=True)
            st.success("Extraction Successful.")
            
            # Download buttons
            st.write("### Download Extracted Images")
            d1, d2 = st.columns(2)
            with d1:
                host_img = Image.fromarray(host_image)
                buf_host = io.BytesIO()
                host_img.save(buf_host, format="PNG")
                st.download_button("📥 Download Host Image", buf_host.getvalue(), "extracted_host_dct.png", "image/png", key="dl_host_dwt")
            with d2:
                secret_img = Image.fromarray(extracted_arr)
                buf_secret = io.BytesIO()
                secret_img.save(buf_secret, format="PNG")
                st.download_button("📥 Download Secret Image", buf_secret.getvalue(), "extracted_secret_dct.png", "image/png", key="dl_secret_dwt")