import streamlit as st
import login
import numpy as np
import cv2
from PIL import Image, ImageOps
import io

# ### --- ALGORITHM CORE START ---

def preprocess_secret_digital(pil_image):
    """
    1. Resizes to 64x64.
    2. Quantizes to 3-bit color (8 Solid Colors).
    3. Flattens to a binary bitstream.
    """
    target_size = (64, 64)
    img = ImageOps.pad(pil_image, target_size, color='white')
    img_rgb = np.array(img.convert('RGB'))
    
    # Quantize to 3-bit (Values: 0, 36, 72... 255)
    img_3bit = img_rgb // 32 
    
    flat_vals = img_3bit.flatten() # 0-7 integers
    
    # Convert to bits (3 bits per pixel channel)
    bitstream = []
    for val in flat_vals:
        # 3 bits: 4, 2, 1
        bitstream.append((val >> 2) & 1)
        bitstream.append((val >> 1) & 1)
        bitstream.append(val & 1)
        
    return np.array(bitstream, dtype=np.uint8)

def reconstruct_secret_digital(bitstream):
    """
    Reconstructs 3-bit image from bitstream.
    """
    pixels = []
    for i in range(0, len(bitstream), 3):
        if i+2 >= len(bitstream): break
        
        b2 = bitstream[i]
        b1 = bitstream[i+1]
        b0 = bitstream[i+2]
        
        # Reassemble 0-7 value
        val = (b2 << 2) | (b1 << 1) | b0
        
        # Scale 0-7 back to 0-255
        pixels.append(val * 32)
        
    img_flat = np.array(pixels, dtype=np.uint8)
    return img_flat.reshape((64, 64, 3))

def logistic_map(seed, length):
    """Chaotic shuffle."""
    np.random.seed(seed)
    indices = np.arange(length)
    np.random.shuffle(indices)
    return indices

def embed_bits_multi_coeff(cover_channel, secret_bits, chaos_seed):
    """
    Embeds bits into 5 coefficients per 8x8 block.
    Targeting: (3,3), (3,4), (4,3), (4,4), (5,5).
    """
    h, w = cover_channel.shape
    img_float = cover_channel.astype(float)
    
    total_blocks = 4096
    indices = logistic_map(chaos_seed, total_blocks)
    coeffs_list = [(3,3), (3,4), (4,3), (4,4), (5,5)]
    Q = 40.0
    
    bit_idx = 0
    total_bits = len(secret_bits)
    
    for k in range(total_blocks):
        if bit_idx >= total_bits: break
        idx = indices[k]
        r = (idx // 64) * 8
        c = (idx % 64) * 8
        
        block = img_float[r:r+8, c:c+8]
        dct_block = cv2.dct(block)
        
        for (cr, cc) in coeffs_list:
            if bit_idx >= total_bits: break
            bit = secret_bits[bit_idx]
            val = dct_block[cr, cc]
            
            base = np.round(val / Q) * Q
            if bit == 1:
                dct_block[cr, cc] = base + (0.5 * Q)
            else:
                dct_block[cr, cc] = base - (0.5 * Q)
                
            bit_idx += 1
            
        img_float[r:r+8, c:c+8] = cv2.idct(dct_block)
        
    return np.clip(img_float, 0, 255).astype(np.uint8)

def extract_bits_multi_coeff(cover_channel, max_bits, chaos_seed):
    """
    Extracts bits from 5 coefficients.
    """
    h, w = cover_channel.shape
    img_float = cover_channel.astype(float)
    
    total_blocks = 4096
    indices = logistic_map(chaos_seed, total_blocks)
    coeffs_list = [(3,3), (3,4), (4,3), (4,4), (5,5)]
    Q = 40.0
    
    extracted_bits = []
    
    for k in range(total_blocks):
        if len(extracted_bits) >= max_bits: break
        idx = indices[k]
        r = (idx // 64) * 8
        c = (idx % 64) * 8
        
        block = img_float[r:r+8, c:c+8]
        dct_block = cv2.dct(block)
        
        for (cr, cc) in coeffs_list:
            if len(extracted_bits) >= max_bits: break
            val = dct_block[cr, cc]
            
            base = np.round(val / Q) * Q
            diff = val - base
            
            if diff > 0:
                extracted_bits.append(1)
            else:
                extracted_bits.append(0)
                
    return np.array(extracted_bits, dtype=np.uint8)

def dct_embed_digital_spectrum(cover_rgb, secret_pil, chaos_seed):
    cover_rgb = cv2.resize(cover_rgb, (512, 512))
    ycc = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycc)
    
    full_bits = preprocess_secret_digital(secret_pil)
    
    split_point = len(full_bits) // 2
    bits_cr = full_bits[:split_point]
    bits_cb = full_bits[split_point:]
    
    cr_stego = embed_bits_multi_coeff(cr, bits_cr, chaos_seed)
    cb_stego = embed_bits_multi_coeff(cb, bits_cb, chaos_seed + 99) 
    
    stego_ycc = cv2.merge([y, cr_stego, cb_stego])
    return cv2.cvtColor(stego_ycc, cv2.COLOR_YCrCb2RGB)

def dct_extract_digital_spectrum(stego_rgb, chaos_seed):
    stego_rgb = cv2.resize(stego_rgb, (512, 512))
    ycc = cv2.cvtColor(stego_rgb, cv2.COLOR_RGB2YCrCb)
    _, cr, cb = cv2.split(ycc)
    
    total_bits = 36864
    split_point = total_bits // 2
    
    bits_cr = extract_bits_multi_coeff(cr, split_point, chaos_seed)
    bits_cb = extract_bits_multi_coeff(cb, total_bits - split_point, chaos_seed + 99)
    
    full_bits = np.concatenate([bits_cr, bits_cb])
    
    img = reconstruct_secret_digital(full_bits)
    return cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)

# ### --- ALGORITHM CORE END ---

def run(user=None):
    if user is None:
        user = st.session_state.get("user")

    st.info("ℹ️ **Algorithm:** Digital Spectrum DCT (Universal Input)")
    st.write("Accepts **JPG, PNG, JPEG, BMP**. Saves as **PNG** to preserve data integrity.")

    action = st.radio("Choose Action", ["Encrypt (Embed)", "Decrypt (Extract)"], horizontal=True, key="dct_act")

    if action == "Encrypt (Embed)":
        st.subheader("📉 Embed Data")
        
        # --- UPDATE: ACCEPT ALL FORMATS ---
        col1, col2 = st.columns(2)
        with col1: cover_f = st.file_uploader("Cover Image", type=["jpg", "png", "jpeg", "bmp", "tiff"], key="dct_c")
        with col2: secret_f = st.file_uploader("Secret Image", type=["jpg", "png", "jpeg", "bmp", "tiff"], key="dct_s")
        
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
                    
                    result_arr = dct_embed_digital_spectrum(cover_arr, secret_pil, seed)
                    
                    st.success(f"Watermark embedded for {receiver}.")
                    
                    st.write("### 2. Encryption Result")
                    c1, c2 = st.columns(2)
                    with c1: st.image(original_host, caption="Original Host Image", use_column_width=True)
                    with c2: st.image(result_arr, caption="Watermarked Host Image", use_column_width=True)
                    
                    # --- CRITICAL: SAVE AS PNG ---
                    res_img = Image.fromarray(result_arr)
                    buf = io.BytesIO()
                    res_img.save(buf, format="PNG") 
                    
                    st.download_button("Download Watermarked Image (PNG Only)", buf.getvalue(), "dct_digital_result.png", "image/png")
                    st.warning("⚠️ Note: Please do not convert this downloaded image to JPG, or the hidden data will be lost.")
                else:
                    st.error("User not found.")
            else:
                st.warning("Please upload images.")

    else:
        st.subheader("🔓 Extract Data")
        # Extract allows all formats, but warns if they upload JPG
        stego_f = st.file_uploader("Upload Stego Image", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="dct_d")
        
        if st.button("Extract") and stego_f:
            if stego_f.name.lower().endswith(('.jpg', '.jpeg')):
                st.warning("⚠️ Warning: You uploaded a JPEG. Extraction might fail due to compression artifacts. Use PNG for best results.")
                
            my_key = login.get_user_key(user)
            seed = int(my_key, 16) % 100000
            stego_arr = np.array(Image.open(stego_f).convert("RGB"))
            
            try:
                extracted_arr = dct_extract_digital_spectrum(stego_arr, seed)
                
                # Store in session state to persist after download
                st.session_state['dct_dwt_host_image'] = stego_arr.copy()
                st.session_state['dct_dwt_secret_image'] = extracted_arr
                st.session_state['dct_dwt_extraction_done'] = True
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Display images if extraction was done
        if st.session_state.get('dct_dwt_extraction_done', False):
            host_image = st.session_state['dct_dwt_host_image']
            extracted_arr = st.session_state['dct_dwt_secret_image']
            
            st.write("### Decryption Result")
            c1, c2 = st.columns(2)
            with c1:
                st.image(host_image, caption="Extracted Host Image", use_column_width=True)
            with c2:
                st.image(extracted_arr, caption="Extracted Secret Image (Sharp Digital)", use_column_width=True)
            st.success("Extraction Successful.")
            
            # Download buttons
            st.write("### Download Extracted Images")
            d1, d2 = st.columns(2)
            with d1:
                host_img = Image.fromarray(host_image)
                buf_host = io.BytesIO()
                host_img.save(buf_host, format="PNG")
                st.download_button("📥 Download Host Image", buf_host.getvalue(), "extracted_host_dct_dwt.png", "image/png", key="dl_host_dct_dwt")
            with d2:
                secret_img = Image.fromarray(extracted_arr)
                buf_secret = io.BytesIO()
                secret_img.save(buf_secret, format="PNG")
                st.download_button("📥 Download Secret Image", buf_secret.getvalue(), "extracted_secret_dct_dwt.png", "image/png", key="dl_secret_dct_dwt")