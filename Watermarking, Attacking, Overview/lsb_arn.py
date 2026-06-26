import streamlit as st
import login
import numpy as np
import cv2
from PIL import Image
import io

# ### --- ALGORITHM CORE START ---

def resize_to_square(image, size=None):
    h, w = image.shape[:2]
    min_dim = min(h, w)
    if size: min_dim = size
    return cv2.resize(image, (min_dim, min_dim))

def arnold_scramble(image, iterations):
    """
    Scrambles RGB image using Arnold Cat Map.
    """
    h, w = image.shape[:2]
    N = h
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    x_new = (x + y) % N
    y_new = (x + 2*y) % N
    
    scrambled = np.zeros_like(image)
    for c in range(3): # Loop R, G, B
        scrambled[x_new, y_new, c] = image[x, y, c]
        
    for _ in range(iterations - 1):
        temp = np.zeros_like(scrambled)
        for c in range(3):
            temp[x_new, y_new, c] = scrambled[x, y, c]
        scrambled = temp
    return scrambled

def arnold_inverse(image, iterations):
    """
    Descrambles RGB image.
    """
    h, w = image.shape[:2]
    N = h
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    x_old = (2*x - y) % N
    y_old = (-x + y) % N
    
    restored = np.zeros_like(image)
    for c in range(3):
        restored[x_old, y_old, c] = image[x, y, c]
        
    for _ in range(iterations - 1):
        temp = np.zeros_like(restored)
        for c in range(3):
            temp[x_old, y_old, c] = restored[x, y, c]
        restored = temp
    return restored

def lsb_embed_hifi(cover, secret, key_hash):
    """
    Embeds 3 bits per channel (High Fidelity) to reduce noise in Secret.
    """
    h_c, w_c, _ = cover.shape
    
    # 1. Resize Secret to max safe square size (512 max to prevent lag)
    SQ_SIZE = min(h_c, w_c, 512) 
    secret_sq = cv2.resize(secret, (SQ_SIZE, SQ_SIZE))
    
    # 2. Scramble
    iterations = int(key_hash, 16) % 10 + 1
    secret_scrambled = arnold_scramble(secret_sq, iterations)
    
    # 3. Prepare Bits (Top 3 Bits of Secret)
    # 11100000 = 0xE0
    # Shift >> 5 moves them to 00000111
    secret_bits = (secret_scrambled & 0xE0) >> 5
    
    # 4. Get ROI
    cover_roi = cover[0:SQ_SIZE, 0:SQ_SIZE]
    
    # 5. Clear LSBs (Bottom 3 bits)
    # 11111000 = 0xF8
    cover_clean = cover_roi & 0xF8
    
    # 6. Embed
    stego_roi = cover_clean | secret_bits
    
    # 7. Reconstruct
    cover[0:SQ_SIZE, 0:SQ_SIZE] = stego_roi
    
    return cover, SQ_SIZE

def lsb_extract_hifi(stego, key_hash, sq_size=None):
    """
    Extracts 3 bits and applies Denoising.
    """
    if sq_size is None: sq_size = 512
        
    # 1. Extract Bits
    stego_roi = stego[0:sq_size, 0:sq_size]
    extracted_bits = stego_roi & 0x07 # Bottom 3 bits
    
    # 2. Shift to MSB (00000111 -> 11100000)
    recovered_scrambled = extracted_bits << 5
    
    # 3. Inverse Scramble
    iterations = int(key_hash, 16) % 10 + 1
    restored_img = arnold_inverse(recovered_scrambled, iterations)
    
    # 4. Post-Processing: Denoise
    # This removes the "grain" caused by quantization
    clean_img = cv2.medianBlur(restored_img, 3)
    
    return clean_img

# ### --- ALGORITHM CORE END ---

def run(user=None):
    if user is None:
        user = st.session_state.get("user")

    st.info("ℹ️ **Algorithm:** Hi-Fi LSB + Arnold (RGB)")
    st.write("Enhanced with 3-bit depth and Denoising for cleaner images.")
    
    action = st.radio("Choose Action", ["Encrypt (Hide)", "Decrypt (Extract)"], horizontal=True, key="lsb_action")

    if action == "Encrypt (Hide)":
        st.subheader("🔒 Hide Image")
        col1, col2 = st.columns(2)
        with col1: cover_f = st.file_uploader("Cover (Color)", type=["png", "jpg"], key="lsb_c")
        with col2: secret_f = st.file_uploader("Secret (Color)", type=["png", "jpg"], key="lsb_s")
        
        # Show uploaded image preview
        if cover_f or secret_f:
            st.divider()
            st.write("### Input Preview")
            c1, c2 = st.columns(2)
            with c1:
                if cover_f:
                    st.image(Image.open(cover_f), caption="Cover Image", use_column_width=True)
                    cover_f.seek(0)
            with c2:
                if secret_f:
                    st.image(Image.open(secret_f), caption="Secret Image", use_column_width=True)
                    secret_f.seek(0)
        
        receiver = st.text_input("Send to User (Receiver)")
        
        if st.button("Encrypt Image") and cover_f and secret_f and receiver:
            key = login.get_user_key(receiver)
            if key:
                cover = np.array(Image.open(cover_f).convert("RGB"))
                secret = np.array(Image.open(secret_f).convert("RGB"))
                
                # Keep original for comparison
                original_host = cover.copy()
                
                # Run Hi-Fi Embedding
                result, used_size = lsb_embed_hifi(cover, secret, key)
                
                # Save size for session
                st.session_state['rgb_size'] = used_size
                
                st.success(f"Encrypted! (Size: {used_size}x{used_size})")
                
                # Comparison
                st.write("### Result Comparison")
                c1, c2 = st.columns(2)
                with c1: 
                    st.image(original_host, caption="Original Host Image", use_column_width=True)
                with c2: 
                    st.image(result, caption="Watermarked Host Image", use_column_width=True)
                
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("Download Watermarked Image", buf.getvalue(), "lsb_hifi.png", "image/png")
            else:
                st.error("User not found.")

    else:
        st.subheader("🔓 Decrypt Image")
        stego_f = st.file_uploader("Upload Encrypted Image", key="lsb_d")
        
        if st.button("Decrypt") and stego_f:
            my_key = login.get_user_key(user)
            stego = np.array(Image.open(stego_f).convert("RGB"))
            
            sq_size = st.session_state.get('rgb_size', 512)
            
            try:
                extracted = lsb_extract_hifi(stego, my_key, sq_size)
                
                # Store in session state to persist after download
                st.session_state['lsb_host_image'] = stego.copy()
                st.session_state['lsb_secret_image'] = extracted
                st.session_state['lsb_extraction_done'] = True
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Display images if extraction was done
        if st.session_state.get('lsb_extraction_done', False):
            host_image = st.session_state['lsb_host_image']
            extracted = st.session_state['lsb_secret_image']
            
            st.write("### Decryption Result")
            c1, c2 = st.columns(2)
            with c1: 
                st.image(host_image, caption="Extracted Host Image", use_column_width=True)
            with c2: 
                st.image(extracted, caption="Extracted Secret Image (Denoised)", use_column_width=True)
            st.success("Color image restored with enhanced clarity.")
            
            # Download buttons
            st.write("### Download Extracted Images")
            d1, d2 = st.columns(2)
            with d1:
                host_img = Image.fromarray(host_image)
                buf_host = io.BytesIO()
                host_img.save(buf_host, format="PNG")
                st.download_button("📥 Download Host Image", buf_host.getvalue(), "extracted_host_lsb.png", "image/png", key="dl_host_lsb")
            with d2:
                secret_img = Image.fromarray(extracted)
                buf_secret = io.BytesIO()
                secret_img.save(buf_secret, format="PNG")
                st.download_button("📥 Download Secret Image", buf_secret.getvalue(), "extracted_secret_lsb.png", "image/png", key="dl_secret_lsb")