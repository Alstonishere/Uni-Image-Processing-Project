"""
Digital Watermarking Attack & Quality Measurement Module
Streamlit-based User Interface
"""

import streamlit as st
import cv2
import numpy as np
import io

# ============== MEASUREMENT FUNCTIONS ==============

def calculate_psnr(original, watermarked):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) - Imperceptibility measure"""
    if original.shape != watermarked.shape:
        watermarked = cv2.resize(watermarked, (original.shape[1], original.shape[0]))
    
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    
    mse = np.mean((original - watermarked) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

def calculate_ber(original_watermark, extracted_watermark):
    """Calculate Bit Error Rate (BER) - Robustness measure"""
    original_flat = original_watermark.flatten()
    extracted_flat = extracted_watermark.flatten()
    
    min_len = min(len(original_flat), len(extracted_flat))
    original_flat = original_flat[:min_len]
    extracted_flat = extracted_flat[:min_len]
    
    original_bits = (original_flat > 128).astype(np.uint8)
    extracted_bits = (extracted_flat > 128).astype(np.uint8)
    
    bit_errors = np.sum(original_bits != extracted_bits)
    total_bits = len(original_bits)
    
    ber = bit_errors / total_bits if total_bits > 0 else 0
    return ber, bit_errors, total_bits

def calculate_bpp(watermark, host_image):
    """Calculate Bits Per Pixel (BPP) - Capacity measure"""
    if len(watermark.shape) == 3:
        watermark_bits = watermark.shape[0] * watermark.shape[1] * watermark.shape[2] * 8
    else:
        watermark_bits = watermark.shape[0] * watermark.shape[1] * 8
    
    host_pixels = host_image.shape[0] * host_image.shape[1]
    
    bpp = watermark_bits / host_pixels if host_pixels > 0 else 0
    return bpp, watermark_bits, host_pixels

def calculate_ssim(original, watermarked):
    """Calculate Structural Similarity Index (SSIM)"""
    if original.shape != watermarked.shape:
        watermarked = cv2.resize(watermarked, (original.shape[1], original.shape[0]))
    
    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = cv2.GaussianBlur(original, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(watermarked, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(original ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(watermarked ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(original * watermarked, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

def calculate_nc(original_watermark, extracted_watermark):
    """Calculate Normalized Correlation (NC)"""
    original_flat = original_watermark.flatten().astype(np.float64)
    extracted_flat = extracted_watermark.flatten().astype(np.float64)
    
    min_len = min(len(original_flat), len(extracted_flat))
    original_flat = original_flat[:min_len]
    extracted_flat = extracted_flat[:min_len]
    
    original_norm = original_flat - np.mean(original_flat)
    extracted_norm = extracted_flat - np.mean(extracted_flat)
    
    numerator = np.sum(original_norm * extracted_norm)
    denominator = np.sqrt(np.sum(original_norm ** 2) * np.sum(extracted_norm ** 2))
    
    nc = numerator / denominator if denominator != 0 else 0
    return nc

# ============== ATTACK FUNCTIONS ==============

def gaussian_noise_attack(image, mean=0, sigma=25):
    """Add Gaussian noise to the image"""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def salt_pepper_noise_attack(image, amount=0.02):
    """Add salt and pepper noise to the image"""
    noisy_image = image.copy()
    total_pixels = image.size
    
    num_salt = int(total_pixels * amount / 2)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    if len(image.shape) == 3:
        noisy_image[coords[0], coords[1], :] = 255
    else:
        noisy_image[coords[0], coords[1]] = 255
    
    num_pepper = int(total_pixels * amount / 2)
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    if len(image.shape) == 3:
        noisy_image[coords[0], coords[1], :] = 0
    else:
        noisy_image[coords[0], coords[1]] = 0
    
    return noisy_image

def median_filter_attack(image, kernel_size=3):
    """Apply median filtering to the image"""
    return cv2.medianBlur(image, kernel_size)

def gaussian_blur_attack(image, kernel_size=5):
    """Apply Gaussian blur to the image"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def jpeg_compression_attack(image, quality=50):
    """Simulate JPEG compression attack"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR if len(image.shape) == 3 else cv2.IMREAD_GRAYSCALE)

def rotation_attack(image, angle=5):
    """Rotate the image by a given angle"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

def scaling_attack(image, scale_factor=0.8):
    """Scale image down and back up"""
    h, w = image.shape[:2]
    small = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
    return cv2.resize(small, (w, h))

def cropping_attack(image, crop_percent=10):
    """Crop edges of the image and resize back"""
    h, w = image.shape[:2]
    crop_h = int(h * crop_percent / 100)
    crop_w = int(w * crop_percent / 100)
    cropped = image[crop_h:h-crop_h, crop_w:w-crop_w]
    return cv2.resize(cropped, (w, h))

def histogram_equalization_attack(image):
    """Apply histogram equalization"""
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(image)

def brightness_attack(image, value=30):
    """Adjust brightness of the image"""
    adjusted = image.astype(np.float32) + value
    return np.clip(adjusted, 0, 255).astype(np.uint8)

def contrast_attack(image, factor=1.3):
    """Adjust contrast of the image"""
    adjusted = image.astype(np.float32) * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)

# ============== HELPER FUNCTIONS ==============

def load_image(uploaded_file):
    """Load image from uploaded file"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)
    return image

def load_image_grayscale(uploaded_file):
    """Load image as grayscale from uploaded file"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    uploaded_file.seek(0)
    return image

def image_to_bytes(image, format='.png', quality=95):
    """Convert image to bytes for download"""
    if format.lower() in ['.jpg', '.jpeg']:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode(format, image, encode_param)
    else:
        _, buffer = cv2.imencode(format, image)
    return buffer.tobytes()

def get_quality_label_psnr(psnr):
    """Get quality label based on PSNR value"""
    if psnr == float('inf'):
        return "Identical", "success"
    elif psnr > 40:
        return "Excellent", "success"
    elif psnr > 30:
        return "Good", "info"
    elif psnr > 20:
        return "Fair", "warning"
    else:
        return "Poor", "error"

def get_quality_label_ber(ber):
    """Get quality label based on BER value"""
    if ber < 0.01:
        return "Excellent", "success"
    elif ber < 0.05:
        return "Good", "info"
    elif ber < 0.10:
        return "Fair", "warning"
    else:
        return "Poor", "error"

def get_quality_label_ssim(ssim):
    """Get quality label based on SSIM value"""
    if ssim > 0.95:
        return "Excellent", "success"
    elif ssim > 0.85:
        return "Good", "info"
    elif ssim > 0.70:
        return "Fair", "warning"
    else:
        return "Poor", "error"

def get_quality_label_nc(nc):
    """Get quality label based on NC value"""
    if nc > 0.95:
        return "Excellent", "success"
    elif nc > 0.85:
        return "Good", "info"
    elif nc > 0.70:
        return "Fair", "warning"
    else:
        return "Poor", "error"

# ============== MAIN APP ==============

def run():
    """Main function to run the attack and measurement module"""
    st.title("🔒 Watermark Attack & Quality Measurement")

    # In-page selection instead of sidebar
    mode = st.radio("Select Mode", ["🎯 Attack Simulation", "📊 Quality Measurements"], horizontal=True)
    
    st.divider()

    # ============== ATTACK SIMULATION ==============
    if mode == "🎯 Attack Simulation":
        st.header("Attack Simulation")
        st.write("Apply various attacks to test watermark robustness")
        
        uploaded_image = st.file_uploader("Upload Watermarked Image", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])
        
        if uploaded_image:
            image = load_image(uploaded_image)
            
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.divider()
            st.subheader("Select Attack")
            
            attack_type = st.selectbox("Attack Type", [
                "Gaussian Noise",
                "Salt & Pepper Noise",
                "Median Filter",
                "Gaussian Blur",
                "JPEG Compression",
                "Rotation",
                "Scaling",
                "Cropping",
                "Histogram Equalization",
                "Brightness Adjustment",
                "Contrast Adjustment"
            ])
            
            # Attack parameters
            st.subheader("Attack Parameters")
            
            if attack_type == "Gaussian Noise":
                sigma = st.slider("Noise Sigma", 5, 50, 25)
                attacked_image = gaussian_noise_attack(image, sigma=sigma)
                
            elif attack_type == "Salt & Pepper Noise":
                amount = st.slider("Noise Amount", 0.01, 0.10, 0.02)
                attacked_image = salt_pepper_noise_attack(image, amount=amount)
                
            elif attack_type == "Median Filter":
                kernel = st.select_slider("Kernel Size", options=[3, 5, 7, 9], value=3)
                attacked_image = median_filter_attack(image, kernel_size=kernel)
                
            elif attack_type == "Gaussian Blur":
                kernel = st.select_slider("Kernel Size", options=[3, 5, 7, 9, 11], value=5)
                attacked_image = gaussian_blur_attack(image, kernel_size=kernel)
                
            elif attack_type == "JPEG Compression":
                quality = st.slider("JPEG Quality", 10, 95, 50)
                attacked_image = jpeg_compression_attack(image, quality=quality)
                
            elif attack_type == "Rotation":
                angle = st.slider("Rotation Angle (degrees)", -45, 45, 5)
                attacked_image = rotation_attack(image, angle=angle)
                
            elif attack_type == "Scaling":
                scale = st.slider("Scale Factor", 0.5, 0.95, 0.8)
                attacked_image = scaling_attack(image, scale_factor=scale)
                
            elif attack_type == "Cropping":
                crop_pct = st.slider("Crop Percentage", 5, 30, 10)
                attacked_image = cropping_attack(image, crop_percent=crop_pct)
                
            elif attack_type == "Histogram Equalization":
                attacked_image = histogram_equalization_attack(image)
                
            elif attack_type == "Brightness Adjustment":
                brightness = st.slider("Brightness Value", -100, 100, 30)
                attacked_image = brightness_attack(image, value=brightness)
                
            elif attack_type == "Contrast Adjustment":
                contrast = st.slider("Contrast Factor", 0.5, 2.0, 1.3)
                attacked_image = contrast_attack(image, factor=contrast)
            
            # Display result
            st.divider()
            st.subheader("Attack Result")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original**")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.write("**Attacked**")
                st.image(cv2.cvtColor(attacked_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Quick PSNR comparison
            psnr = calculate_psnr(image, attacked_image)
            st.metric("PSNR (Attack Impact)", f"{psnr:.2f} dB" if psnr != float('inf') else "∞ dB")
            
            # Download button - use JPG for JPEG compression attack, PNG for others
            if attack_type == "JPEG Compression":
                st.download_button(
                    label="📥 Download Attacked Image",
                    data=image_to_bytes(attacked_image, '.jpg', quality),
                    file_name=f"attacked_{attack_type.lower().replace(' ', '_')}_q{quality}.jpg",
                    mime="image/jpeg"
                )
            else:
                st.download_button(
                    label="📥 Download Attacked Image",
                    data=image_to_bytes(attacked_image, '.png'),
                    file_name=f"attacked_{attack_type.lower().replace(' ', '_')}.png",
                    mime="image/png"
                )

    # ============== QUALITY MEASUREMENTS ==============
    elif mode == "📊 Quality Measurements":
        st.header("Quality Measurements")
        
        measurement_type = st.selectbox("Select Measurement", [
            "PSNR - Imperceptibility",
            "BER - Robustness",
            "BPP - Capacity",
            "SSIM - Structural Similarity",
            "NC - Normalized Correlation",
            "Full Analysis"
        ])
        
        st.divider()
        
        # ============== PSNR ==============
        if measurement_type == "PSNR - Imperceptibility":
            st.subheader("PSNR (Peak Signal-to-Noise Ratio)")
            st.info("Measures how imperceptible the watermark is. **Higher PSNR = better invisibility**")
            
            col1, col2 = st.columns(2)
            with col1:
                original_file = st.file_uploader("Original Host Image", type=['png', 'jpg', 'jpeg', 'bmp'], key="psnr_orig")
            with col2:
                extracted_file = st.file_uploader("Extracted Host Image", type=['png', 'jpg', 'jpeg', 'bmp'], key="psnr_ext")
            
            if original_file and extracted_file:
                original = load_image(original_file)
                extracted = load_image(extracted_file)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original Host", use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB), caption="Extracted Host", use_container_width=True)
                
                psnr = calculate_psnr(original, extracted)
                label, status = get_quality_label_psnr(psnr)
                
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("PSNR", f"{psnr:.2f} dB" if psnr != float('inf') else "∞ dB (Identical)")
                with col2:
                    st.metric("Quality", label)
                
                st.markdown("""
                **Interpretation:**
                - PSNR > 40 dB: Excellent imperceptibility
                - PSNR 30-40 dB: Good imperceptibility
                - PSNR 20-30 dB: Fair imperceptibility
                - PSNR < 20 dB: Poor imperceptibility
                """)
        
        # ============== BER ==============
        elif measurement_type == "BER - Robustness":
            st.subheader("BER (Bit Error Rate)")
            st.info("Measures watermark survival after attacks. **Lower BER = better robustness**")
            
            col1, col2 = st.columns(2)
            with col1:
                original_wm_file = st.file_uploader("Original Watermark", type=['png', 'jpg', 'jpeg', 'bmp'], key="ber_orig")
            with col2:
                extracted_wm_file = st.file_uploader("Extracted Watermark (after attack)", type=['png', 'jpg', 'jpeg', 'bmp'], key="ber_ext")
            
            if original_wm_file and extracted_wm_file:
                original_wm = load_image_grayscale(original_wm_file)
                extracted_wm = load_image_grayscale(extracted_wm_file)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_wm, caption="Original Watermark", use_container_width=True)
                with col2:
                    st.image(extracted_wm, caption="Extracted Watermark", use_container_width=True)
                
                ber, bit_errors, total_bits = calculate_ber(original_wm, extracted_wm)
                nc = calculate_nc(original_wm, extracted_wm)
                label, status = get_quality_label_ber(ber)
                
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("BER", f"{ber:.4f} ({ber*100:.2f}%)")
                with col2:
                    st.metric("Bit Errors", f"{bit_errors:,} / {total_bits:,}")
                with col3:
                    st.metric("NC", f"{nc:.4f}")
                with col4:
                    st.metric("Quality", label)
                
                st.markdown("""
                **Interpretation:**
                - BER < 1%: Excellent robustness
                - BER 1-5%: Good robustness
                - BER 5-10%: Fair robustness
                - BER > 10%: Poor robustness
                """)
        
        # ============== BPP ==============
        elif measurement_type == "BPP - Capacity":
            st.subheader("BPP (Bits Per Pixel)")
            st.info("Measures embedding capacity. **Higher BPP = more data can be embedded**")
            
            col1, col2 = st.columns(2)
            with col1:
                host_file = st.file_uploader("Host Image", type=['png', 'jpg', 'jpeg', 'bmp'], key="bpp_host")
            with col2:
                watermark_file = st.file_uploader("Watermark Image", type=['png', 'jpg', 'jpeg', 'bmp'], key="bpp_wm")
            
            if host_file and watermark_file:
                host = load_image(host_file)
                watermark = load_image_grayscale(watermark_file)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(host, cv2.COLOR_BGR2RGB), caption="Host Image", use_container_width=True)
                with col2:
                    st.image(watermark, caption="Watermark", use_container_width=True)
                
                bpp, watermark_bits, host_pixels = calculate_bpp(watermark, host)
                
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("BPP", f"{bpp:.4f} bits/pixel")
                with col2:
                    st.metric("Watermark Bits", f"{watermark_bits:,}")
                with col3:
                    st.metric("Host Pixels", f"{host_pixels:,}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Host Size", f"{host.shape[1]} x {host.shape[0]}")
                with col2:
                    st.metric("Watermark Size", f"{watermark.shape[1]} x {watermark.shape[0]}")
                
                st.markdown("""
                **Interpretation:**
                - BPP < 0.1: Low capacity (small watermark)
                - BPP 0.1-1.0: Moderate capacity
                - BPP > 1.0: High capacity (large watermark)
                """)
        
        # ============== SSIM ==============
        elif measurement_type == "SSIM - Structural Similarity":
            st.subheader("SSIM (Structural Similarity Index)")
            st.info("Measures structural similarity. **Higher SSIM (closer to 1) = better quality**")
            
            col1, col2 = st.columns(2)
            with col1:
                original_file = st.file_uploader("Original Host Image", type=['png', 'jpg', 'jpeg', 'bmp'], key="ssim_orig")
            with col2:
                extracted_file = st.file_uploader("Extracted Host Image", type=['png', 'jpg', 'jpeg', 'bmp'], key="ssim_ext")
            
            if original_file and extracted_file:
                original = load_image(original_file)
                extracted = load_image(extracted_file)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original Host", use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB), caption="Extracted Host", use_container_width=True)
                
                ssim = calculate_ssim(original, extracted)
                label, status = get_quality_label_ssim(ssim)
                
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("SSIM", f"{ssim:.4f}")
                with col2:
                    st.metric("Quality", label)
                
                st.markdown("""
                **Interpretation:**
                - SSIM > 0.95: Excellent structural similarity
                - SSIM 0.85-0.95: Good structural similarity
                - SSIM 0.70-0.85: Fair structural similarity
                - SSIM < 0.70: Poor structural similarity
                """)
        
        # ============== NC ==============
        elif measurement_type == "NC - Normalized Correlation":
            st.subheader("NC (Normalized Correlation)")
            st.info("Measures watermark correlation. **Higher NC (closer to 1) = better robustness**")
            
            col1, col2 = st.columns(2)
            with col1:
                original_wm_file = st.file_uploader("Original Watermark", type=['png', 'jpg', 'jpeg', 'bmp'], key="nc_orig")
            with col2:
                extracted_wm_file = st.file_uploader("Extracted Watermark", type=['png', 'jpg', 'jpeg', 'bmp'], key="nc_ext")
            
            if original_wm_file and extracted_wm_file:
                original_wm = load_image_grayscale(original_wm_file)
                extracted_wm = load_image_grayscale(extracted_wm_file)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_wm, caption="Original Watermark", use_container_width=True)
                with col2:
                    st.image(extracted_wm, caption="Extracted Watermark", use_container_width=True)
                
                nc = calculate_nc(original_wm, extracted_wm)
                label, status = get_quality_label_nc(nc)
                
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("NC", f"{nc:.4f}")
                with col2:
                    st.metric("Quality", label)
                
                st.markdown("""
                **Interpretation:**
                - NC > 0.95: Excellent correlation
                - NC 0.85-0.95: Good correlation
                - NC 0.70-0.85: Fair correlation
                - NC < 0.70: Poor correlation
                """)
        
        # ============== FULL ANALYSIS ==============
        elif measurement_type == "Full Analysis":
            st.subheader("Full Watermarking Quality Analysis")
            st.info("Comprehensive analysis of imperceptibility, robustness, and capacity")
            
            st.write("**Host Images**")
            col1, col2 = st.columns(2)
            with col1:
                original_host_file = st.file_uploader("Original Host Image", type=['png', 'jpg', 'jpeg', 'bmp'], key="full_orig_host")
            with col2:
                extracted_host_file = st.file_uploader("Extracted Host Image", type=['png', 'jpg', 'jpeg', 'bmp'], key="full_ext_host")
            
            st.write("**Watermarks**")
            col1, col2 = st.columns(2)
            with col1:
                original_wm_file = st.file_uploader("Original Watermark", type=['png', 'jpg', 'jpeg', 'bmp'], key="full_orig_wm")
            with col2:
                extracted_wm_file = st.file_uploader("Extracted Watermark", type=['png', 'jpg', 'jpeg', 'bmp'], key="full_ext_wm")
            
            if original_host_file and extracted_host_file and original_wm_file and extracted_wm_file:
                original_host = load_image(original_host_file)
                extracted_host = load_image(extracted_host_file)
                original_wm = load_image(original_wm_file)
                extracted_wm = load_image(extracted_wm_file)
                
                # Also load grayscale versions for BER/NC calculations
                original_wm_file.seek(0)
                extracted_wm_file.seek(0)
                original_wm_gray = load_image_grayscale(original_wm_file)
                extracted_wm_gray = load_image_grayscale(extracted_wm_file)
                
                # Display images
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.image(cv2.cvtColor(original_host, cv2.COLOR_BGR2RGB), caption="Original Host", use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(extracted_host, cv2.COLOR_BGR2RGB), caption="Extracted Host", use_container_width=True)
                with col3:
                    st.image(cv2.cvtColor(original_wm, cv2.COLOR_BGR2RGB), caption="Original Watermark", use_container_width=True)
                with col4:
                    st.image(cv2.cvtColor(extracted_wm, cv2.COLOR_BGR2RGB), caption="Extracted Watermark", use_container_width=True)
                
                # Calculate all metrics
                psnr = calculate_psnr(original_host, extracted_host)
                ssim = calculate_ssim(original_host, extracted_host)
                ber, bit_errors, total_bits = calculate_ber(original_wm_gray, extracted_wm_gray)
                nc = calculate_nc(original_wm_gray, extracted_wm_gray)
                bpp, watermark_bits, host_pixels = calculate_bpp(original_wm, original_host)
                
                # Display results
                st.divider()
                
                st.subheader("📊 Imperceptibility")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("PSNR", f"{psnr:.2f} dB" if psnr != float('inf') else "∞ dB")
                with col2:
                    st.metric("SSIM", f"{ssim:.4f}")
                with col3:
                    label, _ = get_quality_label_psnr(psnr)
                    st.metric("Quality", label)
                
                st.subheader("🛡️ Robustness")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("BER", f"{ber:.4f} ({ber*100:.2f}%)")
                with col2:
                    st.metric("NC", f"{nc:.4f}")
                with col3:
                    label, _ = get_quality_label_ber(ber)
                    st.metric("Quality", label)
                
                st.subheader("💾 Capacity")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("BPP", f"{bpp:.4f} bits/pixel")
                with col2:
                    st.metric("Watermark Bits", f"{watermark_bits:,}")
                with col3:
                    st.metric("Host Pixels", f"{host_pixels:,}")

    # Footer
    st.divider()
    st.caption("Digital Watermarking Attack & Quality Measurement Module")
