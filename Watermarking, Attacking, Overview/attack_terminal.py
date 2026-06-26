import cv2
import numpy as np
from tkinter import filedialog, messagebox
import os

# ============== MEASUREMENT FUNCTIONS ==============

def calculate_psnr(original, watermarked):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) - Imperceptibility measure
    Higher PSNR = better imperceptibility (less visible watermark)
    """
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
    """
    Calculate Bit Error Rate (BER) - Robustness measure
    Lower BER = better robustness (watermark survives attacks)
    BER = number of bit errors / total number of bits
    """
    # Flatten arrays
    original_flat = original_watermark.flatten()
    extracted_flat = extracted_watermark.flatten()
    
    # Ensure same size
    min_len = min(len(original_flat), len(extracted_flat))
    original_flat = original_flat[:min_len]
    extracted_flat = extracted_flat[:min_len]
    
    # Convert to binary (threshold at 128 for grayscale)
    original_bits = (original_flat > 128).astype(np.uint8)
    extracted_bits = (extracted_flat > 128).astype(np.uint8)
    
    # Calculate bit errors
    bit_errors = np.sum(original_bits != extracted_bits)
    total_bits = len(original_bits)
    
    ber = bit_errors / total_bits if total_bits > 0 else 0
    return ber, bit_errors, total_bits

def calculate_bpp(watermark, host_image):
    """
    Calculate Bits Per Pixel (BPP) - Capacity measure
    Higher BPP = more data can be embedded
    BPP = total watermark bits / total host pixels
    """
    # Calculate watermark bits (assuming 8 bits per pixel for grayscale)
    if len(watermark.shape) == 3:
        watermark_bits = watermark.shape[0] * watermark.shape[1] * watermark.shape[2] * 8
    else:
        watermark_bits = watermark.shape[0] * watermark.shape[1] * 8
    
    # Calculate host pixels
    host_pixels = host_image.shape[0] * host_image.shape[1]
    
    bpp = watermark_bits / host_pixels if host_pixels > 0 else 0
    return bpp, watermark_bits, host_pixels

def calculate_ssim(original, watermarked):
    """
    Calculate Structural Similarity Index (SSIM) - Additional imperceptibility measure
    Higher SSIM = better structural similarity (closer to 1)
    """
    if original.shape != watermarked.shape:
        watermarked = cv2.resize(watermarked, (original.shape[1], original.shape[0]))
    
    # Convert to grayscale if color
    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    
    # SSIM constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Mean
    mu1 = cv2.GaussianBlur(original, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(watermarked, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Variance and covariance
    sigma1_sq = cv2.GaussianBlur(original ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(watermarked ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(original * watermarked, (11, 11), 1.5) - mu1_mu2
    
    # SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

def calculate_nc(original_watermark, extracted_watermark):
    """
    Calculate Normalized Correlation (NC) - Robustness measure
    Higher NC = better robustness (closer to 1)
    """
    original_flat = original_watermark.flatten().astype(np.float64)
    extracted_flat = extracted_watermark.flatten().astype(np.float64)
    
    min_len = min(len(original_flat), len(extracted_flat))
    original_flat = original_flat[:min_len]
    extracted_flat = extracted_flat[:min_len]
    
    # Normalize
    original_norm = original_flat - np.mean(original_flat)
    extracted_norm = extracted_flat - np.mean(extracted_flat)
    
    numerator = np.sum(original_norm * extracted_norm)
    denominator = np.sqrt(np.sum(original_norm ** 2) * np.sum(extracted_norm ** 2))
    
    nc = numerator / denominator if denominator != 0 else 0
    return nc

# ============== ATTACK FUNCTIONS ==============

def select_image():
    """Open file dialog to select an image"""
    file_path = filedialog.askopenfilename(
        title="Select Watermarked Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    return file_path

def save_image(image, original_path, attack_name):
    """Save attacked image with appropriate naming"""
    directory = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(directory, f"{name}_{attack_name}{ext}")
    cv2.imwrite(output_path, image)
    return output_path

def gaussian_noise_attack(image, mean=0, sigma=25):
    """Add Gaussian noise to the image"""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def salt_pepper_noise_attack(image, amount=0.02):
    """Add salt and pepper noise to the image"""
    noisy_image = image.copy()
    total_pixels = image.size
    
    # Salt (white pixels)
    num_salt = int(total_pixels * amount / 2)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    if len(image.shape) == 3:
        noisy_image[coords[0], coords[1], :] = 255
    else:
        noisy_image[coords[0], coords[1]] = 255
    
    # Pepper (black pixels)
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
    compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR if len(image.shape) == 3 else cv2.IMREAD_GRAYSCALE)
    return compressed

def rotation_attack(image, angle=5):
    """Rotate the image by a given angle"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

def scaling_attack(image, scale_factor=0.8):
    """Scale image down and back up"""
    h, w = image.shape[:2]
    # Scale down
    small = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
    # Scale back up
    restored = cv2.resize(small, (w, h))
    return restored

def cropping_attack(image, crop_percent=10):
    """Crop edges of the image and resize back"""
    h, w = image.shape[:2]
    crop_h = int(h * crop_percent / 100)
    crop_w = int(w * crop_percent / 100)
    
    cropped = image[crop_h:h-crop_h, crop_w:w-crop_w]
    restored = cv2.resize(cropped, (w, h))
    return restored

def histogram_equalization_attack(image):
    """Apply histogram equalization"""
    if len(image.shape) == 3:
        # Convert to YCrCb and equalize Y channel
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

def run_attack():
    """Main function to run watermarking attacks"""
    # Select image
    image_path = select_image()
    if not image_path:
        print("No image selected.")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Failed to load image.")
        return
    
    print("\n" + "="*50)
    print("DIGITAL WATERMARKING ATTACK MODULE")
    print("="*50)
    print(f"Image loaded: {image_path}")
    print(f"Image size: {image.shape}")
    print("\nAvailable Attacks:")
    print("1. Gaussian Noise")
    print("2. Salt & Pepper Noise")
    print("3. Median Filter")
    print("4. Gaussian Blur")
    print("5. JPEG Compression")
    print("6. Rotation")
    print("7. Scaling")
    print("8. Cropping")
    print("9. Histogram Equalization")
    print("10. Brightness Adjustment")
    print("11. Contrast Adjustment")
    print("12. Apply All Attacks")
    print("0. Exit")
    
    choice = input("\nSelect attack (0-12): ").strip()
    
    if choice == "0":
        return
    
    attacked_image = None
    attack_name = ""
    
    if choice == "1":
        attacked_image = gaussian_noise_attack(image)
        attack_name = "gaussian_noise"
    elif choice == "2":
        attacked_image = salt_pepper_noise_attack(image)
        attack_name = "salt_pepper"
    elif choice == "3":
        attacked_image = median_filter_attack(image)
        attack_name = "median_filter"
    elif choice == "4":
        attacked_image = gaussian_blur_attack(image)
        attack_name = "gaussian_blur"
    elif choice == "5":
        attacked_image = jpeg_compression_attack(image)
        attack_name = "jpeg_compression"
    elif choice == "6":
        attacked_image = rotation_attack(image)
        attack_name = "rotation"
    elif choice == "7":
        attacked_image = scaling_attack(image)
        attack_name = "scaling"
    elif choice == "8":
        attacked_image = cropping_attack(image)
        attack_name = "cropping"
    elif choice == "9":
        attacked_image = histogram_equalization_attack(image)
        attack_name = "histogram_eq"
    elif choice == "10":
        attacked_image = brightness_attack(image)
        attack_name = "brightness"
    elif choice == "11":
        attacked_image = contrast_attack(image)
        attack_name = "contrast"
    elif choice == "12":
        # Apply all attacks and save each
        attacks = [
            (gaussian_noise_attack, "gaussian_noise"),
            (salt_pepper_noise_attack, "salt_pepper"),
            (median_filter_attack, "median_filter"),
            (gaussian_blur_attack, "gaussian_blur"),
            (jpeg_compression_attack, "jpeg_compression"),
            (rotation_attack, "rotation"),
            (scaling_attack, "scaling"),
            (cropping_attack, "cropping"),
            (histogram_equalization_attack, "histogram_eq"),
            (brightness_attack, "brightness"),
            (contrast_attack, "contrast"),
        ]
        
        print("\nApplying all attacks...")
        for attack_func, name in attacks:
            result = attack_func(image)
            output_path = save_image(result, image_path, name)
            print(f"  - {name}: saved to {output_path}")
        
        print("\nAll attacks completed!")
        messagebox.showinfo("Success", "All attacks applied and saved!")
        return
    else:
        print("Invalid choice.")
        return
    
    # Save single attack result
    if attacked_image is not None:
        output_path = save_image(attacked_image, image_path, attack_name)
        print(f"\nAttack applied successfully!")
        print(f"Saved to: {output_path}")
        messagebox.showinfo("Success", f"Attack applied!\nSaved to: {output_path}")

def run_measurements():
    """Main function to run watermarking measurements"""
    print("\n" + "="*50)
    print("WATERMARKING QUALITY MEASUREMENTS")
    print("="*50)
    print("\nMeasurement Options:")
    print("1. PSNR (Imperceptibility) - Compare original & watermarked image")
    print("2. BER (Robustness) - Compare original & extracted watermark")
    print("3. BPP (Capacity) - Calculate embedding capacity")
    print("4. SSIM (Structural Similarity) - Compare original & watermarked image")
    print("5. NC (Normalized Correlation) - Compare original & extracted watermark")
    print("6. Full Analysis (All metrics)")
    print("0. Back")
    
    choice = input("\nSelect measurement (0-6): ").strip()
    
    if choice == "0":
        return
    
    if choice == "1":
        # PSNR - Imperceptibility
        print("\n--- PSNR (Imperceptibility) ---")
        print("Select ORIGINAL host image:")
        original_path = filedialog.askopenfilename(title="Select Original Host Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not original_path:
            return
        
        print("Select WATERMARKED host image:")
        watermarked_path = filedialog.askopenfilename(title="Select Watermarked Host Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not watermarked_path:
            return
        
        original = cv2.imread(original_path)
        watermarked = cv2.imread(watermarked_path)
        
        psnr = calculate_psnr(original, watermarked)
        ssim = calculate_ssim(original, watermarked)
        
        print(f"\n{'='*40}")
        print("IMPERCEPTIBILITY RESULTS")
        print(f"{'='*40}")
        print(f"PSNR: {psnr:.2f} dB" if psnr != float('inf') else "PSNR: Identical (∞)")
        print(f"SSIM: {ssim:.4f}")
        print(f"\nInterpretation:")
        if psnr > 40:
            print("  PSNR > 40 dB: Excellent imperceptibility")
        elif psnr > 30:
            print("  PSNR 30-40 dB: Good imperceptibility")
        elif psnr > 20:
            print("  PSNR 20-30 dB: Fair imperceptibility")
        else:
            print("  PSNR < 20 dB: Poor imperceptibility")
        
    elif choice == "2":
        # BER - Robustness
        print("\n--- BER (Robustness) ---")
        print("Select ORIGINAL watermark:")
        original_wm_path = filedialog.askopenfilename(title="Select Original Watermark",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not original_wm_path:
            return
        
        print("Select EXTRACTED watermark (after attack):")
        extracted_wm_path = filedialog.askopenfilename(title="Select Extracted Watermark",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not extracted_wm_path:
            return
        
        original_wm = cv2.imread(original_wm_path, cv2.IMREAD_GRAYSCALE)
        extracted_wm = cv2.imread(extracted_wm_path, cv2.IMREAD_GRAYSCALE)
        
        ber, bit_errors, total_bits = calculate_ber(original_wm, extracted_wm)
        nc = calculate_nc(original_wm, extracted_wm)
        
        print(f"\n{'='*40}")
        print("ROBUSTNESS RESULTS")
        print(f"{'='*40}")
        print(f"BER: {ber:.4f} ({ber*100:.2f}%)")
        print(f"Bit Errors: {bit_errors} / {total_bits}")
        print(f"NC: {nc:.4f}")
        print(f"\nInterpretation:")
        if ber < 0.01:
            print("  BER < 1%: Excellent robustness")
        elif ber < 0.05:
            print("  BER 1-5%: Good robustness")
        elif ber < 0.10:
            print("  BER 5-10%: Fair robustness")
        else:
            print("  BER > 10%: Poor robustness")
        
    elif choice == "3":
        # BPP - Capacity
        print("\n--- BPP (Capacity) ---")
        print("Select HOST image:")
        host_path = filedialog.askopenfilename(title="Select Host Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not host_path:
            return
        
        print("Select WATERMARK image:")
        watermark_path = filedialog.askopenfilename(title="Select Watermark Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not watermark_path:
            return
        
        host = cv2.imread(host_path)
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        
        bpp, watermark_bits, host_pixels = calculate_bpp(watermark, host)
        
        print(f"\n{'='*40}")
        print("CAPACITY RESULTS")
        print(f"{'='*40}")
        print(f"BPP: {bpp:.4f} bits/pixel")
        print(f"Watermark Bits: {watermark_bits:,}")
        print(f"Host Pixels: {host_pixels:,}")
        print(f"Host Size: {host.shape[1]}x{host.shape[0]}")
        print(f"Watermark Size: {watermark.shape[1]}x{watermark.shape[0]}")
        print(f"\nInterpretation:")
        if bpp < 0.1:
            print("  BPP < 0.1: Low capacity (small watermark)")
        elif bpp < 1.0:
            print("  BPP 0.1-1.0: Moderate capacity")
        else:
            print("  BPP > 1.0: High capacity (large watermark)")
        
    elif choice == "4":
        # SSIM
        print("\n--- SSIM (Structural Similarity) ---")
        print("Select ORIGINAL host image:")
        original_path = filedialog.askopenfilename(title="Select Original Host Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not original_path:
            return
        
        print("Select WATERMARKED host image:")
        watermarked_path = filedialog.askopenfilename(title="Select Watermarked Host Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not watermarked_path:
            return
        
        original = cv2.imread(original_path)
        watermarked = cv2.imread(watermarked_path)
        
        ssim = calculate_ssim(original, watermarked)
        
        print(f"\n{'='*40}")
        print("SSIM RESULTS")
        print(f"{'='*40}")
        print(f"SSIM: {ssim:.4f}")
        print(f"\nInterpretation:")
        if ssim > 0.95:
            print("  SSIM > 0.95: Excellent structural similarity")
        elif ssim > 0.85:
            print("  SSIM 0.85-0.95: Good structural similarity")
        elif ssim > 0.70:
            print("  SSIM 0.70-0.85: Fair structural similarity")
        else:
            print("  SSIM < 0.70: Poor structural similarity")
        
    elif choice == "5":
        # NC - Normalized Correlation
        print("\n--- NC (Normalized Correlation) ---")
        print("Select ORIGINAL watermark:")
        original_wm_path = filedialog.askopenfilename(title="Select Original Watermark",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not original_wm_path:
            return
        
        print("Select EXTRACTED watermark:")
        extracted_wm_path = filedialog.askopenfilename(title="Select Extracted Watermark",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not extracted_wm_path:
            return
        
        original_wm = cv2.imread(original_wm_path, cv2.IMREAD_GRAYSCALE)
        extracted_wm = cv2.imread(extracted_wm_path, cv2.IMREAD_GRAYSCALE)
        
        nc = calculate_nc(original_wm, extracted_wm)
        
        print(f"\n{'='*40}")
        print("NC RESULTS")
        print(f"{'='*40}")
        print(f"NC: {nc:.4f}")
        print(f"\nInterpretation:")
        if nc > 0.95:
            print("  NC > 0.95: Excellent correlation")
        elif nc > 0.85:
            print("  NC 0.85-0.95: Good correlation")
        elif nc > 0.70:
            print("  NC 0.70-0.85: Fair correlation")
        else:
            print("  NC < 0.70: Poor correlation")
        
    elif choice == "6":
        # Full Analysis
        print("\n--- FULL ANALYSIS ---")
        print("Select ORIGINAL host image:")
        original_host_path = filedialog.askopenfilename(title="Select Original Host Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not original_host_path:
            return
        
        print("Select WATERMARKED host image:")
        watermarked_host_path = filedialog.askopenfilename(title="Select Watermarked Host Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not watermarked_host_path:
            return
        
        print("Select ORIGINAL watermark:")
        original_wm_path = filedialog.askopenfilename(title="Select Original Watermark",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not original_wm_path:
            return
        
        print("Select EXTRACTED watermark:")
        extracted_wm_path = filedialog.askopenfilename(title="Select Extracted Watermark",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not extracted_wm_path:
            return
        
        original_host = cv2.imread(original_host_path)
        watermarked_host = cv2.imread(watermarked_host_path)
        original_wm = cv2.imread(original_wm_path, cv2.IMREAD_GRAYSCALE)
        extracted_wm = cv2.imread(extracted_wm_path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate all metrics
        psnr = calculate_psnr(original_host, watermarked_host)
        ssim = calculate_ssim(original_host, watermarked_host)
        ber, bit_errors, total_bits = calculate_ber(original_wm, extracted_wm)
        nc = calculate_nc(original_wm, extracted_wm)
        bpp, watermark_bits, host_pixels = calculate_bpp(original_wm, original_host)
        
        print(f"\n{'='*50}")
        print("FULL WATERMARKING QUALITY ANALYSIS")
        print(f"{'='*50}")
        
        print(f"\n--- IMPERCEPTIBILITY ---")
        print(f"PSNR: {psnr:.2f} dB" if psnr != float('inf') else "PSNR: Identical (∞)")
        print(f"SSIM: {ssim:.4f}")
        
        print(f"\n--- ROBUSTNESS ---")
        print(f"BER: {ber:.4f} ({ber*100:.2f}%)")
        print(f"NC: {nc:.4f}")
        
        print(f"\n--- CAPACITY ---")
        print(f"BPP: {bpp:.4f} bits/pixel")
        
        print(f"\n{'='*50}")
    else:
        print("Invalid choice.")

def main_menu():
    """Main menu for attack and measurement module"""
    while True:
        print("\n" + "="*50)
        print("WATERMARKING ATTACK & MEASUREMENT MODULE")
        print("="*50)
        print("1. Run Attacks")
        print("2. Run Measurements")
        print("0. Exit")
        
        choice = input("\nSelect option (0-2): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            run_attack()
        elif choice == "2":
            run_measurements()
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main_menu()
