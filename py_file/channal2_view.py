# -*- coding: utf-8 -*-
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

# ================================
# Modify here: your .mat file path
# ================================
mat_path = r"D:\Variable_speed\University_of_Ottawa\Original_data\UO_Bearing\1 Data collected from a healthy bearing\H-A-1.mat"

# Encoder parameters - University of Ottawa dataset
ENCODER_CPR = 1024  # Counts Per Revolution (confirmed for Ottawa dataset)

# Sampling frequency (Ottawa dataset is 20kHz)
FS = 20000


# ================================
# 1. Read .mat file
# ================================
data = scio.loadmat(mat_path)

# In Ottawa dataset, channel2 name might be "ch2" or "Channel_2"
possible_keys = ['channel2', 'Channel2', 'ch2', 'Ch2',
                 'channel_2', 'Channel_2']

for key in possible_keys:
    if key in data:
        ch2 = data[key].squeeze()
        break
else:
    raise KeyError("Channel 2 not found, please check variable names in .mat file")

# Create time array
t = np.arange(len(ch2)) / FS

print(f"Signal length: {len(ch2)} samples ({len(ch2)/FS:.2f} seconds)")
print(f"Sampling frequency: {FS} Hz")
print(f"Encoder CPR: {ENCODER_CPR} counts per revolution")


# ================================
# 2. Detect encoder pulse edges and calculate instantaneous RPM
# ================================
def detect_encoder_edges_robust(signal, method='median_threshold'):
    # """
    # Robust edge detection for encoder pulse signal
    
    # Parameters:
    #     signal: Encoder signal array
    #     method: 'median_threshold' or 'zero_crossing'
    
    # Returns:
    #     edge_indices: Array of rising edge positions
    # """
    if method == 'median_threshold':
        # Method 1: Use median as threshold (more robust to outliers)
        threshold = np.median(signal)
        
        # Detect rising edges: signal crosses threshold from below to above
        rising_edges = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0]
        
        # Detect falling edges: signal crosses threshold from above to below
        falling_edges = np.where((signal[:-1] >= threshold) & (signal[1:] < threshold))[0]
        
        # Use the edge type with more detections
        if len(rising_edges) >= len(falling_edges):
            return rising_edges, 'rising'
        else:
            return falling_edges, 'falling'
    
    else:
        # Method 2: Zero crossing (for AC-coupled signals)
        # Remove DC offset
        signal_centered = signal - np.mean(signal)
        
        # Detect rising zero crossings
        rising_edges = np.where((signal_centered[:-1] < 0) & (signal_centered[1:] >= 0))[0]
        
        # Detect falling zero crossings
        falling_edges = np.where((signal_centered[:-1] >= 0) & (signal_centered[1:] < 0))[0]
        
        if len(rising_edges) >= len(falling_edges):
            return rising_edges, 'rising'
        else:
            return falling_edges, 'falling'


def calculate_instantaneous_rpm(signal, fs, encoder_cpr):
    # """
    # Calculate instantaneous RPM from encoder pulse signal
    
    # Steps:
    # 1. Detect pulse edges (rising or falling)
    # 2. Calculate pulse intervals (periods)
    # 3. Convert periods to instantaneous RPM
    # 4. Align time coordinates
    
    # Parameters:
    #     signal: Encoder signal array
    #     fs: Sampling frequency (Hz)
    #     encoder_cpr: Counts per revolution
    
    # Returns:
    #     rpm_time: Time array aligned with RPM values (seconds)
    #     rpm: Instantaneous RPM array
    #     edge_type: Type of edges used ('rising' or 'falling')
    # """
    # Step 1: Detect encoder pulse edges
    edge_indices, edge_type = detect_encoder_edges_robust(signal, method='median_threshold')
    
    if len(edge_indices) < 2:
        print(f"Warning: Not enough edges detected ({len(edge_indices)} edges)")
        print("Trying zero-crossing method...")
        edge_indices, edge_type = detect_encoder_edges_robust(signal, method='zero_crossing')
        
        if len(edge_indices) < 2:
            print(f"Warning: Still not enough edges ({len(edge_indices)} edges)")
            return np.array([]), np.array([]), edge_type
    
    print(f"Detected {len(edge_indices)} {edge_type} edges")
    
    # Step 2: Calculate pulse intervals (periods between consecutive edges)
    # Each edge represents one pulse/count
    pulse_intervals = np.diff(edge_indices) / fs  # Period in seconds
    
    # Filter out unrealistic periods
    # For variable speed bearing: typical RPM range might be 100-3000 RPM
    # At 100 RPM: period per pulse = 60/(100*CPR) = 0.6/CPR �� 0.0006 seconds
    # At 3000 RPM: period per pulse = 60/(3000*CPR) = 0.02/CPR �� 0.00002 seconds
    # But we should be more lenient for variable speed
    min_period = 0.00001  # Very short period (very high RPM)
    max_period = 0.1      # Longer period (very low RPM or missing pulses)
    
    valid_mask = (pulse_intervals > min_period) & (pulse_intervals < max_period)
    
    if np.sum(valid_mask) == 0:
        print("Warning: No valid pulse intervals found after filtering")
        print(f"  Pulse interval range: {np.min(pulse_intervals):.6f} - {np.max(pulse_intervals):.6f} seconds")
        # Try with relaxed constraints
        valid_mask = pulse_intervals > 0
    
    pulse_intervals = pulse_intervals[valid_mask]
    edge_indices = edge_indices[1:][valid_mask]  # Align with intervals
    
    if len(pulse_intervals) == 0:
        print("Error: No valid pulse intervals found")
        return np.array([]), np.array([]), edge_type
    
    # Step 3: Convert pulse period to instantaneous RPM
    # Each pulse represents 1/CPR revolution
    # Period for one revolution = pulse_period * CPR
    # Frequency (revolutions per second) = 1 / (pulse_period * CPR)
    # RPM = frequency * 60
    revolution_periods = pulse_intervals * encoder_cpr  # Time for one full revolution
    rpm = 60.0 / revolution_periods  # Convert to RPM
    
    # Step 4: Align time coordinates
    # Time points at each edge (representing the RPM at that moment)
    rpm_time = t[edge_indices]
    
    return rpm_time, rpm, edge_type


# Calculate instantaneous RPM
rpm_time, rpm, edge_type_used = calculate_instantaneous_rpm(ch2, FS, ENCODER_CPR)

if len(rpm) > 0:
    print(f"\nRPM Calculation Results:")
    print(f"  Edge type used: {edge_type_used}")
    print(f"  Number of RPM data points: {len(rpm)}")
    print(f"  RPM range: {np.min(rpm):.2f} - {np.max(rpm):.2f} RPM")
    print(f"  Average RPM: {np.mean(rpm):.2f} RPM")
    print(f"  Std RPM: {np.std(rpm):.2f} RPM")
    print(f"  Median RPM: {np.median(rpm):.2f} RPM")
    
    # ================================
    # 3. Plot results
    # ================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Original encoder pulse signal with detected edges
    display_samples = min(2 * FS, len(ch2))
    axes[0].plot(t[:display_samples], ch2[:display_samples], 
                 linewidth=0.5, alpha=0.7, color='blue', label='Encoder Signal')
    
    # Mark detected edges in the displayed range
    edge_indices_display, _ = detect_encoder_edges_robust(ch2[:display_samples], method='median_threshold')
    if len(edge_indices_display) > 0:
        axes[0].plot(t[edge_indices_display], ch2[edge_indices_display], 
                    'ro', markersize=3, label=f'{edge_type_used.capitalize()} edges', alpha=0.6)
    
    # Draw threshold line
    threshold = np.median(ch2)
    axes[0].axhline(y=threshold, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (Median)')
    
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Signal Amplitude')
    axes[0].set_title('Channel 2 Encoder Pulse Signal with Detected Edges (First 2 seconds)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Instantaneous RPM (raw, no smoothing to show true instantaneous speed)
    axes[1].plot(rpm_time, rpm, 'b-', linewidth=1.0, alpha=0.8, label='Instantaneous RPM')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Speed (RPM)')
    axes[1].set_title('Instantaneous Speed (RPM) vs Time - Calculated from Pulse Intervals')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Add statistics text
    stats_text = (f'Mean: {np.mean(rpm):.2f} RPM, Std: {np.std(rpm):.2f} RPM, '
                  f'Range: [{np.min(rpm):.2f}, {np.max(rpm):.2f}] RPM')
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=9)
    
    # Plot 3: Instantaneous RPM - full view
    axes[2].plot(rpm_time, rpm, 'b-', linewidth=1.0, label='Instantaneous RPM', alpha=0.7)
    
    # Add trend line (moving average for visualization)
    if len(rpm) > 20:
        window = min(50, len(rpm) // 20)
        if window > 1:
            kernel = np.ones(window) / window
            rpm_trend = np.convolve(rpm, kernel, mode='same')
            axes[2].plot(rpm_time, rpm_trend, 'r-', linewidth=2, 
                        label='Trend (Moving Average)', alpha=0.8)
    
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Speed (RPM)')
    axes[2].set_title('Instantaneous Speed (RPM) - Full Time Range')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print interpretation
    print("\n" + "="*60)
    print("Interpretation:")
    print("="*60)
    print("- Instantaneous RPM is calculated from pulse intervals")
    print("- Each pulse represents 1/{} revolution".format(ENCODER_CPR))
    print("- RPM = 60 / (pulse_period * {})".format(ENCODER_CPR))
    print("- The plot shows:")
    print("  * Acceleration: RPM gradually increases")
    print("  * Constant speed: RPM remains stable")
    print("  * Variable speed: RPM fluctuates")
    print("  * Vibration noise: Eliminated (calculated from pulses)")
    print("="*60)
    
else:
    print("\nError: Could not calculate RPM from encoder signal")
    print("Diagnostic information:")
    print(f"  Signal range: [{np.min(ch2):.2f}, {np.max(ch2):.2f}]")
    print(f"  Signal mean: {np.mean(ch2):.2f}")
    print(f"  Signal std: {np.std(ch2):.2f}")
    print("\nTrying to plot raw signal for inspection...")
    
    # Just plot the raw signal
    display_samples = min(5 * FS, len(ch2))
    plt.figure(figsize=(12, 4))
    plt.plot(t[:display_samples], ch2[:display_samples], linewidth=0.5, color='blue')
    plt.title("Channel 2 Encoder Signal (First 5 seconds)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Signal Amplitude")
    plt.grid(True)
    
    # Draw threshold
    threshold = np.median(ch2)
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (Median)')
    plt.legend()
    plt.tight_layout()
    plt.show()
