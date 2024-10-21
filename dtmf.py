import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit as st

# Constants
DTMF_FREQS = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
    '0': (941, 1336), '*': (941, 1209), '#': (941, 1477)
}

# Utility Functions
def generate_tone(frequency, t):
    """Generate a sine wave for a given frequency and time array."""
    return np.sin(2 * np.pi * frequency * t)

def normalize_audio(audio):
    """Normalize audio to 16-bit PCM range."""
    return np.int16(audio / np.max(np.abs(audio)) * 32767)

def sanitize_filename(key):
    """Generate sanitized filenames for DTMF tones."""
    return f"dtmf_tone_{key.replace('*', 'star').replace('#', 'hash')}.wav"

def read_audio_file(uploaded_file):
    """Read audio data from a .wav file."""
    return sf.read(uploaded_file)

def save_audio_file(audio_data, sample_rate, filename):
    """Save audio data to a .wav file."""
    sf.write(filename, audio_data, sample_rate)

# DTMF Tone Generation
def generate_dtmf_tone(key, sample_rate=8000, duration=0.5):
    """Generate a DTMF tone for a given key."""
    if key not in DTMF_FREQS:
        raise ValueError(f"Invalid key: {key}. Choose from 0-9, *, #.")
    
    low_freq, high_freq = DTMF_FREQS[key]
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    dtmf_tone = generate_tone(low_freq, t) + generate_tone(high_freq, t)
    return normalize_audio(dtmf_tone), sample_rate

def generate_and_save_all_tones():
    """Generate and save DTMF tones for all keys."""
    for key in DTMF_FREQS:
        audio_data, sample_rate = generate_dtmf_tone(key)
        save_audio_file(audio_data, sample_rate, sanitize_filename(key))
        print(f"Saved: {sanitize_filename(key)}")

# Plotting Functions
def plot_signal(data, x, title, xlabel, ylabel):
    """Helper function to plot signals."""
    plt.figure(figsize=(10, 4))
    plt.plot(x, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    st.pyplot(plt)

def plot_time_domain(audio_data, sample_rate):
    """Plot the time-domain signal with labels for each sample."""
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    samples_to_plot = audio_data[:100]
    times_to_plot = time[:100]

    plt.figure(figsize=(12, 6))
    plt.plot(times_to_plot, samples_to_plot, marker='o')
    for i, (x, y) in enumerate(zip(times_to_plot, samples_to_plot)):
        plt.text(x, y, f"{y:.2f}", fontsize=8, ha='center', va='bottom')

    plt.title("Time Domain Signal (First 100 Samples)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    st.pyplot(plt)

def plot_frequency_domain(audio_data, sample_rate):
    """Plot the frequency-domain (FFT) spectrum with labels for dominant frequencies."""
    fft_result = np.fft.fft(audio_data)
    freq = np.fft.fftfreq(len(audio_data), 1 / sample_rate)[:len(fft_result) // 2]
    fft_magnitude = np.abs(fft_result[:len(fft_result) // 2])

    dominant_indices = np.argsort(fft_magnitude)[-2:]
    dominant_freqs = freq[dominant_indices]
    plt.figure(figsize=(10, 4))
    plt.plot(freq, fft_magnitude, label='FFT Magnitude')
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    for i in range(2):
        plt.text(dominant_freqs[i], fft_magnitude[dominant_indices[i]], 
                 f"{dominant_freqs[i]:.1f} Hz", 
                 fontsize=10, ha='center', va='bottom', color='red')

    st.pyplot(plt)

# DTMF Identification
def identify_dtmf_key(audio_data, sample_rate, tolerance=20):
    """Identify the DTMF key from the audio data."""
    fft_result = np.fft.fft(audio_data)
    freq = np.fft.fftfreq(len(audio_data), 1 / sample_rate)[:len(fft_result) // 2]
    fft_magnitude = np.abs(fft_result[:len(fft_result) // 2])
    dominant_freqs = freq[np.argsort(fft_magnitude)[-2:]]

    for key, (low_freq, high_freq) in DTMF_FREQS.items():
        if (np.isclose(dominant_freqs, [low_freq, high_freq], atol=tolerance).all() or
            np.isclose(dominant_freqs, [high_freq, low_freq], atol=tolerance).all()):
            return key
    return None

# Streamlit Application
def dtmf_app():
    """Streamlit GUI application for DTMF decoding."""
    st.title('DTMF Decoder Application')
    uploaded_file = st.file_uploader("Upload a DTMF tone (.wav file)", type="wav")

    if uploaded_file:
        audio_data, sample_rate = read_audio_file(uploaded_file)
        st.write(f"Sampling Rate: {sample_rate} Hz")
        st.audio(uploaded_file, format='audio/wav')

        st.subheader("Time Domain Signal")
        plot_time_domain(audio_data, sample_rate)

        st.subheader("Frequency Domain Analysis")
        plot_frequency_domain(audio_data, sample_rate)

        identified_key = identify_dtmf_key(audio_data, sample_rate)
        st.subheader(f"Identified DTMF Key: {identified_key}" if identified_key else 
                     "Unable to identify the DTMF key.")
    else:
        st.warning("Please upload a DTMF tone (.wav file) to analyze.")

# Main Entry Point
if __name__ == "__main__":
    generate_and_save_all_tones()
    dtmf_app()
