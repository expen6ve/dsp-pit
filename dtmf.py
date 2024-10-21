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
SAMPLE_RATE = 8000

# Helper Functions
def generate_tone(key, duration=0.5):
    """Generate a DTMF tone for a given key."""
    if key not in DTMF_FREQS:
        raise ValueError(f"Invalid key: {key}. Choose from 0-9, *, #.")
    
    low_freq, high_freq = DTMF_FREQS[key]
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    tone = np.sin(2 * np.pi * low_freq * t) + np.sin(2 * np.pi * high_freq * t)
    return np.int16(tone / np.max(np.abs(tone)) * 32767)

def sanitize_filename(key):
    """Sanitize filenames by replacing * and #."""
    return f"dtmf_tone_{'star' if key == '*' else 'hash' if key == '#' else key}.wav"

def save_tone(key):
    """Generate and save a DTMF tone to a .wav file."""
    filename = sanitize_filename(key)
    sf.write(filename, generate_tone(key), SAMPLE_RATE)
    return filename

def plot_signal(audio_data, sample_rate, title, xlabel, ylabel, x_data=None):
    """Reusable function to plot signals."""
    plt.figure(figsize=(10, 4))
    plt.plot(x_data or np.linspace(0, len(audio_data) / sample_rate, len(audio_data)), audio_data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    st.pyplot(plt)

def identify_key(audio_data, tolerance=20):
    """Identify DTMF key from the audio data using frequency analysis."""
    fft_result = np.abs(np.fft.fft(audio_data)[:len(audio_data) // 2])
    freqs = np.fft.fftfreq(len(audio_data), 1 / SAMPLE_RATE)[:len(audio_data) // 2]
    dominant_freqs = freqs[np.argsort(fft_result)[-2:]]
    
    for key, (low_freq, high_freq) in DTMF_FREQS.items():
        if all(np.isclose(dominant_freqs, [low_freq, high_freq], atol=tolerance)):
            return key
    return None

# Streamlit App
def dtmf_app():
    st.title('DTMF Decoder Application')

    uploaded_file = st.file_uploader("Upload a DTMF tone (.wav file)", type="wav")
    if not uploaded_file:
        st.warning("Please upload a DTMF tone (.wav file) to analyze.")
        return

    audio_data, sample_rate = sf.read(uploaded_file)
    st.write(f"Sample Rate: {sample_rate} Hz")

    st.subheader("Time Domain Signal (First 100 Samples)")
    plot_signal(audio_data[:100], sample_rate, "Time Domain Signal", "Time (s)", "Amplitude")

    st.subheader("Frequency Spectrum")
    fft_result = np.abs(np.fft.fft(audio_data)[:len(audio_data) // 2])
    freqs = np.fft.fftfreq(len(audio_data), 1 / sample_rate)[:len(audio_data) // 2]
    plot_signal(fft_result, sample_rate, "Frequency Spectrum", "Frequency (Hz)", "Magnitude", freqs)

    key = identify_key(audio_data)
    st.subheader(f"Identified DTMF Key: {key or 'Not Identified'}")

if __name__ == "__main__":
    for key in DTMF_FREQS:
        save_tone(key)  # Generate and save tones only once
    dtmf_app()
