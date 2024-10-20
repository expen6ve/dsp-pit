import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit as st

# Define the frequencies for the DTMF tones (low and high frequencies for each key)
DTMF_FREQS = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
    '0': (941, 1336), '*': (941, 1209), '#': (941, 1477)
}

# Generate a DTMF tone for a specific key
def generate_dtmf_tone(key, sample_rate=8000, duration=0.5):
    if key not in DTMF_FREQS:
        raise ValueError(f"Invalid key: {key}. Choose from 0-9, *, #.")
    
    low_freq, high_freq = DTMF_FREQS[key]
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate sine waves for the low and high frequencies
    low_tone = np.sin(2 * np.pi * low_freq * t)
    high_tone = np.sin(2 * np.pi * high_freq * t)
    
    # Combine both tones (sum of sine waves)
    dtmf_tone = low_tone + high_tone
    # Normalize the tone to prevent clipping
    return np.int16(dtmf_tone / np.max(np.abs(dtmf_tone)) * 32767), sample_rate

# Function to read audio data from a .wav file
def read_audio_file(uploaded_file):
    audio_data, sample_rate = sf.read(uploaded_file)
    return audio_data, sample_rate

# Function to save audio data to a .wav file
def save_audio_file(audio_data, sample_rate, filename):
    sf.write(filename, audio_data, sample_rate)

# Function to plot the time-domain signal (First 100 samples)
def plot_time_domain_signal(audio_data, sample_rate):
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    plt.figure(figsize=(10, 4))
    plt.plot(time[:100], audio_data[:100])  # Plot only the first 100 samples
    plt.title("Time Domain Signal (First 100 Samples)")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    st.pyplot(plt)

# Function to plot the frequency-domain signal (FFT magnitude spectrum)
def plot_frequency_domain_signal(audio_data, sample_rate):
    fft_result = np.fft.fft(audio_data)
    freq = np.fft.fftfreq(len(audio_data), d=1/sample_rate)
    fft_magnitude = np.abs(fft_result[:len(fft_result) // 2])
    freq = freq[:len(freq) // 2]

    plt.figure(figsize=(10, 4))
    plt.plot(freq, fft_magnitude)
    plt.title("Frequency Spectrum")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    st.pyplot(plt)

# Function to identify the DTMF key based on the dominant frequencies in the tone
def identify_dtmf_key(audio_data, sample_rate=8000, tolerance=20):
    fft_result = np.fft.fft(audio_data)
    freq = np.fft.fftfreq(len(audio_data), d=1/sample_rate)
    fft_magnitude = np.abs(fft_result[:len(fft_result) // 2])
    freq = freq[:len(freq) // 2]

    # Identify the two dominant frequencies
    dominant_freqs_indices = np.argsort(fft_magnitude)[-2:]
    dominant_freqs = freq[dominant_freqs_indices]

    # Try to match the frequencies within the tolerance range
    for key, (low_freq, high_freq) in DTMF_FREQS.items():
        if (np.isclose(dominant_freqs[0], low_freq, atol=tolerance) and np.isclose(dominant_freqs[1], high_freq, atol=tolerance)) or \
           (np.isclose(dominant_freqs[0], high_freq, atol=tolerance) and np.isclose(dominant_freqs[1], low_freq, atol=tolerance)):
            return key
    
    return None

# Function to sanitize filenames (replace * and #)
def sanitize_filename(key):
    if key == '*':
        return 'dtmf_tone_star.wav'
    elif key == '#':
        return 'dtmf_tone_hash.wav'
    else:
        return f'dtmf_tone_{key}.wav'

# Generate DTMF tones for all keys and save them as .wav files
def generate_and_save_all_dtms():
    for key in DTMF_FREQS:
        dtmf_tone, _ = generate_dtmf_tone(key, duration=0.5)
        filename = sanitize_filename(key)
        sf.write(filename, dtmf_tone, 8000)
        print(f"DTMF tone for key '{key}' generated and saved as '{filename}'.")

# Streamlit GUI application
def dtmf_app():
    st.title('DTMF Decoder Application')

    # File uploader widget to accept the user's .wav file
    uploaded_file = st.file_uploader("Upload a DTMF tone (.wav file)", type="wav")

    # Initialize audio_data and sample_rate
    audio_data = None
    sample_rate = None
    
    if uploaded_file is not None:
        # Read the audio data from the file
        audio_data, sample_rate = read_audio_file(uploaded_file)
    else:
        st.warning("Please upload a DTMF tone (.wav file) to analyze.")

    if audio_data is not None:
        st.write(f"Sampling Rate: {sample_rate} Hz")
        
        # Time-domain visualization
        st.subheader("Time Domain Signal")
        plot_time_domain_signal(audio_data, sample_rate)
        
        # Frequency-domain visualization
        st.subheader("Frequency-Domain Analysis")
        plot_frequency_domain_signal(audio_data, sample_rate)
        
        # Identify DTMF key based on frequency analysis
        identified_key = identify_dtmf_key(audio_data, sample_rate, tolerance=20)
        
        if identified_key:
            st.subheader(f"Identified DTMF Key: {identified_key}")
        else:
            st.subheader("Unable to identify the DTMF key.")

# Main entry point for the Streamlit app
if __name__ == "__main__":
    generate_and_save_all_dtms()  # Generate and save DTMF tones
    dtmf_app()  # Start the Streamlit app
