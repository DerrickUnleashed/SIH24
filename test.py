# Import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

freq = 44100

duration = 10

print("Recording...")
try:
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1, dtype='int16')

    sd.wait()
    print("Recording complete.")

    write("/Users/derricksamuel/Desktop/BreakingBonds/temp.wav", freq, recording)

    print("Audio files saved.")
except Exception as e:
    print(f"An error occurred: {e}")
