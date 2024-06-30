import numpy as np
import matplotlib.pyplot as plt

# Oscillator Function
def generate_wave(wave_type, t, freq=440, phase=0, volume=1.0, detune_factor=0, num_detune=5):
    if wave_type == 'sine':
        return volume * np.sin(2 * np.pi * freq * t + phase)
    elif wave_type == 'sawtooth':
        return volume * (2 * ((freq * t + phase) % 1) - 1)
    elif wave_type == 'triangular':
        return volume * (2 * abs(2 * ((freq * t + phase) % 1) - 1) - 1)
    elif wave_type == 'square':
        return volume * np.sign(np.sin(2 * np.pi * freq * t + phase))
    elif wave_type == 'noise':
        return volume * np.random.normal(0, 1, len(t))
    elif wave_type == 'detuned':
        result = np.zeros_like(t)
        for i in range(num_detune):
            detuned_freq = freq * (1 + (i - num_detune // 2) * detune_factor / 100)
            result += volume * np.sin(2 * np.pi * detuned_freq * t + phase) / num_detune
        return result
    else:
        raise ValueError("Invalid wave type selected.")

# Envelope Function
def enveloped_audio(input_signal, A, H, D, S, duration, shift, velocity=1.0, sample_rate=44100):
    R = duration -( A + H + D + S )  # Total duration of the envelope
    t = np.linspace(0, duration, num=int(duration * sample_rate))  # Time vector
    shifted_t = t - shift
    shifted_t[shifted_t < 0] = 0
    envelope = np.zeros_like(t)
    amplitude = (max(input_signal) - min(input_signal)) / 2
    # Control the speed
    a = np.log(amplitude + 1) * velocity / A  
    b = -1 / D
    j = -(np.exp(b * (A + H + D)) + np.exp(a * A) - np.exp(b * (A + H)) - 1) / R
    i = -j * (A + H + D + S + R)

    # Attack phase
    envelope[shifted_t <= A] = np.exp(a * shifted_t[shifted_t <= A]) - 1

    # Hold phase
    envelope[(shifted_t > A) & (shifted_t < A + H)] = np.exp(a * A) - 1

    # Decay phase
    envelope[(shifted_t > A + H) & (shifted_t <= A + H + D)] = (np.exp(b * (shifted_t[(shifted_t > A + H) & (shifted_t <= A + H + D)])) +
                                                np.exp(a * A) - np.exp(b * (A + H)) - 1)

    # Sustain phase
    envelope[(shifted_t > A + H + D) & (shifted_t <= A + H + D + S)] = (np.exp(b * (A + H + D)) +
                                                        np.exp(a * A) - np.exp(b * (A + H)) - 1)

    # Release phase
    envelope[shifted_t > A + H + D + S] = j * shifted_t[shifted_t > A + H + D + S] + i

    return envelope[:len(input_signal)] * input_signal[:len(envelope)]

# Effect Classes
class Saturator:
    def __init__(self, threshold, bias=0.0, drive=1.0, a=1.0, wet_dry=1.0):
        self.threshold = threshold
        self.bias = bias
        self.drive = drive
        self.a = a
        self.wet_dry = wet_dry

    def hard_clipping(self, x):
        T = self.threshold
        return np.clip(x, -T, T)

    def soft_clipping(self, x):
        T = self.threshold
        return T * np.tanh(x / T)

    def variable_saturation(self, x):
        T = self.threshold
        a = self.a
        return T * (np.abs(x / T) ** a) * np.sign(x)

    def process(self, x, mode='hard'):
        x_processed = self.drive * x + self.bias
        if mode == 'hard':
            wet_signal = self.hard_clipping(x_processed)
        elif mode == 'soft':
            wet_signal = self.soft_clipping(x_processed)
        elif mode == 'variable':
            wet_signal = self.variable_saturation(x_processed)
        else:
            raise ValueError("Invalid mode selected. Choose 'hard', 'soft', or 'variable'.")
        return self.wet_dry * wet_signal + (1 - self.wet_dry) * x

class Compressor:
    def __init__(self, threshold, ratio, attack, release, wet_dry=1.0, sample_rate=44100):
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release
        self.wet_dry = wet_dry
        self.sample_rate = sample_rate
        self.gain_reduction_db = 0.0

    def compute_gain_reduction(self, x):
        T = self.threshold
        R = self.ratio
        A = self.attack
        Rl = self.release

        attack_samples = self.sample_rate * (A / 1000)
        release_samples = self.sample_rate * (Rl / 1000)

        x_db = 20 * np.log10(np.abs(x) + 1e-6)
        gain_reduction = np.zeros_like(x_db)

        for i in range(1, len(x_db)):
            if x_db[i] > T:
                desired_gain_reduction = T + (x_db[i] - T) / R
            else:
                desired_gain_reduction = 0

            if x_db[i] > T:
                gain_reduction[i] = gain_reduction[i - 1] - (1 / attack_samples)
            else:
                gain_reduction[i] = gain_reduction[i - 1] + (1 / release_samples)

            gain_reduction[i] = max(gain_reduction[i], desired_gain_reduction)

        self.gain_reduction_db = gain_reduction

    def apply_gain_reduction(self, x):
        linear_gain = 10 ** (self.gain_reduction_db / 20)
        return x * linear_gain

    def process(self, x):
        self.compute_gain_reduction(x)
        wet_signal = self.apply_gain_reduction(x)
        return self.wet_dry * wet_signal + (1 - self.wet_dry) * x

class Delay:
    def __init__(self, delay_time, wet_dry=1.0, sample_rate=44100):
        self.delay_time = delay_time
        self.wet_dry = wet_dry
        self.sample_rate = sample_rate
        self.delay_samples = int(self.delay_time * self.sample_rate)
        self.buffer = np.zeros(self.delay_samples)
        self.buffer_index = 0

    def process(self, x):
        output = np.zeros_like(x)
        buffer_length = len(self.buffer)

        for i in range(len(x)):
            
            delayed_sample = self.buffer[self.buffer_index]
            output[i] = self.wet_dry * delayed_sample + (1 - self.wet_dry) * x[i]
            self.buffer[self.buffer_index] = x[i]
            self.buffer_index = (self.buffer_index + 1) % buffer_length

        return output

class ReverbEffect:
    def __init__(self, wet_dry=0.5):
        self.wet_dry = wet_dry

    def add_reverb(self, input_signal):
        impulse_response = np.random.normal(0, 0.5, 1000)                        # Example impulse response
        output_signal = np.convolve(input_signal, impulse_response, mode='full')
        output_signal[:len(input_signal)] = (1 - self.wet_dry) * input_signal + self.wet_dry * output_signal[:len(input_signal)]
        return output_signal[:len(input_signal)]

# Example usage
sample_rate = 44100  # Sample rate in Hz
duration = 10.0  # Duration in seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate a test signal: a sawtooth wave with varying amplitude
wave_type = 'sine'
frequency = 25.0  # Frequency of the sawtooth wave in Hz
input_signal = generate_wave(wave_type, t, freq=frequency)

# Apply the envelope to the input signal
A, H, D, S = 0.5, 0.5, 1, 1.5
enveloped_signal = enveloped_audio(input_signal, A, H, D, S, duration, shift=0, sample_rate=sample_rate)

# Instantiate the effects
saturator = Saturator(threshold=0.5, bias=0.0, drive=1.0, a=2.0, wet_dry=1)
compressor = Compressor(threshold=-3, ratio=4, attack=20, release=100, wet_dry=1, sample_rate=sample_rate)
delay_effect = Delay(delay_time=0.5, wet_dry=1, sample_rate=sample_rate)
reverb_effect = ReverbEffect(wet_dry=0.99)

# Process the enveloped signal through each effect
saturated_signal = saturator.process(enveloped_signal, mode='hard')
compressed_signal = compressor.process(saturated_signal)
delayed_signal = delay_effect.process(compressed_signal)
reverberated_signal = reverb_effect.add_reverb(delayed_signal)

# Plot the original and processed signals for comparison
plt.figure(figsize=(15, 10))

plt.subplot(5, 1, 1)
plt.plot(t, enveloped_signal, label="Enveloped Signal")
plt.title("Enveloped Signal")
plt.ylabel("Amplitude")

plt.subplot(5, 1, 2)
plt.plot(t, saturated_signal, label="Saturated Signal (Hard Clipping)", color='orange')
plt.title("Saturated Signal (Hard Clipping)")
plt.ylabel("Amplitude")

plt.subplot(5, 1, 3)
plt.plot(t, compressed_signal, label="Compressed Signal", color='green')
plt.title("Compressed Signal")
plt.ylabel("Amplitude")

plt.subplot(5, 1, 4)
plt.plot(t, delayed_signal, label="Delayed Signal", color='blue')
plt.title("Delayed Signal")
plt.ylabel("Amplitude")

plt.subplot(5, 1, 5)
plt.plot(t, reverberated_signal , label="Reverberated Signal", color='red')
plt.title("Reverberated Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()