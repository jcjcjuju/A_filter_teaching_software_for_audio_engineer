import sys
import numpy as np
import sounddevice as sd
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import lfilter, filtfilt, freqz, firwin2, minimum_phase
from scipy.signal import remez
import soundfile as sf
import time
import math
 
# ========= Constants =========
# Canvas size constants
PZ_CANVAS_WIDTH, PZ_CANVAS_HEIGHT = 300, 300
TF_CANVAS_HEIGHT = 40
BODE_CANVAS_WIDTH, BODE_CANVAS_HEIGHT = 600, 400

# ========= DSP Modules =========

class AudioEffect:
    """Base class for all effects."""
    def process(self, audio_buffer: np.ndarray) -> np.ndarray:
        # Default: pass through the audio buffer.
        return audio_buffer

class Equalizer(AudioEffect):
    # ------------------------------------------------------------------
    def _ensure_linear_phase_taps(self):
        """
        Compute or reuse linear/max‑phase FIR taps.  
        Only redesigns when any of the key EQ parameters have changed.
        """
        key = (self.eq_type,
               self.center_freq,
               self.Q,
               self.gain,
               self.samplerate)
        if getattr(self, "_fir_cache_key", None) != key or \
           not hasattr(self, "linear_taps") or not hasattr(self, "max_phase_taps"):
            self._design_linear_phase_filter()
            self._fir_cache_key = key
    # ------------------------------------------------------------------
    def _center_taps(self, taps: np.ndarray, length: int) -> np.ndarray:
        """
        Return `taps` zero‑padded to `length` with its peak centred.
        Caches the result so repeated calls with the same (id(taps), length)
        reuse the array instead of re‑slicing and reallocating.
        """
        cache_key = (id(taps), length)
        # Return a cached centered IR if available
        if cache_key in self._ir_cache:
            return self._ir_cache[cache_key].copy()

        taps_len = len(taps)
        ir = np.zeros(length, dtype=float)
        # Compute start index so the tap peak is centered
        start = length // 2 - taps_len // 2
        end   = start + taps_len
        if start < 0:
            taps = taps[-start:]
            start = 0
        if end > length:
            taps = taps[:length - start]
            end = length
        ir[start:end] = taps

        # Cache for future reuse
        self._ir_cache[cache_key] = ir
        return ir.copy()
    # ------------------------------------------------------------------
    def _mixed_phase_scale(self, ratio: float):
        # Mixed-phase scaling: cache anchor frequency gains to avoid repeated freqz calls
        cache_key = (self.eq_type, self.center_freq, self.Q, self.gain, self.samplerate)
        if cache_key in self._anchor_gain_cache:
            H_iir_anchor, H_lin_anchor = self._anchor_gain_cache[cache_key]
        else:
            # Determine anchor frequency
            if self.eq_type in ("Low-Pass", "Low-Shelf"):
                f_anchor = 0.0
            elif self.eq_type in ("High-Pass", "High-Shelf"):
                f_anchor = self.samplerate / 2.0
            else:
                f_anchor = self.center_freq
            # IIR anchor response
            b, a = self.coeffs["b"], self.coeffs["a"]
            _, H_iir = freqz(b, a, worN=[f_anchor], fs=self.samplerate)
            H_iir_anchor = H_iir[0]
            # Ensure linear-phase taps exist
            self._ensure_linear_phase_taps()
            _, H_lin = freqz(self.linear_taps, [1.0], worN=[f_anchor], fs=self.samplerate)
            H_lin_anchor = H_lin[0]
            # Cache the anchor responses
            self._anchor_gain_cache[cache_key] = (H_iir_anchor, H_lin_anchor)

        # Compute mixed-phase anchor gain
        H_mix = ratio * H_iir_anchor + (1.0 - ratio) * H_lin_anchor
        mag_mix   = np.abs(H_mix)
        mag_target = np.abs(H_iir_anchor)
        # Return scaling factor, avoid zero division
        if mag_mix > 1e-6 and mag_target > 1e-6:
            return mag_target / mag_mix
        else:
            return 1.0
    def __init__(self, center_freq=1000, Q=1, gain=0, eq_type="Peaking EQ (Bell)", phase_type="Minimum Phase"):
        self.center_freq = center_freq   # Hz
        self.Q = Q                       # Q-factor (dimensionless)
        self.gain = gain                 # dB
        self.eq_type = eq_type  # "Peaking EQ (Bell)", "Low Shelf", "High Shelf", "Notch"
        self.phase_type = phase_type  # "Minimum Phase", "Zero Phase", or "Linear Phase"
        self.mix_ratio = 0.5  # proportion of IIR in Mixed Phase (0…1)
        self.polarity_invert = False  # global polarity flip
        # Default audio sample rate for coefficient calculations
        self.samplerate = 44100
        # Cache for dense magnitude spectra to speed up FIR design
        self._lpf_magnitude_cache = {}
        # Cache for centered impulse responses
        self._ir_cache = {}
        # Cache for anchor frequency responses for mixed-phase scaling
        self._anchor_gain_cache = {}
        # Recompute coefficients with the chosen type
        self.coeffs = self._design_filter()
        # For linear-phase, precompute FIR taps
        if self.phase_type == "Linear Phase":
            self._design_linear_phase_filter()

    def _design_filter(self):
        # RBJ Audio EQ Cookbook filter coefficient design
        A = 10**(self.gain / 40.0)
        w0 = 2 * np.pi * self.center_freq / self.samplerate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * self.Q)
        
        if self.eq_type in ["Bell", "Peaking EQ (Bell)"]:
        # Peaking EQ (Bell)
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A

        elif self.eq_type in ["Notch", "Notch (band-stop)"]:
            # Notch filter
            b0 = 1
            b1 = -2 * cos_w0
            b2 = 1
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha

        elif self.eq_type == "Low-Shelf":
            # Low Shelf
            beta = np.sqrt(A) / self.Q
            b0 =    A*((A+1) - (A-1)*cos_w0 + beta*sin_w0)
            b1 = 2*A*((A-1) - (A+1)*cos_w0)
            b2 =    A*((A+1) - (A-1)*cos_w0 - beta*sin_w0)
            a0 =        (A+1) + (A-1)*cos_w0 + beta*sin_w0
            a1 =   -2*((A-1) + (A+1)*cos_w0)
            a2 =        (A+1) + (A-1)*cos_w0 - beta*sin_w0

        elif self.eq_type == "High-Shelf":
            # High Shelf
            beta = np.sqrt(A) / self.Q
            b0 =    A*((A+1) + (A-1)*cos_w0 + beta*sin_w0)
            b1 = -2*A*((A-1) + (A+1)*cos_w0)
            b2 =    A*((A+1) + (A-1)*cos_w0 - beta*sin_w0)
            a0 =        (A+1) - (A-1)*cos_w0 + beta*sin_w0
            a1 =    2*((A-1) - (A+1)*cos_w0)
            a2 =        (A+1) - (A-1)*cos_w0 - beta*sin_w0

        elif self.eq_type == "Low-Pass":
            # 2nd-order Low-Pass
            b0 = (1 - cos_w0) / 2
            b1 = 1 - cos_w0
            b2 = (1 - cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha

        elif self.eq_type == "High-Pass":
            # 2nd-order High-Pass
            b0 = (1 + cos_w0) / 2
            b1 = -(1 + cos_w0)
            b2 = (1 + cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha

        elif self.eq_type == "Band-Pass (constant skirt gain)":
            # Band-Pass, constant skirt gain (with user-adjustable gain)
            b0 = alpha * A
            b1 = 0
            b2 = -alpha * A
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha

        elif self.eq_type == "Band-Pass (constant peak gain)":
            # Band-Pass, constant peak gain
            b0 = alpha * A
            b1 = 0
            b2 = -alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A

        elif self.eq_type == "All-Pass":
            # All-Pass filter
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            b0 = a2
            b1 = a1
            b2 = a0

        else:
            # Fallback to Peaking EQ
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
        # Normalize filter coefficients and return
        b = [b0 / a0, b1 / a0, b2 / a0]
        a = [1.0, a1 / a0, a2 / a0]
        return {'b': b, 'a': a}

    def update_parameters(self, center_freq, Q, gain):
        self.center_freq = center_freq
        self.Q = Q
        self.gain = gain
        self.coeffs = self._design_filter()

    def _design_linear_phase_filter(self):
        """
        Design a linear‑phase FIR filter that closely matches the current IIR magnitude
        response.  A higher tap‑count is chosen adaptively so that large boost/cut
        values (or narrow Q) are captured without the “washed‑out” 10 dB cap you
        noticed.

        Strategy
        --------
        * Measure the desired magnitude on a dense 4096‑point grid.
        * Choose the number of taps proportional to bandwidth ­– but never
          less than 255 and always odd so we get a Type‑I linear‑phase FIR.
        * Use `firwin2` with a Kaiser window (beta=7.5 ≈ 60 dB sidelobes),
          or equiripple (remez) for peaking/notch filters.
        """
        # Dense measurement of the target magnitude (cached)
        cache_key = (self.eq_type, self.center_freq, self.Q, self.gain, self.samplerate)
        if cache_key in self._lpf_magnitude_cache:
            freq, gain = self._lpf_magnitude_cache[cache_key]
        else:
            n_fft = 4096
            w, h = freqz(self.coeffs['b'], self.coeffs['a'], worN=n_fft, fs=self.samplerate)
            mag = np.abs(h)
            freq = w[: n_fft // 2 + 1]
            gain = mag[: n_fft // 2 + 1]
            # Cache half-spectrum
            self._lpf_magnitude_cache[cache_key] = (freq, gain)

        # Ensure exact endpoints
        freq[0]           = 0.0
        freq[-1]          = self.samplerate / 2.0
        gain[0]           = gain[1]        # stabilise DC
        gain[-1]          = gain[-2]       # stabilise Nyquist

        # Adaptive tap-count:  fs / transition_bw  (heuristic)
        transition_bw = max(20.0, self.center_freq / max(1, self.Q) / 2)
        # Increase tap count: 8× for Maximum Phase to achieve sharper curves, 4× for others
        if self.phase_type == "Maximum Phase":
            est_taps = int(8 * self.samplerate / transition_bw)
        else:
            est_taps = int(4 * self.samplerate / transition_bw)
        numtaps = max(255, (est_taps // 2) * 2 + 1)   # make it odd

        # Use equiripple FIR (Parks–McClellan) for peaking/notch filters
        if self.eq_type in ("Bell", "Peaking EQ (Bell)", "Notch", "Notch (band-stop)"):
            f0 = self.center_freq
            bw = f0 / max(self.Q, 1e-6)
            f1 = max(0.0, f0 - bw/2)
            f2 = min(self.samplerate/2.0, f0 + bw/2)
            nyq = self.samplerate / 2.0
            bands = [0.0, f1, f1, f2, f2, nyq]
            if self.eq_type in ("Bell", "Peaking EQ (Bell)"):
                # Three bands: passband, peak band, stopback
                desired = [1, 10**(self.gain/20), 1]
                weight = [1, 1, 1]
            else:
                # Three bands: passband, notch band, passband
                desired = [1, 0, 1]
                weight = [1, 10, 1]
            try:
                taps = remez(numtaps, bands, desired, weight=weight, fs=self.samplerate)
            except ValueError:
                # Fall back to windowed FIR on convergence failure
                taps = firwin2(numtaps=numtaps,
                               freq=freq,
                               gain=gain,
                               fs=self.samplerate,
                               window=('kaiser', 7.5))
        else:
            taps = firwin2(numtaps=numtaps,
                           freq=freq,
                           gain=gain,
                           fs=self.samplerate,
                           window=('kaiser', 7.5))
        # ------------------------------------------------------------------
        # Choose an *anchor frequency* where we know the desired gain exactly
        #   • 0 Hz for low‑pass / low‑shelf
        #   • Nyquist for high‑pass / high‑shelf
        #   • the user‑specified centre frequency for everything else
        if self.eq_type in ("Low-Pass", "Low-Shelf"):
            f_anchor = 0.0
        elif self.eq_type in ("High-Pass", "High-Shelf"):
            f_anchor = self.samplerate / 2.0
        else:
            f_anchor = self.center_freq
        # Desired magnitude from the current IIR prototype
        _, h_des = freqz(self.coeffs['b'], self.coeffs['a'],
                         worN=[f_anchor], fs=self.samplerate)
        desired_mag = np.abs(h_des[0])
        # Actual FIR magnitude at the same frequency
        _, h_fir = freqz(taps, [1.0], worN=[f_anchor], fs=self.samplerate)
        fir_mag  = np.abs(h_fir[0])
        # Avoid zeroing out linear taps for notch filters (desired_mag ≈ 0)
        if fir_mag > 0 and desired_mag > 1e-6:
            taps *= desired_mag / fir_mag
        # ------------------------------------------------------------------
        self.linear_taps = taps
        from scipy.signal import minimum_phase  # already imported at top
        # --- derive a *true* maximum-phase companion --------------------------
        min_taps = minimum_phase(taps, method="homomorphic")
        max_taps = min_taps[::-1]
        # Re-normalize max-phase taps to match desired magnitude at anchor frequency
        _, h_max = freqz(max_taps, [1.0], worN=[f_anchor], fs=self.samplerate)
        max_mag = np.abs(h_max[0])
        # Avoid zeroing out max-phase taps for notch filters
        if max_mag > 0 and desired_mag > 1e-6:
            max_taps *= desired_mag / max_mag
        self.max_phase_taps = max_taps

    def get_impulse_response(self, length=512):
        """
        Compute the impulse response of the equalizer based on the current phase type, ensuring the main impulse is centered.
        """
        if self.phase_type in ("Linear Phase", "Zero Phase", "Mixed Phase"):
            self._ensure_linear_phase_taps()
            taps = self.linear_taps
            ir = self._center_taps(taps, length)
            if self.phase_type == "Mixed Phase":
                impulse = np.zeros(length); impulse[length//2] = 1.0
                ir_iir  = lfilter(self.coeffs['b'], self.coeffs['a'], impulse)
                r = self.mix_ratio
                ir = r * ir_iir + (1.0 - r) * ir
            if self.polarity_invert:
                ir = -ir
            return ir
        elif self.phase_type == "Maximum Phase":
            self._ensure_linear_phase_taps()
            taps = self.max_phase_taps
            ir = self._center_taps(taps, length)
            if self.polarity_invert:
                ir = -ir
            return ir
        else:
            # IIR modes (Minimum Phase, etc.)
            impulse = np.zeros(length)
            impulse[0] = 1.0
            b = self.coeffs['b']
            a = self.coeffs['a']
            ir = lfilter(b, a, impulse)
            # Shift the IIR response to center the peak at length // 2
            ir_shifted = np.zeros(length)
            ir_shifted[length // 2:] = ir[:length - (length // 2)]
            if self.polarity_invert:
                ir_shifted = -ir_shifted
            return ir_shifted

    # ------------------------------------------------------------------
    def _zero_phase_scaled_b(self):
        """
        Return a copy of the IIR numerator `b` that is pre‑scaled so that a
        forward‑plus‑reverse (`filtfilt`) pass gives the *same* in‑band gain
        that the user dialled in for a single‑pass filter.

        Strategy
        --------
        Let |H(jω₀)| be the single‑pass magnitude at an anchor frequency ω₀.
        After filtfilt it becomes |H|².  We want |H|²·s² = |H| ⇒ s = 1/√|H|.
        """
        b = np.array(self.coeffs["b"], dtype=float)
        a = np.array(self.coeffs["a"], dtype=float)
        # Choose the same anchor freq logic used elsewhere
        if self.eq_type in ("Low-Pass", "Low-Shelf"):
            f_anchor = 0.0
        elif self.eq_type in ("High-Pass", "High-Shelf"):
            f_anchor = self.samplerate / 2.0
        else:
            f_anchor = self.center_freq
        _, H = freqz(b, a, worN=[f_anchor], fs=self.samplerate)
        mag = np.abs(H[0])
        # Avoid infinite scaling for notch filters (mag ≈ 0)
        if mag > 1e-6:
            scale = 1.0 / np.sqrt(mag)
            b *= scale
        return b

    def process(self, audio_buffer):
        # Apply filtering based on selected phase mode
        mode = self.phase_type
        # Ensure mono input is 2D for uniform handling
        is_mono = (audio_buffer.ndim == 1)
        x = audio_buffer if not is_mono else audio_buffer[:, np.newaxis]

        # Prepare output container
        out = np.empty_like(x)

        # Ensure FIR taps for FIR modes if needed
        if mode in ("Zero Phase", "Linear Phase", "Maximum Phase", "Mixed Phase"):
            self._ensure_linear_phase_taps()

        # Process per channel
        for ch in range(x.shape[1]):
            channel_data = x[:, ch]
            if mode == "Zero Phase":
                # True zero‑phase (non‑causal) filtering: forward + reverse IIR
                b_scaled = self._zero_phase_scaled_b()
                filtered = filtfilt(b_scaled, self.coeffs['a'], channel_data, method='pad')
            elif mode == "Linear Phase":
                filtered = lfilter(self.linear_taps, [1.0], channel_data)
            elif mode == "Maximum Phase":
                filtered = lfilter(self.max_phase_taps, [1.0], channel_data)
            elif mode == "Mixed Phase":
                # Adjustable blend of IIR minimum-phase and FIR linear-phase, with gain correction
                iir_out = lfilter(self.coeffs['b'], self.coeffs['a'], channel_data)
                lin_out = lfilter(self.linear_taps, [1.0], channel_data)
                r = self.mix_ratio
                scale = self._mixed_phase_scale(r)
                filtered = scale * (r * iir_out + (1.0 - r) * lin_out)
            else:
                # Default minimum-phase IIR
                filtered = lfilter(self.coeffs['b'], self.coeffs['a'], channel_data)
            # Apply polarity flip at the very end
            if self.polarity_invert:
                filtered = -filtered
            out[:, ch] = filtered

        # Convert back to 1D for mono
        return out[:, 0] if is_mono else out

class Delay(AudioEffect):
    def __init__(self, delay_ms=500, feedback=0.5, mix=0.5, samplerate=44100):
        self.samplerate = samplerate
        self.delay_ms = delay_ms
        self.feedback = feedback
        self.mix = mix
        self.delay_samples = int(delay_ms * samplerate / 1000)
        # Use shared Delay_Line for buffering
        self.delay_line = Delay_Line(self.delay_samples)
        self.channel_lines = {}
        # Compute filter coefficients for pole-zero plotting
        self.coeffs = self._compute_coeffs()
        # Precompute analytic poles & zeros
        self.zeros, self.poles = self._compute_poles_zeros()

    def update_parameters(self, delay_ms, feedback, mix):
        self.delay_ms = delay_ms
        self.feedback = feedback
        self.mix = mix
        self.delay_samples = int(delay_ms * self.samplerate / 1000)
        # Update shared Delay_Line instances
        self.delay_line = Delay_Line(self.delay_samples)
        self.channel_lines = {}
        # Recompute coefficients for PZ plot
        self.coeffs = self._compute_coeffs()
        # Recompute analytic poles & zeros
        self.zeros, self.poles = self._compute_poles_zeros()

    def _compute_coeffs(self):
        # Approximate comb filter transfer function H(z) ≈ (1-mix) + mix·z^{-D} / (1 - feedback·z^{-D})
        # Numerator: (1-mix) at z^0 and mix at z^{-delay_samples}; Denominator: 1 - feedback·z^{-delay_samples}
        b = [1 - self.mix] + [0] * (self.delay_samples - 1) + [self.mix]
        a = [1] + [0] * (self.delay_samples - 1) + [-self.feedback]
        return {'b': b, 'a': a}

    def _compute_poles_zeros(self):
        # Analytically compute comb-filter poles & zeros to avoid high-order root finding
        D = self.delay_samples
        # Zeros: solve (1-mix) + mix·z^{-D} = 0 ⇒ z^D = -mix/(1-mix)
        if self.mix != 1.0:
            mag_z = (self.mix / (1 - self.mix)) ** (1.0 / D)
            angles_z = (np.pi + 2 * np.pi * np.arange(D)) / D
            zeros = mag_z * np.exp(1j * angles_z)
        else:
            zeros = np.array([], dtype=complex)
        # Poles: solve 1 - feedback·z^{-D} = 0 ⇒ z^D = feedback
        mag_p = self.feedback ** (1.0 / D)
        angles_p = 2 * np.pi * np.arange(D) / D
        poles = mag_p * np.exp(1j * angles_p)
        return zeros, poles
    
    def process(self, audio_buffer):
        """
        Apply a feedback delay with adjustable wet/dry mix.

        Algorithm (per sample):
            d[n]   = current delayed sample from buffer
            buf_in = x[n] + self.feedback * d[n]      # feedback written to buffer
            y[n]   = (1‑mix) * x[n] + mix * d[n]      # output

        Works for both mono (1‑D) and multi‑channel (2‑D) numpy arrays.
        """
        # ---------- MONO ----------
        if audio_buffer.ndim == 1:
            x = audio_buffer
            y = np.empty_like(x)
            for i in range(len(x)):
                delayed = self.delay_line.buffer[self.delay_line.ptr]
                # write input + feedback into buffer
                self.delay_line.buffer[self.delay_line.ptr] = x[i] + self.feedback * delayed
                # advance pointer
                self.delay_line.ptr = (self.delay_line.ptr + 1) % self.delay_line.delay_length
                # wet/dry mix
                y[i] = (1.0 - self.mix) * x[i] + self.mix * delayed
            return y

        # ---------- MULTI‑CHANNEL ----------
        else:
            x = audio_buffer
            y = np.empty_like(x)
            n_channels = x.shape[1]

            for ch in range(n_channels):
                # create per‑channel delay line if absent
                if ch not in self.channel_lines:
                    self.channel_lines[ch] = Delay_Line(self.delay_samples)
                line = self.channel_lines[ch]

                for i in range(x.shape[0]):
                    delayed = line.buffer[line.ptr]
                    line.buffer[line.ptr] = x[i, ch] + self.feedback * delayed
                    line.ptr = (line.ptr + 1) % line.delay_length
                    y[i, ch] = (1.0 - self.mix) * x[i, ch] + self.mix * delayed
            return y

class Allpass_Linear:
    # linearly interpolated allpass
    def __init__(self, max_delay_length, delay_length, g):
        self.max_delay_length = max_delay_length
        self.delay_length = int(delay_length)
        self.delay_buffer = np.zeros(max_delay_length)
        self.write_pointer = 0
        self.read_pointer = max_delay_length - self.delay_length
        self.g = g

    def next(self, in_sample):
        frac, integer = np.modf(self.read_pointer)
        idx = int(integer)
        next_idx = (idx + 1) % self.max_delay_length
        delay_out = frac * self.delay_buffer[next_idx] + (1 - frac) * self.delay_buffer[idx]
        middle = in_sample + self.g * delay_out
        out = -self.g * middle + delay_out
        self.delay_buffer[self.write_pointer] = middle
        self.read_pointer = (self.read_pointer + 1) % self.max_delay_length
        self.write_pointer = (self.write_pointer + 1) % self.max_delay_length
        return out

    def at(self, index):
        i = int(index)
        pos = (self.write_pointer - i) % self.max_delay_length
        return self.delay_buffer[pos]

    def setDelayLength(self, new_length):
        self.delay_length = int(new_length)
        self.read_pointer = (self.write_pointer + (self.max_delay_length - self.delay_length)) % self.max_delay_length

class LPF_Bandwidth:
    def __init__(self, bw):
        self.bw = bw
        self.one_bw = 1 - bw
        self.x_past = 0.0

    def next(self, in_sample):
        filt_in = in_sample * self.bw
        out = filt_in + self.x_past * self.one_bw
        self.x_past = out
        return out

class LPF_Damping:
    def __init__(self, d):
        self.d = d
        self.one_d = 1 - d
        self.x_past = 0.0

    def next(self, in_sample):
        filt_in = in_sample * self.one_d
        out = filt_in + self.x_past * self.d
        self.x_past = out
        return out

class Delay_Line:
    def __init__(self, delay_length):
        self.delay_length = int(delay_length)
        self.buffer = np.zeros(self.delay_length)
        self.ptr = 0

    def next(self, in_sample):
        out = self.buffer[self.ptr]
        self.buffer[self.ptr] = in_sample
        self.ptr = (self.ptr + 1) % self.delay_length
        return out

    def at(self, index):
        i = int(index)
        pos = (self.ptr - i) % self.delay_length
        return self.buffer[pos]

class Allpass:
    def __init__(self, delay_length, g):
        self.delay_length = int(delay_length)
        self.buffer = np.zeros(self.delay_length)
        self.ptr = 0
        self.g = g

    def next(self, in_sample):
        buf_out = self.buffer[self.ptr]
        middle = in_sample + self.g * buf_out
        out = -self.g * middle + buf_out
        self.buffer[self.ptr] = middle
        self.ptr = (self.ptr + 1) % self.delay_length
        return out

    def at(self, index):
        i = int(index)
        pos = (self.ptr - i) % self.delay_length
        return self.buffer[pos]

class Reverb(AudioEffect):
    def __init__(self, decay=0.5, mix=0.5, samplerate=44100):
        self.decay = decay
        self.mix = mix
        self.sr = samplerate
        # early‑reflection chain
        self.pre_delay = Delay_Line(int(0.011 * samplerate))  # ≈11 ms
        self.lpf1      = LPF_Bandwidth(0.2)
        self.apf1      = Allpass(210, 0.75)
        self.apf2      = Allpass(158, 0.75)
        self.apf3      = Allpass(561, 0.625)
        self.apf4      = Allpass(410, 0.625)
        self.g5        = 0.7
        # two parallel modulated comb lines
        self.mapf1     = Allpass_Linear(1500, 1343, 0.7)
        self.delay1    = Delay_Line(6241)
        self.lpf2      = LPF_Damping(decay)
        self.apf5      = Allpass(3931, 0.5)
        self.delay2    = Delay_Line(4681)
        self.mapf2     = Allpass_Linear(1100, 995, 0.7)
        self.delay3    = Delay_Line(6590)
        self.lpf3      = LPF_Damping(decay)
        self.apf6      = Allpass(2664, 0.5)
        self.delay4    = Delay_Line(5505)
        # LFO settings
        self.n         = 0
        self.lfo1_f    = 0.6
        self.lfo2_f    = 0.4
        self.lfo1_d    = 12
        self.lfo2_d    = 12

    def update_parameters(self, decay=None, mix=None):
        if decay is not None:
            self.decay = decay
            self.lpf2 = LPF_Damping(decay)
            self.lpf3 = LPF_Damping(decay)
        if mix is not None:
            self.mix = mix

    def process(self, buf: np.ndarray) -> np.ndarray:
        # ensure shape (n,2)
        x = buf if buf.ndim>1 else np.stack([buf,buf],1)
        out = np.zeros_like(x)
        Fs = self.sr
        for i in range(x.shape[0]):
            # update modulated delays
            w1 = 2*np.pi*self.lfo1_f/Fs
            w2 = 2*np.pi*self.lfo2_f/Fs
            M1 = 1343 + self.lfo1_d * np.sin(w1*self.n)
            M2 = 1100 + self.lfo2_d * np.sin(w2*self.n)
            self.mapf1.setDelayLength(M1)
            self.mapf2.setDelayLength(M2)
            self.n += 1
            for ch in (0,1):
                s = x[i,ch]
                # early reflections
                s = self.pre_delay.next(s)
                s = self.lpf1.next(s)
                for ap in (self.apf1,self.apf2,self.apf3,self.apf4):
                    s = ap.next(s)
                # comb network
                one_in = s + self.g5 * getattr(self,f'two_out_{ch}',0.0)
                two_in = s + self.g5 * getattr(self,f'one_out_{ch}',0.0)
                one_s = self.mapf1.next(one_in)
                one_s = self.delay1.next(one_s)
                one_s = self.lpf2.next(one_s)
                one_s = self.apf5.next(one_s)
                one_out = self.delay2.next(one_s)
                two_s = self.mapf2.next(two_in)
                two_s = self.delay3.next(two_s)
                two_s = self.lpf3.next(two_s)
                two_s = self.apf6.next(two_s)
                two_out = self.delay4.next(two_s)
                setattr(self,f'one_out_{ch}',one_out)
                setattr(self,f'two_out_{ch}',two_out)
                # mix
                out[i,ch] = (1-self.mix)*x[i,ch] + self.mix*(one_out+two_out)
        return out

class Compressor(AudioEffect):
    def __init__(self, threshold=0.5, ratio=4, attack=0.01, release=0.1):
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release
        self.env = 0  # Envelope follower (simple)

    def update_parameters(self, threshold, ratio, attack, release):
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release

    def process(self, audio_buffer):
        if audio_buffer.ndim == 1:
            processed = np.copy(audio_buffer)
            for i, sample in enumerate(audio_buffer):
                self.env = max(abs(sample), self.env * 0.99)
                if self.env > self.threshold:
                    gain = 1 / (1 + (self.ratio - 1) * ((self.env - self.threshold) / self.env))
                else:
                    gain = 1
                processed[i] = sample * gain
            return processed
        else:
            processed = np.copy(audio_buffer)
            n_channels = audio_buffer.shape[1]
            for ch in range(n_channels):
                if not hasattr(self, 'channel_env'):
                    self.channel_env = {}
                if ch not in self.channel_env:
                    self.channel_env[ch] = 0.0
                env = self.channel_env[ch]
                for i, sample in enumerate(audio_buffer[:, ch]):
                    env = max(abs(sample), env * 0.99)
                    if env > self.threshold:
                        gain = 1 / (1 + (self.ratio - 1) * ((env - self.threshold) / env))
                    else:
                        gain = 1
                    processed[i, ch] = sample * gain
                self.channel_env[ch] = env
            return processed

# ========= Global Effects Chain =========
class EffectsChain:
    def __init__(self):
        self.effects = []  # list of AudioEffect objects

    def add_effect(self, effect: AudioEffect):
        self.effects.append(effect)

    def remove_effect(self, index: int):
        if 0 <= index < len(self.effects):
            del self.effects[index]

    def swap_effects(self, idx1: int, idx2: int):
        if 0 <= idx1 < len(self.effects) and 0 <= idx2 < len(self.effects):
            self.effects[idx1], self.effects[idx2] = self.effects[idx2], self.effects[idx1]

    def process(self, audio_buffer: np.ndarray) -> np.ndarray:
        out_buffer = audio_buffer
        for effect in self.effects:
            out_buffer = effect.process(out_buffer)
        return out_buffer

global_chain = EffectsChain()  # shared instance across UI and audio callback
global_input_level = np.array([0.0, 0.0])
global_output_level = np.array([0.0, 0.0])

# ========= Real-Time Audio Callback =========
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print("Audio callback status:", status)
    processed = global_chain.process(indata)
    rms_input = np.sqrt(np.mean(np.square(indata), axis=0))
    rms_output = np.sqrt(np.mean(np.square(processed), axis=0))
    global global_input_level, global_output_level
    global_input_level = rms_input
    global_output_level = rms_output
    print("RMS Input:", rms_input, "RMS Output:", rms_output)
    outdata[:] = processed

# ========= UI: Effect Module Widget =========
class EffectModuleWidget(QtWidgets.QWidget):
    def __init__(self, effect, parent_ui):
        super().__init__()
        self.effect = effect
        self.parent_ui = parent_ui  # reference to main UI for callbacks on deletion/reordering
        # Control flag for showing legend on PZ plot
        self.show_legend = True
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        # --- 100 ms throttling for slider‑driven EQ updates ---
        self._deferred_timer = QtCore.QTimer(self)
        self._deferred_timer.setSingleShot(True)
        self._deferred_timer.setInterval(100)  # 0.1 s
        self._deferred_timer.timeout.connect(self._apply_deferred_eq_update)
        header = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(self.effect.__class__.__name__)
        label.setStyleSheet("font-weight: bold;")
        header.addWidget(label)
        # Up, Down, Delete buttons
        self.upButton = QtWidgets.QPushButton("Up")
        self.downButton = QtWidgets.QPushButton("Down")
        self.deleteButton = QtWidgets.QPushButton("Delete")
        header.addWidget(self.upButton)
        header.addWidget(self.downButton)
        header.addWidget(self.deleteButton)
        layout.addLayout(header)

        # Split parameters into two columns
        self.leftForm = QtWidgets.QFormLayout()
        self.rightForm = QtWidgets.QFormLayout()
        paramsContainer = QtWidgets.QHBoxLayout()
        paramsContainer.addLayout(self.leftForm)
        paramsContainer.addLayout(self.rightForm)
        layout.addLayout(paramsContainer)

        # Depending on effect type, add parameter controls.
        if isinstance(self.effect, Equalizer):
            # === EQ type selector ===
            type_label = QtWidgets.QLabel("Filter Type")
            self.typeCombo = QtWidgets.QComboBox()
            self.typeCombo.addItems([
                "Low-Pass",
                "High-Pass",
                "Band-Pass (constant skirt gain)",
                "Band-Pass (constant peak gain)",
                "Notch (band-stop)",
                "All-Pass",
                "Peaking EQ (Bell)",
                "Low-Shelf",
                "High-Shelf"
            ])
            self.typeCombo.setCurrentText(self.effect.eq_type)
            self.typeCombo.currentTextChanged.connect(self.update_eq_type)

            # === Phase Mode selector ===
            phase_label = QtWidgets.QLabel("Phase Mode")
            self.phaseCombo = QtWidgets.QComboBox()
            self.phaseCombo.addItems([
                "Minimum Phase",
                "Zero Phase",
                "Linear Phase",
                "Mixed Phase",
                "Maximum Phase"
            ])
            self.phaseCombo.setCurrentText(self.effect.phase_type)
            self.phaseCombo.currentTextChanged.connect(self.update_eq_phase)

            # === Polarity Flip (independent of phase mode) ===
            polarity_cb = QtWidgets.QCheckBox("Invert Polarity (–1)")
            polarity_cb.setChecked(self.effect.polarity_invert)
            self.leftForm.addRow("", polarity_cb)
            polarity_cb.stateChanged.connect(self.update_polarity)
            self.polarity_cb = polarity_cb

            # === IIR proportion slider (for Mixed Phase) ===
            mix_label = QtWidgets.QLabel("IIR % (Mixed Phase)")
            self.mixSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.mixSlider.setRange(0,100)
            self.mixSlider.setValue(int(self.effect.mix_ratio * 100))
            self.mixSlider.valueChanged.connect(self.update_mix_ratio)
            self.mixSlider.setEnabled(self.effect.phase_type == "Mixed Phase")

            # === Pole/Zero plot ===
            self.pzFig = Figure(figsize=(3, 3))
            self.pzCanvas = FigureCanvas(self.pzFig)
            # Fix the canvas size to prevent resizing
            self.pzCanvas.setFixedSize(PZ_CANVAS_WIDTH, PZ_CANVAS_HEIGHT)
            self.leftForm.addRow(QtWidgets.QLabel("Pole/Zero Plot"), self.pzCanvas)
            self.plot_poles_zeros()
            # === Legend visibility toggle ===
            self.legendCheckbox = QtWidgets.QCheckBox("Show Legend")
            self.legendCheckbox.setChecked(True)
            self.leftForm.addRow(QtWidgets.QLabel(""), self.legendCheckbox)
            self.legendCheckbox.stateChanged.connect(self.toggle_legend)
            self.aLines = []
            self.bLines = []
            # === Denominator (a) coefficient editors ===
            aLayout = QtWidgets.QHBoxLayout()
            for i, val in enumerate(self.effect.coeffs['a']):
                label = QtWidgets.QLabel(f"a{i}")
                line = QtWidgets.QLineEdit(str(val))
                line.setFixedWidth(180)
                line.textChanged.connect(lambda text, idx=i: self.on_coeff_text_changed('a', idx, text))
                self.aLines.append(line)
                aLayout.addWidget(label)
                aLayout.addWidget(line)
            self.leftForm.addRow("Denominator Coefficients (a)", aLayout)

            # === Numerator (b) coefficient editors ===
            bLayout = QtWidgets.QHBoxLayout()
            for i, val in enumerate(self.effect.coeffs['b']):
                label = QtWidgets.QLabel(f"b{i}")
                line = QtWidgets.QLineEdit(str(val))
                line.setFixedWidth(180)
                line.textChanged.connect(lambda text, idx=i: self.on_coeff_text_changed('b', idx, text))
                self.bLines.append(line)
                bLayout.addWidget(label)
                bLayout.addWidget(line)
            self.leftForm.addRow("Numerator Coefficients (b)", bLayout)
            # === Transfer Function Render ===
            self.tfFig = Figure(figsize=(4, 1))
            self.tfCanvasTF = FigureCanvas(self.tfFig)
            # Fix the height so it stays inline
            self.tfCanvasTF.setFixedHeight(TF_CANVAS_HEIGHT)
            self.tfCanvasTF.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            self.leftForm.addRow("Transfer Function", self.tfCanvasTF)
            self.update_transfer_function_display()

            # === Q Factor (float) ===
            qSpin = QtWidgets.QDoubleSpinBox()
            qSpin.setRange(0.1, 20.0)
            qSpin.setSingleStep(0.1)
            qSpin.setValue(self.effect.Q)
            qSpin.valueChanged.connect(self.update_eq_Q)
            self.leftForm.addRow("Q Factor", qSpin)
            self.qSpin = qSpin  # keep a reference for deferred updates

            # === Q Factor Slider (log scale; update on release) ===
            qSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            qSlider.setRange(0, 1000)
            qSlider.setTracking(True)            # allow continuous movement
            log_q_min = math.log10(0.1)
            log_q_max = math.log10(20.0)
            init_q_norm = (math.log10(self.effect.Q) - log_q_min) / (log_q_max - log_q_min)
            qSlider.setValue(int(init_q_norm * 1000))
            def on_q_slider_move(val):
                q_norm = val / 1000.0
                q_value = 10 ** (log_q_min + q_norm * (log_q_max - log_q_min))
                qSpin.blockSignals(True)
                qSpin.setValue(q_value)
                qSpin.blockSignals(False)
                self._schedule_deferred_eq_update()
            qSlider.valueChanged.connect(on_q_slider_move)
            # No longer connect sliderReleased to immediate update
            def on_q_spin(val):
                q_norm = (math.log10(val) - log_q_min) / (log_q_max - log_q_min)
                qSlider.blockSignals(True)
                qSlider.setValue(int(q_norm * 1000))
                qSlider.blockSignals(False)
            qSpin.valueChanged.connect(on_q_spin)
            self.leftForm.addRow("Q Factor Slider", qSlider)

            # === Gain (dB) (float) ===
            self.gainSpin = QtWidgets.QDoubleSpinBox()
            self.gainSpin.setRange(-24.0, 24.0)
            self.gainSpin.setSingleStep(0.1)
            self.gainSpin.setValue(self.effect.gain)
            self.gainSpin.valueChanged.connect(self.update_eq_gain)
            self.leftForm.addRow("Gain (dB)", self.gainSpin)
            # Disable gain control for LP, HP, and All-Pass types
            if self.effect.eq_type in [
                "Low-Pass",
                "High-Pass",
                "All-Pass"
            ]:
                self.gainSpin.setEnabled(False)
            # === Gain Slider ===
            gainSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            gainSlider.setRange(-240, 240)  # represents gain * 10
            gainSlider.setValue(int(self.effect.gain * 10))
            gainSlider.setTracking(True)            # allow continuous movement
            gainSlider.valueChanged.connect(lambda val: (
                self.gainSpin.setValue(val / 10),
                self._schedule_deferred_eq_update()
            ))
            self.gainSpin.valueChanged.connect(lambda val: gainSlider.setValue(int(val * 10)))
            self.leftForm.addRow("Gain (dB) Slider", gainSlider)
            # [Filter Order control removed]
            # === Center Frequency (Hz) ===
            freqSpin = QtWidgets.QDoubleSpinBox()
            freqSpin.setRange(20.0, 20000.0)
            freqSpin.setSingleStep(1.0)
            freqSpin.setValue(self.effect.center_freq)
            freqSpin.valueChanged.connect(self.update_eq_center)
            self.leftForm.addRow("Center Frequency (Hz)", freqSpin)
            self.freqSpin = freqSpin  # keep a reference for deferred updates

            # === Center Frequency Slider (log scale; update on release) ===
            freqSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            freqSlider.setRange(0, 1000)
            freqSlider.setTracking(True)            # allow continuous movement
            log_f_min = math.log10(20.0)
            log_f_max = math.log10(20000.0)
            init_f_norm = (math.log10(self.effect.center_freq) - log_f_min) / (log_f_max - log_f_min)
            freqSlider.setValue(int(init_f_norm * 1000))
            def on_freq_slider_move(val):
                f_norm = val / 1000.0
                f_value = 10 ** (log_f_min + f_norm * (log_f_max - log_f_min))
                freqSpin.blockSignals(True)
                freqSpin.setValue(f_value)
                freqSpin.blockSignals(False)
                self._schedule_deferred_eq_update()
            freqSlider.valueChanged.connect(on_freq_slider_move)
            # No longer connect sliderReleased to immediate update
            def on_freq_spin(val):
                f_norm = (math.log10(val) - log_f_min) / (log_f_max - log_f_min)
                freqSlider.blockSignals(True)
                freqSlider.setValue(int(f_norm * 1000))
                freqSlider.blockSignals(False)
            freqSpin.valueChanged.connect(on_freq_spin)
            self.leftForm.addRow("Center Frequency (Hz) Slider", freqSlider)

            # === Filter Type, Phase Mode, and IIR % (moved below Center Frequency slider) ===
            self.leftForm.addRow(type_label, self.typeCombo)
            self.leftForm.addRow(phase_label, self.phaseCombo)
            self.leftForm.addRow(mix_label, self.mixSlider)
            self.mixSlider.valueChanged.connect(lambda _val: self._schedule_deferred_eq_update())

            # === Composite Bode Plot ===
            self.bodeFig   = Figure(figsize=(5, 5))
            self.bodeCanvas = FigureCanvas(self.bodeFig)
            # Set a taller fixed height for better visibility
            self.bodeCanvas.setFixedHeight(BODE_CANVAS_HEIGHT)
            self.bodeCanvas.setFixedWidth(BODE_CANVAS_WIDTH)
            self.bodeCanvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            self.rightForm.addRow("Bode Plot", self.bodeCanvas)
            # create two stacked axes sharing the x‑axis
            self.ax_mag   = self.bodeFig.add_subplot(2, 1, 1)
            self.ax_phase = self.bodeFig.add_subplot(2, 1, 2, sharex=self.ax_mag)

            # Draw initial Bode plots
            self.update_bode_plots()

            # === Impulse Response Plot ===
            self.irFig = Figure(figsize=(5, 4))  # Doubled from (5, 2) to (10, 4)
            self.irCanvas = FigureCanvas(self.irFig)
            self.irCanvas.setFixedHeight(400)  # Doubled from 200 to 400 pixels
            self.rightForm.addRow("Impulse Response", self.irCanvas)
            self.plot_impulse_response()
        elif isinstance(self.effect, Delay):
            self.addDial("Delay (ms)", 10, 1000, self.effect.delay_ms, self.update_delay_ms)
            self.addDial("Feedback", 0, 95, int(self.effect.feedback * 100), self.update_delay_feedback)
            self.addDial("Mix", 0, 100, int(self.effect.mix * 100), self.update_delay_mix)
        elif isinstance(self.effect, Reverb):
            self.addDial("Decay", 0, 100, int(self.effect.decay * 100), self.update_reverb_decay)
            self.addDial("Mix", 0, 100, int(self.effect.mix * 100), self.update_reverb_mix)
        elif isinstance(self.effect, Compressor):
            self.addDial("Threshold", 0, 100, int(self.effect.threshold * 100), self.update_comp_threshold)
            self.addDial("Ratio", 1, 20, self.effect.ratio, self.update_comp_ratio)
            self.addDial("Attack (ms)", 1, 100, int(self.effect.attack * 1000), self.update_comp_attack)
            self.addDial("Release (ms)", 10, 1000, int(self.effect.release * 1000), self.update_comp_release)

        # Connect buttons to actions
        self.upButton.clicked.connect(lambda: self.parent_ui.move_effect(self, up=True))
        self.downButton.clicked.connect(lambda: self.parent_ui.move_effect(self, up=False))
        self.deleteButton.clicked.connect(lambda: self.parent_ui.delete_effect(self))

    def addDial(self, label_text, min_val, max_val, init_val, slot_func):
        # Create a dial with a corresponding LCD display and add it to the left column.
        dial = QtWidgets.QDial()
        dial.setMinimum(min_val)
        dial.setMaximum(max_val)
        dial.setValue(init_val)
        dial.setNotchesVisible(True)
        dial.setWrapping(False)
        lcd = QtWidgets.QLCDNumber()
        lcd.display(init_val)
        # Connect dial changes to update the LCD and call slot_func.
        def on_value_change(val):
            lcd.display(val)
            slot_func(val)
        dial.valueChanged.connect(on_value_change)
        # Allow double-click on the display to input a value directly
        def on_lcd_double_click(event):
            val, ok = QtWidgets.QInputDialog.getInt(self, "Set Value", label_text + ":", dial.value(), min_val, max_val)
            if ok:
                dial.setValue(val)
        lcd.mouseDoubleClickEvent = on_lcd_double_click
        # Create a horizontal container for dial and LCD.
        container = QtWidgets.QHBoxLayout()
        container.addWidget(dial)
        container.addWidget(lcd)
        self.leftForm.addRow(label_text, container)

    # ========== EQ parameter update functions ==========
    def update_eq_parameters(self, center=None, Q=None, gain=None):
        if center is None:
            center = self.effect.center_freq
        if Q is None:
            Q = self.effect.Q
        if gain is None:
            gain = self.effect.gain
        self.effect.update_parameters(center_freq=center, Q=Q, gain=gain)
        # Keep FIR taps in sync for FIR‑based modes
        if self.effect.phase_type in ("Linear Phase", "Zero Phase", "Maximum Phase", "Mixed Phase"):
            self.effect._ensure_linear_phase_taps()
        self.plot_poles_zeros()
        self.sync_coeff_fields()
        self.update_transfer_function_display()
        self.update_bode_plots()
        self.plot_impulse_response()

    def update_eq_center(self, val):
        self.update_eq_parameters(center=val)

    def update_eq_Q(self, val):
        self.update_eq_parameters(Q=val)

    def update_eq_gain(self, val):
        self.update_eq_parameters(gain=val)

    # ========== Delay parameter update functions ==========
    def update_delay_ms(self, val):
        self.effect.update_parameters(delay_ms=val, feedback=self.effect.feedback, mix=self.effect.mix)

    def update_delay_feedback(self, val):
        # val is percent, convert to 0-1
        feedback = val / 100.0
        self.effect.update_parameters(delay_ms=self.effect.delay_ms, feedback=feedback, mix=self.effect.mix)

    def update_delay_mix(self, val):
        mix = val / 100.0
        self.effect.update_parameters(delay_ms=self.effect.delay_ms, feedback=self.effect.feedback, mix=mix)

    # ========== Reverb parameter update functions ==========
    def update_reverb_decay(self, val):
        decay = val / 100.0
        self.effect.update_parameters(decay=decay, mix=self.effect.mix)

    def update_reverb_mix(self, val):
        mix = val / 100.0
        self.effect.update_parameters(decay=self.effect.decay, mix=mix)

    # ========== Compressor parameter update functions ==========
    def update_comp_threshold(self, val):
        threshold = val / 100.0
        self.effect.update_parameters(threshold=threshold, ratio=self.effect.ratio,
                                      attack=self.effect.attack, release=self.effect.release)

    def update_comp_ratio(self, val):
        self.effect.update_parameters(threshold=self.effect.threshold, ratio=val,
                                      attack=self.effect.attack, release=self.effect.release)

    def update_comp_attack(self, val):
        attack = val / 1000.0
        self.effect.update_parameters(threshold=self.effect.threshold, ratio=self.effect.ratio,
                                      attack=attack, release=self.effect.release)

    def update_comp_release(self, val):
        release = val / 1000.0
        self.effect.update_parameters(threshold=self.effect.threshold, ratio=self.effect.ratio,
                                      attack=self.effect.attack, release=release)

    def update_eq_type(self, text):
        # Update the effect's filter type and redraw PZ plot
        self.effect.eq_type = text
        self.effect.coeffs = self.effect._design_filter()
        # Re‑design FIR taps if the current phase mode relies on them
        if self.effect.phase_type in ("Linear Phase", "Zero Phase", "Maximum Phase", "Mixed Phase"):
            self.effect._ensure_linear_phase_taps()
        self.plot_poles_zeros()
        self.sync_coeff_fields()
        self.update_transfer_function_display()
        self.update_bode_plots()
        self.plot_impulse_response()
        # Enable or disable gain control based on filter type
        if text in ["Low-Pass", "High-Pass", "All-Pass"]:
            self.gainSpin.setEnabled(False)
        else:
            self.gainSpin.setEnabled(True)
 
    def update_filter_base(self, text):
        # Update filter implementation base
        self.effect.filter_base = text
        # Recompute coefficients based on new base and redraw
        self.effect.coeffs = self.effect._design_filter()
        self.plot_poles_zeros()

    def toggle_legend(self, state):
        # Toggle legend visibility and redraw plot
        self.show_legend = (state == QtCore.Qt.Checked)
        self.plot_poles_zeros()

    def update_eq_phase(self, text):
        # Update the EQ phase type and redesign filter
        self.effect.phase_type = text
        # Always redesign IIR coefficients
        self.effect.coeffs = self.effect._design_filter()
        # Design FIR taps based on phase mode
        if text == "Linear Phase":
            self.effect._ensure_linear_phase_taps()
        elif text == "Maximum Phase":
            # Design linear-phase taps then maximum-phase taps are handled in _ensure_linear_phase_taps()
            self.effect._ensure_linear_phase_taps()
        elif text in ("Zero Phase", "Mixed Phase"):
            # These modes also need a fresh FIR magnitude match
            self.effect._ensure_linear_phase_taps()
        self.plot_poles_zeros()
        self.sync_coeff_fields()
        self.update_transfer_function_display()
        self.update_bode_plots()
        self.plot_impulse_response()
        # Show slider only in Mixed Phase
        if hasattr(self, 'mixSlider'):
            self.mixSlider.setEnabled(self.effect.phase_type == "Mixed Phase")
    def update_polarity(self, state):
        self.effect.polarity_invert = (state == QtCore.Qt.Checked)
        # only phase plot changes
        self.update_bode_plots()
        self.plot_impulse_response()
    def update_mix_ratio(self, val):
        """Slider callback for IIR proportion in Mixed Phase (0-100)."""
        self.effect.mix_ratio = val / 100.0
        if self.effect.phase_type == "Mixed Phase":
            self.update_bode_plots()
            self.plot_impulse_response()

    # ---------- deferred update helpers ----------
    def _schedule_deferred_eq_update(self):
        """Restart the 0.5 s single‑shot timer so heavy redraw happens sparsely."""
        self._deferred_timer.start()

    def _apply_deferred_eq_update(self):
        """
        Perform the actual heavy update once the user has paused sliding for
        0.5 seconds.  Uses the current spin‑box values to update all EQ parameters
        in one shot.
        """
        if isinstance(self.effect, Equalizer):
            center = float(self.freqSpin.value())
            Q      = float(self.qSpin.value())
            gain   = float(self.gainSpin.value())
            self.update_eq_parameters(center=center, Q=Q, gain=gain)

    # update_eq_order method removed

    def plot_poles_zeros(self):
        # Determine poles/zeros based on effect type and phase mode
        if isinstance(self.effect, Equalizer):
            phase = self.effect.phase_type
            if phase == "Linear Phase":
                # FIR linear-phase: zeros from taps, no poles
                zeros = np.roots(self.effect.linear_taps)
                poles = np.array([])
            elif phase == "Maximum Phase":
                # FIR maximum-phase: zeros from max-phase taps, no poles
                zeros = np.roots(self.effect.max_phase_taps)
                poles = np.array([])
            else:
                # IIR-based modes (Minimum, Zero, Phase-Invert, Mixed): use IIR coeffs
                coeffs_b = self.effect.coeffs['b']
                coeffs_a = self.effect.coeffs['a']
                zeros = np.roots(coeffs_b)
                poles = np.roots(coeffs_a)
        elif hasattr(self.effect, 'zeros') and hasattr(self.effect, 'poles'):
            # Other effects with explicit zeros/poles (Delay, Reverb)
            zeros = self.effect.zeros
            poles = self.effect.poles
        else:
            # Fallback to IIR coeffs
            coeffs_b = self.effect.coeffs['b']
            coeffs_a = self.effect.coeffs['a']
            zeros = np.roots(coeffs_b)
            poles = np.roots(coeffs_a)
        ax = self.pzFig.subplots()
        ax.clear()
        # Ensure full unit circle is visible
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        # Hide axis spines and ticks for an unobstructed view
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        # draw unit circle
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), linestyle='--')
        # plot zeros and poles
        if zeros.size > 0:
            ax.scatter(zeros.real, zeros.imag, marker='o', label='Zeros')
        if poles.size > 0:
            ax.scatter(poles.real, poles.imag, marker='x', label='Poles')
        ax.set_aspect('equal', 'box')
        ax.set_title('Pole-Zero Plot')
        # self.pzFig.tight_layout()
        # Display legend in the center with transparency if enabled
        if self.show_legend:
            ax.legend(loc='center', framealpha=0.3)
        self.pzCanvas.draw()

    def on_coeff_text_changed(self, section, idx, text):
        # Update single coefficient and refresh plot
        try:
            val = float(text)
            coeffs = self.effect.coeffs
            coeffs[section][idx] = val
            self.effect.coeffs = coeffs
            self.plot_poles_zeros()
            self.update_transfer_function_display()
            self.update_bode_plots()
            self.plot_impulse_response()
        except ValueError:
            pass

    def sync_coeff_fields(self):
        # Update coefficient QLineEdit fields to match current coeffs
        for i, line in enumerate(self.aLines):
            line.blockSignals(True)
            line.setText(str(self.effect.coeffs['a'][i]))
            line.blockSignals(False)
        for i, line in enumerate(self.bLines):
            line.blockSignals(True)
            line.setText(str(self.effect.coeffs['b'][i]))
            line.blockSignals(False)

    def update_transfer_function_display(self):
        # Draw H(z) as LaTeX fraction using Matplotlib mathtext
        coeffs = self.effect.coeffs
        b = coeffs['b']
        a = coeffs['a']
        # Build LaTeX strings
        num_terms = []
        for i, val in enumerate(b):
            if i == 0:
                num_terms.append(f"{val:.3f}")
            else:
                num_terms.append(f"{val:.3f}z^{{-{i}}}")
        den_terms = []
        for i, val in enumerate(a):
            if i == 0:
                den_terms.append(f"{val:.3f}")
            else:
                den_terms.append(f"{val:.3f}z^{{-{i}}}")
        num_latex = " + ".join(num_terms)
        den_latex = " + ".join(den_terms)
        latex = rf"$H(z)=\frac{{{num_latex}}}{{{den_latex}}}$"
        # Render to canvas
        self.tfFig.clear()
        ax = self.tfFig.subplots()
        ax.axis("off")
        ax.text(0.5, 0.5, latex, ha="center", va="center", fontsize=12)
        self.tfCanvasTF.draw()

    def update_bode_plots(self):
        # Prepare frequency grid
        min_f = 20.0
        max_f = self.effect.samplerate / 2.0
        freqs = np.logspace(np.log10(min_f), np.log10(max_f), 512)

        # Make sure FIR taps are current if the active phase mode needs them
        if self.effect.phase_type in ("Linear Phase", "Maximum Phase", "Mixed Phase"):
            self.effect._ensure_linear_phase_taps()

        # Compute frequency response based on phase mode
        if self.effect.phase_type == "Maximum Phase":
            taps = self.effect.max_phase_taps
            w, h = freqz(taps, [1.0], worN=freqs, fs=self.effect.samplerate)
            mag = 20 * np.log10(np.abs(h))
            phase = np.unwrap(np.angle(h)) * 180 / np.pi
        elif self.effect.phase_type == "Linear Phase":
            taps = self.effect.linear_taps
            w, h = freqz(taps, [1.0], worN=freqs, fs=self.effect.samplerate)
            mag = 20 * np.log10(np.abs(h))
            phase = np.unwrap(np.angle(h)) * 180 / np.pi
        elif self.effect.phase_type == "Zero Phase":
            b = self.effect._zero_phase_scaled_b()
            a = self.effect.coeffs['a']
            w, h = freqz(b, a, worN=freqs, fs=self.effect.samplerate)
            # forward + reverse ⇒ square of magnitude, convert to dB correctly
            mag = 20 * np.log10(np.abs(h))
            phase = np.zeros_like(mag)
        elif self.effect.phase_type == "Mixed Phase":
            # Minimum-phase
            b, a = self.effect.coeffs['b'], self.effect.coeffs['a']
            w, h_iir = freqz(b, a, worN=freqs, fs=self.effect.samplerate)
            # Linear-phase
            taps = self.effect.linear_taps
            _, h_lin = freqz(taps, [1.0], worN=freqs, fs=self.effect.samplerate)
            # Apply same gain correction as in audio path
            r = self.effect.mix_ratio
            scale = self.effect._mixed_phase_scale(r)
            h_mix = scale * (r * h_iir + (1.0 - r) * h_lin)
            mag = 20 * np.log10(np.abs(h_mix))
            phase = np.unwrap(np.angle(h_mix)) * 180 / np.pi
        else:
            # Minimum-phase IIR
            b, a = self.effect.coeffs['b'], self.effect.coeffs['a']
            w, h = freqz(b, a, worN=freqs, fs=self.effect.samplerate)
            mag = 20 * np.log10(np.abs(h))
            phase = np.unwrap(np.angle(h)) * 180 / np.pi

        # Apply polarity inversion as a 180° phase shift
        if self.effect.polarity_invert:
            phase = phase + 180
        # Wrap phase into [-180°, +180°] for display continuity
        phase = (phase + 180) % 360 - 180
        # clear axes
        self.ax_mag.clear()
        self.ax_phase.clear()
        # Ensure positive x-limits for log scale; fallback to linear if invalid
        if min_f > 0 and max_f > min_f:
            self.ax_mag.set_xscale('log')
            self.ax_phase.set_xscale('log')
            self.ax_mag.set_xlim(min_f, max_f)
            self.ax_phase.set_xlim(min_f, max_f)
        else:
            # fallback to linear scale to avoid warnings
            self.ax_mag.set_xscale('linear')
            self.ax_phase.set_xscale('linear')

        # magnitude (top)
        self.ax_mag.semilogx(w, mag)
        self.ax_mag.set_ylabel("Magnitude (dB)")
        self.ax_mag.grid(True, which='both', linestyle='--')
        # Fix magnitude y-axis limits for better visibility
        self.ax_mag.set_ylim(-60, 30)

        # phase (bottom)
        self.ax_phase.semilogx(w, phase)
        self.ax_phase.set_xlabel("Frequency (Hz)")
        self.ax_phase.set_ylabel("Phase (°)")
        self.ax_phase.grid(True, which='both', linestyle='--')
        # Fix phase y-axis limits for better visibility
        self.ax_phase.set_ylim(-180, 180)

        # draw once
        self.bodeCanvas.draw()


    def plot_impulse_response(self):
        """
        Plot the impulse response of the equalizer with x=0 centered in the plot.
        """
        length = 512  # Length of impulse response
        ir = self.effect.get_impulse_response(length=length)
        
        # Create a symmetric time axis centered at 0
        half_length = length // 2
        time_axis = np.arange(-half_length, half_length)
        
        # The impulse response is already centered by get_impulse_response,
        # so we can directly plot it with the symmetric time axis.
        
        # Clear any old axes and create a fresh single subplot
        self.irFig.clear()
        ax = self.irFig.add_subplot(1, 1, 1)
        ax.plot(time_axis, ir)
        ax.set_title("Impulse Response")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        
        # Set y-axis limits to ensure visibility of small amplitudes
        max_amplitude = np.max(np.abs(ir))
        ax.set_ylim(-max_amplitude * 1.2, max_amplitude * 1.2) if max_amplitude > 0 else ax.set_ylim(-1, 1)
        
        # Add vertical line at x=0 for clarity
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        ax.grid(True, linestyle='--')
        # Improve layout to prevent overlapping labels
        self.irFig.tight_layout()
        self.irCanvas.draw()

# ========= UI: Main Effects Chain Interface =========
class EffectsChainUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modular Audio Plugin")
        self.setGeometry(50, 50, 900, 600)
        self.effectWidgets = []  # list of EffectModuleWidget instances
        self.init_ui()
        # --- offline processing state ---
        self.audio_data = None
        self.samplerate = 44100
        self.processed_audio = None
        # Ask for an audio file as soon as the UI opens
        self.load_audio()

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        control_layout = QtWidgets.QHBoxLayout()
        
        # Panel to add different effect modules
        self.add_eq_btn = QtWidgets.QPushButton("Add Equalizer")
        self.add_delay_btn = QtWidgets.QPushButton("Add Delay")
        self.add_reverb_btn = QtWidgets.QPushButton("Add Reverb")
        self.add_comp_btn = QtWidgets.QPushButton("Add Compressor")
        control_layout.addWidget(self.add_eq_btn)
        control_layout.addWidget(self.add_delay_btn)
        control_layout.addWidget(self.add_reverb_btn)
        control_layout.addWidget(self.add_comp_btn)
        # Button to load an audio file
        self.load_audio_btn = QtWidgets.QPushButton("Load Audio")
        control_layout.addWidget(self.load_audio_btn)
        self.load_audio_btn.clicked.connect(self.load_audio)
        main_layout.addLayout(control_layout)
        
        # Scroll area for the list of modules
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.modulesContainer = QtWidgets.QWidget()
        self.modulesLayout = QtWidgets.QVBoxLayout(self.modulesContainer)
        self.scrollArea.setWidget(self.modulesContainer)
        main_layout.addWidget(self.scrollArea)
        
        
        
        # Offline render & playback controls
        offline_layout = QtWidgets.QHBoxLayout()
        self.render_btn = QtWidgets.QPushButton("Render")
        self.play_btn = QtWidgets.QPushButton("Play Processed")
        offline_layout.addWidget(self.render_btn)
        offline_layout.addWidget(self.play_btn)
        self.pause_btn = QtWidgets.QPushButton("Pause")
        offline_layout.addWidget(self.pause_btn)
        main_layout.addLayout(offline_layout)
        # Playback seek slider
        self.playback_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.playback_slider.setRange(0, 100)
        main_layout.addWidget(self.playback_slider)

        # Timer for updating playback position
        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.setInterval(100)  # update every 100 ms
        self.playback_timer.timeout.connect(self.update_playback_progress)
        # Signal connections
        self.render_btn.clicked.connect(self.render_audio)
        self.play_btn.clicked.connect(self.play_processed_audio)
        
        
        # Connect button signals
        self.add_eq_btn.clicked.connect(self.add_equalizer)
        self.add_delay_btn.clicked.connect(self.add_delay)
        self.add_reverb_btn.clicked.connect(self.add_reverb)
        self.add_comp_btn.clicked.connect(self.add_compressor)
        self.pause_btn.clicked.connect(self.pause_playback)
        # Seek slider signals
        self.playback_slider.sliderPressed.connect(self.seek_start)
        self.playback_slider.sliderReleased.connect(self.seek_end)
        
        # self.level_timer = QtCore.QTimer(self)
        # self.level_timer.setInterval(50)  # update every 50 milliseconds
        # self.level_timer.timeout.connect(self.update_level_meters)
        # self.level_timer.start()
    
    def add_effect_module(self, effect):
        # Update the global chain.
        global_chain.add_effect(effect)
        # Create and add a new widget.
        widget = EffectModuleWidget(effect, self)
        self.effectWidgets.append(widget)
        self.modulesLayout.addWidget(widget)
    
    def add_equalizer(self):
        eq = Equalizer()
        self.add_effect_module(eq)
    
    def add_delay(self):
        delay = Delay()
        self.add_effect_module(delay)
    
    def add_reverb(self):
        reverb = Reverb()
        self.add_effect_module(reverb)
    
    def add_compressor(self):
        comp = Compressor()
        self.add_effect_module(comp)
    
    def delete_effect(self, widget):
        index = self.effectWidgets.index(widget)
        # Remove from UI layout.
        widget.setParent(None)
        self.effectWidgets.pop(index)
        # Remove from global chain.
        global_chain.remove_effect(index)
    
    def move_effect(self, widget, up=True):
        idx = self.effectWidgets.index(widget)
        if up and idx > 0:
            # Swap UI widgets.
            self.effectWidgets[idx], self.effectWidgets[idx-1] = \
                self.effectWidgets[idx-1], self.effectWidgets[idx]
            # Swap in the layout.
            self.modulesLayout.insertWidget(idx-1, widget)
            # Swap in the global chain.
            global_chain.swap_effects(idx, idx-1)
        elif not up and idx < len(self.effectWidgets) - 1:
            self.effectWidgets[idx], self.effectWidgets[idx+1] = \
                self.effectWidgets[idx+1], self.effectWidgets[idx]
            self.modulesLayout.insertWidget(idx+1, widget)
            global_chain.swap_effects(idx, idx+1)

    # ---------- Offline processing helpers ----------
    def load_audio(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav *.flac *.aiff *.mp3)"
        )
        if not path:
            return
        self.audio_data, self.samplerate = sf.read(path, always_2d=False)
        self.processed_audio = None  # clear any previous render

    def render_audio(self):
        if self.audio_data is None:
            QtWidgets.QMessageBox.warning(self, "No Audio Loaded", "Please load an audio file first.")
            return
        self.processed_audio = global_chain.process(self.audio_data)
        # Pre‑compute duration so the seek slider can work even before first play
        self.playback_duration = len(self.processed_audio) / self.samplerate
        QtWidgets.QMessageBox.information(self, "Render Complete", "Processing finished!")

    def play_processed_audio(self):
        if self.processed_audio is None:
            QtWidgets.QMessageBox.warning(self, "Not Rendered", "Render the audio first.")
            return
        # reset slider to start
        self.playback_slider.setValue(0)
        sd.play(self.processed_audio, self.samplerate)
        self.playback_duration = len(self.processed_audio) / self.samplerate
        self.play_start_time = time.time()
        self.playback_timer.start()

    # NEW top‑level method
    def pause_playback(self):
        sd.stop()
        self.playback_timer.stop()

    # NEW top‑level method
    def update_playback_progress(self):
        if not hasattr(self, "play_start_time") or not hasattr(self, "playback_duration"):
            return
        elapsed = time.time() - self.play_start_time
        percent = int(min(100, (elapsed / self.playback_duration) * 100))
        self.playback_slider.setValue(percent)
        if elapsed >= self.playback_duration:
            self.playback_timer.stop()

    def seek_start(self):
        # stop while the user is dragging
        self.playback_timer.stop()
        sd.stop()

    def seek_end(self):
        if self.processed_audio is None:
            return

        # Ensure playback duration is known
        if not hasattr(self, "playback_duration"):
            self.playback_duration = len(self.processed_audio) / self.samplerate

        percent = self.playback_slider.value() / 100.0
        new_sample = int(percent * len(self.processed_audio))

        # Start playback from the new position
        sd.play(self.processed_audio[new_sample:], self.samplerate)

        # Adjust start time so progress keeps in sync
        self.play_start_time = time.time() - percent * self.playback_duration
        self.playback_timer.start()

# ========= Main Application =========
def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = EffectsChainUI()
    ui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
