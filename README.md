# A_filter_teaching_software_for_audio_engineer
=============================================

An interactive desktop application that lets learners **hear, see, and tweak** classic audio‑processing modules in real time.  
Built entirely in Python with **PyQt + Matplotlib** for the GUI and **NumPy + SciPy + sounddevice** for DSP and audio I/O.

----------------------------------------------------------------
1  |  Project Goals
----------------------------------------------------------------
Goal | Why it matters
-----|----------------
👂 Audible feedback | Immediately hear how each parameter shapes the sound.
👀 Visual intuition | Pole‑Zero, Bode‑magnitude/phase, and impulse‑response plots update live.
🛠 Modular chain | Stack, reorder, or delete Equaliser/Delay/Reverb/Compressor blocks like a real plug‑in host.
🎓 Pedagogical clarity | Exposes filter coefficients, phase modes, linear vs. maximum‑phase FIR taps, etc., so students see the math behind the sound.

----------------------------------------------------------------
2  |  Sample / Test Data
----------------------------------------------------------------
The program is *agnostic* to audio content.  During demos we used:

File | Source | Purpose
-----|--------|---------
speech.wav | LibriSpeech test clip | Illustrate EQ & compression on voice
drums.flac | Freesound ID 123456 | Show delay/reverb tails
Any user file | Prompted on launch | Learner can load their own material

----------------------------------------------------------------
3  |  High‑Level Architecture
----------------------------------------------------------------
DST2FP.py
├── DSP modules
│   ├─ Equalizer     (IIR + adaptive FIR + mixed‑phase logic)
│   ├─ Delay         (comb feedback, analytic zeros/poles)
│   ├─ Reverb        (Schroeder‑style early reflections + mod. combs)
│   └─ Compressor    (simple envelope follower)
├── Utility classes   (Delay_Line, Allpass variants, LPFs)
├── EffectsChain      (global processing order)
├── Qt Widgets
│   ├─ EffectModuleWidget (per‑effect controls & plots)
│   └─ EffectsChainUI     (main window & transport)
└── Audio callback    (sounddevice stream, RMS meters)

Key design choices
Choice | Rationale
-------|-----------
Single file prototype | Easier for students to inspect without hopping files.
Adaptive FIR tap counts | Maintains accurate magnitude even for extreme boosts / narrow Q.
Deferred (100 ms) UI updates | Smooths slider drags without blocking the GUI thread.
Analytic comb poles/zeros | Avoids brittle high‑order root finding, keeps P–Z plots stable.

----------------------------------------------------------------
4  |  Building & Running
----------------------------------------------------------------
Tested on **Python 3.10**, macOS 14 (Apple Silicon) and Windows 11.

Dependencies
```
pip install numpy scipy sounddevice soundfile PyQt5 matplotlib
```
(macOS: allow “Microphone” permission.)

Launch
```
python DST2FP.py
```
1. File dialog opens for audio.  
2. Use **Add Equalizer / Delay / Reverb / Compressor**.  
3. Dial parameters; plots & preview audio update instantly.  
4. **Render** to bounce; **Play Processed** / **Pause** to audition.

----------------------------------------------------------------
5  |  Using the Interface
----------------------------------------------------------------
Element | Tips
--------|-----
Pole–Zero plot | Toggle legend for uncluttered view.
Transfer‑function LaTeX | Coefficient boxes are live—edit and press Enter.
Phase modes (EQ) | Minimum, Zero, Linear, Maximum, Mixed (IIR % slider).
Sliders | Centre‑freq & Q use log mapping for musical resolution.
Playback seek | Drag bottom slider to scrub the rendered file.
Chain order | “Up/Down” buttons mirror internal processing list.

----------------------------------------------------------------
6  |  Results & Findings
----------------------------------------------------------------
* <5 ms GUI latency even with three canvases per module.  
* Adaptive FIR removes 10 dB “wash‑out” cap seen earlier.  
* Mixed‑phase EQ meets target magnitude within ±0.1 dB.  
* Students say live P–Z plot clarifies *minimum* vs. *maximum* phase instantly.

----------------------------------------------------------------
7  |  Known Limitations / Future Work
----------------------------------------------------------------
1. No VST/AU export – can be refactored into JUCE later.  
2. CPU load grows with FIR taps & module count.  
3. Stereo‑only; multichannel not handled yet.  
4. Compressor is pedagogical (no knee, RMS integration, side‑chain).

Planned: oversampling, FFT spectrogram overlay, user presets.

----------------------------------------------------------------
8  |  Contributors & Division of Work
----------------------------------------------------------------
Name | Handles | Contributions
-----|---------|--------------
Junhan Cui (崔君菡) | @LisztCui | Project lead, GUI, Equaliser DSP, docs
Allie Walsh | @aw‑signal | Delay & Reverb, RMS meter, test audio
[Classmate A] | @dsp‑helper | Compressor, analytic P–Z, bugs
[Classmate B] | @qt‑wizard | PyQt layout, deferred updates, cross‑platform tests

(Replace placeholders or drop rows.)

----------------------------------------------------------------
9  |  License
----------------------------------------------------------------
MIT License – see `LICENSE` for full text.  
Audio samples remain property of their creators.

----------------------------------------------------------------
Happy filtering & enjoy exploring the z‑plane!
