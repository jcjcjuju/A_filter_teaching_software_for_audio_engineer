# A_filter_teaching_software_for_audio_engineer
=============================================

An interactive desktop application that lets learners **hear, see, and tweak** classic audio‑processing modules in real time.  
Built entirely in Python with **PyQt + Matplotlib** for the GUI and **NumPy + SciPy + sounddevice** for DSP and audio I/O.

Tested on **Python 3.10**, macOS 14 (Apple Silicon) and Windows 11.

## Dependencies
```bash
pip install numpy scipy sounddevice soundfile PyQt5 matplotlib
```
(macOS: allow “Microphone” permission.)

## Launch
```bash
python DST2FP.py
```
1. File dialog opens for audio.  
2. Use **Add Equalizer / Delay / Reverb / Compressor**.  
3. Dial parameters; plots & preview audio update instantly.  
4. **Render** to bounce; **Play Processed** / **Pause** to audition.

## Goals
| Goal | Why it matters |
|------|----------------|
|👂 Audible feedback | Immediately hear how each parameter shapes the sound.|
|👀 Visual intuition | Pole‑Zero, Bode‑magnitude/phase, and impulse‑response plots update live.|
|🛠 Modular chain | Stack, reorder, or delete Equaliser/Delay/Reverb/Compressor blocks like a real plug‑in host.|
|🎓 Pedagogical clarity | Exposes filter coefficients, phase modes, linear vs. maximum‑phase FIR taps, etc., so students see the math behind the sound.|

## High‑Level Architecture
- **DST2FP.py**
  - **DSP modules**
    - `Equalizer` – IIR + adaptive FIR + mixed‑phase logic  
    - `Delay` – comb feedback, analytic zeros/poles  
    - `Reverb` – Schroeder‑style early reflections + modulated combs  
    - `Compressor` – simple envelope follower  
  - **Utility classes** – `Delay_Line`, `Allpass` variants, LPFs  
  - **EffectsChain** – global processing order  
  - **Qt Widgets**
    - `EffectModuleWidget` – per‑effect controls & plots  
    - `EffectsChainUI` – main window & transport  
  - **Audio callback** – `sounddevice` stream with RMS metering

----------------------------------------------------------------
Happy filtering & enjoy exploring the z‑plane!
