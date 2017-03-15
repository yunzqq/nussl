import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

from nussl import AudioSignal, Melodia, Evaluation

mixture = AudioSignal('../input/mixture/mixture.wav', offset = 45, duration = 30)
vocals = AudioSignal('../input/mixture/vocals.wav', offset = 45, duration = 30)

melodia = Melodia(input_audio_signal=mixture)
melodia.run()
melodia.melody_signal.write_audio_to_file('melody.wav')
sources = melodia.make_audio_signals()
estimated = []
for i,s in enumerate(sources):
    s.write_audio_to_file('output/Melodia %d.wav' % i)
