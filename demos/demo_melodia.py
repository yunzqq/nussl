import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

from nussl import AudioSignal, Melodia

mixture = AudioSignal('../input/mixture/mixture.wav', duration = 60, offset = 45)

melodia = Melodia(input_audio_signal=mixture)
melodia.run()
sources = melodia.make_audio_signals()
estimated = []
for i,s in enumerate(sources):
    s.write_audio_to_file('output/Melodia %d.wav' % i)