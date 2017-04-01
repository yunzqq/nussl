import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import nussl
print nussl

mixture = nussl.AudioSignal('../input/blondiecallme.mp3', duration = 30)

projet = nussl.ProjetRepet(mixture, verbose = True, num_iterations = 50, num_sources = 2)
projet.run()
sources = projet.make_audio_signals()

for i,m in enumerate(sources):
    m.write_audio_to_file('output/projet_repet_%d.wav' % i)
