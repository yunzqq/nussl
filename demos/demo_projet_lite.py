import os
import sys
import time

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import nussl

# mixture = nussl.AudioSignal('../input/panned_mixture_four_sources.wav')
mixture = nussl.AudioSignal('../input/mysharona_solo.mp3', duration=30)

start_time = time.time()
projet = nussl.ProjetLite(mixture, verbose = True, num_iterations = 200, num_sources = 4)
projet.run()
end_time = time.time()
sources = projet.make_audio_signals()

print 'Took %f seconds' % (end_time - start_time)

for i,m in enumerate(sources):
    m.write_audio_to_file('output/projet_lite_%d.wav' % i)