from nussl import AlgorithmSwitcher, AudioSignal, RepetSim, Melodia, Repet, FT2D, Projet
mix = AudioSignal('../input/synth_mix.wav')
separated = {'bg':{}, 'fg':{}}

def separate(mix, approach):
    print 'Separating with %s' % approach.__name__
    if approach.__name__ == 'Projet':
        s = approach(mix, num_sources=2, num_iterations=20)
        separated['projet'] = s.run()
    else:
        s = approach(mix)
        s.run()
        separated['bg'][approach.__name__], separated['fg'][approach.__name__] = s.make_audio_signals()

approaches = [Melodia, RepetSim, Projet]

for a in approaches:
    separate(mix, a)

separated['fg']['Projet'] = separated['projet'][1]

import matplotlib.pyplot as plt
switcher = AlgorithmSwitcher(mix,
                             [separated['fg'][a.__name__] for a in approaches], [a.__name__ for a in approaches],
                            model = '/home/prem/research/nussl/nussl/separation/models/vocal_sdr_predictor.model')
bg_s, fg_s = switcher.run()
plt.figure(figsize=(20, 8))
plt.subplot(211)
switcher.plot(None)

plt.subplot(212)
for x in switcher.sdrs.T:
    plt.plot(x)
plt.xlim([0, 30])
plt.legend([a.__name__ for a in approaches])

plt.show()

