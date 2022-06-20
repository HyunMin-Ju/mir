import numpy as np
import IPython.display as ipd
import muspy

BASE_DIR = './MIDI-BERT-CP/Data/CP_data/'

data = np.load(BASE_DIR + 'pop909_train.npy')
song = data[100]

x = song.T

bar, sub_beat, pitch, dur = x[0], x[1], x[2], x[3]
start_timestep = []

i=0
for j in range(len(pitch)):
    if bar[j]==0:
        i += 1
    start_timestep.append(84 * i + (sub_beat[j] - 1) * 12)


start_timestep = np.array(start_timestep)
velocity = np.array([64] * (j+1))

#print(velocity.shape)
#print(start_timestep.shape)
#print(pitch.shape)

note_repr = np.stack([start_timestep, pitch[:j+1], dur[:j+1]*12, velocity]).T

print(note_repr)

####
muspy.download_bravura_font()
muspy.download_musescore_soundfont()
gen_music = muspy.from_note_representation(note_repr)
gen_audio = gen_music.synthesize().T
ipd.Audio(gen_audio/2**10, rate=44100)

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [30., 20.]
gen_music.show_score(font_scale=500)