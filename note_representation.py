import numpy as np
import IPython.display as ipd
import muspy

song = np.load('predict.npy')


x = song

bar, sub_beat, pitch, dur = x[0], x[1], x[2], x[3]
start_timestep = []

i=0
for j in range(len(pitch)):
    if bar[j]==0:
        i += 1
    elif bar[j] ==2:
      break
    start_timestep.append(128 * i + (sub_beat[j] - 1) * 8)


start_timestep = np.array(start_timestep)
velocity = np.array([64] * (j+1))

#print(velocity.shape)
#print(start_timestep.shape)
#print(pitch.shape)

note_repr = np.stack([start_timestep, pitch[:j+1]+24, dur[:j+1]*2, velocity]).T

print(note_repr)

####
muspy.download_bravura_font()
muspy.download_musescore_soundfont()
!sudo apt-get install fluidsynth
gen_music = muspy.from_note_representation(note_repr)
gen_audio = gen_music.synthesize().T
ipd.Audio(gen_audio/2**15, rate=44100)

