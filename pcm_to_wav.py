import wave

for i in range(1, 7):
    path = "C:\projects\\voice_cloning\\2.pcm"
    path = "%d.pcm" % i
    # f = open(path, 'rb')
    # pcm_data = f.read()
    # ff = wave.open("test.wav", "wb")
    # ff.setparams((2, 2, 16000, 0, 'NONE', 'NONE'))
    # ff.writeframes(pcm_data)
    import numpy as np
    from scipy.io.wavfile import write

    data = np.memmap(path, dtype='h', mode='r')
    print(data.max())
    print(data.min())
    print(data.mean())
    print(len(data))
    write("test%d.wav" % i, 16000, data)
