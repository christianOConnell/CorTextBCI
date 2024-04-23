from pylsl import StreamInlet, resolve_stream
import time

numStreams = 3
# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
stream_1 = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(stream_1[0])

def collectData(): #collects 3.75 seconds of data, returns 2D array
    start = time.time()
    dataSquare = []
    count = 0
    sums = [0, 0, 0, 0, 0, 0, 0, 0]  # 8 0s
    while time.time() < start + 3.75:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
        chunk, timestamps = inlet.pull_chunk()

        for sample in chunk: #sample is array of channel data
            if count >= 9:
                for i in range(8):
                    sums[i] = int(sums[i]) // 10
                dataSquare.append(sums)
                count = 0
                sums = [0, 0, 0, 0, 0, 0, 0, 0]  # 8 0s
            for i in range(8):
                sums[i] += float(sample[i])
            count+=1

            #print(sample)

    return dataSquare

dsq = collectData()
print("Data square:")
print(dsq)