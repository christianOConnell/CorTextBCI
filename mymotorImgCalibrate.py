import random
import time
from pylsl import StreamInfo, StreamOutlet, vectorf

def random_direction():
    return random.choice(['left', 'right'])

info = StreamInfo('MyMarkerStream', 'Markers', 1, channel_format = 'string', source_id='myuidw43536')

outlet = StreamOutlet(info)



for i in range(80):
    direction = (random_direction())
    print(direction)
    outlet.push_sample(direction[0])
    time.sleep(3)
