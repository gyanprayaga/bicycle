# Bicycle Runs
import numpy as np

import random

from main import Bicycle

bike = Bicycle()

bike.engageAutobalance()

# accelerate from stopped position
for t in np.linspace(0, 100):
    bike.pedal(1 * t)

# turn
bike.steer(10)

# introduce wobble
bike.tilt(5)
bike.steer(2)
bike.tilt(7)
bike.steer(5)
bike.tilt(3)
bike.steer(12)
bike.tilt(6)
bike.steer(3)

bike.disengageAutobalance()