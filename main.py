import time
from typing import List

import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R

import matplotlib as mpl
import matplotlib.pyplot as plt

# class Environment:
#     """
#     The manufactured environment created by the physics engine

#     This also includes the rendered system.
#     """

#     def __init__(self):
#         """
#         Load the physics engine.
#         """
#         self.world: int = 0
#         self.bicycle = 1

# consider moving some utility methods out of Bicycle class if we want to use them for control
# for example, assembly of front wheel 


class Bicycle:
    def __init__(self):
        self._autobalance = False
        
        # create an empty vector list
        self._structure = []  

        # consider using dicts for storing vars
        self._structure = {
            'frontwheel': [],
            'rearwheel': []
        }

        pass

    def assemble(self):
        """One-time operation to assemble the bike's parts"""
        self._assembleFrontWheel()
        self._assembleRearWheel()
        self._assembleSteeringColumn()
        self._assembleFrameAxle()

    def __assembleWheel(self, show=True) -> List:
        """Utility function which returns point cloud for a new wheel"""
        


        # show the wheel
        if show:
            plt.show()
        
        return []

    def _assembleSteeringColumn(self):
        """Adds an axle directly above the front wheel"""
        pass

    def _assembleFrameAxle(self):
        """Adds axle which connects the rear wheel to the steering column"""
        pass

    def _assembleFrontWheel(self):
        """Draw the front wheel vectors"""
        wheel = self.__assembleWheel(self)
        self._structure += self.__translate(wheel, 20, 20, 20)

    def _assembleRearWheel(self):
        """Draw the rear wheel vectors"""
        wheel = self.__assembleWheel(self)
        self._structure += self.__translate(wheel, -20, 20, 20)        
        
    def __translate(self, vectors, x, y, z):
        """Utility function which translates the given vectors a certain amount"""
        return translatedVectors

    def tilt(self, degree):
        """Tilt the bicycle some degrees in the x axis"""
        pass

    def steer(self, degree):
        """Turn the front wheel of the bicycle some some degrees in the x axis"""
        pass

    def calculateCOM(self):
        """Calculates and returns the center of mass (COM) of the bicycle in x, y, and z coords"""
        pass

    def pedal(self, distance):
        """Pedal the bicycle forward a specified distance in meters"""
        pass

    # def accelerate(self, intensity: float):
    #     """Speed up the bike"""
    #     if intensity == 0 or intensity > 1:
    #         raise ValueError("Intensity must be between 0 and 1")

    def brake(self, intensity: float):
        """Engage the 'brakes' to decelerate the bicycle"""
        if intensity == 0 or intensity > 1:
            raise ValueError("Intensity must be between 0 and 1")

    def engageAutobalance(self):
        """Engage the autobalance system as a new thread"""
        print("Engaging autobalance system")
        if self._autobalance is True:
            raise AssertionError("Autobalance is already engaged")
        
        self._autobalance = True

        # start autobalance loop
        i = 0
        while True:
            # run loop
            if self._autobalance:
                # TODO: figure out how to break out of the while loop!
                print(f'Balancing... {i}')
                time.sleep(1)
                i += 1
            else:
                break

    def disengageAutobalance(self):
        """Disengage the autobalance system"""
        self._autobalance = False # this should stop 
        print("Disengaging autobalance system")


class Control:
    def __init__(self):
        """
        The control system responsible for stabilizing the bicycle
        """
        
        def stabilize(self, case):
            """
            Entry point
            """
            if case is 1:
                self._model1solver()
            else:
                self.model2solver()

        def _model1solver():
            return True

        def _model2solver():
            return True



if __name__ == '__main__':
    bicycle = Bicycle()

    # In this routine, we accelerate slowly, coast for 10 seconds, and then decelerate.

    bicycle.accelerate(1) # acceleration intensity, 0.0 -> 1.0
    bicycle.engageAutobalance()
    time.sleep(5)
    bicycle.brake(0.5) # brake actuation: 0.0 -> 1.0
    bicycle.disengageAutobalance()

main()