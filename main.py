import time
import math
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
            'front_wheel': [],
            'rear_wheel': [], # these are all x,y,z matrices composed of 1+ vectors
            'frame_axle': [],
            'fork_and_steering_column': [],
        }

        # these could be useful when we want to get out important values
        self._points = {
            'center_of_mass': np.array([]),
            'center_of_rear_wheel': np.array([]),
            'contact_point_of_rear_wheel': np.array([0, 0, 0]),
            'center_of_front_wheel': np.array([]),
            'fork_point': np.array([3, 3, 3])
        }

        pass

    def assemble(self):
        """One-time operation to assemble the bike's parts"""
        self._assembleFrontWheel()
        self._assembleRearWheel()
        self._assembleSteeringColumn()
        self._assembleFrameAxle()

    def structure(self):
        """Get the bicycle structure"""
        return self._structure

    @staticmethod
    def __assemble_wheel(radius: float = 0.5) -> np.ndarray:
        """Utility function which returns point cloud for a new wheel"""

        num_rim = 50 # for now

        spoke_angles = np.linspace(0.0, 2.0 * np.pi, num_rim)

        # create a unit vector from the wheel's center
        yz = np.array([np.sin(spoke_angles), np.cos(spoke_angles)])

        # create our rim point cloud by scaling the unit vector by the radius
        xyz = radius * (np.vstack((np.zeros(num_rim), yz))) # we use vstack to fill out the first row of the matrix
        xyz = (xyz.transpose()) # becomes a zero column

        return xyz # i dont think we need the spoke array

    def _assembleSteeringColumn(self):
        """Adds an axle directly above the front wheel"""
        self._structure['fork_and_steering_column'] = self._points['fork_point'] - self._points['center_of_front_wheel']
        pass

    def _assembleFrameAxle(self):
        """Adds axle which connects the rear wheel to the fork point"""
        self._structure['frame_axle'] = self._points['fork_point'] - self._points['center_of_rear_wheel']
        pass

    def _assembleFrontWheel(self):
        """Draw the front wheel vectors"""
        wheel = self.__assemble_wheel(self)
        self._structure['front_wheel'] += self.__translate(wheel, 0, -2, 0)

    def _assembleRearWheel(self):
        """Draw the rear wheel vectors"""
        wheel = self.__assemble_wheel(self)

    def __translate(self, vector, x, y, z):
        """Utility function which translates the given vector a certain amount"""
        return vector + np.array([x, y, z])

    @staticmethod
    def _euler_rodrigues(axis, degrees):
        """Produces a rotation matrix for a given rotation angle

            Uses the Euler-Rodriguez equation for rotating a matrix:
            https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
        """
        theta = np.radians(degrees)
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def rotate(self, part, degrees):
        """Rotate a part of the bicycle some degrees"""
        rotation_matrix = self._euler_rodrigues('z', degrees)
        self.__structure[part] = np.dot(rotation_matrix, self.__structure[part])

    def tilt(self, degrees):
        """Tilt the bicycle some degrees (about the y axis)"""
        rotation_matrix = self._euler_rodrigues('y', degrees)

        # iterate through bicycle structure and rotate each constitutent vector
        # TODO: better way to decouple this functionality -> what if I just want to tilt the front wheel?
        for partName, partStructure in self.__structure.__dict__.items():
            self.__structure[partName] = np.dot(rotation_matrix, partStructure)

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
    bike = Bicycle()

    # Lets start by building the bike's frame and wheels
    bike.assemble()




    # In this routine, we accelerate slowly, coast for 10 seconds, and then decelerate.

    # bicycle.accelerate(1) # acceleration intensity, 0.0 -> 1.0
    # bicycle.engageAutobalance()
    # time.sleep(5)
    # bicycle.brake(0.5) # brake actuation: 0.0 -> 1.0
    # bicycle.disengageAutobalance()