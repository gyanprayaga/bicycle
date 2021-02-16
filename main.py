import time
import math
from typing import List, Optional

import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
            # 'frame_axle': [],
            # 'fork_and_steering_column': [],
        }

        # these could be useful when we want to get out important values
        self._points = {
            'center_of_mass': np.array([0, 1, 2]), # near the saddle
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
    def __assemble_wheel(radius=0.5):
        """Utility function which returns point cloud for a new wheel"""

        num_rim = 50  # for now

        spoke_angles = np.linspace(0.0, 2.0 * np.pi, num_rim)

        # create a unit vector from the wheel's center
        yz = np.array([np.sin(spoke_angles), np.cos(spoke_angles)])

        # we use vstack to fill out the first row of the matrix
        empty_first_row = np.vstack((np.zeros(num_rim), yz))

        # create our rim point cloud by scaling the unit vector by the radius
        xyz = radius * empty_first_row

        xyz = (xyz.transpose())  # becomes a zero column

        return xyz  # i dont think we need the spoke array

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
        wheel = self.__assemble_wheel()
        self._structure['front_wheel'] = self.__translate(wheel, 0, 2, 0)

    def _assembleRearWheel(self):
        """Draw the rear wheel vectors"""
        self._structure['rear_wheel'] = self.__assemble_wheel()

    def __translate(self, vector, x, y, z):
        """Utility function which translates the given vector a certain amount"""
        return vector + np.array([x, y, z])

    def _rotate_vectors(self, axis_of_rotation: List[float], vectors, angle):
        """Rotate the given vectors some angle (radians) about a specified axis of rotation"""
        n = axis_of_rotation / np.linalg.norm(axis_of_rotation) # this takes the norm (measures the size of the eleents)
        r = R.from_rotvec(angle * n)  # 3d vector co-directional to the axis of rotation
        rotated_vector = r.apply(vectors)  # takes the dot product of the rotation vector & the given vectors
        return rotated_vector

    def rotate(self, degrees: float, part: str):
        """Rotate a part of the bicycle some degrees (about the z axis)"""
        z_axis = [0, 0, 1]
        radians = np.radians(degrees)

        self._structure[part] = self._rotate_vectors(z_axis, self._structure[part], radians)

    def tilt(self, degrees: float, part: Optional[str]):
        """Tilt the bicycle or one of its parts some degrees (about the y axis)"""
        y_axis = [0, 1, 0]
        radians = np.radians(degrees)

        if part is not None:
            # just tilt the one part
            self._structure[part] = self._rotate_vectors(y_axis, self._structure[part], radians)
        else:
            # iterate through bicycle structure and rotate each constitutent vector
            # TODO: better way to decouple this functionality -> what if I just want to tilt the front wheel?
            for partName, partStructure in self._structure.__dict__.items():
                self._structure[partName] = self._rotate_vectors(y_axis, self._structure[partName], radians)

    def steer(self, degree):
        """Turn the front wheel of the bicycle some some degrees in the x axis"""
        pass

    def calculateCOM(self):
        """Calculates and returns the center of mass (COM) of the bicycle in x, y, and z coords"""
        pass

    def pedal(self, distance):
        """Pedal the bicycle forward a specified distance in meters"""
        pass

    def visualize(self, ax):
        """Re-render an image of the bike system "on-demand"""

        # dimensions of the axes
        minmax = [-5.0, 5.0]
        minmaxz = [-3.0, 3.0]

        ax.set_xlim(minmax)
        ax.set_ylim(minmax)
        ax.set_zlim(minmaxz)

        # label each axis
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # these are the vertices of the plane, in an array
        verts = [[(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (0.0, 4.0, 0.0)]]

        # make a collection and add it to the axes instance
        ax.add_collection3d(Poly3DCollection(verts, facecolors='g'))

        # plot the structure by iterating over each part in the dictionary
        for part in self._structure.keys():
            part_array = self._structure[part]
            x = part_array[:, 0]
            y = part_array[:, 1]
            z = part_array[:, 2]
            ax.plot(x, y, z, 'r')

        # have to explicitly call this to show the plot in shell
        plt.show()

    # def calculateCOM(self):
    #     """Calculate where the COM is by iterating over each part of the structure"""
    #     # this should only be done once at the beginning, after that we simply manipulate that point
    #
    #     # initially: start with a vector up to a point, call that COM (it will probably be at the bicycle seat)
    #     # later we can do this dynamically if we add a rider, and want to take into account the frame
    #
    #     # - assembly
    #     # - calculateCOM, prefill into the self._points._COM
    #     # - manipualation

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
            if case == 1:
                self._model1solver()
            else:
                self.model2solver()

        def _model1solver():
            return True

        def _model2solver():
            return True



if __name__ == '__main__':
    bike = Bicycle()

    # setup visualization to inspect bicycle
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    # Lets start by building the bike's frame and wheels
    # bike.assemble()

    bike._assembleRearWheel()
    bike._assembleFrontWheel()
    bike.rotate(40, 'front_wheel')
    bike.tilt(15, 'front_wheel') # TODO: we need to tilt about the new axis vector

    # See what monstrosity we have manufactured
    bike.visualize(ax)





    # In this routine, we accelerate slowly, coast for 10 seconds, and then decelerate.

    # bicycle.accelerate(1) # acceleration intensity, 0.0 -> 1.0
    # bicycle.engageAutobalance()
    # time.sleep(5)
    # bicycle.brake(0.5) # brake actuation: 0.0 -> 1.0
    # bicycle.disengageAutobalance()