"""
Title: Digital Bicycle
Author: Gyan Prayaga
Date: March 22, 2021
Description:
The Bicycle class exposes an interface for assembly, manipulation, visualization, and description
(trail and center of mass) of a digital bicycle.

The included example script assembles the bicycle, tilts it 15 degrees, and then reads its structure and point data.
"""
import time
from typing import List, Optional, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Bicycle:

    """
    The Bicycle class enables us to create a digital clone of a simplified bicycle. The goal of this class was to run
    various real-world transformations (e.g. tilting the bicycle, turning the front wheel, etc.) and get out
    useful data about the bicycle's points and structure. In the future, a separate Control class would interface
    with the Bicycle class to stabilize it.

    Bicycle lifecycle:
    (1) The Bicycle class is instantiated
    (2) Bicycle is assembled using assemble()
    (3) We can get information about the bicycle's poitns or structure matrices at any point by calling map()
    or structure() respectively
    (4) We can get the bicycle's trail using trail()
    (5) We can tilt or rotate any bicycle parts using the tilt() or rotate() methods respectively
    (6) Finally, we can see the bicycle and any transformations applied to it by running visualize(), which
    runs a separate MatPlotLib Python executable
    """
    def __init__(self):
        """
        We construct the bicycle with two dictionaries:
        (1) _structure{} for the bike's structure and
        (2): _points{} for useful points in the bike's geometry
        """

        # 3D matrices for the bicycle's parts
        self._structure = {
            'front_wheel': [],
            'rear_wheel': [],
            'frame_axle': [],
            'fork_and_steering_column': [],
        }

        # vectors for important points, relative to the origin (0,0,0)
        self._points = {
            'center_of_mass': np.array([0, 1, 2]),  # arbitrary, we can put near the saddle
            'center_of_rear_wheel': np.array([0, 0.5, 0]),
            'contact_point_of_rear_wheel': np.array([0, 0, 0]),
            'center_of_front_wheel': np.array([]),
            'handlebars': np.array([0, 1.75, 1.5])
        }

    def assemble(self):
        """One-time operation to assemble the bike's parts. Assembles front wheel, rear wheel, steering column,
        and frame axle (in that order)."""
        self._assemble_front_wheel()
        self._assemble_rear_wheel()
        self._assemble_steering_column()
        self._assemble_frame_axle()

    def structure(self) -> Dict:
        """Get the bicycle structure dictionary"""
        return self._structure

    def __assemble_wheel(self, radius=0.5):
        """Utility function which returns the rim point vectors for a new wheel"""
        num_rim = 50  # used in making the spokes

        spoke_angles = np.linspace(0.0, 2.0 * np.pi, num_rim)

        # create a unit vector from the wheel's center
        yz = np.array([np.sin(spoke_angles), np.cos(spoke_angles)])

        # we use vstack to fill out the first row of the matrix
        empty_first_row = np.vstack((np.zeros(num_rim), yz))

        # create our rim point cloud by scaling the unit vector by the radius
        xyz = radius * empty_first_row

        xyz = (xyz.transpose())  # becomes a zero column

        return xyz

    def _assemble_steering_column(self):
        """Adds an axle directly above the front wheel"""
        self._structure['fork_and_steering_column'] = np.array([self._points['center_of_front_wheel'], self._points['handlebars']])
        pass

    def _assemble_frame_axle(self):
        """Adds axle which connects the rear wheel to the fork point"""
        self._structure['frame_axle'] = np.array([self._points['center_of_rear_wheel'], self._points['handlebars']])
        pass

    def _assemble_front_wheel(self):
        """Draw the front wheel vectors"""
        wheel = self.__assemble_wheel()
        self._structure['front_wheel'] = self.__translate(wheel, 0, 2, 0.5)

        self._points['center_of_front_wheel'] = np.array([0, 2, 0.5])

    def _assemble_rear_wheel(self):
        """Draw the rear wheel vectors"""
        wheel = self.__assemble_wheel()

        self._points['center_of_rear_wheel'] = np.array([0, 0, 0.5])

        # we want the contact point to be on the origin
        self._structure['rear_wheel'] = self.__translate(wheel, 0, 0, 0.5)

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

    def tilt(self, degrees: float, part: Optional[str] = None):
        """Tilt the bicycle or one of its parts some degrees (about the y axis)

        The degrees can be any value from 0-360.

        The part can be any part in the structure dictionary (e.g. 'front_wheel', 'fork_and_steering_columm')
        """
        y_axis = [0, 1, 0]
        radians = np.radians(degrees)

        if part is not None:
            # just tilt the one part
            self._structure[part] = self._rotate_vectors(y_axis, self._structure[part], radians)
        else:
            # iterate through bicycle structure and rotate each constituent vector
            for partName, partStructure in self._structure.items():
                self._structure[partName] = self._rotate_vectors(y_axis, self._structure[partName], radians)

            # also tilt the points about the y-axis accordingly
            for pointName, pointVector in self._points.items():
                self._points[pointName] = self._rotate_vectors(y_axis, self._points[pointName], radians)

    def map(self) -> Dict:
        """Simply returns the structure and points dictionary"""
        return {
            'structure': self._structure,
            'points': self._points
        }

    def trail(self):
        """
        Gets the trail of the bike. This is computed as a one-off measurement when the bike is first assembled.

        For the full sequence of operations used to calculate the trail, refer to Section 3.2.1 in the report.
        """
        cfw = self._points['center_of_front_wheel']
        fsc = self._structure['fork_and_steering_column']
        fsc_concise = fsc[1] - fsc[0]

        v20 = cfw
        
        n_hat = fsc_concise / np.linalg.norm(fsc)  # unit vector for steering column
        k_hat = np.array([0, 0, 1])

        v30 = v20 - n_hat * (- (np.dot(k_hat, v20))/(np.dot(n_hat, k_hat)))

        origin_to_intersection = v30

        # keep x and y but set z = 0
        center_of_front_wheel_on_ground = np.array([cfw[0,], cfw[1,], 0])

        trail_v = origin_to_intersection - center_of_front_wheel_on_ground

        # trail is the magnitude of the trail vector above
        trail = np.linalg.norm(trail_v)

        return trail

    def visualize(self, ax):
        """Re-render an image of the bike system "on-demand"""

        # dimensions of the axes
        minmax = [-2.0, 5.0]
        minmaxz = [-3.0, 3.0]

        ax.set_xlim(minmax)
        ax.set_ylim(minmax)
        ax.set_zlim(minmaxz)

        # label each axis
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # these are the vertices of the plane, in an array
        verts = [[(0.0, -2.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (0.0, 4.0, 0.0)]]

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


class Control:
    def __init__(self):
        """
        The control system responsible for stabilizing the bicycle
        """
        self._autobalance = False

    def engage_autobalance(self):
        """
        Engage the autobalance system as a new thread

        Note: This is scaffolding for an auto-balancing process which will run when the bicycle is instantiated.
        Refer to the stabilize() method for more detail on how this process could be designed.
        """
        print("Engaging autobalance system")
        if self._autobalance is True:
            raise AssertionError("Autobalance is already engaged")

        self._autobalance = True

        # start autobalance loop
        i = 0
        while True:
            # run loop
            if self._autobalance:
                print(f'Balancing... {i}')
                time.sleep(1)
                i += 1
            else:
                break

    def disengage_autobalance(self):
        """Disengage the autobalance system (scaffolding)"""
        self._autobalance = False
        print("Disengaging autobalance system")

    def stabilize(self, case):
        """
        Scaffolding for an entry point based on some case

        :param case: "Case" is used ambiguously here to mean the context by which a specific equation would be engaged.
        For example, we might use Eq 16 (see report) with tilt angle and tilt velocity as inputs in a specific context,
        but an entirely different physics model in a different context.

        Proposed steps for implementing the control system include:
        (1) Some asynchronous/multiprocessing Python execution, such that the solver can respond dynamically to
        the changing tilt angle in a separate thread
        (2) Some mechanism to continuously output data about the bicycle's structure/point vectors, as well as
        re-render the visualization

        Due to the complexity of this programming problem, the integration between the control system and the digital
        bicycle has not yet been finished. However, this scaffolding has been left as a future project.
        """
        if case == 1:
            self._model_1_solver()
        else:
            self._model_2_solver()

    def _model_1_solver(self):
        """Scaffolding for a solver under some condition X"""
        pass

    def _model_2_solver(self):
        """Scaffolding for a solver under some condition Y"""
        pass


if __name__ == '__main__':
    """
    Example script for assembly of the digital bicycle, applying some transformations, and visualizing it.
    """
    bike = Bicycle()

    # setup visualization to inspect bicycle
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    # building the bike's frame and wheels
    bike.assemble()

    # get the trail after assembly
    print('trail:', bike.trail())

    # tilt the bike 15 degrees
    bike.tilt(15)

    # see how our points (e.g. COM) have changed after the transformation
    print('map: ', bike.map())

    # render the bicycle
    bike.visualize(ax)