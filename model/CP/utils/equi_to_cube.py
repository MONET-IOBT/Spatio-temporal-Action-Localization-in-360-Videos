import numpy as np
import cv2
import math as m

from scipy.interpolate import RegularGridInterpolator as interp2d
from scipy.interpolate import interp1d
from pylab import *
import sys
sys.path.insert(0, '/home/bo/research/realtime-action-detection')
from model.CP.utils.sph_utils import rotx, roty, rotz


class Equi2Cube:
    def __init__(self, output_width, in_image, vfov=90):

        self.out = {}
        assert in_image.shape[0]*2 == in_image.shape[1]
        self.cube_out = np.array([])
        views = [[180, 0, 0],  # Back
                 [0, -90, 0],     # Bottom
                 [0, 0, 0],  # Front
                 [-90, 0, 0],  # Left
                 [90, 0, 0],  # Right
                 [0, 90, 0]]  # Top

        self.inXs = []
        self.inYs = []

        vfov = vfov*np.pi/180
        views = np.array(views)*np.pi/180
        output_width = output_width
        output_height = output_width
        input_width = in_image.shape[1]
        input_height = in_image.shape[0]

        self.in_image = in_image
        self.views = views
        self.output_width = output_width
        self.output_height = output_height
        self.input_width = input_width
        self.input_height = input_height

        topLeft = np.array(
            [-m.tan(vfov/2)*(output_width/output_height), -m.tan(vfov/2), 1])

        # Scaling factor for grabbing pixel co-ordinates
        uv = np.array([-2*topLeft[0]/output_width, -
                       2*topLeft[1]/output_height, 0])

        # Equirectangular lookups
        res_acos = 2*input_width
        res_atan = 2*input_height
        step_acos = np.pi / res_acos
        step_atan = np.pi / res_atan
        lookup_acos = np.append(
            np.array(-np.cos(np.array(np.arange(0, res_acos))*step_acos)), 1.)
        lookup_atan = np.append(np.append(np.tan(step_atan/2-pi/2), np.tan(
            np.array(np.arange(1, res_atan))*step_atan-pi/2)), np.tan(-step_atan/2+pi/2))
        #lookup_atan = np.array([np.tan(step_atan/2-pi/2), np.tan(np.array(range(res_atan))*step_atan-pi/2), np.tan(-step_atan/2+pi/2)])

        X, Y = np.meshgrid(range(output_width), range(
            output_height))  # for output grid
        X = X.flatten()
        Y = Y.flatten()
        self.X = X
        self.Y = Y
        XImage, YImage = np.meshgrid(range(input_height), range(input_width))

        for idx in range(views.shape[0]):
            yaw = views[idx, 0]
            pitch = views[idx, 1]
            roll = views[idx, 2]
            transform = np.dot(np.dot(roty(yaw), rotx(pitch)), rotz(roll))

            points = np.concatenate((np.concatenate((topLeft[0] + uv[0] * np.expand_dims(X, axis=0),
                                                     topLeft[1] + uv[1]*np.expand_dims(Y, axis=0)), axis=0),
                                     topLeft[2] + uv[2]*np.ones((1, X.shape[0]))), axis=0)
            moved_points = np.dot(transform, points)

            x_points = moved_points[0, :]
            y_points = moved_points[1, :]
            z_points = moved_points[2, :]

            nxz = sqrt(x_points**2 + z_points**2)
            phi = zeros(X.shape[0])
            theta = zeros(X.shape[0])

            ind = nxz < 10e-10
            phi[ind & (y_points > 0)] = pi/2
            phi[ind & (y_points <= 0)] = -pi/2

            ind = np.logical_not(ind)
            phi_interp = interp1d(
                lookup_atan, np.arange(0, res_atan+1), 'linear')
            phi[ind] = phi_interp(y_points[ind]/nxz[ind])*step_atan - (pi/2)
            theta_interp = interp1d(
                lookup_acos, np.arange(0, res_acos+1), 'linear')
            theta[ind] = theta_interp(-z_points[ind]/nxz[ind])*step_acos
            theta[ind & (x_points < 0)] = -theta[ind & (x_points < 0)]

            # Find equivalent pixel co-ordinates
            inX = (theta / pi) * (input_width/2) + (input_width/2) + 1
            inY = (phi / (pi/2)) * (input_height/2) + (input_height/2) + 1

            # Cap if out of bounds
            inX[inX < 1] = 1
            inX[inX >= input_width-1] = input_width - \
                1  # not equl -> out of range
            inY[inY < 1] = 1
            inY[inY >= input_height-1] = input_height-1
            self.inXs.append(inX)
            self.inYs.append(inY)

    def to_cube(self, in_image):

        for idx in range(self.views.shape[0]):
            # Initialize output image
            out = self.out
            out[idx] = np.zeros(
                (self.output_height, self.output_width, in_image.shape[2]), in_image.dtype)

            out_pix = zeros((self.X.shape[0], in_image.shape[2]))

            inX = self.inXs[idx].reshape(
                self.output_width, self.output_height).astype('float32')
            inY = self.inYs[idx].reshape(
                self.output_width, self.output_height).astype('float32')
            for c in range(in_image.shape[2]):
                out[idx][:, :, c] = cv2.remap(
                    in_image[:, :, c], inX, inY, cv2.INTER_LINEAR)
        return out


if __name__ == '__main__':
    # Simple test
    input_img = cv2.imread("1.jpg")
    print(input_img.shape)
    e2c = Equi2Cube(512, input_img)

    output_cubeset = e2c.to_cube(input_img)

    for i in output_cubeset:
        cv2.imwrite(str(i)+'_cube.jpg',output_cubeset[i])

    from cube_to_equi import Cube2Equi
    import torch
    cubes = torch.zeros((6,3,512,512)).cuda()
    for i in range(6):
        cubes[i] = torch.from_numpy(output_cubeset[i]).permute(2,0,1)
    print(cubes.shape)

    c2e = Cube2Equi(cubes.shape[2])

    equi = c2e.to_equi_nn(cubes)

    print(equi.shape)

    equi = equi[0].permute(1,2,0).cpu().detach().numpy()
    print(equi.shape)
    cv2.imwrite('d.jpg',equi)
