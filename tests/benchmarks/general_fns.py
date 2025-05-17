#!/usr/bin/env python3

# Python imports
import logging
import math
import time
from PIL import Image

# Module imports
import numpy as np
import scipy

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__file__)


def test_colour_extraction():
    def colour_extract(input_image_obj, target_rgb, cyl_length, cyl_radius):
        """OLD FUNCTION: Function for extracting target colours from image
        converts image to RGB space and find coordinates of pixels within a
        cylinder whose center is the target triplet. You can visualise this
        by removing commented plot statements throughout this function.

        Args:
            input_image_obj (str) : Name of file within current directory, or path to a file.
            target_rgb (list) : triplet of target Red Green Blue colour [Red,Green,Blue].
            cyl_length (int) : length of cylinder.
            cyl_radius (int) : radius of cylinder.

        Returns:
            COL (JpegImageFile) : PIL JpegImageFile of the filtered image highlighting yellow text.
        """
        # col4 = input_image_obj
        # pix4 = col4.load()

        # DEFINE END POINTS OF CYLINDER
        np.set_printoptions(suppress=True)

        def append_spherical_1point(xyz):
            ptsnew = np.hstack(
                (xyz, np.zeros(xyz.shape), np.zeros(xyz.shape), np.zeros(xyz.shape))
            )
            xy = xyz[0] ** 2 + xyz[1] ** 2

            ptsnew[3] = np.sqrt(xy)  # xy length
            ptsnew[4] = np.sqrt(xy + xyz[2] ** 2)  # magnitude of vector (radius)
            # angles:
            ptsnew[5] = np.arctan(np.divide(ptsnew[1], ptsnew[0])) * (180 / math.pi)  # theta
            ptsnew[6] = np.arcsin(np.divide(ptsnew[2], ptsnew[4])) * (180 / math.pi)  # alpha

            return ptsnew

        targ = np.array(target_rgb)
        out2 = append_spherical_1point(targ)

        H2 = cyl_length  # This is our Hypotenuse
        O2 = math.sin(math.radians(out2[6])) * H2  # Opposite length 2
        A2 = math.cos(math.radians(out2[6])) * H2  # Adjecent length 2 == Hypotenuse1
        O1 = math.sin(math.radians(out2[5])) * A2
        A1 = math.cos(math.radians(out2[5])) * A2

        R2 = out2[0] + A1
        G2 = out2[1] + O1
        B2 = out2[2] + O2

        R1 = out2[0] - A1
        G1 = out2[1] - O1
        B1 = out2[2] - O2

        Ctemp = np.array([out2[0], out2[1], out2[2]])
        ## Visualise target pixel in the RGB space
        # fig = plt.figure(1)
        # ax = fig.add_subplot(111, projection='3d')
        # # ax.scatter(R1,G1,B1,color="black")
        # # ax.scatter(R2,G2,B2,color="black")
        # ax.scatter(out2[0],out2[1],out2[2],c = Ctemp/255)
        # ax.plot([0,0],[0,0],[0,250], color="blue")
        # ax.plot([0,250],[0,0],[0,0], color="red")
        # ax.plot([0,0],[0,250],[0,0], color="green")
        # ax.plot([0,out2[0]],[0,out2[1]],[0,0], color="black")
        # ax.plot([0,out2[0]],[0,out2[1]],[0,out2[2]], color="black")
        # ax.plot([out2[0],out2[0]],[out2[1],out2[1]],[0,out2[2]], color="black")
        # ax.plot([out2[0],out2[0]],[0,out2[1]],[0,0], color="black")
        # ax.plot([0,out2[0]],[0,0],[0,0], color="black")
        # ax.set_xlabel('Red x')
        # ax.set_ylabel('Green y')
        # ax.set_zlabel('Blue z')
        # ax.xaxis.label.set_color('red')
        # ax.tick_params(axis='x', colors='red')
        # ax.yaxis.label.set_color('green')
        # ax.tick_params(axis='y', colors='green')
        # ax.zaxis.label.set_color('blue')
        # ax.tick_params(axis='z', colors='blue')
        # ax.view_init(20, -45)
        # plt.show()

        # CYLINDER STUFF
        # defining mask
        # shape = (260, 260, 260)
        # image = np.zeros(shape=shape)

        # set radius and centres values
        # r = 100
        start = [R1, G1, B1]
        end = [R2, G2, B2]
        p1 = np.array(start)
        p2 = np.array(end)

        # vector in direction of axis
        v = p2 - p1
        # find magnitude of vector
        mag = scipy.linalg.norm(v)
        # unit vector in direction of axis
        v = v / mag
        # make some vector not in the same direction as v
        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])
        # make vector perpendicular to v
        n1 = np.cross(v, not_v)
        # normalize n1
        n1 /= scipy.linalg.norm(n1)
        # make unit vector perpendicular to v and n1
        n2 = np.cross(v, n1)
        # surface ranges over t from 0 to length of axis and 0 to 2*pi
        t = np.linspace(0, mag, 20)
        theta = np.linspace(0, 2 * np.pi, 20)
        rsample = np.linspace(0, r, 2)

        # use meshgrid to make 2d arrays
        t, theta2 = np.meshgrid(t, theta)
        rsample, theta = np.meshgrid(rsample, theta)

        # generate coordinates for surface
        # "Tube"
        X, Y, Z = [
            p1[i] + v[i] * t + r * np.sin(theta2) * n1[i] + r * np.cos(theta2) * n2[i]
            for i in [0, 1, 2]
        ]
        # "Bottom"
        X2, Y2, Z2 = [
            p1[i] + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i]
            for i in [0, 1, 2]
        ]
        # "Top"
        X3, Y3, Z3 = [
            p1[i]
            + v[i] * mag
            + rsample[i] * np.sin(theta) * n1[i]
            + rsample[i] * np.cos(theta) * n2[i]
            for i in [0, 1, 2]
        ]

        # ax.plot_surface(X, Y, Z,alpha=.2)
        # ax.plot_surface(X2, Y2, Z2,alpha=.2)
        # ax.plot_surface(X3, Y3, Z3,alpha=.2)
        def points_in_cylinder(pt1, pt2, r, q):
            pt1 = np.array(pt1)
            pt2 = np.array(pt2)
            vec = pt2 - pt1
            const = r * np.linalg.norm(vec)

            if (
                    np.dot(q - pt1, vec) >= 0 >= np.dot(q - pt2, vec)
                    and np.linalg.norm(np.cross(q - pt1, vec)) <= const
            ):
                # print("is inside")
                logi = 1
            else:
                # print("not inside")
                logi = 0

            return logi

        col4 = input_image_obj
        pix4 = col4.load()
        # gry4 = col4.convert("L")  # returns grayscale version.

        Rmat = []
        Gmat = []
        Bmat = []
        Xs = []
        Ys = []
        C = []
        alpha = []
        beta = []
        gamma = []

        for y in range(col4.size[1]):
            for x in range(col4.size[0]):
                Rmat.append(pix4[x, y][0])
                Gmat.append(pix4[x, y][1])
                Bmat.append(pix4[x, y][2])
                Xs.append(x)
                Ys.append(y)
                C.append([pix4[x, y][0], pix4[x, y][1], pix4[x, y][2]])

        C = np.array(C)

        logi = []
        for i in range(len(Rmat)):
            logi.append(
                points_in_cylinder(start, end, r, np.array([Rmat[i], Gmat[i], Bmat[i]]))
            )

        ids = np.where(np.array(logi) == 1)
        Rmat = np.array(Rmat)
        Gmat = np.array(Gmat)
        Bmat = np.array(Bmat)
        Xs = np.array(Xs)
        Ys = np.array(Ys)
        # ax.scatter(Rmat[ids],Gmat[ids],Bmat[ids],c = C[ids]/255) # plot the select points in RGB coords
        # ax.scatter(Rmat,Gmat,Bmat,c = C/255)
        # plt.show()

        # COMBINE
        COL = input_image_obj
        COL = COL.convert("RGB")  # We need RGB, so convert here.
        PIX = COL.load()

        for id in ids[0]:
            PIX[Xs[id], Ys[id]] = (255, 255, 255)  # (Rmat[id],Gmat[id],Bmat[id])

        for y in range(COL.size[1]):
            for x in range(COL.size[0]):
                if PIX[x, y] != (255, 255, 255):
                    PIX[x, y] = (0, 0, 0)

        # plt.figure(2)
        # plt.imshow(COL)
        # plt.show()
        return COL

    image = Image.open('Lt_test_image.png')
    target_colour = (255, 255, 100)
    rad = 100
    height = 80
    num_iter = 3
    img_extracted_orig = None
    img_extracted_new = None

    t_init = time.time()
    for _ in range(num_iter):
        img_extracted_new = colour_extract(image, target_colour, rad, height)
    t_fin_new = (time.time() - t_init) / num_iter

    t_init = time.time()
    for _ in range(num_iter):
        img_extracted_orig = colour_extract(image, target_colour, rad, height)
    t_fin_old = (time.time() - t_init) / num_iter

    logging.info(f"Average run time of test method: {t_fin_old:.3f}")
    logging.info(f"Average run time of func method: {t_fin_new:.3f}")

    img_extracted_orig = np.array(img_extracted_orig, dtype=np.float_)
    img_extracted_new = np.array(img_extracted_new, dtype=np.float_)

    mape = np.abs(img_extracted_new[:, :, None] - img_extracted_orig)
    mape[img_extracted_orig != 0] /= img_extracted_orig[img_extracted_orig != 0]
    mape = np.mean(mape)
    logger.info(f"MAPE between methods: {mape}")

    overlap = 100 * np.sum(img_extracted_new[:, :, None] == img_extracted_orig) / img_extracted_orig.size
    logger.info(f"Overlap between images {overlap:.3f}%")


if __name__ == "__main__":
    test_colour_extraction()
