import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig


# alterar o diretório de acordo com o local das imagens
file = "C:/Users/Fefon/OneDrive/Documentos/eu/USP/Jupiter/Nyx/Fotos_voo/Imagem_9084.jpg"

# -1 = cv.IMREAD_COLOR : RGB
# 0 = cv.IMREAD_GRAYSCALE : BW
# 1 = cv.IMREAD_UNCHANGED : RGBA

img = cv.imread(file, -1)
cv.imshow('Original Image', img)
cv.waitKey(0)

img_original = img
oimg = img[750 - 600:750 - 200, 200:900]
# oimg = img[750 - 515:750 - 300, 200:900]
cv.imshow('Cut Image', oimg)
cv.waitKey(0)

img = cv.cvtColor(oimg, cv.COLOR_BGR2GRAY)
cv.imshow('Blurred Image ', img)
cv.waitKey(0)

img = cv.GaussianBlur(img, (15, 15), 0)
cv.imshow('Blurred Image ', img)
cv.waitKey(0)

edges = cv.Canny(img, 60, 60)
cv.imshow('Edge Detection', edges)
cv.waitKey(0)

kernel = np.ones((15, 15), np.uint8)
img = cv.dilate(edges, kernel, iterations=1)
cv.imshow('Eroded Image', img)
cv.waitKey(0)

kerne = np.ones((15, 15), np.uint8)
img = cv.erode(img, kerne, iterations=1)
cv.imshow('Eroded Image', img)
cv.waitKey(0)


def fit_ellipse(x, y):
    """
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses".
    """

    D1 = np.vstack([x ** 2, x * y, y ** 2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def cart_to_pol(coeffs):
    """
    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.
    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b ** 2 - a * c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c * d - b * f) / den, (a * f - b * d) / den

    num = 2 * (a * f ** 2 + c * d ** 2 + g * b **
               2 - 2 * b * d * f - a * c * g)
    fac = np.sqrt((a - c) ** 2 + 4 * b ** 2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp / ap) ** 2
    if r > 1:
        r = 1 / r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi / 2
    else:
        phi = np.arctan((2. * b) / (a - c)) / 2
        if a > c:
            phi += np.pi / 2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi / 2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

def get_ellipse_pts(params, npts=600, tmin=0, tmax=2 * np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.
    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

points = np.argwhere(img > 0)

x = points[:, 1]
y = points[:, 0]

def limpeza(x, y):
    tamanho = len(x)
    x_medio, y_medio = int(np.average(x)), int(np.average(y))
    indices = []
    for i in range(len(x)):
        if x[i] < x_medio - 350 or x[i] > x_medio + 400 or y[i] < y_medio - 20 or y[i] > y_medio + 100:
            indices.append(i)

    x = np.delete(x, indices)
    y = np.delete(y, indices)
    if tamanho - len(x) > 30:
        x, y = limpeza(x, y)
    return x, y

x, y = limpeza(x, y)

coeffs = fit_ellipse(x, y)

x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)

x_elipse, y_elipse = get_ellipse_pts((x0, y0, ap, bp, e, phi))

for i in range(len(x_elipse)):
    image = cv.circle(oimg, (round(x_elipse[i]), round(
        y_elipse[i])), radius=1, color=(0, 0, 255), thickness=-1)

cv.imshow('Final Image', image)
cv.waitKey(0)

pontos_elipse = np.array((x_elipse, y_elipse))

def cam2world(m):
    n_points = m.shape[1]
    ss = [-553.5804672814066, 0, 0.0003042253410,
          0.0000007453406, -0.0000000006921]
    xc = 532.2073
    yc = 921.8853
    width = 1920
    height = 1080
    c = 0.9985
    d = 1.4388e-04
    e = 3.7199e-04

    A = np.array([[c, d], [e, 1]])
    T = np.array([xc, yc]).reshape(-1, 1) @ np.ones((1, n_points))

    m = np.linalg.inv(A) @ (m - T)
    M = getpoint(ss, m)
    # normalizes coordinates so that they have unit length (projection onto the unit sphere)
    M = normc(M)

    return M

def getpoint(ss, m):
    # Given an image point it returns the 3D coordinates of its correspondent optical ray
    
    w = np.vstack((m[0, :], m[1, :], np.polyval(
        np.flip(ss), np.sqrt(m[0, :] ** 2 + m[1, :] ** 2))))
    return w

def normc(x):
    # normalize columns of matrix
    return x / np.linalg.norm(x, axis=0)

M = cam2world(pontos_elipse)


def affine_fit(M):
    p = []
    for i in range(3):
        avg = 0
        for j in range(len(M)):
            avg += M[j][i]
        avg /= len(M)
        p.append(avg)
    R = np.zeros((len(M), 3))
    for i in range(len(M))  :
        for j in range(3):
            R[i][j] = M[i][j] - p[j]
    E = np.matmul(R.transpose(), R)
    d, V = eig(np.matmul(R.transpose(), R))
    D = np.zeros((len(R[0]), len(R[0])),dtype=complex)
    for x in range(len(d)):
        D[x][x] = d[x]
    n = V[0]
    V = np.delete(V,0,0)
    V.transpose()
    return n,V,p

n,V,p = affine_fit(M.transpose())


def plane_fit(n, V, p, M):
    ax = plt.axes(projection='3d')

    # desenha a elipse
    ax.plot3D(M[0], M[1], M[2], 'red')
    n, V, p = affine_fit(M.transpose())
    d = -(n[0] * p[0] + n[1] * p[1] + n[2] * p[2])
    z = -(n[0] * M[0] + n[1] * M[1] + d) / (n[2])
    y = -(n[0] * M[0] + n[2] * M[2] + d) / (n[1])
    x = -(n[1] * M[1] + n[2] * M[2] + d) / (n[0])
    theta = np.arcsin(n[1])
    phi = -np.arctan(n[0]/n[2])

    # desenha o vetor normal
    ax.plot3D(x, y, z, 'blue')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    a = np.outer(np.cos(u), np.sin(v))
    b = np.outer(np.sin(u), np.sin(v))
    c = np.outer(np.ones(np.size(u)), np.cos(v))

    # desenha a esfera unitária
    ax.plot_surface(a, b, c, linewidth=0.0, alpha=0.4)
    plt.show()
    return print(f"Theta = {theta} Phi = {phi}")

plane_fit(n,V,p,M)


def combina_img(original, elipse):
    h1, w1 = elipse.shape[:2]
    h2, w2 = original.shape[:2]

    # create empty matrix
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

    # combine 2 images
    vis[:h1, :w1, :3] = elipse
    vis[:h2, w1:w1 + w2, :3] = original
    cv.imshow('Combined Image', original)


combina_img(img_original, oimg)

cv.waitKey(0)
cv.destroyAllWindows()

