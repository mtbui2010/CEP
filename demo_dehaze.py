from time import time
import cv2
import numpy as np
from cv2.ximgproc import guidedFilter


def truncate(input, vmax=None, vmin=None):
    out = np.copy(input)
    is_ndarray = isinstance(input, np.ndarray)
    if vmin is not None and vmax is None:
        if is_ndarray:
            out[np.where(input < vmin)] = vmin
        else:
            out = max(out, vmin)
    if vmin is None and vmax is not None:
        if is_ndarray:
            out[np.where(input > vmax)] = vmax
        else:
            out = min(out, vmax)
    if vmin is not None and vmax is not None:
        if vmin > vmax: return out
        if is_ndarray:
            out[np.where(input < vmin)] = vmin
            out[np.where(input > vmax)] = vmax
        else:
            out = max(out, vmin)
            out = min(out, vmax)
    return out


class CEP():
    def __init__(self):

        self.radius = 15
        self.ksize = 2 * self.radius + 1, 2 * self.radius + 1
        self.eps = 0.008
        self.k = 0.95
        self.t_min = 0.001
        self.t_max = 1

        self.air_top_n = 0.001
        self.fast_trans_estimate = True
        self.show_airlight_loc = True

    def enhance_rgb(self, rgb):
        tt = time()
        self.data_type = rgb.dtype.name
        self.vmin, self.vmax = np.iinfo(self.data_type).min, np.iinfo(self.data_type).max
        rgb = rgb.astype('float')

        # estimate airlight
        air_locs = self.estimate_airlight(rgb)

        b, g, r = cv2.split(rgb)
        air_r, air_g, air_b = np.mean(r[air_locs]), np.mean(g[air_locs]), np.mean(b[air_locs])

        # estimate transmission
        if self.fast_trans_estimate:
            t = self.estimate_trans_fast(rgb, (air_r, air_g, air_b))
        else:
            t = self.estimate_trans(rgb, (air_r, air_g, air_b))

        # dehaze
        out = np.zeros(rgb.shape, self.data_type)
        out[:, :, 0] = self.dehaze(b, air_b, t)
        out[:, :, 1] = self.dehaze(g, air_g, t)
        out[:, :, 2] = self.dehaze(r, air_r, t)

        if self.show_airlight_loc:
            out[air_locs] = (0, 255, 0)

        print('Dehazing elapsed: %.3fs ...' % (time() - tt))

        return out

    def estimate_airlight(self, rgb):
        min_ch = np.amin(rgb, axis=2)
        h, w = min_ch.shape[:2]
        ss = max(4 * self.radius + 1, 61)
        mean_min_ch = cv2.boxFilter(min_ch, -1, (ss, ss))
        # air_locs = np.where(mean_min_ch> (1-self.air_top_n)*np.amax(mean_min_ch))
        mean_min_ch = mean_min_ch.flatten()
        argsort = np.argsort(mean_min_ch)[::-1]
        air_locs_1D = argsort[:int(h * w * self.air_top_n)]
        air_Y = air_locs_1D // w
        air_X = air_locs_1D - air_Y * w

        return (air_Y, air_X)

    def estimate_trans(self, rgb, airlight):
        b, g, r = np.copy(rgb)
        air_r, air_g, air_b = airlight
        rn, gn, bn = r / air_r, g / air_g, b / air_b

        minr = self.get_mine(rn)
        ming = self.get_mine(gn)
        minb = self.get_mine(bn)

        t = 1 - self.k * np.minimum(np.minimum(minr, ming), minb)
        return truncate(t, vmin=self.t_min, vmax=self.t_max)

    def get_mine(self, im):
        im_float = im.astype('float32')
        u = guidedFilter(guide=im_float, src=im_float, radius=self.radius, eps=self.eps)

        if self.fast_trans_estimate:
            sig = cv2.boxFilter((im_float - u) * (im_float - u), -1, self.ksize)
        else:
            sig = guidedFilter(guide=im_float, src=(im_float - u) * (im_float - u), radius=self.radius, eps=self.eps)

        return u - np.abs(np.sqrt(sig))

    def dehaze(self, im, air, t):
        return truncate(np.divide(im - air, t) + air, vmin=self.vmin, vmax=self.vmax)

    def estimate_trans_fast(self, rgb, airlight):
        b, g, r = cv2.split(rgb)
        air_r, air_g, air_b = airlight
        rn, gn, bn = r / air_r, g / air_g, b / air_b

        min_ch = np.minimum(np.minimum(rn, gn), bn)
        mine = self.get_mine(min_ch)

        t = 1 - self.k * mine
        return truncate(t, vmin=self.t_min, vmax=self.t_max)

if __name__=='__main__':
    im = cv2.imread('CEP/test_images/bus.bmp')

    out = CEP().enhance_rgb(im)

    cv2.imshow('input', im)
    cv2.imshow('output', out)
    cv2.waitKey()
