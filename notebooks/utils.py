
import numpy as np


from scipy.ndimage import rotate, map_coordinates, gaussian_filter
import matplotlib.colors as mpc

def get_hue_image(shape, hue_direction = 'vertical'):
    hue_axis = 0 if hue_direction == 'vertical' else 1
    hue_lenght = shape[hue_axis]
    hue_vec = np.linspace(0,1,hue_lenght)
    hue =  np.tile(hue_vec, (shape[not hue_axis],1)).T
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
    return mpc.hsv_to_rgb(np.stack((hue,sat, val), axis = -1))




def vector_2_rgb(dx,dy):
    dz = dx+1j*dy
    hue = np.angle(dz)
    sat = np.absolute(dz)
    lig = 1 * np.ones_like(sat)

    hue = (hue + np.pi)/(2*np.pi)
    sat = sat/np.max(sat)

    hsv = np.stack((hue, sat,lig), axis = -1)
    return mpc.hsv_to_rgb(hsv)


class ElasticDeformation2D:
    """
    Cpoied and alltered from:
    https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/augment/transforms.py
    """

    def __init__(self, random_state, spline_order=2, alpha=2000, sigma=50,**kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        :param apply_3d: if True apply deformations in each axis
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, m):
        assert m.ndim in [2, 3]

        # Assume shape (C,H,W)
        if m.ndim == 2:
            volume_shape = m.shape
        else:
            volume_shape = m[0].shape

        dy, dx = [
            gaussian_filter(
                self.random_state.randn(*volume_shape),
                self.sigma, mode="reflect"
            ) * self.alpha for _ in range(2)
        ]

        y_dim, x_dim = volume_shape
        y, x = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing='ij')
        indices = y + dy, x + dx

        if m.ndim == 2:
            return map_coordinates(m, indices, order=self.spline_order, mode='reflect')
        else:
            channels = [map_coordinates(c, indices, order=self.spline_order, mode='reflect') for c in m]
            return np.stack(channels, axis=0)


    def call_verbose(self, m):
        assert m.ndim in [2, 3]
        output = {}

        # Assume shape (C,H,W)
        if m.ndim == 2:
            volume_shape = m.shape
        else:
            volume_shape = m[0].shape
        output['image_shape'] = volume_shape

        # Draw distortion from a gaussian distribution and apply a gaussian filter over it to 
        # remove higher frequencies and make it more continues. Increase the distortion by 
        # a factor of alpha.
        dy, dx = [
            gaussian_filter(
                self.random_state.randn(*volume_shape),
                self.sigma, mode="reflect"
            ) * self.alpha for _ in range(2)
        ]
        output['dx'] = dx
        output['dy'] = dy

        y_dim, x_dim = volume_shape
        y, x = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing='ij')
        indices = y + dy, x + dx

        output['x+dx'] = x+dx
        output['y+dy'] = y+dy

        if m.ndim == 2:
            m_ = map_coordinates(m, indices, order=self.spline_order, mode='reflect')
        else:
            channels = [map_coordinates(c, indices, order=self.spline_order, mode='reflect') for c in m]
            m_ = np.stack(channels, axis=0)
        
        output['m_'] = m_
        return output