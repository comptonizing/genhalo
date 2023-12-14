import math
import numpy as np
import numbers
import cv2 as cv
from descriptors import cachedproperty
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astropy.io import fits

class Image:
    def __init__(self, img,
            detectionSigma = 3.0,
            detectionFwhm = 3.0,
            detectionThresholdFactor = 5.0
            ):
        self.__img = img.copy()
        self.__sigma = detectionSigma
        self.__fwhm = detectionSigma
        self.__thresholdFactor = detectionThresholdFactor

    @cachedproperty
    def stats(self):
        mean, median, std = sigma_clipped_stats(self.__img, sigma=self.__sigma)
        return dict(mean=mean, median=median, std=std)

    @cachedproperty
    def stars(self):
        daofind = DAOStarFinder(fwhm=self.__fwhm, threshold=self.__thresholdFactor * self.stats['std'])
        return daofind(self.__img - self.stats['median'])

    @property
    def data(self):
        return self.__img

class Star:
    pass

def isarray(thing):
    return isinstance(thing, list) or isinstance(thing, np.ndarray)

def isnumber(thing):
    return isinstance(thing, numbers.Number)

# Image dimensions
# Inset position
# Radius
# Intensity (mono or rgb)
# Blur
# Number of vanes
# Angle of vanes
# Thickness of vanes
# Shadow size
class Halo:

    def __init__(self,
            imageDimensions,
            insertPosition,
            radius,
            intensity,
            blur = 10,
            noise = 0,
            shadowSize = 0,
            vanesNumber = 0,
            vanesAngle = 0,
            vanesThickness = 50
            ):

        if not isarray(imageDimensions) or len(imageDimensions) != 2:
            raise ValueError("imageDimensions needs to be a list/array of length 2")
        for thing in imageDimensions:
            if not isnumber(thing):
                raise ValueError("imageDimensions values need to be numbers")
            if thing <= 0:
                raise ValueError("imageDimensions need to be larger than 0")

        if not isarray(insertPosition) or len(insertPosition) != 2:
            raise ValueError("insertPosition needs to be a ist/array of length 2")
        for thing in insertPosition:
            if not isnumber(thing):
                raise ValueError("insertPosition values need to be numbers")
            if thing < 0:
                raise ValueError("insertPosition need to be at least 0")

        if not isnumber(radius):
            raise ValueError("radius needs to be a number")
        if radius <= 0:
            raise ValueError("radius needs to be larger than 0")

        if isnumber(intensity):
            self.__nChannels = 1
            intensity = [intensity]
        elif isarray(intensity):
            self.__nChannels = len(intensity)
        else:
            raise ValueError("intensity needs to be a single value or a list/array")
        for thing in intensity:
            if thing <= 0:
                raise ValueError("All intensities must be larger than 0")
        if self.__nChannels != 1 and self.__nChannels != 3:
            raise ValueError("Exactly 1 or 3 channels are supported")

        if isnumber(noise):
            self.__noise = np.empty(self.__nChannels)
            self.__noise.fill(noise)
        elif isarray(noise):
            if len(noise) != self.__nChannels:
                raise ValueError("Number of noise values and channels don't match")
        else:
            raise ValueError("noise needs to be a single number of a list/array with the same length as intensity")
        for thing in self.__noise:
            if thing < 0:
                raise ValueError("noise values need to be at least 0")

        if not isnumber(shadowSize):
            raise ValueError("shadowSize needs to be a number")
        if shadowSize < 0:
            raise ValueError("shadowSize needs to be at least 0")

        if type(vanesNumber) is not int:
            raise ValueError("vanesNumber needs to be an int")
        if vanesNumber < 0:
            raise ValueError("vanesNumber needs to be at least 0")

        if not isnumber(vanesAngle):
            raise ValueError("vanesAngle needs to be a number")
        if vanesAngle < 0 or vanesAngle > 360:
            raise ValueError("vanesAngle needs to be between 0 and 360")

        if not isnumber(vanesThickness):
            raise ValueError("vanesThickness needs to be a number")
        if vanesThickness < 0:
            raise ValueError("vanesThickness needs to be at least 0")

        self.__imageDimensions = imageDimensions
        self.__insertPosition = insertPosition
        self.__radius = radius
        self.__intensity = intensity
        self.__blur = blur
        self.__shadowSize = shadowSize
        self.__vanesNumber = vanesNumber
        self.__vanesAngle = vanesAngle
        self.__vanesThickness = vanesThickness
        self.__img = None

    def __addVane(self, angle):
        x0 = self.__insertPosition[0]
        y0 = self.__insertPosition[1]
        x1 = int(round(x0 + 1.1 * self.__radius * math.cos(angle / 180. * math.pi), 0))
        y1 = int(round(y0 + 1.1 * self.__radius * math.sin(angle / 180. * math.pi), 0))
        cv.line(self.__img, [x0, y0], [x1, y1], 0, self.__vanesThickness)

    def __addVanes(self):
        angleStep = 360. / self.__vanesNumber
        for ii in range(0, self.__vanesNumber):
            angle = self.__vanesAngle + ii * angleStep
            self.__addVane(angle)

    def __addNoise(self):
        s = list(cv.split(self.__img))
        for ii in range(self.__nChannels):
            if self.__noise[ii] == 0:
                continue
            mask = (s[ii] > 0).astype(int).astype(np.float32)
            tmp = np.zeros(self.__imageDimensions, dtype=np.float32)
            tmp = cv.randn(tmp, 0., self.__noise[ii])
            tmp = cv.multiply(mask, tmp)
            s[ii] = np.add(s[ii], tmp)
            s[ii][ (s[ii] < 0) ] = 0.
        self.__img = cv.merge(s)

    def __setup_halo(self):
        center = [self.__insertPosition[0], self.__insertPosition[1]]
        self.__img = np.zeros([self.__imageDimensions[0], self.__imageDimensions[1], self.__nChannels], dtype=np.float32)
        cv.circle(self.__img, center, self.__radius, self.__intensity, -1)

        if self.__shadowSize > 0:
            cv.circle(self.__img, center, self.__shadowSize, 0, -1)

        if self.__vanesNumber > 0:
            self.__addVanes()

        if self.__blur > 0:
            self.__img = cv.blur(self.__img, [self.__blur, self.__blur])

        if (self.__noise > 0).any():
            self.__addNoise()

    @property
    def img(self):
        if self.__img is None:
            self.__setup_halo()
        return self.__img

    def save(self, fname):
        print(np.min(self.img))
        print(np.max(self.img))
        hdu = fits.PrimaryHDU(cv.split(self.img))
        hdul = fits.HDUList([hdu])
        hdul.writeto(fname, overwrite=True)
