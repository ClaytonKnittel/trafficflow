from pygame import KEYUP, KEYDOWN

class camera:

    def __init__(self, bbox, screenWidth, screenHeight):
        self.__origin = bbox[0:2]
        self.__dim = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        self.__sDim = (screenWidth, screenHeight)
        self.__pos = [0, 0]
        self.__swid = self.__dim[0]
        self._snap()

    def aspectRatio(self):
        return self.__dim[1] / self.__dim[0]

    def bottomLeft(self):
        return (self.__pos[0] - self.__swid / 2, self.__pos[1] - self.__swid * self.aspectRatio() / 2)

    def widthHeight(self):
        return (self.__swid, self.__swid * self.aspectRatio())

    def zoom(self):
        return self.__dim[0] / self.__swid

    def _snap(self):
        hwid = self.__swid / 2
        if self.__pos[0] - hwid < self.__origin[0]:
            self.__pos[0] = hwid + self.__origin[0]
        elif self.__pos[0] + hwid > self.__dim[0] + self.__origin[0]:
            self.__pos[0] = self.__dim[0] + self.__origin[0] - hwid

        hhei = self.__swid * self.aspectRatio() / 2
        if self.__pos[1] - hhei < self.__origin[1]:
            self.__pos[1] = hhei + self.__origin[1]
        elif self.__pos[1] + hhei > self.__dim[1] + self.__origin[1]:
            self.__pos[1] = self.__dim[1] + self.__origin[1] - hhei

    def move(self, dx=0, dy=0):
        factor = self.__swid
        self.__pos[0] += dx * factor
        self.__pos[1] += dy * factor
        self._snap()

    def zoom_in(self, factor):
        self.__swid /= factor
        if (self.__swid > self.__dim[0]):
            self.__swid = self.__dim[0]
        self._snap()

    def screenCoords(self, pos):
        bbox = self.bbox()
        return ((pos[0] - bbox[0]) * self.__sDim[0] / (bbox[2] - bbox[0]),
                self.__sDim[1] - (pos[1] - bbox[1]) * self.__sDim[1] / (bbox[3] - bbox[1]))

    def worldCoords(self, pos):
        return (self.__pos[0] + self.__swid * (pos[0] / self.__sDim[0] - .5),
                self.__pos[1] + self.__swid * self.aspectRatio() * (.5 - pos[1] / self.__sDim[1]))

    def bbox(self):
        xmin = self.__pos[0] - self.__swid / 2
        xmax = self.__pos[0] + self.__swid / 2
        ymin = self.__pos[1] - self.__swid * self.aspectRatio() / 2
        ymax = self.__pos[1] + self.__swid * self.aspectRatio() / 2
        return (xmin, ymin, xmax, ymax)


class keyListener:

    RELEASED = 0
    JUST_PRESSED = 1
    PRESSED = 2

    def __init__(self):
        self.__dict = {}

    def pressAction(self, key, val):
        self.__dict[key] = [val, keyListener.RELEASED, False]

    def tapAction(self, key, val):
        self.__dict[key] = [val, keyListener.RELEASED, True]

    def pollEvent(self, event):
        if event.key in self.__dict:
            if event.type == KEYUP:
                self.__dict[event.key][1] = keyListener.RELEASED
            elif event.type == KEYDOWN:
                self.__dict[event.key][1] = keyListener.JUST_PRESSED

    def act(self):
        for pair in self.__dict.values():
            if pair[1] == keyListener.JUST_PRESSED:
                pair[0]()
                if pair[2]:
                    pair[1] = keyListener.PRESSED
