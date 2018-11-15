import shapefile
import os.path
from rtree import index

def _gen_file(file, num):
    file += '/Trans_RoadSegment.shp'
    if num == 0:
        return file
    return file[:-4] + str(num + 1) + file[-4:]

class shapefiles:

    def __init__(self, firstFileLoc):
        self._sfiles = []
        i = 0
        self._len = 0
        while True:
            file = _gen_file(firstFileLoc, i)
            if not os.path.isfile(file):
                break
            self._sfiles.append(shapefile.Reader(file))
            self._len += len(self._sfiles[-1])
            i += 1
        if self._len == 0:
            raise FileNotFoundError('No shapefile found at ' + firstFileLoc)

    def __del__(self):
        for file in self._sfiles:
            file.close()

    def __len__(self):
        return self._len

    def _determine_bucket(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(str(index) + ' out of range of geomap of length ' + str(len(self)))
        b = 0
        begin = 0
        while index >= begin + len(self._sfiles[b]):
            begin += len(self._sfiles[b])
            b += 1
        return b, begin

    def __getitem__(self, item):
        b, begin = self._determine_bucket(item)
        return self._sfiles[b].shape(item - begin)

    def shape(self, item):
        return self[item]

    def record(self, item):
        b, begin = self._determine_bucket(item)
        return self._sfiles[b].record(item - begin)

    def shapeRecord(self, item):
        b, begin = self._determine_bucket(item)
        return self._sfiles[b].shapeRecord(item - begin)

    def iterShapes(self):
        for shape_file in self._sfiles:
            for shape in shape_file.iterShapes():
                yield shape

    def iterShapeRecords(self):
        for shape_file in self._sfiles:
            for shape in shape_file.iterShapeRecords():
                yield shape

    def fields(self):
        return self._sfiles[0].fields


class _rtree_generator:

    def __init__(self, firstFileLoc):
        self.s = shapefiles(firstFileLoc)

    def load_into_file(self, file):
        r = index.Rtree(file)
        i = 0
        for shape in self.s.iterShapes():
            r.insert(i, shape.bbox)
            i = i + 1


class geomap:

    def __init__(self, firstFileLoc, rtree_loc):
        self.shapes = shapefiles(firstFileLoc)
        self.rtree = index.Rtree(rtree_loc)
        self._gen_bbox()

    def _gen_bbox(self):
        self.bbox = self.shapes._sfiles[0].bbox
        for sfile in self.shapes._sfiles[1:]:
            b = sfile.bbox
            if b[0] < self.bbox[0]:
                self.bbox[0] = b[0]
            if b[1] < self.bbox[1]:
                self.bbox[1] = b[1]
            if b[2] > self.bbox[2]:
                self.bbox[2] = b[2]
            if b[3] > self.bbox[3]:
                self.bbox[3] = b[3]


if __name__ == '__main__':
    # r = _rtree_generator('/users/claytonknittel/downloads/Shape-3/Trans_RoadSegment.shp')
    # r.load_into_file('rtrees/pennsylvania')
    g = geomap('rtrees/Pennsylvania/Trans_RoadSegment.shp', 'rtrees/pennsylvania')
    print([i for i in g.rtree.intersection((-90.34, 38.63, -90.3, 38.67))])
