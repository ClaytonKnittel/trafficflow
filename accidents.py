import csv
import random
from enum import Enum


class IntersectionType(Enum):
    MID_BLOCK = 0
    FOUR_WAY_INTERSECTION = 1
    T_INTERSECTION = 2
    Y_INTERSECTION = 3
    ROUNDABOUT = 4
    MULTI_LEG_INTERSECTION = 5
    ON_RAMP = 6
    OFF_RAMP = 7
    CROSSOVER = 8
    RAILROAD_CROSSING = 9
    OTHER = 10
    UNKNOWN = 99


class RelationToRoad(Enum):
    ON_ROADWAY = 1
    SHOULDER = 2
    MEDIAN = 3
    ROADSIDE = 4
    OUTSIDE_TRAFFIC = 5
    PARKING_LANE = 6
    GORE = 7
    UNKNOWN = 9


class CollitionType(Enum):
    NO_COLLISION = 0
    REAR_END = 1
    HEAD_ON = 2
    REAR_TO_REAR = 3
    ANGLE = 4
    SIDESWIPE_SAME_DIR = 5
    SIDESWIPE_OPPOSITE_DIR = 6
    HIT_FIXED_OBJ = 7
    HIT_PEDESTRIAN = 8
    OTHER = 9


def _get_age(dict):
    if dict['DRIVER_COUNT_16YR']:
        return 16


def _int_cast(string):
    try:
        return int(string)
    except ValueError:
        return -1


class _acc:

    # specifically designed for Allegheny crash reports

    def __init__(self, r_dict):
        x = float(r_dict['DEC_LONG'])
        y = float(r_dict['DEC_LAT'])
        self.pos = (x, y)
        self.street_name = r_dict['STREET_NAME']
        self.speed_limit = _int_cast(r_dict['SPEED_LIMIT'])
        self.lanes = _int_cast(r_dict['LANE_COUNT'])
        self.severity = _int_cast(r_dict['MAX_SEVERITY_LEVEL'])
        self.deaths = _int_cast(r_dict['FATAL_COUNT'])
        self.urban_rural = _int_cast(r_dict['URBAN_RURAL'])
        self.intersection = _int_cast(r_dict['INTERSECT_TYPE'])
        self.relation_to_road = _int_cast(r_dict['RELATION_TO_ROAD'])
        self.collision_type = _int_cast(r_dict['COLLISION_TYPE'])
        self.hour_of_day = _int_cast(r_dict['HOUR_OF_DAY'])

    def attr_vector(self):
        return [self.speed_limit, self.lanes, self.severity, self.deaths, self.urban_rural, self.intersection,
                self.relation_to_road, self.collision_type, self.hour_of_day]

    def bbox(self):
        return (*self.pos, *self.pos)


def gen_accidents(path):
    with open(path) as file:
        read = csv.DictReader(file, delimiter=',')

        for r in read:
            x = r['DEC_LONG']
            y = r['DEC_LAT']
            if x == '' or y == '':
                continue
            try:
                x = float(x)
                y = float(y)
            except ValueError:
                print('could not format', x, y, 'as integers')
                continue
            yield r


class data_generator:

    # paths are to csv files with traffic accident data
    def __init__(self, *paths):
        self.accs = []
        for path in paths:
            for a in gen_accidents(path):
                self.accs.append(_acc(a))
        self.ratio = 9

    def _percent_train(self):
        return self.ratio / (1 + self.ratio)

    def __len__(self):
        if self.ratio == 0:
            return len(self.accs)
        return int(len(self.accs) * self._percent_train())

    def __iter__(self):
        yield from self.accs

    # a ratio of 0 means use all data as training data.
    # otherwise, ratio is training data : testing data
    def set_cross_validation_ratio(self, ratio):
        self.ratio = ratio

    # without replacement
    def random_sample(self, sample_size):
        if sample_size > len(self):
            raise IndexError('Cannot take bootstrap sample of size ' + str(sample_size)
                             + ' from dataset of size ' + str(len(self)))
        for sample in random.sample(range(0, len(self)), sample_size):
            yield self.accs[sample]

    def _random_gen(self, samples):
        for sample in samples:
            yield self.accs[sample]

    def random_samples(self, sample_size, num_bootstraps):
        total = sample_size * num_bootstraps
        if total > len(self):
            raise IndexError('Cannot take ' + str(num_bootstraps) + ' bootstrap samples of size ' + str(sample_size)
                             + ' from dataset of size ' + str(len(self)))
        samples = random.sample(range(0, len(self)), total)
        for b in range(0, num_bootstraps):
            yield self._random_gen(samples[b * sample_size: (b + 1) * sample_size])

    # with replacement
    def bootstrap_sample(self, sample_size):
        for x in range(0, sample_size):
            yield random.randint(0, len(self))


def get_road(geomap, accident):
    return geomap.rtree.nearest(accident.pos, 1)
