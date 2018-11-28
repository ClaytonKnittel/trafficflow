
class _pt:

    def __init__(self, label, probability):
        self.label = label
        self.probability = probability


class discreet_distribution:

    def __init__(self, name=None):
        self.name = name
        self._dict = []
        self._size = 0

    def add(self, item, occurences=1):
        self._size += occurences
        for i in self._dict:
            if i[0] == item:
                i[1] += occurences
                return
        self._dict.append([item, occurences])
        self._dict.sort(key=lambda i: -i[1])

    def __getitem__(self, item):
        return self.probability(item)

    def probability(self, item):
        for i in self._dict:
            if i[0] == item:
                return i[1] / len(self)
        raise IndexError('no such index ' + str(item) + ' in distribution')

    def most_likely(self):
        return _pt(self._dict[0][0], self._dict[0][1] / len(self))

    def pts(self):
        for i in self._dict:
            yield _pt(i[0], i[1] / len(self))

    def __len__(self):
        return self._size

    def __repr__(self):
        ret = ''
        if self.name is not None:
            ret += str(self.name) + ' '
        for pt in self.pts():
            ret += '({}: {:.2f}) '.format(pt.label, 100 * pt.probability)
        return ret + str(len(self))
