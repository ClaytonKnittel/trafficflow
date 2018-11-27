import numpy as np
import itertools
import operator
import csv


def most_common_label(vector, attr):
    vector.sort(key=lambda x: int(x[attr]))
    groups = itertools.groupby(vector, key=operator.itemgetter(attr))
    maxels = 0
    maxlabel = -1
    for g in groups:
        length = 0
        for _ in g[1]:
            length += 1
        if length > maxels:
            maxels = length
            maxlabel = g[0]
    return maxlabel, maxels / len(vector)


def intval(str):
    try:
        return int(str)
    except ValueError:
        try:
            return int('0' + str)
        except ValueError:
            return hash(str)


def all_labels_with_probs(vector, attr):
    vector.sort(key=lambda x: intval(x[attr]))
    groups = itertools.groupby(vector, key=operator.itemgetter(attr))
    group = []
    for g in groups:
        l = []
        for item in g[1]:
            l.append(item[attr])
        group.append((l[0], len(l) / len(vector)))
    group.sort(key=lambda x: -x[1])
    return group


def split(vectors, category):
    vectors.sort(key=lambda x: intval(x[category]))
    groups = itertools.groupby(vectors, key=operator.itemgetter(category))
    group = {}
    for g in groups:
        l = []
        for item in g[1]:
            l.append(item)
        group[g[0]] = l
    return group
    # split the vectors by the category at index category
    # dic = {}
    # for vector in vectors:
    #     if not vector[category] in dic:
    #         dic[vector[category]] = [vector,]
    #     else:
    #         dic[vector[category]].append(vector)
    # return dic


def entropy(vectors, cluster):
    # determines entropy of vectors split by the given cluster
    entr = 0
    splits = split(vectors, cluster)
    for sublist in splits:
        p = len(splits[sublist]) / len(vectors)
        if p != 0:
            entr -= p * np.log2(p)
    return entr


def info(vectors, split_attribute, cluster):
    # split_attribute is the attribute you want to split by,
    # cluster is the category that determines the group that
    # each data point belongs to (for calculating the entropy)
    splits = split(vectors, split_attribute)
    _info = 0
    for sublist in splits:
        p = len(splits[sublist]) / len(vectors)
        _info += p * entropy(splits[sublist], cluster)
    return _info


def info_gain(vectors, split_attribute, cluster):
    return entropy(vectors, cluster) - info(vectors, split_attribute, cluster)


def split_info(vectors, split_attribute):
    return entropy(vectors, split_attribute)


def gain_ratio(vectors, split_attribute, cluster):
    ig = info_gain(vectors, split_attribute, cluster)
    si = split_info(vectors, split_attribute)
    if ig == si == 0:
        return 0
    return ig / si


class decision_tree:

    def __init__(self, vectors, cluster_location=None, attrlist=None, min_leaf_size=1):
        # vectors is a list of data points of categorical (integral)
        # data. The attribute that is assumed to be the cluster identifier
        # is the first unless otherwise specified
        self.min_leaf_size = min_leaf_size
        if cluster_location is None:
            cluster_location = 0
        self.cluster_loc = cluster_location
        if attrlist is None:
            attrlist = list(x for x in range(0, len(vectors[0])) if x != cluster_location)
        self._root = self._create_tree(None, vectors, attrlist)

    def _create_tree(self, parent, subset, attrlist):
        n = decision_tree._node()
        n.parent = parent
        if len(subset) == 0:
            n.cluster_labels.append((-1, 0))
            return n
        if len(subset) <= self.min_leaf_size or len(attrlist) == 0:
            n.cluster_labels = all_labels_with_probs(subset, self.cluster_loc)
            return n
        if entropy(subset, self.cluster_loc) == 0:
            n.cluster_labels.append((subset[0][self.cluster_loc], 1))
            return n

        max_gainratio = -1
        best_attrloc = -1
        for i in range(0, len(attrlist)):
            gainratio = gain_ratio(subset, attrlist[i], self.cluster_loc)
            if gainratio > max_gainratio:
                max_gainratio = gainratio
                best_attrloc = i

        n.cluster_labels = all_labels_with_probs(subset, self.cluster_loc)
        splits = split(subset, attrlist[best_attrloc])
        n.val = attrlist[best_attrloc]
        newattrs = attrlist[:best_attrloc] + attrlist[best_attrloc + 1:]
        for sub in splits:
            n.children.append((self._create_tree(n, splits[sub], newattrs), sub))
        return n

    def classify(self, datapt):
        n = self._get_node(datapt, self._root)
        return n.val

    def _get_node(self, datapt, node):
        if node.val is None:
            return node
        return self._get_node(datapt, node[datapt[self.cluster_loc]])

    def __repr__(self):
        return self._rep(self._root)

    def _rep(self, node, depth=0, split_by=None):
        ret = '-' * depth
        if split_by is not None:
            ret += '(' + split_by + ')' + ' '
        ret += str(node) + '\n'
        for c in node.children:
            ret += self._rep(c[0], depth+1, split_by=str(c[1]))
        return ret

    class _node:

        def __init__(self):
            # the attribute loc to split by
            self.val = None
            # used if this is a leaf node, to determine which cluster
            # the leaf belongs to (gives sorted list of clusters by
            # their probabilities/confidences)
            self.cluster_labels = []
            self.children = []
            self.parent = None

        def __getitem__(self, item):
            for child in self.children:
                if child[1] == item:
                    return child[0]
            return None

        def __repr__(self):
            if self.val is not None:
                return str(self.val) + ' ' + str(self.cluster_labels)
            else:
                return 'l ' + str(self.cluster_labels)


if __name__ == '__main__':

    # v = [[0, 'a', 'x'],
    #      [0, 'b', 'x'],
    #      [1, 'a', 'y'],
    #      [1, 'b', 'y'],
    #      [2, 'a', 'x'],
    #      [2, 'b', 'x'],
    #      [2, 'b', 'x']]

    with open('/users/claytonknittel/pycharmprojects/trafficflow/data/allegheny.csv') as f:
        c = csv.DictReader(f, delimiter=',')

        v = []
        count = 0
        for f in c:
            if f['severity'] == '-1':
                continue
            v.append({})
            for k, val in f.items():
                v[-1][k] = val
            count += 1
            if count == -1:
                break

        l = list(c.fieldnames)
        l.remove('severity')
        l.remove('street name')
        l.remove('injuries')
        l.remove('deaths')
        l.remove('time')
        l.remove('lat')
        l.remove('lon')
        l.remove('month')
        l.remove('year')
        l.remove('collision type')
        tree = decision_tree(v, cluster_location='severity', attrlist=l, min_leaf_size=1000)
        print(tree)


