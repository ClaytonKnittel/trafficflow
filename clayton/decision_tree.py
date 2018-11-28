import numpy as np
import itertools
import operator
import csv
import random
from clayton.stats import discreet_distribution


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
    group = discreet_distribution(name=attr)
    for g in groups:
        l = []
        for item in g[1]:
            l.append(item[attr])
        group.add(l[0], occurences=len(l))
    return group


def split(vectors, category):
    # split the vectors by the category at index category
    vectors.sort(key=lambda x: intval(x[category]))
    groups = itertools.groupby(vectors, key=operator.itemgetter(category))
    group = {}
    for g in groups:
        l = []
        for item in g[1]:
            l.append(item)
        group[g[0]] = l
    return group


def entropy(vectors, cluster):
    # determines entropy of vectors split by the given cluster
    entr = 0
    map = {}
    for v in vectors:
        if v[cluster] in map:
            map[v[cluster]] += 1
        else:
            map[v[cluster]] = 1
    for k in map:
        p = map[k] / len(vectors)
        if p != 0:
            entr -= p * np.log2(p)
    # splits = split(vectors, cluster)
    # for sublist in splits:
    #     p = len(splits[sublist]) / len(vectors)
    #     if p != 0:
    #         entr -= p * np.log2(p)
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


def bootstrap_sample(vectors, size):
    ret = []
    for x in range(0, size):
        ret.append(vectors[random.randint(0, len(vectors) - 1)])
    return ret


class decision_tree:

    def __init__(self, vectors, cluster_location=None, attrlist=None, min_leaf_size=1, m=-1):
        # vectors is a list of data points of categorical (integral)
        # data. The attribute that is assumed to be the cluster identifier
        # is the first unless otherwise specified
        # n is the number of features to be considered per split
        self.m = m
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
            n.cluster_labels.add(-1, 1)
            return n
        if len(subset) <= self.min_leaf_size or len(attrlist) == 0:
            n.cluster_labels = all_labels_with_probs(subset, self.cluster_loc)
            return n
        if entropy(subset, self.cluster_loc) == 0:
            n.cluster_labels.add(subset[0][self.cluster_loc], occurences=len(subset))
            return n

        max_gainratio = -1
        best_attrloc = -1

        if self.m == -1:
            at = attrlist
        else:
            if self.m >= len(attrlist):
                at = attrlist
            else:
                at = random.sample(attrlist, self.m)

        for i in range(0, len(at)):
            gainratio = gain_ratio(subset, at[i], self.cluster_loc)
            if gainratio > max_gainratio:
                max_gainratio = gainratio
                best_attrloc = i

        n.cluster_labels = all_labels_with_probs(subset, at[best_attrloc])
        splits = split(subset, at[best_attrloc])
        n.val = at[best_attrloc]
        newattrs = attrlist.copy()
        newattrs.remove(at[best_attrloc])
        for sub in splits:
            n.children.append((self._create_tree(n, splits[sub], newattrs), sub))
            # print(sub)
        return n

    def classify(self, datapt):
        n = self._get_node(datapt, self._root)
        if n is None:
            return discreet_distribution()
        return n.cluster_labels

    def _get_node(self, datapt, node):
        # print(str(node))
        if node is None:
            return node
        if node.val is None:
            return node
        # print(datapt[node.val])
        return self._get_node(datapt, node[datapt[node.val]])

    def __repr__(self):
        return self._rep(self._root)

    def _rep(self, node, depth=0, split_by=None):
        ret = '-' * depth
        if split_by is not None:
            ret += '(' + split_by + ') '
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
            self.cluster_labels = discreet_distribution()
            self.children = []
            self.parent = None

        def __getitem__(self, item):
            for child in self.children:
                if child[1] == item:
                    return child[0]
            return None

        def __repr__(self):
            if self.val is not None:
                return str(self.cluster_labels)
            else:
                return 'l ' + str(self.cluster_labels)


class random_forest:

    def __init__(self, vectors, cluster_location=None, attrlist=None, min_leaf_size=1, T=1, n=-1, m=-1):
        # T is total number of trees
        # n is the size of the bagged clusters (number of data pts per bag)
        # m is number of attributes to choose from attrlist for each split
        if n == -1:
            self.n = len(vectors) / 3
        else:
            self.n = n
        self.m = m
        self.trees = []
        for x in range(0, T):
            sample = bootstrap_sample(vectors, n)
            self.trees.append(decision_tree(sample, cluster_location=cluster_location, attrlist=attrlist,
                                            min_leaf_size=min_leaf_size, m=m))

    def classify(self, datapt):
        ret = {}
        for tree in self.trees:
            res = tree.classify(datapt)
            for vp in res.pts():
                if vp.label in ret:
                    ret[vp.label] += vp.probability
                else:
                    ret[vp.label] = vp.probability
        for i in ret.keys():
            ret[i] /= len(self.trees)
        return ret


if __name__ == '__main__':

    # v = [[0, 'a', 'x'],
    #      [0, 'b', 'x'],
    #      [1, 'a', 'y'],
    #      [1, 'b', 'y'],
    #      [2, 'a', 'x'],
    #      [2, 'b', 'x'],
    #      [2, 'b', 'x']]
    #
    # r = random_forest(v, cluster_location=0, T=12, n=3, m=1)
    #
    # print(r.trees[0])
    #
    # print(r.classify({1: 'a', 2: 'y'}))
    #
    # exit(0)

    with open('/users/claytonknittel/pycharmprojects/trafficflow/data/allegheny.csv') as f:
        c = csv.DictReader(f, delimiter=',')

        v = []
        count = 0
        tester = None
        for f in c:
            if count == 1:
                tester = f
            if f['severity'] == '-1':
                continue
            v.append({})
            for k, val in f.items():
                v[-1][k] = val
            count += 1
            if count == -1:
                break
        # print(list(x['speed limit'] for x in v))

        l = list(c.fieldnames)
        print(all_labels_with_probs(v, attr='severity'))
        l.remove('severity')
        l.remove('street name')
        l.remove('injuries')
        l.remove('deaths')
        l.remove('time')
        l.remove('lat')
        l.remove('lon')
        l.remove('day')
        l.remove('month')
        l.remove('year')
        # l.remove('collision type')
        tree = random_forest(v, cluster_location='severity', attrlist=l, min_leaf_size=1000, T=60, n=10000, m=int(len(l) / 3))
        # tree = random_forest(v, cluster_location='severity', attrlist=l, min_leaf_size=1000, T=10, n=50000, m=10)
        # print(tree)
        print(tester)
        print('clas', tree.classify(tester))

