

def numDifferences(v1, v2, attrlist):
    count = 0
    for attr in attrlist:
        if v1[attr] != v2[attr]:
            count += 1
    return count


def k_means(vectors, k, attrlist=None, similarity_measure=None):

    if similarity_measure is None:
        similarity_measure = numDifferences
    if attrlist is None:
        attrlist = list(range(0, len(vectors[0])))

    supports = []
    for x in range(0, k):
        supports.append([])
        for y in range(0, len(attrlist)):
            pass

