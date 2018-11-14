from rtree import index

i = index.Rtree()
file_i = index.Rtree('rtrees/rtree_ex')

i.insert(0, (0, 0, 1, 1))
# file_i.insert(0, (0, 0, 1, 1), obj=objex('hille'))

l = i.intersection((1, 1, 2, 2))
l2 = file_i.intersection((1, 1, 2, 2))

print([n.object for n in i.intersection((1, 1, 2, 2), objects=True)])
print([n.object.name for n in file_i.intersection((1, 1, 2, 2), objects=True)])

