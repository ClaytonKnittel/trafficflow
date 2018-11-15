from accidents import *
from map import geomap

# here you need to supply two things. The first is the location of the folder
# containing all of the shape (.shp) files that you can download from
# https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Tran/Shape/
#
# the second thing is the location of the r tree data for those shape files,
# which I have been able to include in the github, as they are only a measly
# 100 MB or so (as opposed to the shapefile folders which are about
# 4-20 GB each)
#
# if you need r trees for more states, just ask me and I can make it
g = geomap('/users/claytonknittel/downloads/Pennsylvania', 'rtrees/pennsylvania')

# here, you need to supply at least one source of traffic accident data. So far,
# the only format supported is the one used by Allegheny County, so download
# the csv files from
# https://catalog.data.gov/dataset/allegheny-county-crash-data
data = data_generator('/users/claytonknittel/downloads/alleghenyAccidents.csv')


# set the ratio of testing data to training data. 9 or 10 was
# the one he suggested in class
data.set_cross_validation_ratio(9)


# get a random sample of 10 data points and print out the street
# name on which the accident happened and the speed limit of that street
for datapt in data.random_sample(10):
    print(datapt.street_name, datapt.speed_limit)

print()

print('average speed limits from 3 samples of size 10:')
# print the average speed limit from 3 random
# samples, each of size 10. Note, for random sampling,
# all data is chosen without replacement (even across
# multiple batches)
for mini_batch in data.random_samples(10, 3):
    avg_speed_limit = 0
    for datapt in mini_batch:
        avg_speed_limit += datapt.speed_limit
    print(avg_speed_limit / 10)


print()

print('bootstrap sample, print number of deaths')
# can also take a bootstrap sample, simply meaning
# you choose with replacement
for datapt in data.bootstrap_sample(10):
    print(datapt.deaths)


# look in accidents.py under _acc to see what other
# attributes datapts have


# additionally, you can find the road segment from the r-tree
# corresponding to the site of the accident via the get_road method.
# You must supply the geomap that the road will be coming from along
# with the accident object (datapt)
datapt = list([d for d in data.random_sample(1)])[0]
road = get_road(g, datapt)

print()

# note, the get_road method returns a ShapeRecord object. These have two
# instance variables, a Shape object (representing the geometric data of
# the road) and a Record object (which contains the qualitative and
# descriptive data of the road, like its length, name, etc)
# you can access the two with road.shape and road.record, respectively

# Record objects behave like dictionaries. To see what attributes Record
# objects for a certain geomap have, you can invoke
print(g.shapes.fields())
# And to see the value of some attribute of a road, for example, its name,
# you can call
print(road.record['FULL_STREE'])
