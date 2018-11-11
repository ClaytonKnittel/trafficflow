import shapefile

sf1 = shapefile.Reader('/users/claytonknittel/downloads/Shape/Trans_RoadSegment.shp')
# sf2 = shapefile.Reader('/users/claytonknittel/downloads/Shape/Trans_RoadSegment2.shp')
# sf3 = shapefile.Reader('/users/claytonknittel/downloads/Shape/Trans_RoadSegment3.shp')
# sf4 = shapefile.Reader('/users/claytonknittel/downloads/Shape/Trans_RoadSegment4.shp')
# sf5 = shapefile.Reader('/users/claytonknittel/downloads/Shape/Trans_RoadSegment5.shp')
# sf6 = shapefile.Reader('/users/claytonknittel/downloads/Shape/Trans_RoadSegment6.shp')

print(sf1)
print(sf1.bbox)
print(sf1.shape(1))

shap = sf1.shape(1)

for name in dir(shap):
    if not name.startswith('_'):
        print(name)

print(shap.shapeTypeName)
for point in shap.points:
    print(['%.5f' % coord for coord in point])

fields = sf1.fields
print(fields)

# very large, takes a long time
# records = sf1.records()
# print(len(records))

r1 = sf1.record(1)
print(r1.as_dict())
