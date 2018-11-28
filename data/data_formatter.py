import csv
from datetime import datetime
from map import geomap


attrrow = ['severity', 'deaths', 'injuries', 'passengers', 'lat', 'lon', 'time', 'collision type', 'road type',
           'year', 'month', 'day', 'street name', 'speed limit', 'lane count', 'urban/rural', 'intersection type',
           'weather', 'road condition', 'illumination', 'location', 'unbelted', 'driver 16', 'driver 17',
           'driver 18', 'driver 19', 'driver 20', 'driver 50-64', 'driver 65-74', 'driver 75+']


class severity:
    no_injury = 0
    minor_injury = 1
    moderate_injury = 2
    major_injury = 3
    fatal = 4
    unknown = -1


class col_type:
    no_collision = 0
    rear_end = 1
    head_on = 2
    rear_to_rear = 3
    angle = 4
    sideswipe_same_dir = 5
    sideswipe_opp_dir = 6
    hit_fixed_obj = 7
    hid_ped = 8
    other = 9


class road_type:
    interstate = 0
    US_route = 1
    state_route = 2
    county_route = 3
    local_street = 4
    private = 5
    unknown = -1


class urban_rural:
    rural = 0
    small_urban = 1
    urban = 2
    large_urban = 3


class intersection_type:
    mid_block = 0
    four_way_intersection = 1
    t_intersection = 2
    y_intersection = 3
    roundabout = 4
    multi_leg_intersection = 5
    on_ramp = 6
    off_ramp = 7
    crossover = 8
    railroad_crossing = 9
    unknown = -1


class weather:
    clear = 0
    rain = 1
    sleet = 2
    snow = 3
    fog = 4
    rain_and_fog = 5
    sleet_and_fog = 6
    unknown = -1


class road_condition:
    dry = 0
    wet = 1
    sand_mud_dirt_oil_gravel = 2
    snow = 3
    slush = 4
    ice = 5
    ice_patches = 6
    water = 7
    unknown = -1


class illumination:
    daylight = 0
    dark_no_street_lights = 1
    dark_street_lights = 2
    dusk = 3
    dawn = 4
    dark_unknown_road_lighting = 5
    unknown = -1


class location:
    na = 0
    underpass = 1
    ramp = 2
    bridge = 3
    tunnel = 4
    toll_booth = 5
    crossover = 6
    driveway_parkinglot = 7
    ramp_and_bridge = 8
    unknown = -1


def colon_sep_vals_to_csv(path):
    f = open(path, 'r')

    strs = f.readlines()
    newstrs = []

    for str in strs:
        newstrs.append(str.replace(',', '/').replace(';', ','))

    f.close()
    f = open(path, 'w')

    for str in newstrs:
        f.write(str)


def fill_in_street_names(csv_path, geomap, new_csv_path=None):
    rtree = geomap.rtree
    if new_csv_path is None:
        new_csv_path = csv_path
    with open(csv_path) as file:
        f = csv.DictReader(file, delimiter=',')
        with open(new_csv_path, 'w') as file2:
            wr = csv.DictWriter(file2, fieldnames=attrrow, delimiter=',')
            wr.writeheader()
            for row in f:
                loc = [row['lon'], row['lat']]

                if loc[0] == '' or loc[1] == '':
                    continue
                loc = [float(f) for f in loc]
                dic = row.copy()
                for road in rtree.nearest(loc, 10):
                    rec = geomap.shapes.record(road)['FULL_STREE']
                    if rec != '':
                        dic['street name'] = rec
                wr.writerow(dic)


if __name__ == '__main__':
    # g = geomap('/users/claytonknittel/downloads/Shape', '/users/claytonknittel/PycharmProjects/trafficflow/rtrees/north_carolina')
    # fill_in_street_names('/users/claytonknittel/downloads/test.csv', g, '/users/claytonknittel/downloads/test2.csv')

    with open('/users/claytonknittel/downloads/alleghenyAccidents.csv', 'r') as file:
        f = csv.DictReader(file, delimiter=',')
        with open('/users/claytonknittel/PycharmProjects/trafficflow/data/allegheny.csv', 'w') as file:
            wr = csv.DictWriter(file, fieldnames=attrrow, delimiter=',')
            wr.writeheader()
            for row in f:
                dic = {}
                if row['MAX_SEVERITY_LEVEL'] == '0':
                    dic['severity'] = severity.no_injury
                elif row['MAX_SEVERITY_LEVEL'] == '1':
                    dic['severity'] = severity.fatal
                elif row['MAX_SEVERITY_LEVEL'] == '2':
                    dic['severity'] = severity.major_injury
                elif row['MAX_SEVERITY_LEVEL'] == '3':
                    dic['severity'] = severity.moderate_injury
                elif row['MAX_SEVERITY_LEVEL'] == '4':
                    dic['severity'] = severity.minor_injury
                else:
                    dic['severity'] = severity.unknown

                dic['deaths'] = row['FATAL_COUNT']
                dic['injuries'] = row['INJURY_COUNT']

                passengers = row['PERSON_COUNT']
                if passengers == '':
                    dic['passengers'] = 0
                else:
                    dic['passengers'] = passengers

                lat = row['DEC_LAT']
                lon = row['DEC_LONG']
                dic['lat'] = lat
                dic['lon'] = lon

                dic['year'] = row['CRASH_YEAR']
                dic['month'] = row['CRASH_MONTH']
                if row['TIME_OF_DAY'] == '9999' or len(row['TIME_OF_DAY']) < 3:
                    dic['time'] = -1
                else:
                    t = row['TIME_OF_DAY']
                    if len(t) == 3:
                        t = '0' + t
                    dt = datetime.strptime(t, '%H%M')
                    dic['time'] = dt.hour * 60 + dt.minute

                dic['lane count'] = row['LANE_COUNT']
                dic['speed limit'] = row['SPEED_LIMIT']
                dic['urban/rural'] = row['URBAN_RURAL']

                int_type = int(row['INTERSECT_TYPE'])
                if int_type == 0:
                    dic['intersection type'] = intersection_type.mid_block
                elif int_type == 1:
                    dic['intersection type'] = intersection_type.four_way_intersection
                elif int_type == 2:
                    dic['intersection type'] = intersection_type.t_intersection
                elif int_type == 3:
                    dic['intersection type'] = intersection_type.y_intersection
                elif int_type == 4:
                    dic['intersection type'] = intersection_type.roundabout
                elif int_type == 5:
                    dic['intersection type'] = intersection_type.multi_leg_intersection
                elif int_type == 6:
                    dic['intersection type'] = intersection_type.on_ramp
                elif int_type == 7:
                    dic['intersection type'] = intersection_type.off_ramp
                elif int_type == 8:
                    dic['intersection type'] = intersection_type.crossover
                elif int_type == 9:
                    dic['intersection type'] = intersection_type.railroad_crossing
                elif int_type == 10 or int_type == 99:
                    dic['intersection type'] = intersection_type.unknown
                else:
                    print(int_type)

                if row['WEATHER'] != '':
                    cond = int(row['WEATHER'])
                    if cond >= 8:
                        dic['weather'] = weather.unknown
                    else:
                        dic['weather'] = cond

                dic['collision type'] = row['COLLISION_TYPE']
                dic['street name'] = row['STREET_NAME']

                o = row['ROAD_OWNER']
                if o == '1' or o == '5' or o == '6':
                    dic['road type'] = road_type.interstate
                elif o == '2':
                    dic['road type'] = road_type.state_route
                elif o == '3':
                    dic['road type'] = road_type.county_route
                elif o == '4':
                    dic['road type'] = road_type.local_street
                elif o == '7':
                    dic['road type'] = road_type.private
                else:
                    dic['road type'] = road_type.unknown

                o = row['ROAD_CONDITION']
                try:
                    if int(o) > 8:
                        o = road_condition.unknown
                except ValueError:
                    pass
                if o == '':
                    o = road_condition.unknown
                dic['road condition'] = o

                o = row['ILLUMINATION']
                try:
                    o = int(o) - 1
                    if o >= 8:
                        o = illumination.unknown
                except ValueError:
                    pass
                if o == '':
                    o = illumination.unknown
                dic['illumination'] = o

                o = row['LOCATION_TYPE']
                try:
                    o = int(o)
                    if o > 8:
                        o = location.unknown
                except ValueError:
                    pass
                if o == '':
                    o = location.unknown
                dic['location'] = o

                dic['unbelted'] = row['UNBELTED']
                dic['driver 16'] = row['DRIVER_16YR']
                dic['driver 17'] = row['DRIVER_17YR']
                dic['driver 18'] = row['DRIVER_18YR']
                dic['driver 19'] = row['DRIVER_19YR']
                dic['driver 20'] = row['DRIVER_20YR']
                dic['driver 50-64'] = row['DRIVER_50_64YR']
                dic['driver 65-74'] = row['DRIVER_65_74YR']
                dic['driver 75+'] = row['DRIVER_75PLUS']

                wr.writerow(dic)

    # with open('/users/claytonknittel/downloads/cpd-crash-incidents2.csv', 'r') as file:
    #     f = csv.DictReader(file, delimiter=',')
    #     with open('/users/claytonknittel/downloads/test.csv', 'w') as file:
    #         wr = csv.DictWriter(file, fieldnames=attrrow, delimiter=',')
    #         wr.writeheader()
    #         for row in f:
    #             dic = {}
    #
    #             deaths = int(row['fatality'])
    #             inj = int(row['possblinj'])
    #             if deaths > 0:
    #                 if deaths > 1:
    #                     dic['severity'] = severity.multiple_fatal
    #                 else:
    #                     dic['severity'] = severity.fatal
    #             elif inj > 0:
    #                 if inj > 1:
    #                     dic['severity'] = severity.major_injury
    #                 else:
    #                     dic['severity'] = severity.minor_injury
    #             else:
    #                 dic['severity'] = severity.no_injury
    #
    #             dic['deaths'] = deaths
    #             dic['injuries'] = inj
    #
    #             passengers = row['numpassengers']
    #             if passengers == '':
    #                 dic['passengers'] = 0
    #             else:
    #                 dic['passengers'] = passengers
    #
    #             lat = row['lat']
    #             lon = row['lon']
    #             if lat == '' or lon == '':
    #                 lat = row['lat2']
    #                 lon = row['lon2']
    #             dic['lat'] = lat
    #             dic['lon'] = lon
    #
    #             rdtype = row['rdclass']
    #             if rdtype == '"PUBLIC VEHICULAR AREA"':
    #                 dic['road type'] = road_type.offroad
    #             elif rdtype == '"NC ROUTE"':
    #                 dic['road type'] = road_type.state_route
    #             elif rdtype == '"STATE SECONDARY ROUTE"':
    #                 dic['road type'] = road_type.state_route
    #             elif rdtype == '"US ROUTE"':
    #                 dic['road type'] = road_type.US_route
    #             elif rdtype == '"LOCAL STREET"':
    #                 dic['road type'] = road_type.local_street
    #             elif rdtype == '"INTERSTATE"':
    #                 dic['road type'] = road_type.interstate
    #             elif rdtype == '"PRIVATE ROAD,DRIVEWAY"':
    #                 dic['road type'] = road_type.private
    #             elif rdtype == '' or rdtype == '"OTHER *"':
    #                 pass
    #             else:
    #                 print(rdtype)
    #
    #             dt = datetime.strptime(row['crash_date'][:-6], '%Y-%m-%d %H:%M:%S')
    #             dic['year'] = dt.year
    #             dic['month'] = dt.month
    #             dic['day'] = dt.day
    #             dic['time'] = dt.hour * 60 + dt.minute
    #
    #             int_type = row['rdfeature']
    #             if int_type == 'FOUR-WAY INTERSECTION':
    #                 dic['intersection type'] = intersection_type.four_way_intersection
    #             elif int_type == 'T-INTERSECTION':
    #                 dic['intersection type'] = intersection_type.t_intersection
    #             elif int_type == 'RELATED TO INTERSECTION' or int_type == 'NO SPECIAL FEATURE'\
    #                     or int_type == 'ALLEY INTERSECTION' or int_type == 'SHARED-USE PATHS OR TRAILS'\
    #                     or int_type == 'MERGE LANE BETWEEN ON AND OFF RAMP'\
    #                     or int_type == 'NON-INTERSECTION MEDIAN CROSSING' or int_type == 'TUNNEL'\
    #                     or int_type == 'END OR BEGINNING-DIVIDED HIGHWAY':
    #                 dic['intersection type'] = intersection_type.mid_block
    #             elif int_type == 'FOUR-WAY INTERSECTION':
    #                 dic['intersection type'] = intersection_type.four_way_intersection
    #             elif int_type == 'DRIVEWAY/ PUBLIC' or int_type == 'DRIVEWAY/ PRIVATE':
    #                 dic['intersection type'] = intersection_type.mid_block
    #             elif int_type == 'Y-INTERSECTION':
    #                 dic['intersection type'] = intersection_type.y_intersection
    #             elif int_type == 'OFF-RAMP PROPER' or int_type == 'OFF-RAMP TERMINAL ON CROSSROAD' or int_type == 'OFF-RAMP ENTRY':
    #                 dic['intersection type'] = intersection_type.off_ramp
    #             elif int_type == 'OTHER *' or int_type == '':
    #                 dic['intersection type'] = intersection_type.unknown
    #             elif int_type == 'ON-RAMP ENTRY' or int_type == 'ON-RAMP TERMINAL ON CROSSROAD' or int_type == 'ON-RAMP PROPER':
    #                 dic['intersection type'] = intersection_type.on_ramp
    #             elif int_type == 'RAILROAD CROSSING':
    #                 dic['intersection type'] = intersection_type.railroad_crossing
    #             elif int_type == 'BRIDGE APPROACH' or int_type == 'BRIDGE' or int_type == 'UNDERPASS':
    #                 dic['intersection type'] = intersection_type.mid_block
    #             elif int_type == 'TRAFFIC CIRCLE/ROUNDABOUT':
    #                 dic['intersection type'] = intersection_type.roundabout
    #             elif int_type == 'FIVE-POINT/ OR MORE':
    #                 dic['intersection type'] = intersection_type.multi_leg_intersection
    #             else:
    #                 print(int_type)
    #
    #             cond = row['rdcondition']
    #             if cond == 'DRY':
    #                 dic['weather'] = weather.clear
    #             elif cond == 'CLOUDY':
    #                 dic['weather'] = weather.cloudy
    #             elif cond == 'RAIN':
    #                 dic['weather'] = weather.rain
    #             elif cond == 'WET' or cond == 'SLUSH' or cond == 'WATER (STANDING/ MOVING)':
    #                 dic['weather'] = weather.wet
    #             elif cond == 'FOG/ SMOG/ SMOKE':
    #                 dic['weather'] = weather.smog
    #             elif cond == 'SNOW':
    #                 dic['weather'] = weather.snow
    #             elif cond == 'ICE':
    #                 dic['weather'] = weather.ice
    #             elif cond == 'SLEED/ HAIL':
    #                 dic['weather'] = weather.sleet
    #             elif cond == '' or cond == 'UNKNOWN' or cond == 'OTHER *' or cond == 'SAND/ MUD/ DIRT/ GRAVEL':
    #                 dic['weather'] = weather.unknown
    #             else:
    #                 print(cond)
    #
    #             wr.writerow(dic)

