
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


