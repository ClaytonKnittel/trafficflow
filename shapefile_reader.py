from input import *
import pygame
from map import geomap
from accidents import *

g = geomap('/users/claytonknittel/downloads/Pennsylvania', 'rtrees/pennsylvania')
data = data_generator('/users/claytonknittel/downloads/alleghenyAccidents.csv')

shaps = []

def inside(bbox1, bbox2):
    return bbox1[2] >= bbox2[0] and bbox1[3] >= bbox2[1] and bbox1[0] <= bbox2[2] and bbox1[1] <= bbox2[3]


run = True

width = 640
height = 480

# bb = [-90.34, 38.63, -90.3, 38.67]
# bb = [-90.6, 38.56, -90.56, 38.6]
allegheny_bbox = [-80.4866, 40.0970993, -74.94589996, 41.01869965]
bb = [-80., 40.5, -79.8, 40.65]

print(g.shapes._sfiles[0].fields)

pygame.init()
screen = pygame.display.set_mode((width, height))
screen.convert()
pygame.display.set_caption('Map')

for shape in g.rtree.intersection(bb):
    shaps.append(g.shapes.shape(shape))

print(len(shaps), len(data))

# b = [-91, 38, -89, 40]

print(g.bbox)

cam = camera(bb, width, height)

draw_bb = False

def swap():
    global draw_bb
    draw_bb = not draw_bb

def draw(draw, screen, shape):
    pts = shape.points

    if draw_bb:
        b = shape.bbox
        draw.line(screen, (255, 0, 0), cam.screenCoords((b[0], b[1])), cam.screenCoords((b[0], b[3])))
        draw.line(screen, (255, 0, 0), cam.screenCoords((b[0], b[3])), cam.screenCoords((b[2], b[3])))
        draw.line(screen, (255, 0, 0), cam.screenCoords((b[2], b[3])), cam.screenCoords((b[2], b[1])))
        draw.line(screen, (255, 0, 0), cam.screenCoords((b[2], b[1])), cam.screenCoords((b[0], b[1])))

    pt = pts[0]
    for pt2 in pts[1:]:
        draw.line(screen, (0, 0, 0), cam.screenCoords(pt), cam.screenCoords(pt2))
        pt = pt2

kl = keyListener()
kl.pressAction(pygame.K_w, lambda: cam.move(dy=.05))
kl.pressAction(pygame.K_s, lambda: cam.move(dy=-.05))
kl.pressAction(pygame.K_d, lambda: cam.move(dx=.05))
kl.pressAction(pygame.K_a, lambda: cam.move(dx=-.05))
kl.tapAction(pygame.K_SPACE, lambda: swap())

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYDOWN:
            kl.pollEvent(event)
        elif event.type == pygame.KEYUP:
            kl.pollEvent(event)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                shap = g.shapes.shapeRecord(list(g.rtree.nearest(cam.worldCoords(pygame.mouse.get_pos())))[0])
                print(shap.record['FULL_STREE'])
            if event.button == 4:
                cam.zoom_in(1.03)
            elif (event.button == 5):
                cam.zoom_in(.97)
    kl.act()
    screen.fill((20, 240, 250))

    for a in data:
        if not inside(a.bbox(), cam.bbox()):
            continue
        p = cam.screenCoords(a.pos)
        pp = (int(p[0]), int(p[1]))
        pygame.draw.circle(screen, (230, 40, 20), pp, 2)
    for s in shaps:
        if not inside(s.bbox, cam.bbox()):
            continue
        draw(pygame.draw, screen, s)

    pygame.display.flip()

pygame.display.quit()
pygame.quit()
