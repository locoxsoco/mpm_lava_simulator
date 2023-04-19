import math

def pixelToRay(camera, px, py, width, height):
    # Get coordinates
    view = (camera.curr_lookat - camera.curr_position).normalized()
    horizontal = (view.cross(camera.curr_up)).normalized()
    vertical = (horizontal.cross(view)).normalized()
    # print(f'view: {view} horizontal: {horizontal} vertical: {vertical}')

    length = 1.0

    rad = math.radians(55)

    vLength = math.tan(rad / 2.0) * length
    hLength = vLength * width / height
    vertical = vertical*vLength
    horizontal = horizontal*hLength

    #  Translate mouse coordinates so that the origin lies in the center of the view port
    x = px - width / 2.0
    y = py - height / 2.0
    # print(f'px: {px} py: {py} x: {x} y: {y}')

    # Scale mouse coordinates so that half the view port width and height becomes 1.0
    x /= width / 2.0
    y /= height / 2.0
    # print(f'px: {px} py: {py} new_x: {x} new_y: {y}')

    # Direction is a linear combination to compute intersection of picking ray with view port plane
    return camera.curr_position, (view * length + horizontal * x + vertical * y).normalized()