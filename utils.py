import taichi as ti

def pixelToRay(camera, px, py, width, height):
    # Get coordinates
    view = (camera.lookat() - camera.position()).normalized()
    horizontal = (view.cross(camera.up())).normalized()
    vertical = (horizontal.cross(camera.view())).normalized()

    length = 1.0

    rad = ti.math.radians(camera.fov)

    vLength = ti.math.tan(rad / 2.0) * length
    hLength = vLength * width / height
    vertical = vertical*vLength
    horizontal = horizontal*hLength

    #  Translate mouse coordinates so that the origin lies in the center of the view port
    x = px - width / 2.0
    y = height / 2.0 - py

    # Scale mouse coordinates so that half the view port width and height becomes 1.0
    x /= width / 2.0
    y /= height / 2.0

    # Direction is a linear combination to compute intersection of picking ray with view port plane
    return camera.position(), (view * length + horizontal * x + vertical * y).normalized()