import colorsys


def scale_lightness(rgb, scale_l):
    """
    Lightness scaler

    Takes a given color in rgb-format and scales changes it lightness, due to a given parameter n.

    Arguments:
        rgb -- rgb color
        scale_l -- lightness scaling value

    Returns:
        tuple (r, g, b)
    """
    # cite start
    # code was puplished on
    # https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    h, l, s = colorsys.rgb_to_hls(*rgb)  # convert rgb to hls
    c = colorsys.hls_to_rgb(
        h, min(1, l * scale_l), s=s
    )  # manipulate h, l, s values and return as rgb

    # cite end
    return c
