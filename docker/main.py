import click
import numpy as np
from aicsimageio import AICSImage
from skimage.io import imsave

from basicpy import BaSiC


@click.command
@click.argument("input_path")
@click.argument("output_folder")
@click.option("smoothness_flatfield")
@click.option("smoothness_darkfield")
def main(
    input_path,
    output_path,
    smoothness_flatfield,
    smoothness_darkfield,
    sparse_cost_darkfield,
):
    basic = BaSiC(
        smoothness_flatfield=smoothness_flatfield,
        smoothness_darkfield=smoothness_darkfield,
        sparse_cost_darkfield=sparse_cost_darkfield,
    )
    images = AICSImage(input_path).data
    images_data = [images[ind] for ind in np.ndindex(images.shape[:-4])]
    basic.fit(images_data)
    flatfield_path = output_path / input_path + "-ffp.tiff"
    darkfield_path = output_path / input_path + "-dfp.tiff"
    imsave(flatfield_path, basic.flatfield)
    imsave(darkfield_path, basic.darkfield)
