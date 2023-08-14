# @File(label="Select a slide to process") filename
# @Float(label="Flat field smoothing parameter (0 for automatic)", value=0.1) lambda_flat
# @Float(label="Dark field smoothing parameter (0 for automatic)", value=0.01) lambda_dark
# @Integer(label="Estimate Darkfield") get_darkfield

# NOTE... modified from https://github.com/labsyspharm/basic-illumination/blob/master/imagej_basic_ashlar.py

# Takes a slide (or other multi-series BioFormats-compatible file set) and
# generates flat- and dark-field correction profile images with BaSiC. The
# output format is two multi-series TIFF files (one for flat and one for dark)
# which is the input format used by Ashlar.

# Invocation for running from the commandline:
#
# ImageJ --ij2 --headless --run imagej_basic_ashlar.py "filename='input.ext',output_dir='output',experiment_name='my_experiment'"

import sys
from ij import IJ, WindowManager, Prefs
from ij.macro import Interpreter
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from loci.formats import ImageReader
from loci.formats.in import DynamicMetadataOptions
import BaSiC_ as Basic
import time

import pdb

def main():
    _get_darkfield = get_darkfield>0
    print(_get_darkfield)
    print(filename)
    
    # The internal initialization of the BaSiC code fails when we invoke it via
    # scripting, unless we explicitly set a the private 'noOfSlices' field.
    # Since it's private, we need to use Java reflection to access it.
    options = ImporterOptions()
    options.id = str(filename)
    input_image = BF.openImagePlus(options)[0]
    dims = input_image.getDimensions()
    print(dims)
    Basic_noOfSlices = Basic.getDeclaredField('noOfSlices')
    Basic_noOfSlices.setAccessible(True)

    basic = Basic()
    Basic_noOfSlices.setInt(basic, dims[4])
    WindowManager.setTempCurrentImage(input_image)
    lambda_estimate = "Manual"

    basic.exec(
        input_image, None, None,
        "Estimate shading profiles", 
        "Estimate both flat-field and dark-field" if _get_darkfield else "Estimate flat-field only (ignore dark-field)",
        lambda_estimate, lambda_flat, lambda_dark,
        "Ignore", "Compute shading only"
    )

    basic = Basic()
    Basic_noOfSlices.setInt(basic, dims[4])
    WindowManager.setTempCurrentImage(input_image)
    lambda_estimate = "Manual"
    start = time.time()
    basic.exec(
        input_image, None, None,
        "Estimate shading profiles", 
        "Estimate both flat-field and dark-field" if _get_darkfield else "Estimate flat-field only (ignore dark-field)",
        lambda_estimate, lambda_flat, lambda_dark,
        "Ignore", "Compute shading only"
    )
    stop = time.time()
    print("erapsed time: %s" % (stop - start))
    input_image.close()

    stem=str(filename).split("/")[-1].split(".")[0]
    output_dir="/".join(str(filename).split("/")[:-1])
    suffix = "with_darkfield" if _get_darkfield else "no_darkfield"

    print(stem)
    flatfield = WindowManager.getImage("Flat-field:%s" % input_image.title)
    IJ.saveAsTiff(flatfield, str(output_dir) + "/%s_flatfield_%s.tif"%(stem, suffix))

    if _get_darkfield:
        darkfield = WindowManager.getImage("Dark-field:%s" % input_image.title)
        IJ.saveAsTiff(darkfield, str(output_dir) + "/%s_darkfield_%s.tif"%(stem, suffix))
main()