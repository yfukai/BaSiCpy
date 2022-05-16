#%%
"""jax_implementation_experimental.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KyM-VBLB3uZ8qjaOyN0V7-xBmwP3Oux-

# install and load data
"""


# copy from basicpy.data

import pooch
from os import path
from skimage.io import imread
import glob

EXPERIMENTAL_TEST_DATA_PROPS = {
    "cell_culture": {
        "filename": "Cell_culture.zip",
        "hash": "md5:797bbc4c891e5fe59f4200e771b55c3a",
    },
    "timelapse_brightfield": {
        "filename": "Timelapse_brightfield.zip",
        "hash": "md5:460e5f78ac69856705704fedad9f9e59",
    },
    "timelapse_nanog": {
        "filename": "Timelapse_nanog.zip.zip",
        "hash": "md5:815d53cac35b671269b17bd627d7baa7",
    },
    "timelapse_pu1": {
        "filename": "Timelapse_Pu1.zip.zip",
        "hash": "md5:bee97561e87c51e90b46da9b439e8b7b",
    },
    "wsi_brain": {
        "filename": "WSI_Brain.zip",
        "hash": "md5:6e163786ddec2a690aa4bb47a64bcded",
    },
}

POOCH = pooch.create(
    path=pooch.os_cache("basicpy"),
    # Use the Zenodo DOI
    base_url="doi:10.5281/zenodo.6334810/",
    registry={v["filename"]: v["hash"] for v in EXPERIMENTAL_TEST_DATA_PROPS.values()},
)

def fetch(data_name: str):
    """Fetch a sample dataset from Zenodo.
    Args:
        data_name: The name of the dataset. Must be one of ["cell_culture",
            "timelapse_brightfield", "timelapse_nanog", "timelapse_pu1",
            "wsi_brain"].

    Returns:
        Iterable[ndarray]: An iterable of uncorrected images.
        Iterable[ndarray]: An iterable of corrected images by the reference
            implementation (Tingying Peng et al., Nature Communication 8:14836 (2017).)

    Raises:
        ValueError: If the dataset name is not one of the allowed values.
    """
    if data_name not in EXPERIMENTAL_TEST_DATA_PROPS.keys():
        raise ValueError(f"{data_name} is not a valid test data name")
    file_name = EXPERIMENTAL_TEST_DATA_PROPS[data_name]["filename"]
    test_file_paths = POOCH.fetch(file_name, processor=pooch.Unzip())
    assert all(path.exists(f) for f in test_file_paths)
    basedir = path.commonpath(test_file_paths)
    uncorrected_paths = sorted(
        glob.glob(path.join(basedir, "Uncorrected*", "**", "*.tif"), recursive=True)
    )
    if len(uncorrected_paths) == 0:
        uncorrected_paths = sorted(
            glob.glob(path.join(basedir, "Uncorrected*", "**", "*.png"), recursive=True)
        )
    corrected_paths = sorted(
        glob.glob(path.join(basedir, "Corrected*", "**", "*.tif"), recursive=True)
    )
    if "WSI_Brain" in file_name:
        uncorrected_paths = list(
            filter(lambda p: "BrainSection" in p, uncorrected_paths)
        )
        corrected_paths = list(filter(lambda p: "BrainSection" in p, corrected_paths))

    assert len(uncorrected_paths) > 0
    assert len(uncorrected_paths) == len(corrected_paths)
    uncorrected = (imread(f) for f in uncorrected_paths)
    corrected = (imread(f) for f in corrected_paths)

    return uncorrected, corrected
#%%
import numpy as np
from matplotlib import pyplot as plt
from jax import numpy as jnp
newax = jnp.newaxis
from skimage.transform import downscale_local_mean

images=np.array(list(fetch("wsi_brain")[0]))
images = np.array([downscale_local_mean(im,(4,4)) for im in images])
print(images.shape)
# %%
plt.imshow(images[10])
#%%
"""# test original implementation"""

images2=np.swapaxes(images,0,-1).astype(np.float32)
W=np.ones_like(images2,dtype=np.float32)
print(images2.shape,W.shape,lambda_flatfield,lambda_darkfield)
#%%
%%time
A1_hat, E1_hat, A_offset, stopCriterion=basicpy.tools.inexact_alm.inexact_alm_rspca_l1(
    images2,
    weight=W,
    lambda_flatfield=lambda_flatfield,
    lambda_darkfield=lambda_darkfield,
    get_darkfield=False,
    optimization_tol=1e-4,
    max_iterations=500,
)
print(A1_hat.shape,E1_hat.shape,A_offset.shape,stopCriterion)
X_A = np.reshape(A1_hat, images2.shape[:2] + (-1,), order="F")
X_E = np.reshape(E1_hat, images2.shape[:2] + (-1,), order="F")
X_A_offset = np.reshape(A_offset, images2.shape[:2], order="F")
print(X_A.shape,X_E.shape,X_A_offset.shape)
flatfield_flatonly_original = np.mean(X_A, axis=2) - X_A_offset

#%%
%%time
A1_hat, E1_hat, A_offset, stopCriterion=basicpy.tools.inexact_alm.inexact_alm_rspca_l1(
    images2,
    weight=W,
    lambda_flatfield=lambda_flatfield,
    lambda_darkfield=lambda_darkfield,
    get_darkfield=True,
    optimization_tol=1e-4,
    max_iterations=500,
)
print(A1_hat.shape,E1_hat.shape,A_offset.shape,stopCriterion)
X_A = np.reshape(A1_hat, images2.shape[:2] + (-1,), order="F")
X_E = np.reshape(E1_hat, images2.shape[:2] + (-1,), order="F")
X_A_offset = np.reshape(A_offset, images2.shape[:2], order="F")
print(X_A.shape,X_E.shape,X_A_offset.shape)
flatfield_withdark_original = np.mean(X_A, axis=2) - X_A_offset
darkfield_withdark_original = X_A_offset

#%%
from basicpy import BaSiC
b=BaSiC(get_darkfield=True,max_reweight_iterations=10,fitting_mode="ladmap")
b.fit(images)
plt.imshow(b.flatfield)
plt.colorbar()
plt.show()
plt.imshow(b.darkfield)
plt.colorbar()
plt.show()
#for w in b._weight:
#    plt.imshow(w)
#    plt.colorbar()
#    plt.show()
#%%
from basicpy import BaSiC
b=BaSiC(get_darkfield=True,max_reweight_iterations=1,fitting_mode="approximate")
b.fit(images)
plt.imshow(b.flatfield)
plt.colorbar()
plt.show()
plt.imshow(b.darkfield)
plt.colorbar()
plt.show()


# %%
