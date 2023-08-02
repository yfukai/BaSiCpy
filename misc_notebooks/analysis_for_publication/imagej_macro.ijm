open("/work/fukai/basicpy/BaSiCPy/misc_notebooks/analysis_for_publication/testdata_for_imagej/wsi_brain.tif");
selectImage("wsi_brain.tif");
//run("BaSiC ", "processing_stack=wsi_brain.tif flat-field=None dark-field=None shading_estimation=[Estimate shading profiles] shading_model=[Estimate flat-field only (ignore dark-field)] setting_regularisationparametes=Manual temporal_drift=Ignore correction_options=[Compute shading only] lambda_flat=0.50 lambda_dark=0.50");
//selectImage("Flat-field:wsi_brain.tif");
//saveAs("Tiff", "/work/fukai/basicpy/BaSiCPy/misc_notebooks/analysis_for_publication/testdata_for_imagej/wsi_brain_flatfield_wo_darkfield.tif");
//run("BaSiC ", "processing_stack=wsi_brain.tif flat-field=None dark-field=None shading_estimation=[Estimate shading profiles] shading_model=[Estimate flat-field only (ignore dark-field)] setting_regularisationparametes=Manual temporal_drift=Ignore correction_options=[Compute shading only] lambda_flat=0.50 lambda_dark=0.50");
//selectImage("Flat-field:wsi_brain.tif");
basic = BaSiC_()
//        basic.exec(
//            input_image, None, None,
//            "Estimate shading profiles", "Estimate both flat-field and dark-field",
//            lambda_estimate, lambda_flat, lambda_dark,
//            "Ignore", "Compute shading only"
//        )