import pandas as pd
import rasterio as rio
from rasterio.plot import reshape_as_raster, reshape_as_image
import numpy as np
import os
import math
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import threading
from tqdm.auto import tqdm
from pathlib import Path

def fcToCsv(fc, properties):
    """
    Convert a featureCollection to a pandas dataframe
    
    Args: 
        fc (ee.FeatureCollection): A feature collection with properties to convert to a dataframe.
        properties (ee.List): The property names to extract
    
    Returns:
        A Pandas DataFrame with each row a feature and each column a property.
        
    Examples
        fcToCsv(training, training.first().propertyNames()).drop('system:index', axis=1)
    """
    props = fc.select(properties).first().propertyNames()
    output = fc.map(lambda ft: ft.set('output', props.map(lambda prop: ft.get(prop))))
    result = output.aggregate_array('output').getInfo()
    result = pd.DataFrame(result, columns= props.getInfo())
    return result

def inference(infile, model, outfile, patchSize, num_workers=4):
    """
    Run inference using model on infile block-by-block and write to a new file (outfile). 
    In the case, that the infile image width/height is not exactly divisible by 32, padding
    is added for inference and removed prior to the outfile being saved.
    
    Args:
        infile (string): Path to input image/covariates
        model (pth file): Loaded trained model/checkpoint
        outfile (string): Path to save predicted image
        patchSize (int): Must be a multiple of 32. Size independent of model input size.
        num_workers (int): Num of workers to parralelise across
        
    Returns:
        A tif saved to the outfile destination
        
    """

    with rio.open(infile) as src:
        
        logger = logging.getLogger(__name__)

        # Create a destination dataset based on source params. The
        # destination will be tiled, and we'll process the tiles
        # concurrently.
        profile = src.profile
        profile.update(blockxsize= patchSize, blockysize= patchSize, tiled=True, count=1)

        with rio.open(Path(outfile), "w", **profile) as dst:
            windows = [window for ij, window in dst.block_windows()]

            # use a lock to protect the DatasetReader/Writer
            read_lock = threading.Lock()
            write_lock = threading.Lock()

            def process(window):
                with read_lock:
                    src_array = src.read(window=window)
                    i_arr = reshape_as_image(src_array)

                    #Format input_image for inference
                    nPixels = i_arr.shape[0]*i_arr.shape[1]
                    nBands = i_arr.shape[-1]
                    # Take full image and reshape into long 2d array (nrow * ncol, nband) for classification
                    new_arr = i_arr.reshape(nPixels, nBands)#reshape 3d array to 2d array that matches the training data table from earlier
                    bandnames = list(src.descriptions)
                    result_ = model.predict(pd.DataFrame(new_arr, columns = bandnames).fillna(0))
                    # # Reshape our classification map back into a 2D matrix so we can save it as an image
                    result = result_.reshape(i_arr[:, :, 0].shape).astype(np.float64)

                with write_lock:
                    dst.write(result, 1, window=window)
                    # bar.update(1)

            # We map the process() function over the list of
            # windows.
            with tqdm(total=len(windows), desc = os.path.basename(outfile)) as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # executor.map(process, windows)
                    futures = {executor.submit(process, window): window for window in windows}
                    try:
                        for future in concurrent.futures.as_completed(futures):
                            future.result()
                            pbar.update(1)

                    except Exception as ex:
                        logger.info('Cancelling...')
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise ex  