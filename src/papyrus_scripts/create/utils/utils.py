# -*- coding: utf-8

import os
import glob
from typing import Callable, Iterator
from shortuuid import ShortUUID

from pandas.io.parsers import TextFileReader as PandasTextFileReader
import pystow


def process_and_write_chunks(data: PandasTextFileReader, func: Callable, dest: str, *pystow_prefixes, **kwargs) -> None:
    """Process incoming chunks of data using the specified function and wrtie the results into the given destination.

    :param data: iterator of data to be processed
    :param func: function to apply to the incoming data
    :param dest: file to write the processed result to
    :param pystow_prefixes: prefixes to store the destination file into
    :param kwargs: keywords arguments to be passed to pystow
    """
    # Create temporary file
    temp_file = f'{dest}.{ShortUUID().random(length=8)}.papyrus_temp'
    # Iterate over chunks
    for i, chunk in enumerate(data):
        # Process the data with the given function
        processed = func(chunk)
        if not isinstance(processed, Iterator):
            # Write the processed chunk to temporary file
            pystow.dump_df(*pystow_prefixes,
                           name=temp_file,
                           obj=processed,
                           to_csv_kwargs={'header': i==0, # ensure header is written only once
                                          'mode': 'w' if i==0 else 'a'} # ensure file is not overwritten each time
                           )
        else:
            # Iterate over result containing chunk
            for j, subchunk in enumerate(processed):
                # Write the processed chunk to temporary file
                pystow.dump_df(*pystow_prefixes,
                               name=temp_file,
                               obj=subchunk,
                               to_csv_kwargs={'header': (i, j) == (0, 0),  # ensure header is written only once
                                              # ensure file is not overwritten each time
                                              'mode': 'w' if (i, j) == (0, 0) else 'a'})

    # Rename to destination file once all data were processed
    os.rename(os.path.join(os.environ['PYSTOW_HOME'], *pystow_prefixes, temp_file),
              os.path.join(os.environ['PYSTOW_HOME'], *pystow_prefixes, dest))


def remove_temporary_files(*pystow_prefixes) -> None:
    """Remove temporary files in the folder defined byb the pystow prefixes.

    :param pystow_prefixes: prefixes used to generate temporary files to be removed
    """
    for tmp_file in glob.glob(os.path.join(os.environ['PYSTOW_HOME'], *pystow_prefixes, '*.papyrus_temp')):
        os.remove(tmp_file)
