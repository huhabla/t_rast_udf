#!/usr/bin/env python
# -*- coding: utf-8 -*-
############################################################################
#
# MODULE:       t.rast.aggr_func
# AUTHOR(S):    Soeren Gebbert
#
# PURPOSE:      Apply a user defined function (UDF) to aggregate a time series into a single output raster map
# COPYRIGHT:    (C) 2017 by the GRASS Development Team
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#############################################################################

#%module
#% description: Apply a user defined function (UDF) to aggregate a time series into a single output raster map
#% keyword: temporal
#% keyword: aggregation
#% keyword: raster
#% keyword: time
#%end

#%option G_OPT_STRDS_INPUT
#%end

#%option G_OPT_R_OUTPUT
#%end

#%option G_OPT_F_INPUT
#% key: pyfile
#% description: The Python file with user defined function to apply to the input STRDS and create an output raster map
#%end

#%option
#% key: nrows
#% type: integer
#% description: Number of rows that should be provided at once to the user defined function
#% required: no
#% multiple: no
#% answer: 1
#%end

#%option G_OPT_T_WHERE
#%end

import pprint
import sys, os
import numpy as np
import pandas as pd
from pandas import DatetimeIndex
import grass.script as gcore
from grass.pygrass.raster import RasterRow
from grass.pygrass.raster.buffer import Buffer
from grass.pygrass.gis.region import Region
from grass.pygrass.raster.raster_type import TYPE as RTYPE

def udf_time_series_to_raster_map(t):
    pass


def create_udf_ts_tile_object(identifier, cell_array, start_time, end_time=None):
    """Create a time series object for a user defined function

    :param identifier: The identifier of the time series, in GRASS GIS is would be
                       the STRDS name
    :param cell_array: A three dimensional cell array. For each time stamp a two dimensional
                       slice of cell values is provided.
                       - First (0) dimension is time, time stamps are located in separate arrays
                       - Second (1) dimension is y
                       - Third (2) dimension is x
    :param start_time: A pandas.DatetimeIndex object that includes the start-time-stamps
                       of the first dimension of the cell array
    :param end_time: A pandas.DatetimeIndex object that includes the end-time-stamps
                       of the first dimension of the cell array

    The provided data will be put into a dictionary that has the following layout:

    .. code: Python
    {
        "identifier":identifier,
        "cell_array":cell_array,
        "start_time":start_time,
        "end_time":end_time
    }

    :return: A dictionary that contains the time series tile
    """

    data_object = {}
    data_object["identifier"] = identifier
    data_object["cell_array"] = cell_array
    data_object["start_time"] = start_time
    data_object["end_time"] = end_time

    return data_object


############################################################################

def main():
    # lazy imports
    import grass.temporal as tgis

    # Get the options
    input = options["input"]
    output = options["output"]
    where = options["where"]
    pyfile = options["pyfile"]
    nrows = int(options["nrows"])

    sys.path.append(os.path.dirname(os.path.abspath(pyfile)))
    # import the user defined function, the name of the file must be udf
    exec('from %s import udf_time_series_to_raster_map' % os.path.basename(pyfile).replace(".py", ""))

    rd = tgis.RasterDataset(None)
    rd.get_id()

    tgis.init()

    dbif = tgis.SQLDatabaseInterfaceConnection()
    dbif.connect()

    sp = tgis.open_old_stds(input, "strds", dbif)

    map_list = sp.get_registered_maps_as_objects(where=where, order="start_time", dbif=dbif)

    if not map_list:
        dbif.close()
        gcore.fatal(_("Space time raster dataset <%s> is empty") % input)

    if nrows == 0:
        dbif.close()
        gcore.fatal(_("Number of rows for the udf must be greater 0."))

    open_maps = []
    start_time = []
    end_time = []
    region = Region()

    # We create a two dimensional array by the size num_timestamps x columns

    mtype = None

    # Open all maps for processing
    for map in map_list:
        start, end = map.get_temporal_extent_as_tuple()
        start_time.append(start)
        end_time.append(end)

        rmap = RasterRow(map.get_id())
        rmap.open(mode='r')
        if mtype is not None:
            if mtype != rmap.mtype:
                dbif.close()
                gcore.fatal(_("Space time raster dataset <%s> is contains map with different type. "
                              "This is not supported.") % input)

        mtype = rmap.mtype
        open_maps.append(rmap)

    start_time = DatetimeIndex(start_time)
    end_time = DatetimeIndex(end_time)

    output_map = RasterRow(name=output)
    output_map.open(mode="w", mtype=mtype, overwrite=gcore.overwrite())

    # Read several wors for each map and load them into the udf
    for index in range(0, region.rows, nrows):
        # Compute the muber of usable rows
        if index + nrows > region.rows:
            usable_rows = index + nrows - region.rows + 1
        else:
            usable_rows = nrows
        # print("Usable rows", usable_rows)

        data = np.ndarray(shape=[len(map_list), usable_rows,
                                 region.cols],
                          dtype=RTYPE[mtype]['numpy'])

        for rmap, tindex in zip(open_maps, range(len(map_list))):
            for n in range(usable_rows):
                row = rmap[index + n]
                data[tindex][n][:] = row[:]

        t = create_udf_ts_tile_object(identifier=sp.get_id(),
                                      cell_array=data,
                                      start_time=start_time,
                                      end_time=end_time)

        # Call the user defined function
        ret = udf_time_series_to_raster_map(t)

        for row in ret:
            # Write the result into the output raster map
            b = Buffer(shape=(region.cols,), mtype=mtype)
            b[:] = row[:]
            output_map.put_row(b)

    output_map.close()

    dbif.close()

if __name__ == "__main__":
    options, flags = gcore.parser()
    main()
