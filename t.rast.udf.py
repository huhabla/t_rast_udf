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

#%option G_OPT_STRDS_OUTPUT
#%end

#%option G_OPT_R_OUTPUT
#% key: basename
#% description: The basename of the output raster maps
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

import numpy as np
from grass.temporal import RasterDataset, SQLDatabaseInterfaceConnection
from openeo_udf.api.base import SpatialExtent, RasterCollectionTile, UdfData
from pandas import DatetimeIndex
import grass.script as gcore
from grass.pygrass.raster import RasterRow
from grass.pygrass.raster.buffer import Buffer
from grass.pygrass.gis.region import Region
from grass.pygrass.raster.raster_type import TYPE as RTYPE
import geopandas
import pandas
import numpy
import xarray
from shapely.geometry import Polygon, Point
import json
import sys
from typing import Optional, List, Dict, Tuple


def create_raster_collection_tile(id: str, region: Region, array, index: int, usable_rows: int,
            start_times: DatetimeIndex, end_times: DatetimeIndex) -> RasterCollectionTile:
    """Create a raster collection tile

    :param id: The id of the strds
    :param region: The GRASS GIS Region
    :param array: The three dimensional array of data
    :param index: The current index
    :param usable_rows: The number of usable rows
    :param start_times: Start timed
    :param end_times: End tied
    :return: The udf data object
    """
    extent = SpatialExtent(top=region.north, bottom=region.south,
                           left=region.west + index,
                           right=region.west + index + usable_rows,
                           height=region.nsres, width=region.ewres)

    return RasterCollectionTile(id=id, data=array,
                                start_times=start_times,
                                end_times=end_times,
                                extent=extent)


def run_udf(code: str, epsg_code: str, raster_collection_tiles: List[RasterCollectionTile]) -> UdfData:
    """Run the user defined code (udf) and  create the required input for the function

    :param code: The UDF code
    :param epsg_code: The EPSG code of the projection
    :param raster_collection_tiles: The id of the strds
    :return: The resulting udf data object
    """

    data = UdfData(proj={"EPSG": epsg_code},
                   raster_collection_tiles=raster_collection_tiles)

    exec(code)

    return data

def open_raster_maps_get_timestamps(map_list: List[RasterDataset],
                                    dbif: SQLDatabaseInterfaceConnection) -> Tuple[List[RasterRow],
                                                                                        DatetimeIndex,
                                                                                        DatetimeIndex, int]:
    """Open all input raster maps, generate the time vectors and return them with the map type as tuple

    :param map_list:
    :param dbif:
    :return:
    """

    print("open_raster_maps_get_timestamps")

    open_maps = []    # Open maps of the existing STRDS
    start_times = []
    end_times = []
    mtype = None

    # Open all existing maps for processing
    for map in map_list:
        start, end = map.get_temporal_extent_as_tuple()
        start_times.append(start)
        end_times.append(end)

        rmap = RasterRow(map.get_id())
        rmap.open(mode='r')
        if mtype is not None:
            if mtype != rmap.mtype:
                dbif.close()
                gcore.fatal(_("Space time raster dataset <%s> is contains map with different type. "
                              "This is not supported.") % input)

        mtype = rmap.mtype
        open_maps.append(rmap)

    start_times = DatetimeIndex(start_times)
    end_times = DatetimeIndex(end_times)

    return open_maps, start_times, end_times, mtype


def count_resulting_maps(map_list: List[RasterDataset], sp,  dbif: SQLDatabaseInterfaceConnection,
                         region: Region, code: str, epsg_code: str):

    print("count_resulting_maps")

    open_maps, start_times, end_times, mtype = open_raster_maps_get_timestamps(map_list=map_list, dbif=dbif)

    # We need to count the number of slices that are returned from the udf, so we feed the first row to
    # the udf
    numberof_slices = 0
    array = np.ndarray(shape=[len(map_list), 1, region.cols], dtype=RTYPE[mtype]['numpy'])
    for rmap, tindex in zip(open_maps, range(len(map_list))):
        row = rmap[0]
        array[tindex][0][:] = row[:]

    raster_collection_tile = create_raster_collection_tile(id=sp.get_id(), region=region, array=array,
                                                           usable_rows=1, index=0, start_times=start_times,
                                                           end_times=end_times)
    data = run_udf(code=code, epsg_code=epsg_code, raster_collection_tiles=[raster_collection_tile,])
    for slice in data.get_raster_collection_tiles()[0].data:
        numberof_slices += 1

    print(f"Number of found slices in the output: {numberof_slices}")
    for rmap in open_maps:
        rmap.close()

    return numberof_slices


############################################################################

def main():
    # lazy imports
    import grass.temporal as tgis
    import sys

    # Get the options
    input = options["input"]
    output = options["output"]
    basename = options["basename"]
    where = options["where"]
    pyfile = options["pyfile"]
    nrows = int(options["nrows"])

    # Import the python code into the current function context
    code = open(pyfile, "r").read()
    projection_kv = gcore.parse_command("g.proj", flags="g")
    epsg_code = projection_kv["epsg"]


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

    open_output_maps = []     # Maps that are newly generated
    region = Region()

    numberof_slices = count_resulting_maps(map_list=map_list, dbif=dbif, sp=sp,
                                           region=region, code=code, epsg_code=epsg_code)
    open_input_maps, start_times, end_times, mtype = open_raster_maps_get_timestamps(map_list=map_list, dbif=dbif)

    if numberof_slices == 1:
        output_map = RasterRow(name=basename)
        output_map.open(mode="w", mtype=mtype, overwrite=gcore.overwrite())
        open_output_maps.append(output_map)
    elif numberof_slices > 1:
        for slice in range(numberof_slices):
            output_map = RasterRow(name=basename + f"_{slice}")
            output_map.open(mode="w", mtype=mtype, overwrite=gcore.overwrite())
            open_output_maps.append(output_map)
    else:
        dbif.close()
        gcore.fatal(_("No result generated") % input)

    # Read several rows for each map and load them into the udf
    for index in range(0, region.rows, nrows):
        if index + nrows > region.rows:
            usable_rows = index + nrows - region.rows + 1
        else:
            usable_rows = nrows

        array = np.ndarray(shape=[len(map_list), usable_rows,
                                  region.cols],
                           dtype=RTYPE[mtype]['numpy'])
        # We support the reading of several rows for a single udf execution
        for rmap, tindex in zip(open_input_maps, range(len(map_list))):
            for n in range(usable_rows):
                row = rmap[index + n]
                array[tindex][n][:] = row[:]

        raster_collection_tile = create_raster_collection_tile(id=sp.get_id(), region=region, array=array,
                                                               usable_rows=usable_rows, index=index,
                                                               start_times=start_times, end_times=end_times)
        data = run_udf(code=code, epsg_code=epsg_code, raster_collection_tiles=[raster_collection_tile, ])

        rtiles = data.get_raster_collection_tiles()
        for count, slice in enumerate(rtiles[0].data):
            output_map = open_output_maps[count]
            # print(f"Write slice at index {index} \n{slice} for map {output_map.name}")
            for row in slice:
                # Write the result into the output raster map
                b = Buffer(shape=(region.cols,), mtype=mtype)
                b[:] = row[:]
                output_map.put_row(b)

    for output_map in open_output_maps:
        output_map.close()

    dbif.close()


if __name__ == "__main__":
    options, flags = gcore.parser()
    main()
