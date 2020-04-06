#!/usr/bin/env python3
# -*- coding: utf-8 -*-
############################################################################
#
# MODULE:       t.rast.udf
# AUTHOR(S):    Soeren Gebbert
#
# PURPOSE:      Apply a user defined function (UDF) to aggregate a time series into a single output raster map
# COPYRIGHT:    (C) 2018 - 2019 by the GRASS Development Team
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

# %module
# % description: Apply a user defined function (UDF) to aggregate a time series into a single output raster map
# % keyword: temporal
# % keyword: aggregation
# % keyword: raster
# % keyword: time
# %end

# %option G_OPT_STRDS_INPUT
# %end

# %option G_OPT_STRDS_OUTPUT
# %end

# %option G_OPT_R_OUTPUT
# % key: basename
# % description: The basename of the output raster maps
# %end

# %option G_OPT_F_INPUT
# % key: pyfile
# % description: The Python file with user defined function to apply to the input STRDS and create an output raster map
# %end

# %option
# % key: nrows
# % type: integer
# % description: Number of rows that should be provided at once to the user defined function
# % required: no
# % multiple: no
# % answer: 1
# %end

# %option G_OPT_T_WHERE
# %end

import numpy as np
from grass.temporal import RasterDataset, SQLDatabaseInterfaceConnection
from openeo_udf.api.datacube import DataCube
from openeo_udf.api.udf_data import UdfData
from openeo_udf.api.run_code import run_user_code
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


def create_datacube(id: str, region: Region, array, index: int, usable_rows: int,
                    start_times: DatetimeIndex, end_times: DatetimeIndex) -> DataCube:
    """Create a data cube

    >>> array = xarray.DataArray(numpy.zeros(shape=(2, 3)), coords={'x': [1, 2], 'y': [1, 2, 3]}, dims=('x', 'y'))
    >>> array.attrs["description"] = "This is an xarray with two dimensions"
    >>> array.name = "testdata"
    >>> h = DataCube(array=array)

    :param id: The id of the strds
    :param region: The GRASS GIS Region
    :param array: The three dimensional array of data
    :param index: The current index
    :param usable_rows: The number of usable rows
    :param start_times: Start timed
    :param end_times: End tied
    :return: The udf data object
    """

    left = region.west
    top = region.north + index * region.nsres

    xcoords = []
    for col in region.cols:
        xcoords.append(left + col * region.ewres)

    ycoords = []
    for row in range(usable_rows):
        ycoords.append(top + row * region.nsres)

    tcoords = start_times.tolist()

    new_array = xarray.DataArray(array, dims=('t', 'y', 'x'), coords={'t': tcoords, 'y': ycoords, 'x' : xcoords})
    new_array.name = id

    return DataCube(array=new_array)


def run_udf(code: str, epsg_code: str, datacube_list: List[DataCube]) -> UdfData:
    """Run the user defined code (udf) and  create the required input for the function

    :param code: The UDF code
    :param epsg_code: The EPSG code of the projection
    :param datacube: The id of the strds
    :return: The resulting udf data object
    """

    data = UdfData(proj={"EPSG": epsg_code}, datacube_list=datacube_list)

    return run_user_code(code=code, data=data)


def open_raster_maps_get_timestamps(map_list: List[RasterDataset],
                                    dbif: SQLDatabaseInterfaceConnection) -> Tuple[List[RasterRow],
                                                                                   DatetimeIndex,
                                                                                   DatetimeIndex, int]:
    """Open all input raster maps, generate the time vectors and return them with the map type as tuple

    :param map_list:
    :param dbif:
    :return:
    """

    open_maps = []  # Open maps of the existing STRDS
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


def count_resulting_maps(map_list: List[RasterDataset], sp, dbif: SQLDatabaseInterfaceConnection,
                         region: Region, code: str, epsg_code: str) -> int:
    """Run the UDF code for a single raster line for each input map and count the
    resulting slices in the first raster collection tile

    :param map_list: The list of maps
    :param sp: The STRDS
    :param dbif: The database interface
    :param region: The current computational region
    :param code: The UDF code
    :param epsg_code: The EPSG code
    :return: The number of slices that were counted
    """

    open_maps, start_times, end_times, mtype = open_raster_maps_get_timestamps(map_list=map_list, dbif=dbif)

    # We need to count the number of slices that are returned from the udf, so we feed the first row to
    # the udf
    numberof_slices = 0
    array = np.ndarray(shape=[len(map_list), 1, region.cols], dtype=RTYPE[mtype]['numpy'])
    for rmap, tindex in zip(open_maps, range(len(map_list))):
        row = rmap[0]
        array[tindex][0][:] = row[:]

    datacube = create_datacube(id=sp.get_id(), region=region, array=array,
                               usable_rows=1, index=0, start_times=start_times,
                               end_times=end_times)
    data = run_udf(code=code, epsg_code=epsg_code, datacube_list=[datacube, ])
    for slice in data.get_datacube_list()[0].array:
        numberof_slices += 1

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

    open_output_maps: List[RasterRow] = []  # Maps that are newly generated
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

    result_start_times = []
    first = False

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

        datacube = create_datacube(id=sp.get_id(), region=region, array=array,
                                   usable_rows=usable_rows, index=index,
                                   start_times=start_times, end_times=end_times)
        data = run_udf(code=code, epsg_code=epsg_code, datacube_list=[datacube, ])

        # Read only the first cube
        datacubes = data.get_datacube_list()
        first_cube_array: xarray.DataArray = datacubes[0].get_array()

        if first is False:
            result_start_times = first_cube_array.coords['t']

        for count, slice in enumerate(first_cube_array):
            output_map = open_output_maps[count]
            # print(f"Write slice at index {index} \n{slice} for map {output_map.name}")
            for row in slice:
                # Write the result into the output raster map
                b = Buffer(shape=(region.cols,), mtype=mtype)
                b[:] = row[:]
                output_map.put_row(b)

        first = True

    # Create new STRDS

    for output_map in open_output_maps:
        output_map.close()

    dbif.close()


if __name__ == "__main__":
    options, flags = gcore.parser()
    main()
