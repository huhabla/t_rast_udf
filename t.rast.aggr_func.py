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

import numpy as np
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
from typing import Optional, List, Dict, Tuple

__license__ = "Apache License, Version 2.0"
__author__ = "Soeren Gebbert"
__copyright__ = "Copyright 2018, Soeren Gebbert"
__maintainer__ = "Soeren Gebbert"
__email__ = "soerengebbert@googlemail.com"


class SpatialExtent(object):

    def __init__(self, top: float, bottom: float, right: float, left: float,
                 height: Optional[float] = None, width: Optional[float] = None):
        """Constructor of the axis aligned spatial extent of a collection tile

        Args:
            top (float): The top (northern) border of the data chunk
            bottom (float): The bottom (southern) border of the data chunk
            right (float): The righ (eastern) border of the data chunk
            left (float): The left (western) border of the data chunk
            height (float): The top-bottom pixel resolution (ignored in case of vector data chunks)
            width (float): The right-left pixel resolution (ignored in case of vector data chunks)

        """

        self.top = top
        self.bottom = bottom
        self.right = right
        self.left = left
        self.height = height
        self.width = width
        self.polygon = self.as_polygon()

    def contains_point(self, top: float, left: float) -> Point:
        """Return True if the provided coordinate is located in the spatial extent, False otherwise

        Args:
           top (float): The top (northern) coordinate of the point
           left (float): The left (western) coordinate of the point


        Returns:
            bool: True if the coordinates are contained in the extent, False otherwise

        """
        return self.polygon.contains(Point(left, top))
        # return self.polygon.intersects(Point(left, top))

    def to_index(self, top: float, left: float) -> Tuple[int, int]:
        """Return True if the provided coordinate is located in the spatial extent, False otherwise

        Args:
           top (float): The top (northern) coordinate
           left (float): The left (western) coordinate

        Returns:
             tuple(int, int): (x, y) The x, y index

        """
        x = int(abs((left - self.left) / self.width))
        y = int(abs((top - self.top) / self.height))
        return (x, y)

    def __str__(self):
        return "top: %(n)s\n" \
               "bottom: %(s)s\n" \
               "right: %(e)s\n" \
               "left: %(w)s\n" \
               "height: %(ns)s\n" \
               "width: %(ew)s" % {"n": self.top, "s": self.bottom, "e": self.right,
                                  "w": self.left, "ns": self.height, "ew": self.width}

    def as_polygon(self) -> Polygon:
        """Return the extent as shapely.geometry.Polygon to perform
        comparison operations between other extents like equal, intersect and so on

        Returns:
            shapely.geometry.Polygon: The polygon representing the spatial extent

        """

        return Polygon([(self.left, self.top), (self.right, self.top),
                        (self.right, self.bottom), (self.left, self.bottom)])

    @staticmethod
    def from_polygon(polygon: Polygon) -> 'SpatialExtent':
        """Convert a polygon with rectangular shape into a spatial extent

        Args:
            polygon (shapely.geometry.Polygon): The polygon that should be converted into a spatial extent

        Returns:
            SpatialExtent: The spatial extent

        """

        coords = list(polygon.exterior.coords)

        top = coords[0][1]
        bottom = coords[2][1]
        right = coords[1][0]
        left = coords[0][0]

        return SpatialExtent(top=top, bottom=bottom, right=right, left=left)

    def to_dict(self) -> Dict:
        """Return the spatial extent as a dict that can be easily converted into JSON

        Returns:
            dict:
            Dictionary representation

        """
        d = dict(extent=dict(top=self.top, bottom=self.bottom, right=self.right,
                             left=self.left))

        if self.width:
            d["extent"].update({"width": self.width})
        if self.height:
            d["extent"].update({"height": self.height})

        return d

    @staticmethod
    def from_dict(extent: Dict):
        """Create a SpatialExtent from a python dictionary that was created from
        the JSON definition of the SpatialExtent

        Args:
            extent (dict): The dictionary that contains the spatial extent definition

        Returns:
            SpatialExtent:
            A new SpatialExtent object

        """

        top = None
        bottom = None
        right = None
        left = None
        width = None
        height = None

        if "top" in extent:
            top = extent["top"]
        if "bottom" in extent:
            bottom = extent["bottom"]
        if "right" in extent:
            right = extent["right"]
        if "left" in extent:
            left = extent["left"]
        if "width" in extent:
            width = extent["width"]
        if "height" in extent:
            height = extent["height"]

        return SpatialExtent(top=top, bottom=bottom, left=left, right=right, height=height, width=width)


class CollectionTile(object):

    def __init__(self, id: str, extent: Optional[SpatialExtent] = None,
                 start_times: Optional[pandas.DatetimeIndex] = None,
                 end_times: Optional[pandas.DatetimeIndex] = None):
        """Constructor of the base class for tile of a collection

        Args:
            id: The unique id of the raster collection tile
            extent: The spatial extent with resolution information, must be of type SpatialExtent
            start_times: The pandas.DateTimeIndex vector with start times for each spatial x,y slice
            end_times: The pandas.DateTimeIndex vector with end times for each spatial x,y slice, if no
                       end times are defined, then time instances are assumed not intervals

        """

        self.id = id
        self._extent: Optional[SpatialExtent] = None
        self._start_times: Optional[pandas.DatetimeIndex] = None
        self._end_times: Optional[pandas.DatetimeIndex] = None
        self._data: List = None

        self.set_extent(extent=extent)
        self.set_start_times(start_times=start_times)
        self.set_end_times(end_times=end_times)

    def check_data_with_time(self):
        """Check if the start and end date vectors have the same size as the data
        """

        if self._data is not None and self.start_times is not None:
            if len(self.start_times) != len(self._data):
                raise Exception("The size of the start times vector just be equal "
                                "to the size of data")

        if self._data is not None and self.end_times is not None:
            if len(self.end_times) != len(self._data):
                raise Exception("The size of the end times vector just be equal "
                                "to the size of data")

    def __str__(self) -> str:
        return "id: %(id)s\n" \
               "extent: %(extent)s\n" \
               "start_times: %(start_times)s\n" \
               "end_times: %(end_times)s" % {"id": self.id,
                                             "extent": self.extent,
                                             "start_times": self.start_times,
                                             "end_times": self.end_times}

    def get_start_times(self) -> Optional[pandas.DatetimeIndex]:
        """Returns the start time vector

        Returns:
            pandas.DatetimeIndex: Start time vector

        """
        return self._start_times

    def set_start_times(self, start_times: Optional[pandas.DatetimeIndex]):
        """Set the start times vector

        Args:
            start_times (pandas.DatetimeIndex): The start times vector

        """
        if start_times is None:
            return

        if isinstance(start_times, pandas.DatetimeIndex) is False:
            raise Exception("The start times vector mus be of type pandas.DatetimeIndex")

        self._start_times = start_times

    def get_end_times(self) -> Optional[pandas.DatetimeIndex]:
        """Returns the end time vector

        Returns:
            pandas.DatetimeIndex: End time vector

        """
        return self._end_times

    def set_end_times(self, end_times: Optional[pandas.DatetimeIndex]):
        """Set the end times vector

        Args:
            end_times (pandas.DatetimeIndex): The  end times vector
        """
        if end_times is None:
            return

        if isinstance(end_times, pandas.DatetimeIndex) is False:
            raise Exception("The start times vector mus be of type pandas.DatetimeIndex")

        self._end_times = end_times

    def get_extent(self) -> SpatialExtent:
        """Return the spatial extent

        Returns:
            SpatialExtent: The spatial extent

        """
        return self._extent

    def set_extent(self, extent: SpatialExtent):
        """Set the spatial extent

        Args:
            extent (SpatialExtent): The spatial extent with resolution information, must be of type SpatialExtent
        """
        if extent is None:
            return

        if isinstance(extent, SpatialExtent) is False:
            raise Exception("extent mus be of type SpatialExtent")

        self._extent = extent

    start_times = property(fget=get_start_times, fset=set_start_times)
    end_times = property(fget=get_end_times, fset=set_end_times)
    extent = property(fget=get_extent, fset=set_extent)

    def extent_to_dict(self) -> Dict:
        """Convert the extent into a dictionary representation that can be converted to JSON

        Returns:
            dict:
            The spatial extent

        """
        return self._extent.to_dict()

    def start_times_to_dict(self) -> Dict:
        """Convert the start times vector into a dictionary representation that can be converted to JSON

        Returns:
            dict:
            The start times vector

        """
        return dict(start_times=[t.isoformat() for t in self._start_times])

    def end_times_to_dict(self) -> Dict:
        """Convert the end times vector into a dictionary representation that can be converted to JSON

        Returns:
            dict:
            The end times vector

        """
        return dict(end_times=[t.isoformat() for t in self._end_times])

    def set_extent_from_dict(self, extent: Dict):
        """Set the spatial extent from a dictionary

        Args:
            extent (dict): The dictionary with the layout of the JSON SpatialExtent definition
        """
        self.set_extent(SpatialExtent.from_dict(extent))

    def set_start_times_from_list(self, start_times: Dict):
        """Set the start times vector from a dictionary

        Args:
            start_times (dict): The dictionary with the layout of the JSON start times vector definition
        """
        self.set_start_times(pandas.DatetimeIndex(start_times))

    def set_end_times_from_list(self, end_times: Dict):
        """Set the end times vector from a dictionary

        Args:
            end_times (dict): The dictionary with the layout of the JSON end times vector definition
        """
        self.set_end_times(pandas.DatetimeIndex(end_times))


class RasterCollectionTile(CollectionTile):

    def __init__(self, id: str, extent: SpatialExtent, data: numpy.ndarray,
                 wavelength: Optional[float] = None,
                 start_times: Optional[pandas.DatetimeIndex] = None,
                 end_times: Optional[pandas.DatetimeIndex] = None):
        """Constructor of the tile of an raster collection

        Args:
            id (str): The unique id of the raster collection tile
            extent (SpatialExtent): The spatial extent with resolution information
            data (numpy.ndarray): The three dimensional numpy.ndarray with indices [t][y][x]
            wavelength (float): The optional wavelength of the raster collection tile
            start_times (pandas.DatetimeIndex): The vector with start times for each spatial x,y slice
            end_times (pandas.DatetimeIndex): The pandas.DateTimeIndex vector with end times for each spatial x,y slice, if no
                       end times are defined, then time instances are assumed not intervals
        """

        CollectionTile.__init__(self, id=id, extent=extent, start_times=start_times, end_times=end_times)

        self.wavelength = wavelength
        self.set_data(data)
        self.check_data_with_time()

    def __str__(self) -> str:
        return "id: %(id)s\n" \
               "extent: %(extent)s\n" \
               "wavelength: %(wavelength)s\n" \
               "start_times: %(start_times)s\n" \
               "end_times: %(end_times)s\n" \
               "data: %(data)s" % {"id": self.id, "extent": self.extent, "wavelength": self.wavelength,
                                   "start_times": self.start_times, "end_times": self.end_times, "data": self.data}

    def sample(self, top: float, left: float):
        """Sample the raster tile at specific top, left coordinates.

        If the coordinates are not in the spatial extent of the tile, then None will be returned.
        Otherwise a list of values, depending on the number of x,y slices are returned.

        The coordinates must be of the same projection as the raster collection.

        Args:
           top (float): The top (northern) coordinate of the point
           left (float): The left (western) coordinate of the point

        Returns:
            numpy.ndarray:
            A one dimensional array of values
        """
        if self.extent.contains_point(top=top, left=left) is True:
            x, y = self.extent.to_index(top, left)

            values = []
            for xy_slice in self.data:
                value = xy_slice[y][x]

                values.append(value)
            return values

        return None

    def get_data(self) -> numpy.ndarray:
        """Return the three dimensional numpy.ndarray with indices [t][y][x]

        Returns:
            numpy.ndarray: The three dimensional numpy.ndarray with indices [t][y][x]

        """
        return self._data

    def set_data(self, data: numpy.ndarray):
        """Set the three dimensional numpy.ndarray with indices [t][y][x]

        This function will check if the provided data is a numpy.ndarray with three dimensions

        Args:
            data (numpy.ndarray): The three dimensional numpy.ndarray with indices [t][y][x]

        """
        if isinstance(data, numpy.ndarray) is False:
            raise Exception("Argument data must be of type numpy.ndarray")

        if len(data.shape) != 3:
            raise Exception("Argument data must have three dimensions")

        self._data = data

    data = property(fget=get_data, fset=set_data)

    def to_dict(self) -> Dict:
        """Convert this RasterCollectionTile into a dictionary that can be converted into
        a valid JSON representation

        Returns:
            dict:
            RasterCollectionTile as a dictionary
        """

        d = {"id": self.id}
        if self._data is not None:
            d["data"] = self._data.tolist()
        if self.wavelength is not None:
            d["wavelength"] = self.wavelength
        if self._start_times is not None:
            d.update(self.start_times_to_dict())
        if self._end_times is not None:
            d.update(self.end_times_to_dict())
        if self._extent is not None:
            d.update(self.extent_to_dict())

        return d

    @staticmethod
    def from_dict(ict_dict: Dict):
        """Create a raster collection tile from a python dictionary that was created from
        the JSON definition of the RasterCollectionTile

        Args:
            ict_dict (dict): The dictionary that contains the raster collection tile definition

        Returns:
            RasterCollectionTile:
            A new RasterCollectionTile object

        """

        if "id" not in ict_dict:
            raise Exception("Missing id in dictionary")

        if "data" not in ict_dict:
            raise Exception("Missing data in dictionary")

        if "extent" not in ict_dict:
            raise Exception("Missing extent in dictionary")

        ict = RasterCollectionTile(id=ict_dict["id"],
                                   extent=SpatialExtent.from_dict(ict_dict["extent"]),
                                   data=numpy.asarray(ict_dict["data"]))

        if "start_times" in ict_dict:
            ict.set_start_times_from_list(ict_dict["start_times"])

        if "end_times" in ict_dict:
            ict.set_end_times_from_list(ict_dict["end_times"])

        if "wavelength" in ict_dict:
            ict.wavelength = ict_dict["wavelength"]

        return ict


class HyperCube:

    def __init__(self, data: xarray.DataArray):

        self.set_data(data)

    def __str__(self):
        return "id: %(id)s\n" \
               "data: %(data)s" % {"id": self.id, "data": self.data}

    def get_data(self) -> xarray.DataArray:
        """Return the xarray.DataArray that contains the data and dimension definition

        Returns:
            xarray.DataArray: that contains the data and dimension definition

        """
        return self._data

    def set_data(self, data: xarray.DataArray):
        """Set the xarray.DataArray that contains the data and dimension definition

        This function will check if the provided data is a geopandas.GeoDataFrame and raises
        an Exception

        Args:
            data: xarray.DataArray that contains the data and dimension definition

        """
        if isinstance(data, xarray.DataArray) is False:
            raise Exception("Argument data must be of type xarray.DataArray")

        self._data = data

    @property
    def id(self):
        return self._data.name

    data = property(fget=get_data, fset=set_data)

    def to_dict(self) -> Dict:
        """Convert this hypercube into a dictionary that can be converted into
        a valid JSON representation

        Returns:
            dict:
            HyperCube as a dictionary

        example = {
            "id": "test_data",
            "data": [
                [
                    [0.0, 0.1],
                    [0.2, 0.3]
                ],
                [
                    [0.0, 0.1],
                    [0.2, 0.3]
                ]
            ],
            "dimension": [{"name": "time", "unit": "ISO:8601", "coordinates":["2001-01-01", "2001-01-02"]},
                          {"name": "X", "unit": "degree", "coordinates":[50.0, 60.0]},
                          {"name": "Y", "unit": "degree"},
                         ]
        }
        """

        d = {"id": "", "data": "", "dimensions": []}
        if self._data is not None:
            xd = self._data.to_dict()

            if "name" in xd:
                d["id"] = xd["name"]

            if "data" in xd:
                d["data"] = xd["data"]

            if "attrs" in xd:
                if "description" in xd["attrs"]:
                    d["description"] = xd["attrs"]["description"]

            if "dims" in xd and "coords" in xd:
                for dim in xd["dims"]:
                    if dim in xd["coords"]:
                        if "data" in xd["coords"][dim]:
                            d["dimensions"].append({"name": dim, "coordinates": xd["coords"][dim]["data"]})
                        else:
                            d["dimensions"].append({"name": dim})

        return d

    @staticmethod
    def from_dict(hc_dict: Dict) -> "HyperCube":
        """Create a hypercube from a python dictionary that was created from
        the JSON definition of the HyperCube

        Args:
            hc_dict (dict): The dictionary that contains the hypercube definition

        Returns:
            HyperCube

        """

        if "id" not in hc_dict:
            raise Exception("Missing id in dictionary")

        if "data" not in hc_dict:
            raise Exception("Missing data in dictionary")

        coords = {}
        dims = list()

        if "dimensions" in hc_dict:
            for dim in hc_dict["dimensions"]:
                dims.append(dim["name"])
                if "coordinates" in dim:
                    coords[dim["name"]] = dim["coordinates"]

        if dims and coords:
            data = xarray.DataArray(numpy.asarray(hc_dict["data"]), coords=coords, dims=dims)
        elif dims:
            data = xarray.DataArray(numpy.asarray(hc_dict["data"]), dims=dims)
        else:
            data = xarray.DataArray(numpy.asarray(hc_dict["data"]))

        if "id" in hc_dict:
            data.name = hc_dict["id"]
        if "description" in hc_dict:
            data.attrs["description"] = hc_dict["description"]

        hc = HyperCube(data=data)

        return hc


class FeatureCollectionTile(CollectionTile):

    def __init__(self, id: str, data: geopandas.GeoDataFrame,
                 start_times: Optional[pandas.DatetimeIndex] = None,
                 end_times: Optional[pandas.DatetimeIndex] = None):
        """Constructor of the tile of a vector collection

        Args:
            id (str): The unique id of the vector collection tile
            data (geopandas.GeoDataFrame): A GeoDataFrame with geometry column and attribute data
            start_times (pandas.DateTimeIndex): The vector with start times for each spatial x,y slice
            end_times (pandas.DateTimeIndex): The pandas.DateTimeIndex vector with end times
                                              for each spatial x,y slice, if no
                       end times are defined, then time instances are assumed not intervals
        """
        CollectionTile.__init__(self, id=id, start_times=start_times, end_times=end_times)

        self.set_data(data)
        self.check_data_with_time()

    def __str__(self):
        return "id: %(id)s\n" \
               "start_times: %(start_times)s\n" \
               "end_times: %(end_times)s\n" \
               "data: %(data)s" % {"id": self.id, "extent": self.extent,
                                   "start_times": self.start_times,
                                   "end_times": self.end_times, "data": self.data}

    def get_data(self) -> geopandas.GeoDataFrame:
        """Return the geopandas.GeoDataFrame that contains the geometry column and any number of attribute columns

        Returns:
            geopandas.GeoDataFrame: A data frame that contains the geometry column and any number of attribute columns

        """
        return self._data

    def set_data(self, data: geopandas.GeoDataFrame):
        """Set the geopandas.GeoDataFrame that contains the geometry column and any number of attribute columns

        This function will check if the provided data is a geopandas.GeoDataFrame and raises
        an Exception

        Args:
            data (geopandas.GeoDataFrame): A GeoDataFrame with geometry column and attribute data

        """
        if isinstance(data, geopandas.GeoDataFrame) is False:
            raise Exception("Argument data must be of type geopandas.GeoDataFrame")

        self._data = data

    data = property(fget=get_data, fset=set_data)

    def to_dict(self) -> Dict:
        """Convert this FeatureCollectionTile into a dictionary that can be converted into
        a valid JSON representation

        Returns:
            dict:
            FeatureCollectionTile as a dictionary
        """

        d = {"id": self.id}
        if self._start_times is not None:
            d.update(self.start_times_to_dict())
        if self._end_times is not None:
            d.update(self.end_times_to_dict())
        if self._data is not None:
            d["data"] = json.loads(self._data.to_json())

        return d

    @staticmethod
    def from_dict(fct_dict: Dict):
        """Create a feature collection tile from a python dictionary that was created from
        the JSON definition of the FeatureCollectionTile

        Args:
            fct_dict (dict): The dictionary that contains the feature collection tile definition

        Returns:
            FeatureCollectionTile:
            A new FeatureCollectionTile object

        """

        if "id" not in fct_dict:
            raise Exception("Missing id in dictionary")

        if "data" not in fct_dict:
            raise Exception("Missing data in dictionary")

        fct = FeatureCollectionTile(id=fct_dict["id"],
                                    data=geopandas.GeoDataFrame.from_features(fct_dict["data"]))

        if "start_times" in fct_dict:
            fct.set_start_times_from_list(fct_dict["start_times"])

        if "end_times" in fct_dict:
            fct.set_end_times_from_list(fct_dict["end_times"])

        return fct


class StructuredData(object):
    """This class represents structured data that is produced by an UDF and can not be represented
    as a RasterCollectionTile or FeatureCollectionTile. For example the result of a statistical
    computation. The data is self descriptive and supports the basic types dict/map, list and table.

    The data field contains the UDF specific values (argument or return) as dict, list or table:

        * A dict can be as complex as required by the UDF
        * A list must contain simple data types example {\"list\": [1,2,3,4]}
        * A table is a list of lists with a header, example {\"table\": [[\"id\",\"value\"],
                                                                           [1,     10],
                                                                           [2,     23],
                                                                           [3,     4]]}

    """

    def __init__(self, description, data, type):
        self.description = description
        self.data = data
        self.type = type

    def to_dict(self) -> Dict:
        return dict(description=self.description, data=self.data, type=self.type)

    @staticmethod
    def from_dict(structured_data: Dict):
        description = structured_data["description"]
        data = structured_data["data"]
        type = structured_data["type"]
        return StructuredData(description=description, data=data, type=type)


class MachineLearnModel(object):

    def __init__(self, framework: str, name: str, description: str, path: str):
        """The constructor to create a machine learn model object

        Args:
            framework: The name of the framework, pytroch and sklearn are supported
            name: The name of the model
            description: The description of the model
            path: The path to the pre-trained machine learn model that should be applied
        """
        self.framework = framework
        self.name = name
        self.description = description
        self.path = path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the machine learn model from the path.

        Supported model:
        - sklearn models that are created with sklearn.externals.joblib
        - pytorch models that are created with torch.save

        """
        if self.framework.lower() in "sklearn":
            from sklearn.externals import joblib
            self.model = joblib.load(self.path)
        if self.framework.lower() in "pytorch":
            import torch
            self.model = torch.load(self.path)

    def get_model(self):
        """Get the loaded machine learn model. This function will return None if the model was not loaded

        :return: the loaded model
        """
        return self.model

    def to_dict(self) -> Dict:
        return dict(description=self.description, name=self.name, framework=self.framework, path=self.path)

    @staticmethod
    def from_dict(machine_learn_model: Dict):
        description = machine_learn_model["description"]
        name = machine_learn_model["name"]
        framework = machine_learn_model["framework"]
        path = machine_learn_model["path"]
        return MachineLearnModel(description=description, name=name, framework=framework, path=path)


class UdfData(object):

    def __init__(self, proj: Dict,
                 raster_collection_tiles: Optional[List[RasterCollectionTile]] = None,
                 hypercube_list: Optional[List[HyperCube]] = None,
                 feature_collection_tiles: Optional[List[FeatureCollectionTile]] = None,
                 structured_data_list: Optional[List[StructuredData]] = None,
                 ml_model_list: Optional[List[MachineLearnModel]] = None):
        """The constructor of the UDF argument class that stores all data required by the
        user defined function.

        Args:
            proj (dict): A dictionary of form {"proj type string": "projection decription"} i. e. {"EPSG":4326}
            raster_collection_tiles (list[RasterCollectionTile]): A list of RasterCollectionTile objects
            hypercube_list (list(HyperCube)): A list of HyperCube objects
            feature_collection_tiles (list[FeatureCollectionTile]): A list of VectorTile objects
            structured_data_list (list[StructuredData]): A list of structured data objects
            ml_model_list (list[MachineLearnModel]): A list of machine learn models
        """

        self._raster_tile_list = []
        self._hypercube_list = []
        self._feature_tile_list = []
        self._raster_tile_dict = {}
        self._hypercube_dict = {}
        self._feature_tile_dict = {}
        self._structured_data_list = []
        self._ml_model_list = []
        self.proj = proj

        if raster_collection_tiles:
            self.set_raster_collection_tiles(raster_collection_tiles=raster_collection_tiles)
        if hypercube_list:
            self.set_hypercube_list(hypercube_list=hypercube_list)
        if feature_collection_tiles:
            self.set_feature_collection_tiles(feature_collection_tiles=feature_collection_tiles)
        if structured_data_list:
            self.set_structured_data_list(structured_data_list=structured_data_list)
        if ml_model_list:
            self.set_ml_model_list(ml_model_list=ml_model_list)

    def get_raster_collection_tile_by_id(self, id: str) -> Optional[RasterCollectionTile]:
        """Get an raster collection tile by its id

        Args:
            id (str): The raster collection tile id

        Returns:
            RasterCollectionTile: the requested raster collection tile of None if not found

        """
        if id in self._raster_tile_dict:
            return self._raster_tile_dict[id]
        return None

    def get_hypercube_by_id(self, id: str) -> Optional[HyperCube]:
        """Get a hypercube by its id

        Args:
            id (str): The raster collection tile id

        Returns:
            HypeCube: the requested raster collection tile of None if not found

        """
        if id in self._hypercube_dict:
            return self._hypercube_dict[id]
        return None

    def get_feature_collection_tile_by_id(self, id: str) -> Optional[FeatureCollectionTile]:
        """Get a vector tile by its id

        Args:
            id (str): The vector tile id

        Returns:
            FeatureCollectionTile: the requested vector tile of None if not found

        """
        if id in self._feature_tile_dict:
            return self._feature_tile_dict[id]
        return None

    def get_raster_collection_tiles(self) -> Optional[List[RasterCollectionTile]]:
        """Get all raster collection tiles

        Returns:
            list[RasterCollectionTile]: The list of raster collection tiles

        """
        return self._raster_tile_list

    def set_raster_collection_tiles(self, raster_collection_tiles: Optional[List[RasterCollectionTile]]):
        """Set the raster collection tiles list

        If raster_collection_tiles is None, then the list will be cleared

        Args:
            raster_collection_tiles (list[RasterCollectionTile]): A list of RasterCollectionTile's
        """

        self.del_raster_collection_tiles()
        if raster_collection_tiles is None:
            return

        for entry in raster_collection_tiles:
            self.append_raster_collection_tile(entry)

    def del_raster_collection_tiles(self):
        """Delete all raster collection tiles
        """
        self._raster_tile_list.clear()
        self._raster_tile_dict.clear()

    def get_hypercube_list(self) -> Optional[List[HyperCube]]:
        """Get the hypercube list
        """
        return self._hypercube_list

    def set_hypercube_list(self, hypercube_list: List[HyperCube]):
        """Set the hypercube list

        If hypercube_list is None, then the list will be cleared

        Args:
            hypercube_list (List[HyperCube]): A list of HyperCube's
        """

        self.del_hypercube_list()
        if hypercube_list is None:
            return

        for hypercube in hypercube_list:
            self.append_hypercube(hypercube)

    def del_hypercube_list(self):
        """Delete all hypercubes
        """
        self._hypercube_list.clear()
        self._hypercube_dict.clear()

    def get_feature_collection_tiles(self) -> Optional[List[FeatureCollectionTile]]:
        """Get all feature collection tiles

        Returns:
            list[FeatureCollectionTile]: The list of feature collection tiles

        """
        return self._feature_tile_list

    def set_feature_collection_tiles(self, feature_collection_tiles: Optional[List[FeatureCollectionTile]]):
        """Set the feature collection tiles

        If feature_collection_tiles is None, then the list will be cleared

        Args:
            feature_collection_tiles (list[FeatureCollectionTile]): A list of FeatureCollectionTile's
        """

        self.del_feature_collection_tiles()
        if feature_collection_tiles is None:
            return

        for entry in feature_collection_tiles:
            self.append_feature_collection_tile(entry)

    def del_feature_collection_tiles(self):
        """Delete all feature collection tiles
        """
        self._feature_tile_list.clear()
        self._feature_tile_dict.clear()

    def get_structured_data_list(self) -> Optional[List[StructuredData]]:
        """Get all structured data entries

        Returns:
            (list[StructuredData]): A list of StructuredData objects

        """
        return self._structured_data_list

    def set_structured_data_list(self, structured_data_list: Optional[List[StructuredData]]):
        """Set the list of structured data

        If structured_data_list is None, then the list will be cleared

        Args:
            structured_data_list (list[StructuredData]): A list of StructuredData objects
        """

        self.del_structured_data_list()
        if structured_data_list is None:
            return

        for entry in structured_data_list:
            self._structured_data_list.append(entry)

    def del_structured_data_list(self):
        """Delete all structured data entries
        """
        self._structured_data_list.clear()

    def get_ml_model_list(self) -> Optional[List[MachineLearnModel]]:
        """Get all machine learn models

        Returns:
            (list[MachineLearnModel]): A list of MachineLearnModel objects

        """
        return self._ml_model_list

    def set_ml_model_list(self, ml_model_list: Optional[List[MachineLearnModel]]):
        """Set the list of machine learn models

        If ml_model_list is None, then the list will be cleared

        Args:
            ml_model_list (list[MachineLearnModel]): A list of MachineLearnModel objects
        """

        self.del_ml_model_list()
        if ml_model_list is None:
            return

        for entry in ml_model_list:
            self._ml_model_list.append(entry)

    def del_ml_model_list(self):
        """Delete all machine learn models
        """
        self._ml_model_list.clear()

    raster_collection_tiles = property(fget=get_raster_collection_tiles,
                                       fset=set_raster_collection_tiles, fdel=del_raster_collection_tiles)
    hypercube_list = property(fget=get_hypercube_list,
                              fset=set_hypercube_list, fdel=del_hypercube_list)
    feature_collection_tiles = property(fget=get_feature_collection_tiles,
                                        fset=set_feature_collection_tiles, fdel=del_feature_collection_tiles)
    structured_data_list = property(fget=get_structured_data_list,
                                    fset=set_structured_data_list, fdel=del_structured_data_list)
    ml_model_list = property(fget=get_ml_model_list,
                             fset=set_ml_model_list, fdel=del_ml_model_list)

    def append_raster_collection_tile(self, raster_collection_tile: RasterCollectionTile):
        """Append a raster collection tile to the list

        It will be automatically added to the dictionary of all raster collection tiles

        Args:
            raster_collection_tile (RasterCollectionTile): The raster collection tile to append
        """
        self._raster_tile_list.append(raster_collection_tile)
        self._raster_tile_dict[raster_collection_tile.id] = raster_collection_tile

    def append_hypercube(self, hypercube: HyperCube):
        """Append a HyperCube to the list

        It will be automatically added to the dictionary of all hypercubes

        Args:
            hypercube (HyperCube): The HyperCube to append
        """
        self._hypercube_list.append(hypercube)
        self._hypercube_dict[hypercube.id] = hypercube

    def append_feature_collection_tile(self, feature_collection_tile: FeatureCollectionTile):
        """Append a feature collection tile to the list

        It will be automatically added to the dictionary of all feature collection tiles

        Args:
            feature_collection_tile (FeatureCollectionTile): The feature collection tile to append
        """
        self._feature_tile_list.append(feature_collection_tile)
        self._feature_tile_dict[feature_collection_tile.id] = feature_collection_tile

    def append_structured_data(self, structured_data: StructuredData):
        """Append a structured data object to the list

        Args:
            structured_data (StructuredData): A StructuredData objects
        """
        self._structured_data_list.append(structured_data)

    def append_machine_learn_model(self, machine_learn_model: MachineLearnModel):
        """Append a machine learn model to the list

        Args:
            machine_learn_model (MachineLearnModel): A MachineLearnModel objects
        """
        self._ml_model_list.append(machine_learn_model)

    def to_dict(self) -> Dict:
        """Convert this UdfData object into a dictionary that can be converted into
        a valid JSON representation

        Returns:
            dict:
            UdfData object as a dictionary
        """

        d = {"proj": self.proj}

        if self._raster_tile_list is not None:
            l = []
            for tile in self._raster_tile_list:
                l.append(tile.to_dict())
            d["raster_collection_tiles"] = l

        if self._hypercube_list is not None:
            l = []
            for hypercube in self._hypercube_list:
                l.append(hypercube.to_dict())
            d["hypercubes"] = l

        if self._feature_tile_list is not None:
            l = []
            for tile in self._feature_tile_list:
                l.append(tile.to_dict())
            d["feature_collection_tiles"] = l

        if self._structured_data_list is not None:
            l = []
            for entry in self._structured_data_list:
                l.append(entry.to_dict())
            d["structured_data_list"] = l

        if self._ml_model_list is not None:
            l = []
            for entry in self._ml_model_list:
                l.append(entry.to_dict())
            d["machine_learn_models"] = l

        return d

    @staticmethod
    def from_dict(udf_dict: Dict):
        """Create a udf data object from a python dictionary that was created from
        the JSON definition of the UdfData class

        Args:
            udf_dict (dict): The dictionary that contains the udf data definition

        Returns:
            UdfData:
            A new UdfData object

        """

        if "proj" not in udf_dict:
            raise Exception("Missing projection in dictionary")

        udf_data = UdfData(proj=udf_dict["proj"])

        if "raster_collection_tiles" in udf_dict:
            l = udf_dict["raster_collection_tiles"]
            for entry in l:
                rct = RasterCollectionTile.from_dict(entry)
                udf_data.append_raster_collection_tile(rct)

        if "hypercubes" in udf_dict:
            l = udf_dict["hypercubes"]
            for entry in l:
                h = HyperCube.from_dict(entry)
                udf_data.append_hypercube(h)

        if "feature_collection_tiles" in udf_dict:
            l = udf_dict["feature_collection_tiles"]
            for entry in l:
                fct = FeatureCollectionTile.from_dict(entry)
                udf_data.append_feature_collection_tile(fct)

        if "structured_data_list" in udf_dict:
            l = udf_dict["structured_data_list"]
            for entry in l:
                sd = StructuredData.from_dict(entry)
                udf_data.append_structured_data(sd)

        if "machine_learn_models" in udf_dict:
            l = udf_dict["machine_learn_models"]
            for entry in l:
                mlm = MachineLearnModel.from_dict(entry)
                udf_data.append_machine_learn_model(mlm)

        return udf_data


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

    # Import the python code into the current function context
    code = open(pyfile, "r").read()

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

    kv = gcore.parse_command("g.proj", flags="g")

    # Read several rows for each map and load them into the udf
    for index in range(0, region.rows, nrows):
        # Compute the number of usable rows
        if index + nrows > region.rows:
            usable_rows = index + nrows - region.rows + 1
        else:
            usable_rows = nrows
        # print("Usable rows", usable_rows)

        array = np.ndarray(shape=[len(map_list), usable_rows,
                                  region.cols],
                           dtype=RTYPE[mtype]['numpy'])

        for rmap, tindex in zip(open_maps, range(len(map_list))):
            for n in range(usable_rows):
                row = rmap[index + n]
                array[tindex][n][:] = row[:]

        extent = SpatialExtent(top=region.north, bottom=region.south,
                               left=region.west + index,
                               right=region.west + index + usable_rows,
                               height=region.nsres, width=region.ewres)

        rtile = RasterCollectionTile(id=sp.get_id(), data=array,
                                     start_times=start_time,
                                     end_times=end_time,
                                     extent=extent)

        data = UdfData(proj={"EPSG": kv["epsg"]},
                       raster_collection_tiles=[rtile, ])

        exec(code)

        rtiles = data.get_raster_collection_tiles()
        for slice in rtiles[0].data:
            print(f"Write slice at index {index} \n{slice}")
            for row in slice:
                # Write the result into the output raster map
                b = Buffer(shape=(region.cols,), mtype=mtype)
                b[:] = row[:]
                output_map.put_row(b)

    output_map.close()

    dbif.close()


if __name__ == "__main__":
    options, flags = gcore.parser()
    main()
