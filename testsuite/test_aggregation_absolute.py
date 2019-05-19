"""Test t.rast.aggr_func

(C) 2017 by the GRASS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Soeren Gebbert
"""
import os
import grass.temporal as tgis
from grass.gunittest.case import TestCase


class TestAggregationAbsolute(TestCase):

    @classmethod
    def setUpClass(cls):
        """Initiate the temporal GIS and set the region
        """
        os.putenv("GRASS_OVERWRITE",  "1")
        tgis.init()
        cls.use_temp_region()
        cls.runModule("g.region",  s=0,  n=80,  w=0,  e=120,  b=0,
                      t=50,  res=10,  res3=10)
        cls.runModule("r.mapcalc", expression="a1 = 100.0",  overwrite=True)
        cls.runModule("r.mapcalc", expression="a2 = 200.0",  overwrite=True)
        cls.runModule("r.mapcalc", expression="a3 = 300.0",  overwrite=True)

        cls.runModule("t.create",  type="strds",  temporaltype="absolute",
                                    output="A",  title="A test",
                                    description="A test",  overwrite=True)

        cls.runModule("t.register", flags="i",  type="raster",  input="A",
                                     maps="a1,a2,a3",
                                     start="2001-01-01",
                                     increment="2 days",
                                     overwrite=True)


    @classmethod
    def tearDownClass(cls):
        """Remove the temporary region
        """
        cls.del_temp_region()
        cls.runModule("t.remove", flags="rf", type="strds", inputs="A")

    def tearDown(self):
        """Remove generated data"""
        self.runModule("g.remove", flags="rf", type="raster", name="aggr_a")

    def test_sum_aggregation_function(self):
        """Simple sum aggregation"""
        udf_file = open("/tmp/udf_ndvi_raster_collection.py", "w")
        code = """
def rct_sum_aggregate(udf_data):
    tile_results = []

    for tile in udf_data.raster_collection_tiles:
        tile_sum = numpy.sum(tile.data, axis=0)
        rows, cols = tile_sum.shape
        array3d = numpy.ndarray([1, rows, cols])
        array3d[0] = tile_sum
        if tile.start_times is not None and tile.end_times is not None:
            starts = pandas.DatetimeIndex([tile.start_times[0]])
            ends = pandas.DatetimeIndex([tile.end_times[-1]])
        else:
            starts = None
            ends = None

        rct = RasterCollectionTile(id=tile.id + "_sum", extent=tile.extent, data=array3d,
                                   start_times=starts, end_times=ends)
        tile_results.append(rct)
    udf_data.set_raster_collection_tiles(tile_results)

rct_sum_aggregate(data)

        """
        udf_file.write(code)
        udf_file.close()

        self.assertModule("t.rast.udf", input="A", output="B",
                          basename="aggr_a", pyfile="/tmp/udf_ndvi_raster_collection.py",
                          overwrite=True, nrows=3)

        self.assertRasterMinMax(map="aggr_a", refmin=600, refmax=600,
                                msg="Minimum must be 600")

    def test_pass_function(self):
        """Pass the input as output"""
        udf_file = open("/tmp/udf_pass_raster_collection.py", "w")
        code = """
def rct_sum(udf_data):
    pass

rct_sum(data)

        """
        udf_file.write(code)
        udf_file.close()

        self.assertModule("t.rast.udf", input="A", output="B",
                          basename="aggr_a", pyfile="/tmp/udf_pass_raster_collection.py",
                          overwrite=True, nrows=1)

        self.assertRasterMinMax(map="aggr_a_0", refmin=100, refmax=100,
                                msg="Minimum must be 100")

        self.assertRasterMinMax(map="aggr_a_1", refmin=200, refmax=200,
                                msg="Minimum must be 200")

        self.assertRasterMinMax(map="aggr_a_2", refmin=300, refmax=300,
                                msg="Minimum must be 300")


if __name__ == '__main__':
    from grass.gunittest.main import test
    test()
