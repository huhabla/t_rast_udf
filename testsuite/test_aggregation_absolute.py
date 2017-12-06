"""Test t.rast.aggregation

(C) 2014 by the GRASS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Soeren Gebbert
"""
import os
import grass.pygrass.modules as pymod
import grass.temporal as tgis
from grass.gunittest.case import TestCase
from grass.gunittest.gmodules import SimpleModule

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
        return
        self.runModule("g.remove", flags="rf", type="raster", name="aggr_a")

    def test_1_aggregation_sum(self):
        """Disaggregation with empty maps"""
        udf_file = open("/tmp/udf_sum.py", "w")
        code = """
import pprint
import numpy as np
def udf_time_series_to_raster_map(t):
    #pprint.pprint(t)
    return np.sum(t["cell_array"], axis=0)
        """
        udf_file.write(code)
        udf_file.close()

        self.assertModule("t.rast.aggr_func", input="A",
                          output="aggr_a",pyfile="/tmp/udf_sum.py",
                          overwrite=True)

        self.assertRasterMinMax(map="aggr_a", refmin=600, refmax=600,
                                msg="Minimum must be 600")

    def test_2_aggregation_mean(self):
        """Disaggregation with empty maps"""
        udf_file = open("/tmp/udf_mean.py", "w")
        code = """
import pprint
import numpy as np
def udf_time_series_to_raster_map(t):
    #pprint.pprint(t)
    return np.mean(t["cell_array"], axis=0)
        """
        udf_file.write(code)
        udf_file.close()

        self.assertModule("t.rast.aggr_func", input="A",
                          output="aggr_a",pyfile="/tmp/udf_mean.py",
                          nrows=2, overwrite=True)

        self.assertRasterMinMax(map="aggr_a", refmin=200, refmax=200,
                                msg="Minimum must be 200")

    def test_3_aggregation_min(self):
        """Disaggregation with empty maps"""
        udf_file = open("/tmp/udf_min.py", "w")
        code = """
import pprint
import numpy as np
def udf_time_series_to_raster_map(t):
    #pprint.pprint(t)
    return np.min(t["cell_array"], axis=0)
        """
        udf_file.write(code)
        udf_file.close()

        self.assertModule("t.rast.aggr_func", input="A",
                          output="aggr_a",pyfile="/tmp/udf_min.py",
                          nrows=3, overwrite=True)

        self.assertRasterMinMax(map="aggr_a", refmin=100, refmax=100,
                                msg="Minimum must be 100")

    def test_4_aggregation_tdmean(self):
        """Disaggregation with empty maps"""
        udf_file = open("/tmp/udf_tdmean.py", "w")
        code = """
import pprint
import numpy as np
def udf_time_series_to_raster_map(t):
    #pprint.pprint(t)

    if t["end_time"] is not None:
        td = t["end_time"][-1] - t["start_time"][0]
    else:
        td = t["start_time"][-1] - t["start_time"][0]

    return np.sum(t["cell_array"], axis=0)/td.days
        """
        udf_file.write(code)
        udf_file.close()

        self.assertModule("t.rast.aggr_func", input="A",
                          output="aggr_a",pyfile="/tmp/udf_tdmean.py",
                          nrows=5, overwrite=True)

        self.assertRasterMinMax(map="aggr_a", refmin=100, refmax=100,
                                msg="Minimum must be 100")

    def test_5_aggregation_sum_where(self):
        """Disaggregation with empty maps"""
        udf_file = open("/tmp/udf_sum.py", "w")
        code = """
import pprint
import numpy as np
def udf_time_series_to_raster_map(t):
    #pprint.pprint(t)
    return np.sum(t["cell_array"], axis=0)
        """
        udf_file.write(code)
        udf_file.close()

        self.assertModule("t.rast.aggr_func", input="A",
                          output="aggr_a",pyfile="/tmp/udf_sum.py",
                          where="start_time > '2001-01-02'", overwrite=True)

        self.assertRasterMinMax(map="aggr_a", refmin=500, refmax=500,
                                msg="Minimum must be 500")


if __name__ == '__main__':
    from grass.gunittest.main import test
    test()
