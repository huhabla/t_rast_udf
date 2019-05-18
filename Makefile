MODULE_TOPDIR = ../grass

PGM = t.rast.aggr_func

include $(MODULE_TOPDIR)/include/Make/Script.make

default: script $(TEST_DST)
