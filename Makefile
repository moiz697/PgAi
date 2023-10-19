# PostgreSQL Extension Makefile for PgAi

MODULE_big = PgAi
OBJS = PgAi.o  # Replace with the list of your object files

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

