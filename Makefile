EXTENSION = PgAi
MODULE_big = PgAi
DATA = PgAi-1.0.sql
OBJS = PgAi.o 
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
