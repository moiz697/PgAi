MODULES = pgai
EXTENSION = pgai
DATA = pgai--1.0.sql
PGFILEDESC = "pgai - example extension for PostgreSQL"

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# Source files for your extension (add more as needed)
OBJS = pgai.o
PG_INCLUDE = -I$(libdir)/postgresql/include

# Compiler flags and options
PG_CFLAGS = $(CFLAGS) -std=c99 -fPIC
SHLIB_LINK = $(LDFLAGS) -lm

# Determine the shared object file extension based on the OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SO_EXT = dylib
else
  SO_EXT = so
endif

# Target shared object file
pgai.$(SO_EXT): $(OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

# Compilation rule for C source files
%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

# Installation targets
DATA_built = pgai--1.0.sql
DATA = $(DATA_built)
PG_VERSION := $(shell $(PG_CONFIG) --version | awk '{print $$2}')
EXTENSION_VERSION = 1.0

EXTENSION_CONTROL = pgai--1.0.control

