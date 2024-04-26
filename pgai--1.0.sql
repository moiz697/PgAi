/* contrib/pg_stat_monitor/pg_stat_monitor--2.0.sql */

-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION pgai" to load this file. \quit

CREATE FUNCTION apple_stock (
out date    date,
out open    numeric,
out high    numeric,
out low     numeric,
out close   numeric,
out volume  bigint,

out close_pred    int
)
RETURNS SETOF record
AS 'MODULE_PATHNAME', 'apple_stock'
LANGUAGE C STRICT VOLATILE PARALLEL SAFE;

CREATE FUNCTION tesla_stock (
out date    date,
out open    numeric,
out high    numeric,
out low     numeric,
out close   numeric,
out volume  bigint,

out close_pred    int
)
RETURNS SETOF record
AS 'MODULE_PATHNAME', 'tesla_stock'
LANGUAGE C STRICT VOLATILE PARALLEL SAFE;

CREATE FUNCTION msci_pak_global_stock (
out date    date,
out open    numeric,
out high    numeric,
out low     numeric,
out close   numeric,
out volume  bigint,

out close_pred    int
)
RETURNS SETOF record
AS 'MODULE_PATHNAME', 'msci_pak_global_stock'
LANGUAGE C STRICT VOLATILE PARALLEL SAFE;
