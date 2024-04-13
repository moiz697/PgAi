/* contrib/pg_stat_monitor/pg_stat_monitor--2.0.sql */

-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION pgai" to load this file. \quit

CREATE FUNCTION pg_test (
out date    date,
out open    numeric,
out high    numeric,
out low     numeric,
out close   numeric,
out volume  bigint,
out name    text,
out close_pred    int
)
RETURNS SETOF record
AS 'MODULE_PATHNAME', 'pg_test'
LANGUAGE C STRICT VOLATILE PARALLEL SAFE;

