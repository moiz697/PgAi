-- Create a C function for the extension
CREATE FUNCTION pgai_hello() RETURNS text
AS 'pgai.so', 'pgai_hello'
LANGUAGE C;

CREATE FUNCTION pgai_loading_data() RETURNS text
AS 'pgai.so','pgai_loading_data'
LANGUAGE C;

CREATE EXTENSION IF NOT EXISTS plpython3u;

CREATE OR REPLACE FUNCTION call_python_script(arg text)
RETURNS void AS
'$libdir/pgai', 
'call_python_script'
LANGUAGE C STRICT;
