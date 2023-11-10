-- Create a C function for the extension
CREATE FUNCTION pgai_hello() RETURNS text
AS 'pgai.so', 'pgai_hello'
LANGUAGE C;

CREATE FUNCTION pgai_loading_data() RETURNS text
AS 'pgai.so','pgai_loading_data'
LANGUAGE C;
