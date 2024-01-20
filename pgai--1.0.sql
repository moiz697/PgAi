-- Create a C function for the extension
CREATE FUNCTION pgai_connect() RETURNS text
AS 'pgai.so', 'pgai_connect'
LANGUAGE C;

CREATE FUNCTION pgai_loading_data() RETURNS text
AS 'pgai.so','pgai_loading_data'
LANGUAGE C;

-- Create the extension if not already installed
CREATE EXTENSION IF NOT EXISTS plpython3u;

-- Create the C function
CREATE OR REPLACE FUNCTION call_python_script(path text)
  RETURNS void
  AS '/Users/moizibrar/Work/fyp/pgai.so', 'call_python_script'
  LANGUAGE C STRICT;

-- Create the PL/Python function
CREATE OR REPLACE FUNCTION run_python_script(path text)
  RETURNS text
  LANGUAGE plpython3u
  AS $$
    # Your Python script code here, using the provided path
    result = "Python script executed successfully"
    return result
  $$;
