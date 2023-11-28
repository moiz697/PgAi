#include "postgres.h"
#include "access/htup_details.h"
#include "access/sysattr.h"
#include "access/xact.h"
#include "catalog/heap.h"
#include "catalog/pg_type.h"
#include "commands/trigger.h"
#include "executor/executor.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/syscache.h"
#include "utils/typcache.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "access/printtup.h"
#include "executor/spi.h"
#include "tcop/pquery.h"
#include "tcop/utility.h"
#include "utils/datum.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/queryjumble.h"
#include 'libpq-fe.h'
#include "fmgr.h"
#include "funcapi.h"
PG_MODULE_MAGIC;



void _PG_init(void);
void _PG_fini(void);

PG_FUNCTION_INFO_V1(pgai_hello);
PG_FUNCTION_INFO_V1(pgai_loading_data);
Datum hello(PG_FUNCTION_ARGS);
extern Datum get_postgres_version(PG_FUNCTION_ARGS);
Datum loading_data(PG_FUNCTION_ARGS);
PG_FUNCTION_INFO_V1(get_postgres_version);
static ProcessUtility_hook_type prev_ProcessUtility = NULL;

static void pgai_ProcessUtility(PlannedStmt *pstmt, const char *queryString,
			 bool readOnlyTree,
			 ProcessUtilityContext context,
			 ParamListInfo params, QueryEnvironment *queryEnv,
			 DestReceiver *dest,
			 QueryCompletion *qc);


/* ... C code here ... */
void _PG_init(void)
{
    /* ... C code here at time of extension loading ... */
    ProcessUtility_hook = prev_ProcessUtility;
}

void _PG_fini(void)
{
    /* ... C code here at time of extension unloading ... */
}

void pgai_ProcessUtility(PlannedStmt *pstmt, const char *queryString,
			 bool readOnlyTree,
			 ProcessUtilityContext context,
			 ParamListInfo params, QueryEnvironment *queryEnv,
			 DestReceiver *dest,
			 QueryCompletion *qc)

 {
	/* ... C code here ... */
    standard_ProcessUtility(pstmt,
                            queryString,
                            readOnlyTree,    
				context,
                            params,
                            queryEnv,
                            dest,
                            qc);
    /* ... C code here ... */
}

Datum pgai_hello(PG_FUNCTION_ARGS)
{
    text *result;

    /* Your custom code goes here, e.g., running a SQL query */
    SPI_connect();

    /* Execute a SQL query to load data from a CSV file into your_table */
    int ret = SPI_exec("COPY stock_data FROM '/Users/moizibrar/Downloads/pgai/archive/individual_stocks_5yr/individual_stocks_5yr/ADSK_data.csv' WITH CSV HEADER;", 0);

    if (ret < 0) {
        elog(ERROR, "Error executing COPY command: %s", SPI_result_code_string(ret));
    }

    SPI_finish();

    /* Return a text result */
    result = cstring_to_text("Hello World");

    PG_RETURN_TEXT_P(result);
}

Datum pgai_loading_data(PG_FUNCTION_ARGS)
{
    
  if (SPI_connect() != SPI_OK_CONNECT)
        ereport(ERROR, (errmsg("Failed to connect to SPI")));

    // Define the SQL statement to create the table
    const char *createTableSQL = 
        "CREATE TABLE stock_data ("
        "    date DATE,"
        "    open NUMERIC,"
        "    high NUMERIC,"
        "    low NUMERIC,"
        "    close NUMERIC,"
        "    volume BIGINT,"
        "    name VARCHAR(5)"
        ");";

    if (SPI_exec(createTableSQL, 0) != SPI_OK_UTILITY)
        ereport(ERROR, (errmsg("Failed to create the table")));

    SPI_finish();
    Datum helloResult = DirectFunctionCall1(pgai_hello, (Datum) 0);
    PG_RETURN_NULL();
}


PG_FUNCTION_INFO_V1(exec_py_script);

Datum
exec_py_script(PG_FUNCTION_ARGS)
{
    char* script_path = PG_GETARG_CSTRING(0);

    Py_Initialize();
    FILE* file = fopen(script_path, "r");

    if (file != NULL) {
        PyRun_SimpleFile(file, script_path);
        fclose(file);
    } else {
        ereport(ERROR,
                (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                 errmsg("Could not open Python script file")));
    }

    Py_Finalize();
    PG_RETURN_VOID();
}
