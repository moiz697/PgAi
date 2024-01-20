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

#include "fmgr.h"
#include "funcapi.h"
#include "tcop/utility.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "utils/datum.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/queryjumble.h"
PG_MODULE_MAGIC;



void _PG_init(void);
void _PG_fini(void);

PG_FUNCTION_INFO_V1(pgai_connect);
PG_FUNCTION_INFO_V1(pgai_loading_data);
Datum pgai_connect(PG_FUNCTION_ARGS);
extern Datum get_postgres_version(PG_FUNCTION_ARGS);
Datum loading_data(PG_FUNCTION_ARGS);
PG_FUNCTION_INFO_V1(call_python_script);
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
    /* Free any dynamically allocated memory */
    if (my_allocated_memory != NULL) {
        pfree(my_allocated_memory);
        my_allocated_memory = NULL;
    }
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

Datum pgai_connect(PG_FUNCTION_ARGS)
{
    text *result;


    SPI_connect();

    
    int ret = SPI_exec("COPY stock_data FROM '/Users/moizibrar/Downloads/pgai/archive/individual_stocks_5yr/individual_stocks_5yr/ADSK_data.csv' WITH CSV HEADER;", 0);

    if (ret < 0) {
        elog(ERROR, "Error executing COPY command: %s", SPI_result_code_string(ret));
    }

    SPI_finish();

  
    result = cstring_to_text("Connected");

    PG_RETURN_TEXT_P(result);
}

Datum pgai_loading_data(PG_FUNCTION_ARGS)
{
    
  if (SPI_connect() != SPI_OK_CONNECT)
        ereport(ERROR, (errmsg("Failed to connect to SPI")));


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
    Datum pgai_connect = DirectFunctionCall1(pgai_connect, (Datum) 0);
    PG_RETURN_NULL();
}


Datum call_python_script(PG_FUNCTION_ARGS) {
    // Check if the path parameter is provided
    if (PG_ARGISNULL(0)) {
        ereport(ERROR, (errmsg("Path to Python script is required")));
        PG_RETURN_NULL();
    }

    // Get the path parameter
    text *path_text = PG_GETARG_TEXT_P(0);
    char *path = text_to_cstring(path_text);

    // Connect to SPI
    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("Failed to connect to SPI")));
        PG_RETURN_NULL();
    }

    // Execute the PL/Python function with the provided script path
    char query[256];
    snprintf(query, sizeof(query), "SELECT run_python_script('%s');", path);

    int ret = SPI_exec(query, 0);
    if (ret < 0) {
        elog(ERROR, "Error executing PL/Python function: %s", SPI_result_code_string(ret));
    }

    // Disconnect from SPI
    SPI_finish();

    PG_RETURN_NULL();
}