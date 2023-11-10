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


PG_MODULE_MAGIC;



void _PG_init(void);
void _PG_fini(void);

PG_FUNCTION_INFO_V1(pgai_hello);
PG_FUNCTION_INFO_V1(pgai_loading_data);
Datum hello(PG_FUNCTION_ARGS);

Datum loading_data(PG_FUNCTION_ARGS);
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
    PG_RETURN_TEXT_P(cstring_to_text("Hello World"));
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
    PG_RETURN_NULL();
}