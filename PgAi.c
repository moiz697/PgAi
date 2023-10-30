#include "postgres.h"
/* OS Includes */
/* PostgreSQL Includes */
PG_MODULE_MAGIC;
void _PG_init(void);
void _PG_fini(void);
Datum pg_all_queries(PG_FUNCTION_ARGS);
PG_FUNCTION_INFO_V1(pg_all_queries);
static void process_utility(PlannedStmt *pstmt, const char *queryString,ProcessUtilityContext context,ParamListInfo params,QueryEnvironment *queryEnv,DestReceiver *dest,       char *completionTag);

void _PG_init(void)
{
    /* ... C code here at time of extension loading ... */
    ProcessUtility_hook = process_utility;
}

Void _PG_fini(void)
{
    /* ... C code here at time of extension unloading ... */
}
static void process_utility(PlannedStmt *pstmt, 
                           const char *queryString,
                           ProcessUtilityContext context,
                           ParamListInfo params,
                           QueryEnvironment *queryEnv,DestReceiver *dest,
                           char *completionTag)
{
    /* ... C code here ... */
    standard_ProcessUtility(pstmt,  &nbsp;
                            queryString,  &nbsp;
                            context,  &nbsp;
                            params,  &nbsp;
                            queryEnv,  &nbsp;
                            dest, 
                            completionTag);
    /* ... C code here ... */
}
Datum pg_all_queries(PG_FUNCTION_ARGS)
{
    /* ... C code here ... */
    tupstore = tuplestore_begin_heap(true, false, work_mem);
    /* ... C code here ... */     
    values[0] = CStringGetTextDatum(query);
    values[1] = CStringGetTextDatum(pid);
    /* ... C code here ... */
    tuplestore_donestoring(tupstore);
    return (Datum) 0;
}
