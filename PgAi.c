
#include "postgres.h"

#include <math.h>
#include <sys/stat.h>
#include <unistd.h>

#include "access/parallel.h"
#include "catalog/pg_authid.h"
#include "common/hashfn.h"
#include "executor/instrument.h"
#include "funcapi.h"
#include "mb/pg_wchar.h"
#include "miscadmin.h"
#include "optimizer/planner.h"
#include "parser/analyze.h"
#include "parser/parsetree.h"
#include "parser/scanner.h"
#include "parser/scansup.h"
#include "pgstat.h"
#include "storage/fd.h"
#include "storage/ipc.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "storage/spin.h"
#include "tcop/utility.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "utils/queryjumble.h"
#include "utils/memutils.h"
#include "utils/timestamp.h"

PG_MODULE_MAGIC;



void _PG_init(void);
void _PG_fini(void);
Datum pg_all_queries(PG_FUNCTION_ARGS);
PG_FUNCTION_INFO_V1(pg_all_queries);


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


