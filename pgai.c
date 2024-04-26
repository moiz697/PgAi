#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "access/heapam.h"
#include "utils/rel.h"
#include "utils/tuplestore.h"
#include "utils/snapmgr.h"
#include "catalog/pg_class.h" 
#include "nodes/makefuncs.h" 
#include "utils/snapshot.h" 

PG_MODULE_MAGIC;

void _PG_init(void);
void _PG_fini(void);

void _PG_init(void)
{
    /* Initialization code goes here */
}

void _PG_fini(void)
{
    /* Finalization code goes here */
}

PG_FUNCTION_INFO_V1(apple_stock);


Datum
apple_stock(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
    TupleDesc tupdesc;
    Tuplestorestate *tupstore;
    MemoryContext per_query_ctx;
	MemoryContext oldcontext;

    /* Check if caller supports returning a tuplestore */
    if (rsinfo == NULL)
        return (Datum) 0;
    /* Check if materialize mode is supported */
    if (!(rsinfo->allowedModes & SFRM_Materialize))
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("Materialize mode required, but it is not allowed in this context.")));
    /* Build a tuple descriptor for our result type */
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        elog(ERROR, "[pgai] apple_stock: Return type must be a row type.");


    if (tupdesc->natts != 7)
		elog(ERROR, "[pgai] apple_stock: Incorrect number of output arguments, received %d, required %d.", tupdesc->natts, 8);

	/* Switch into long-lived context to construct returned data structures */
	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);

   /* Initialize tuplestore */
    tupstore = tuplestore_begin_heap(true, false, 1000);
    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tupstore;
    rsinfo->setDesc = tupdesc;
    MemoryContextSwitchTo(oldcontext);

    /* Open the relation */
    Relation rel = relation_openrv(makeRangeVar("public", "apple_stock", -1), AccessShareLock);
        if (!RelationIsValid(rel))
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_TABLE),
                 errmsg("Relation not found")));
              
    /* Begin table scan */
    Snapshot snapshot = GetTransactionSnapshot();
    TableScanDesc scan = table_beginscan(rel, snapshot, 0, NULL);
    if (!HeapTupleIsValid(scan))
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Table scan failed")));

    /* Fetch rows and add to tuplestore */
    HeapTuple tuple;
    Datum values[8];
    bool nulls[8] = {false};

    while ((tuple = heap_getnext(scan, ForwardScanDirection)) != NULL)
    {
        if (!HeapTupleIsValid(tuple)) {
            elog(ERROR, "Invalid tuple"); // Log error if the tuple is invalid
            break; // Exit the loop if the tuple is invalid
        }

        values[0] = heap_getattr(tuple, 1, tupdesc, &nulls[0]);
        values[1] = heap_getattr(tuple, 2, tupdesc, &nulls[1]);
        values[2] = heap_getattr(tuple, 3, tupdesc, &nulls[2]);
        values[3] = heap_getattr(tuple, 4, tupdesc, &nulls[3]);
        values[4] = heap_getattr(tuple, 5, tupdesc, &nulls[4]);
        values[5] = heap_getattr(tuple, 6, tupdesc, &nulls[5]);
        values[6] = 1000;
      
      
        
        //values[1] =12331;
    
        tuplestore_putvalues(tupstore, tupdesc, values, nulls);
    }
    
 
   /* Cleanup */
    table_endscan(scan);
    relation_close(rel, AccessShareLock);
	tuplestore_donestoring(tupstore);

    return (Datum) 0;
}
PG_FUNCTION_INFO_V1(tesla_stock);


Datum
tesla_stock(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
    TupleDesc tupdesc;
    Tuplestorestate *tupstore;
    MemoryContext per_query_ctx;
	MemoryContext oldcontext;

    /* Check if caller supports returning a tuplestore */
    if (rsinfo == NULL)
        return (Datum) 0;
    /* Check if materialize mode is supported */
    if (!(rsinfo->allowedModes & SFRM_Materialize))
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("Materialize mode required, but it is not allowed in this context.")));
    /* Build a tuple descriptor for our result type */
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        elog(ERROR, "[pgai] tesla_stock: Return type must be a row type.");


    if (tupdesc->natts != 7)
		elog(ERROR, "[pgai] tesla_stock: Incorrect number of output arguments, received %d, required %d.", tupdesc->natts, 8);

	/* Switch into long-lived context to construct returned data structures */
	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);

   /* Initialize tuplestore */
    tupstore = tuplestore_begin_heap(true, false, 1000);
    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tupstore;
    rsinfo->setDesc = tupdesc;
    MemoryContextSwitchTo(oldcontext);

    /* Open the relation */
    Relation rel = relation_openrv(makeRangeVar("public", "tesla_stock", -1), AccessShareLock);
        if (!RelationIsValid(rel))
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_TABLE),
                 errmsg("Relation not found")));
              
    /* Begin table scan */
    Snapshot snapshot = GetTransactionSnapshot();
    TableScanDesc scan = table_beginscan(rel, snapshot, 0, NULL);
    if (!HeapTupleIsValid(scan))
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Table scan failed")));

    /* Fetch rows and add to tuplestore */
    HeapTuple tuple;
    Datum values[8];
    bool nulls[8] = {false};

    while ((tuple = heap_getnext(scan, ForwardScanDirection)) != NULL)
    {
        if (!HeapTupleIsValid(tuple)) {
            elog(ERROR, "Invalid tuple"); // Log error if the tuple is invalid
            break; // Exit the loop if the tuple is invalid
        }

        values[0] = heap_getattr(tuple, 1, tupdesc, &nulls[0]);
        values[1] = heap_getattr(tuple, 2, tupdesc, &nulls[1]);
        values[2] = heap_getattr(tuple, 3, tupdesc, &nulls[2]);
        values[3] = heap_getattr(tuple, 4, tupdesc, &nulls[3]);
        values[4] = heap_getattr(tuple, 5, tupdesc, &nulls[4]);
        values[5] = heap_getattr(tuple, 6, tupdesc, &nulls[5]);
        values[6] = 22;
       
      
        
        //values[1] =12331;
    
        tuplestore_putvalues(tupstore, tupdesc, values, nulls);
    }
    
 
   /* Cleanup */
    table_endscan(scan);
    relation_close(rel, AccessShareLock);
	tuplestore_donestoring(tupstore);

    return (Datum) 0;
}

PG_FUNCTION_INFO_V1(msci_pak_global_stock);


Datum
msci_pak_global_stock(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
    TupleDesc tupdesc;
    Tuplestorestate *tupstore;
    MemoryContext per_query_ctx;
	MemoryContext oldcontext;

    /* Check if caller supports returning a tuplestore */
    if (rsinfo == NULL)
        return (Datum) 0;
    /* Check if materialize mode is supported */
    if (!(rsinfo->allowedModes & SFRM_Materialize))
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("Materialize mode required, but it is not allowed in this context.")));
    /* Build a tuple descriptor for our result type */
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        elog(ERROR, "[pgai] msci_pak_global_stock: Return type must be a row type.");


    if (tupdesc->natts != 7)
		elog(ERROR, "[pgai] msci_pak_global_stock: Incorrect number of output arguments, received %d, required %d.", tupdesc->natts, 8);

	/* Switch into long-lived context to construct returned data structures */
	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);

   /* Initialize tuplestore */
    tupstore = tuplestore_begin_heap(true, false, 1000);
    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tupstore;
    rsinfo->setDesc = tupdesc;
    MemoryContextSwitchTo(oldcontext);

    /* Open the relation */
    Relation rel = relation_openrv(makeRangeVar("public", "msci_pak_global_stock", -1), AccessShareLock);
        if (!RelationIsValid(rel))
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_TABLE),
                 errmsg("Relation not found")));
              
    /* Begin table scan */
    Snapshot snapshot = GetTransactionSnapshot();
    TableScanDesc scan = table_beginscan(rel, snapshot, 0, NULL);
    if (!HeapTupleIsValid(scan))
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Table scan failed")));

    /* Fetch rows and add to tuplestore */
    HeapTuple tuple;
    Datum values[8];
    bool nulls[8] = {false};

    while ((tuple = heap_getnext(scan, ForwardScanDirection)) != NULL)
    {
        if (!HeapTupleIsValid(tuple)) {
            elog(ERROR, "Invalid tuple"); // Log error if the tuple is invalid
            break; // Exit the loop if the tuple is invalid
        }

        values[0] = heap_getattr(tuple, 1, tupdesc, &nulls[0]);
        values[1] = heap_getattr(tuple, 2, tupdesc, &nulls[1]);
        values[2] = heap_getattr(tuple, 3, tupdesc, &nulls[2]);
        values[3] = heap_getattr(tuple, 4, tupdesc, &nulls[3]);
        values[4] = heap_getattr(tuple, 5, tupdesc, &nulls[4]);
        values[5] = heap_getattr(tuple, 6, tupdesc, &nulls[5]);
        values[6] = 1000;
      
      
        
        //values[1] =12331;
    
        tuplestore_putvalues(tupstore, tupdesc, values, nulls);
    }
    
 
   /* Cleanup */
    table_endscan(scan);
    relation_close(rel, AccessShareLock);
	tuplestore_donestoring(tupstore);

    return (Datum) 0;
}

Datum heap_open_and_retrieve_rows(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
    TupleDesc tupdesc = NULL;
    Tuplestorestate *tupstore;

    if (rsinfo == NULL || rsinfo->isDone)
        PG_RETURN_NULL();

    tupdesc = CreateTemplateTupleDesc(1);
    TupleDescInitEntry(tupdesc, (AttrNumber) 1, "value", INT8OID, -1, 0);

    tupstore = tuplestore_begin_heap(true, false, 0);
    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tupstore;
    rsinfo->setDesc = tupdesc;

    /* Open the relation */
    Relation rel = relation_openrv(makeRangeVar("public", "msci_pak_global_stock", -1), AccessShareLock);
    if (!RelationIsValid(rel))
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_TABLE),
                 errmsg("Relation not found")));

    /* Start scanning the relation */
    elog(WARNING, "Starting scan of public.msci_pak_global_stock table");
    Snapshot snapshot = GetTransactionSnapshot();
    elog(WARNING, "Snapshot xmin: %u, xmax: %u, xcnt: %d", snapshot->xmin, snapshot->xmax, snapshot->xcnt);

    TableScanDesc scan = table_beginscan(rel, snapshot, 0, NULL);
    if (!HeapTupleIsValid(scan))
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Table scan failed")));

    /* Fetch rows and add to tuplestore */
    HeapTuple tuple;
    Datum values[1];
    bool nulls[1] = {false};

    while ((tuple = heap_getnext(scan, ForwardScanDirection)) != NULL)
    {
        values[0] = heap_getattr(tuple, 1, tupdesc, &nulls[0]);
        tuplestore_putvalues(tupstore, tupdesc, values, nulls);
    }

    /* Cleanup */
    table_endscan(scan);
    relation_close(rel, AccessShareLock);

    tuplestore_donestoring(tupstore);

    PG_RETURN_NULL();
}        

