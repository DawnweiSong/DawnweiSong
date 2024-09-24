# Databricks notebook source
# MAGIC %md #Get self path

# COMMAND ----------

# DBTITLE 1,Accumulated helper functions for databricks, by Xiaowei Song
#get current notebook fullpath
dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
#Xiaowei Song (Xiaowei.Song@gdit.com)
#Version: 20230808

# COMMAND ----------

#This is just getting the user ID for the workshop so everyone is split
userID = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
# user_id = ''.join(filter(str.isdigit, user_id))
print(userID)

# COMMAND ----------

# logging.basicConfig(level=logging.INFO,  format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',  filename='/Workspace/Users/xiaowei.song@gdit.com/lib/logs/test.log',  filemode='a', force=True)
# logging.info( " ".join(f"'{i}'" if " " in i else i for i in ["a a", "w"]) )
# logging.info("test finished")
# logging.shutdown()

# COMMAND ----------

# %sh
# #cat /Workspace/Users/xiaowei.song@gdit.com/lib/logs/test.log
# ls /Workspace/Users/xiaowei.song@gdit.com/lib/logs/ -hl

# COMMAND ----------

def findFiles(path, verbose=False):
#https://community.databricks.com/t5/data-engineering/how-to-get-the-total-directory-size-using-dbutils/m-p/27293#M19170
#Non-recursive version to get all files under a dir and all its sub-dirs
  dirs = []
  dirs = dbutils.fs.ls(path)
  files = []
  while len(dirs) > 0:
    currentDirs = dirs
    dirs = []
    for d in currentDirs:
      if verbose: print(f"Processing {d.path}")
      else: print('.', end='')
      children = dbutils.fs.ls(d.path)
      for child in children:
        if child.size == 0 and child.path != d.path: #child is a dir
          dirs.append(child)
        elif child.path != d.path: #child is a file
          files.append(child)
  return files
def size4dir(path, verbose=False): 
  allFiles=findFiles(path, verbose=verbose)
  allSize =sum([file.size for file in allFiles])
  print(f'{path} having {len(allFiles)} files, total {allSize/1024/1024/1024:.3g} GB')  
  return allSize, allFiles if verbose else allSize

# COMMAND ----------

# MAGIC %md #Log init

# COMMAND ----------

import sys,os,time, math, re
os.environ['TZ']='US/Eastern'
time.tzset()
from pathlib import Path

__file__='/Workspace/' +dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get() #get Databricks notebook filename
cwd4nb=Path(__file__).parent
import logging 
from datetime import datetime
flog=f'{cwd4nb}/logs/{Path(__file__).stem + datetime.now().strftime("_%Y%m%d.%H%M%S")}.log'
# print(Path(flog).parent)
Path(flog).parent.mkdir(parents=True, exist_ok=True) #permission granted in E2, 20230815
logging.basicConfig(level=logging.INFO,  format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',  filename=flog,  filemode='a')
logging.info( " ".join(f"'{i}'" if " " in i else i for i in sys.argv) )
def log(msg): 
  print(msg + f" | {datetime.now()}")
  logging.info(msg)
def loge(msg): 
  print(msg + f" | {datetime.now()}")
  logging.error(msg)  
def logw(msg): 
  print(msg + f" | {datetime.now()}")
  logging.warning(msg)   


import pandas as pd
log('pandas version: %s' % pd.__version__)
import pyspark.pandas as ps
# log('pyspark pandas version: %s' % ps.__version__)

import numpy as np
log('numpy version: %s' % np.__version__)
import scipy
log(f'scipy version: {scipy.__version__}')


# import databricks.koalas as ks
# log('koalas version: %s' % ks.__version__)

from tqdm import tqdm

import pyspark, gc
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, pandas_udf, col


# log(f'spark: {spark.version} , Databricks: {spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")}') #this will trigger a warning on PYARROW_IGNORE_TIMEZONE

import sklearn
log('sklearn version: %s' % sklearn.__version__)
# print(f'{"----"*20}\nsklearn checking versions:')
# sklearn.show_versions()
# print(f'{"----"*20}\n')

try:
  import hyperopt
  log(f"hyperopt version: {hyperopt.__version__}")
except Exception as e:
  log(f"hyperopt does not exist: {e}")

import xgboost as xgb
log('xgboost version: %s' % xgb.__version__)

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
log(f"matplotlib version: {mpl.__version__} . seaborn version: {sns.__version__}")



import joblib
log(f"joblib version:{joblib.__version__}")

# COMMAND ----------

# MAGIC %md #Tables functions

# COMMAND ----------

def hsize(num, suffix="B", binary='i'): #human readable size
  if binary=='i': pow2_10=1024.0
  else: pow2_10=1000.0 #binary  should=""
  for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
    if abs(num) < pow2_10: return f"{num:3.1f} {unit}{suffix}"
    num /= pow2_10
  return f"{num:.1f} Y{binary}{suffix}"


def getSchemaIfExist(ptn="", schema='FPS_Models'):
  if '.' in ptn:
    ptn2=ptn.split('.')  
    schema=ptn2[0]
    ptn=ptn2[1] #override the input  
  return ptn, schema
def existTable(tbl="", schema='FPS_Models'):
  tbl, schema=getSchemaIfExist(tbl, schema)
  nc=0
  try:
    r=spark.sql(f"select * from {schema}.{tbl} limit 1").to_koalas()
    nc=len(r.columns)
  except Exception as e:
    log(e)
  return nc>0

# def getTableAssociatedNotebook(ptn="", schema='FPS_Models'):
#   tbl, schema=getSchemaIfExist(tbl, schema)
#   r=spark.sql(f"describe history {schema}.{tbl}")


def showTableColumns(tbl="", schema='FPS_Models', catalog="mdev_catalog"):
  tbl, schema=getSchemaIfExist(tbl, schema)
  # r=spark.sql(f"""SELECT ordinal_position,column_name, data_type
  #   FROM sysem.information_schema.columns
  #   WHERE table_catalog= {catalog}
  #     and table_schema = {schema}
  #     AND table_name = {tbl}
  #   ORDER BY ordinal_position""")
  r=spark.sql(f"describe table {schema}.{tbl}")
  return r

def getTableColumns(tbl="", schema='FPS_Models'):
  tbl, schema=getSchemaIfExist(tbl, schema)
  schema=schema.strip()
  table=f"{schema}.{tbl}"
  if len(schema)==0: table=f"{tbl}"
  r=spark.sql(f"select * from {table} limit 1").to_koalas()
  return r.columns
def getTableColumnsCount(tbl="", schema='FPS_Models'):  
  tbl, schema=getSchemaIfExist(tbl, schema)
  return len(getTableColumns(tbl=tbl, schema=schema))
def getTableRowsCount(tbl="", schema='FPS_Models'):
  tbl, schema=getSchemaIfExist(tbl, schema)
  return spark.sql(f"select count(*) from {schema}.{tbl}").collect()[0][0]
def getTableSize(tbl="", schema='FPS_Models'):
  tbl, schema=getSchemaIfExist(tbl, schema)
  return spark.sql(f"describe detail {schema}.{tbl}").select("sizeInBytes").collect()[0]["sizeInBytes"]  

def rmTables(ptn='OpioidSNA_degree*', schema='FPS_MODELS', doRm=False, verbose=False):
#   ptn='OpioidSNA_evenRate*'
  #extract schema if existing in pattern  
  ptn, schema=getSchemaIfExist(ptn, schema)
  print(f"Using schema={schema} and checking tables like '{ptn}'")
  pTables = spark.sql(f"show tables from {schema} like '{ptn}'")
  if pTables.count()==0: return []
  pTables =pTables.orderBy(col('tableName').asc()) #having 3 columns, DW name, tableName, IsTemporary
#   pTables=spark.sql(f"""
#   with t as (select * from information_schema.tables where table_schema='{schema}' and table_name like '{ptn}')
#   select * from t sort by t.tableName asc;
#   """) #this information_schema.tables does not work !!!    
  if verbose or doRm:
    # display(pTables)  
    tinfo = pTables.withColumn("tableSize", F.lit(.0)).withColumn("#row", F.lit(0)).withColumn("#col", F.lit(0)).toPandas()
    # for r in pTables.collect():
    for i in range(len(tinfo.index)):
      r=tinfo.iloc[i,:]
      ridx=tinfo.index[i]
      try:
        ts=getTableSize(tbl=r["tableName"], schema=r["database"])
        tinfo.loc[ridx, "tableSize"]=ts
      except e as Exception:
        log(e) #view does not have describe detail, which only works for table
      tr=getTableRowsCount(tbl=r["tableName"], schema=r["database"])      
      tinfo.loc[ridx, "#row"]=tr
      cr=getTableColumnsCount(tbl=r["tableName"], schema=r["database"])      
      tinfo.loc[ridx, "#col"]=cr
      print(f'Table {r["tableName"]} (isTemporary={r["isTemporary"]}) from {r["database"]} , size= {hsize(ts)} , #rows= {hsize(tr, binary="", suffix="")} , #columns= {hsize(cr, binary="", suffix="")}')
      if doRm: 
        print(f'rm(={doRm}) {r["database"]}.{r["tableName"]}')
        spark.sql(f'drop table if exists {r["database"]}.{r["tableName"]}')
    display(tinfo)
  allTablesList=pTables.select('tableName').rdd.flatMap(lambda x:x).collect()
  print(f"Found {len(allTablesList)} tables in total")
  return allTablesList
showTables=rmTables

# COMMAND ----------

from termcolor import colored
    
def findColumnsContainNull(tbl=''):
#   tbl='FPS_MODELS.OpioidSNA_PharmSUM1'
  print(f"Checking table '{tbl}' to find out which column has null values")
  columns=spark.sql(f"select * from {tbl}").columns  
  redCol=[]
  for i,col in enumerate(columns):
    hasNull=spark.sql(f"select any(isnull({col})) from {tbl}").rdd.flatMap(lambda x:x).collect()[0]
    if hasNull: 
      print(f"{i:03d}: hasNull(col='{col}') is {colored('TRUE', 'red')}")
      redCol.append(col)
    else: print(f"{i:03d}: hasNull(col='{col}') is {hasNull}")
  print(f"#col= {len(redCol)} marked as red: {colored(str(redCol), 'red')}")

# COMMAND ----------

# MAGIC %md ##Join tables' value

# COMMAND ----------

def joinTableValues(tblPtn='*docFrom*', schema='FPS_Models', save2table='docFeatures', verbose=False, prfx4df='', exceptTableName=''):
  docTables = showTables(tblPtn, schema=schema, verbose=verbose)
  docTables.sort(reverse=True) #desc
  if len(exceptTableName)>0:
    idx2rm=[]
    for i in range(len(docTables)):
      if exceptTableName.lower() in docTables[i].lower(): idx2rm.append(i)
    docTables=[ele for i,ele in enumerate(docTables) if i not in idx2rm]
  print(f"Joining tables (len={len(docTables)}): {docTables}")
  t0=docTables.pop(0)
  jsel =f"select t0.id as id, t0.value as {t0}"  + " ".join([f" ,t{1+i}.value as {tbl}" for i,tbl in zip(range(len(docTables)),docTables)])
  jfrom=f"from {schema}.{t0} as t0" + " ".join([f" join {schema}.{tbl} as t{i+1} on t{i+1}.id=t0.id" for i,tbl in zip(range(len(docTables)),docTables)])
  cmd=f"{jsel} {jfrom}"
  if verbose: print(cmd)
  df = spark.sql(cmd)  
  if len(save2table)>0: df.write.mode("overwrite").format('delta').saveAsTable(prfx4df+save2table)
  return df

# COMMAND ----------

# MAGIC %md #DataFrame functions

# COMMAND ----------


#rename DataFrame columns using a dict to do the mapping
def renameColumns(df, mvDict):
  for k,v in mvDict.items():
    df = df.withColumnRenamed(k, v)
  return df
def aliasColumns(df, mvDict):
  return df.select([col(k).alias(v) for k,v in mvDict.items()])

# COMMAND ----------

import contextlib
@contextlib.contextmanager
def cached(df):
  df_cached = df.cache()
  try:
    yield df_cached
  finally:
    df_cached.unpersist()

# COMMAND ----------

def rndRows(df: pyspark.sql.DataFrame, n:int =10)-> pyspark.sql.DataFrame:
  nTotal=df.count()
  n2sample = n if nTotal>n else nTotal
  return df.sample(withReplacement=False, fraction=1.0*n2sample/nTotal).limit(n2sample)
#   return df.sample(withReplacement=False, fraction=1.0).limit(n2sample)

def rndRows4tbl(tbl: str, n:int =10)-> pyspark.sql.DataFrame:
  df=spark.sql(f'select * from {tbl}')
  return rndRows(df, n)

def rndRows4df(df: pyspark.sql.DataFrame, n:int =10, prfx="OpioidSNA_")-> pyspark.sql.DataFrame:
#   tmptbl=f"FPS_Models.{prfx}tmp_rndRows4df"
#   df.write.mode("overwrite").saveAsTable(tmptbl)
#   return rndRows4tbl(tmptbl, n)
#   tdf=df.cache()
  return rndRows(cached(df), n)

# COMMAND ----------

def getNumericFeaturesDf(asql=f"""select * from FPS_Models.OpioidSNA_docFeatures1""", id='id'):
  docFeatures=spark.sql(asql) #.withColumn('prvdr_spclty_cd')
  docFeaturesDf=docFeatures.toPandas().fillna(0)
  if len(id)>0:docFeaturesDf.set_index(id, inplace=True)
  for dtn, dtv in docFeaturesDf.dtypes.items():
    print(f"{dtn} : {dtv}")
    if dtv is np.dtype('object'): 
      docFeaturesDf.loc[:, dtn]=pd.to_numeric(docFeaturesDf.loc[:, dtn])
      print(f" {' '*10} -> {docFeaturesDf.dtypes[dtn]}")    
  return docFeaturesDf

# COMMAND ----------

# DBTITLE 1,sql2pandas
def sql2pandas(asql=f"select * from FPS_Models.DMEReferProviderML_ReferralNPISummary_HCPCS", dataTypeMapping={'Decimal':pyspark.sql.types.FloatType()}):
  from pyspark.sql.types import FloatType
  from pyspark.sql.functions import col
  df=spark.sql(asql)
  #find all decimal columns in your SparkDF and do the mappings specified
  for t in dataTypeMapping:
    # decimals_cols = [c for c in df.columns if 'Decimal' in str(df.schema[c].dataType)]
    decimals_cols = [c for c in df.columns if t in str(df.schema[c].dataType)]
    #convert all decimals columns to floats
    for col in decimals_cols:  df = df.withColumn(col, df[col].cast(dataTypeMapping[t]))

  #Now you can easily convert Spark DF to Pandas DF without decimal errors
  pandas_df = df.toPandas() 
  return pandas_df


# COMMAND ----------

# showTables('*tmp_rnd*', doRm=True)

# COMMAND ----------

def getRDDinfo():
  rddinfo=[{
      "name": s.name(),     
  #     "memSize_MB": float(s.memSize())/ 2**20 , 
      "memSize_GB": float(s.memSize())/ 2**30, 
  #     "diskSize_MB": float(s.diskSize())/ 2**20, 
      "diskSize_GB": float(s.diskSize())/ 2**30, 
      "numPartitions": s.numPartitions(), 
      "numCachedPartitions": s.numCachedPartitions(),
      "callSite": s.callSite(),
  #     "externalBlockStoreSize": s.externalBlockStoreSize(),
      "id": s.id(),
      "isCached": s.isCached(),
      "parentIds": s.parentIds(),
      "scope": s.scope(),
      "storageLevel": s.storageLevel(),
      "toString": s.toString()
  } for s in sc._jsc.sc().getRDDStorageInfo()]
  return rddinfo


# COMMAND ----------

# MAGIC %md ##hist for a table's columns

# COMMAND ----------

import importlib.util
def ExistModule(module="spam"):
  spam_spec = importlib.util.find_spec(module)
  return spam_spec is not None

try:  
  if not ExistModule('pyspark_dist_explore'): 
    print(f"pyspark_dist_explore was not found/installed!")
    exit
  else:
    import pyspark_dist_explore
    from pyspark_dist_explore import hist,distplot
  import matplotlib.pyplot as plt
except Exception as e:
  print(f"Exception occured in detecting and importing pyspark_dist_explore: {pyspark_dist_explore.__file__}")

def findRightTailThrd(bins, edges, thrd=.99):
  nbin=len(bins)
  t=np.sum(bins) #get the total number of samples
  #convert bins to freq/prob
  freq=bins/(t+1e-10)
  cdf = np.cumsum(freq)
  idx=np.argmax(cdf>=thrd)
  return idx, edges[idx]


def hist4table(columns=["x", "y"]+["z"], table=f"pharmFeatures20k", asql=None, nbin=200, width=12, height=6, rightTail=[.99], distFit=False, schema=''):
  if not ExistModule('pyspark_dist_explore'): 
    print(f"pyspark_dist_explore was not found/installed!")    
    exit
  else:
    # from pyspark_dist_explore import hist,distplot #todo in Databricks 13.3 LTS
    pass
  if columns==[]: columns=getTableColumns(tbl=table, schema=schema) #default with all columns
  nCol=len(columns)
  fig, ax = plt.subplots(nrows=nCol, ncols=1)
  fig.set_size_inches([width, height*nCol])  
  hists=[]
  if nCol==1: ax=[ax]
  for i,ncol in zip(range(nCol), columns):
    if asql is None: df=spark.sql(f"select {ncol} from {table}").select(F.col(ncol).cast("float"))
    else: df=spark.sql(asql)
    if not distFit:
      h=hist(ax[i], df, bins = nbin)
    else:
      h=distplot(ax[i], df, bins = nbin)
    ax[i].set_title(ncol)
    hists.append(h)
    if not isinstance(rightTail, list): rightTail=[rightTail]
    for rt in rightTail:
      if rt>0: #plot 5% thrd line
        idx, v4thrd=findRightTailThrd(h[0], h[1], thrd=rt)
        ymin, ymax=ax[i].get_ylim()
        ax[i].vlines(x=[v4thrd], ymin=ymin, ymax=ymax, ls='--', lw=2, alpha=.8, colors=['r'])      
        ax[i].annotate(f"{100*rt:g}%/{v4thrd:g}", xy=[v4thrd, ymax])
  fig.tight_layout()  
  return fig, hists

# COMMAND ----------

# DBTITLE 1,featureInspect for Binary classes
def featureInspect(asql, feature='age', index='rNPI', hist=True, freq=True, thrdName='RiskEst', thrd=0.5, bins=100 , labels=['Normal', 'Risky'], **kwargs):
  # ages=sql2pandas(f'select distinct s.CLM_RFRG_PRVDR_NPI_NUM as rNPI, s.age, coalesce(s.bad+s.cofraud+s.cnc+s.biu, 0) as bad, e.RiskEst  from {prfx}ReferralNPISummary_HCPCS{ptfx} as s join  {prfx}rNPIRiskEst{ptfx} as e on s.CLM_RFRG_PRVDR_NPI_NUM=e.CLM_RFRG_PRVDR_NPI_NUM')
  import pandas as pd
  if isinstance(asql , str):  ages=sql2pandas(asql)
  elif isinstance(asql, pd.DataFrame): ages=asql.copy()
  else: raise Exception("asql should be sql str or a pandas df")
  
  ages.set_index(index, inplace=True)
  
  # display(ages)

  import matplotlib.pyplot as plt
  import seaborn as sns
  fig, ax=plt.subplots()
  sns.distplot(ages.loc[ages[thrdName]<thrd  ,feature],hist=hist,  kde=freq, bins=bins, label=labels[0], **kwargs)
  sns.distplot(ages.loc[ages[thrdName]>=thrd ,feature],hist=hist,  kde=freq, bins=bins, label=labels[1], **kwargs)
  ax.set_xlabel(feature)
  # ax.set_ylabel('Frequency' if freq else 'Count')
  plt.legend()
  return ages
#  featureInspect(f'select distinct s.CLM_RFRG_PRVDR_NPI_NUM as rNPI, s.age, coalesce(s.bad+s.cofraud+s.cnc+s.biu, 0) as bad, e.RiskEst  from {prfx}ReferralNPISummary_HCPCS{ptfx} as s join  {prfx}rNPIRiskEst{ptfx} as e on s.CLM_RFRG_PRVDR_NPI_NUM=e.CLM_RFRG_PRVDR_NPI_NUM', feature='age', index='rNPI', freq=True, thrdName='RiskEst', thrd=params['ReferProvider_riskScoreGe'], bins=100 , labels=['Normal', 'Risky'])
def biViolinInspect(asql, feature='age', index='rNPI', thrdName='RiskEst', thrd=0.5, labels=['Normal', 'Risky'], **kwargs):
  import pandas as pd
  if isinstance(asql , str):  ages=sql2pandas(asql)
  elif isinstance(asql, pd.DataFrame): ages=asql.copy()
  else: raise Exception("asql should be sql str or a pandas df")

  ages.set_index(index, inplace=True)
  # display(ages)
  ages.loc[ages[thrdName]<thrd, 'label']=labels[0]
  ages.loc[ages[thrdName]>=thrd, 'label']=labels[1]  

  import matplotlib.pyplot as plt
  import seaborn as sns
  fig, ax=plt.subplots()
  sns.violinplot(data=ages, x="label", y=feature, hue="label",**kwargs)  
  # ax.set_ylabel(feature)
  plt.legend()
  return ages

# COMMAND ----------

# DBTITLE 1,pptHist4attr, 10 percentiles
def pptHist4attr(asql=f'select riskEst as Risk from fps_models.dmeReferProviderML_rNPIRiskEstTest', attr='Risk', v4thrd=.6, figsize=[8,2], bins=np.arange(0,1,.1), kde=True, ylim=None, **kwargs):  
  import matplotlib.pyplot as plt
  import seaborn as sns
  import numpy as np
  from scipy.stats import percentileofscore
  import pandas as pd 
  rf=sql2pandas(asql)
  fig, ax = plt.subplots(figsize=figsize)
  h=sns.histplot(data=rf, x=attr, bins=bins, kde=kde, stat="percent", element="bars", **kwargs)
  labels = [f'{v:.2f}%' for v in h.containers[0].datavalues]
  h.bar_label(h.containers[0], labels=labels)
  # v4thrd=.6
  if ylim: ax.set_ylim(ylim)
  ymin, ymax=ax.get_ylim()
  h.vlines(x=[v4thrd], ymin=ymin, ymax=ymax, ls='--', lw=1, alpha=.8, colors=['r'])
  ax.annotate(f"{v4thrd:g} | {percentileofscore(rf[attr], v4thrd):.1f}% percentile", xy=[v4thrd, ymax])
  print(f"Percentile of the threshold is: {percentileofscore(rf[attr], v4thrd)}%")
  #report quantiles
  qt={'min':rf[attr].min(), 
      '1%':rf[attr].quantile(.01),
      '5%':rf[attr].quantile(.05),
      '10%':rf[attr].quantile(.1),
      '25%':rf[attr].quantile(.25),
      'mean':rf[attr].mean(),
      '50%':rf[attr].quantile(.5),
      '75%':rf[attr].quantile(.75),
      '90%':rf[attr].quantile(.9),
      '95%':rf[attr].quantile(.95),
      '99%':rf[attr].quantile(.99),
      'max':rf[attr].max(),
      }
  qt=pd.DataFrame([qt])
  display(qt)
  log(f"{len(rf[rf[attr]>=v4thrd])} out of {len(rf)} >=thrd({v4thrd})")
  return qt, (h, ax, fig), rf

#pptHist4attr(asql=f'select riskEst as Risk from {prfx}rNPIRiskEst{ptfx}', attr='Risk', v4thrd=.6)
  

# COMMAND ----------

# MAGIC %md ###Quantile99 mapping

# COMMAND ----------

import pandas as pd
def dfSeriesMapping(dfSeries: pd.Series, q1=0.01, q99=.99, vmin=0, vmax=1):
  return dfSeries.map(lambda color:vmin +(np.max([q1, np.min([color, q99])])-q1)*(vmax-vmin)/(q99 - q1))
def dfSeriesQuantile99mapping(dfSeries: pd.Series, qmin=.01, qmax=.99, vmin=0, vmax=1, verbose=0):
  colorQ1, colorQ99=dfSeries.quantile(q=[qmin, qmax])   #0.01, 0.99
  if verbose>0:
    print(f"qmin={qmin:g}/{colorQ1:g} <-> vmin={vmin:g}")
    print(f"qmax={qmax:g}/{colorQ99:g} <-> vmax={vmax:g}")
  result=dfSeriesMapping(dfSeries, q1=colorQ1, q99=colorQ99, vmin=vmin, vmax=vmax) #.map(lambda c:mpl.colors.to_hex(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1, clip=True), cmap=cmap).to_rgba(c)))
  return result



# COMMAND ----------

# MAGIC %md ##IO

# COMMAND ----------

from glob import glob
from tqdm import tqdm
import subprocess
def splitDF2csv(df=f'select * from FPSModels.OpioidSNA_riskBeneClaims', prfx='/tmp/song/riskBeneClaims', maxRowsPerCsv=1000000, ncsv=None, do7z=False, prfx7z='', volMB=0):
  #Author: Xiaowei Song
  #Version: 20231102
  if isinstance(df, str): pdf=spark.sql(df) #df can be a sql select 
  else: pdf=df
  if not isinstance(pdf, pd.DataFrame): pdf=pdf.toPandas() #convert Spark DF to pandas
  if ncsv is None: ncsv=int(np.ceil(len(pdf)/maxRowsPerCsv)) #ncsv has higher priority
  fl=['']*ncsv
  pbar=tqdm(np.array_split(pdf, ncsv))
  for idx, chunk in enumerate(pbar): #seperate to 2 big csv instead of 1, such that each one has less than 1M rows        
    fn=f'{prfx}{idx:03}.csv'
    pbar.set_description(fn)
    fl[idx]=fn
    chunk.to_csv(fn) #
  del df, idx, chunk #big mem var
  if do7z: #
    if len(prfx7z)==0: prfx7z=prfx        
    if volMB>0: volMB=f'-v{volMB}M'    
    elif volMB<0: volMB=''
    else: volMB='-v10m'
    try:
      # opt7z=f"-sdel -v{volMB} a {prfx7z}.7z {prfx}*.csv"    
      # print(subprocess.check_output(['7z', opt7z, f' && ls -hl {prfx7z}*.7z']))
      results=subprocess.check_output(f"""ls -hl {prfx}*.csv
rm -f {prfx7z}.7z*
7z -sdel {volMB} a {prfx7z}.7z {prfx}*.csv
ls -hl {prfx7z}.7z*
""", shell=True, executable='/bin/bash', universal_newlines=True)
      print(results)
    except subprocess.CalledProcessError as err:
      print(err)
    return glob(f"{prfx7z}.7z*")
  else:
    return fl

# COMMAND ----------

# MAGIC %md ##GraphFrame functions

# COMMAND ----------

def edges2vertices(edges):
  vSrc=edges.select(edges.src.alias('id'))
  vDst=edges.select(edges.dst.alias('id'))
  allNodes=vSrc.union(vDst).distinct()
  del vSrc, vDst
  gc.collect()
  return allNodes
def edges2graph(edges):
  vertices=edges2vertices(edges)
  g=GraphFrame(vertices, edges)
  del vertices
  gc.collect()
  return g
def edges2symmetricalGraph(edges):  
  if 'weight' in set(edges.columns):
    invEdges=edges.select([col('src').alias('dst'),col('dst').alias('src'), 'weight'])
  else:
    invEdges=edges.select([col('src').alias('dst'),col('dst').alias('src')])
  allEdges=edges.union(invEdges)
  allVertices=allEdges.select(col('src').alias('id')).distinct()
  g=GraphFrame(allVertices, allEdges)
  del invEdges, allEdges, allVertices
  gc.collect()
  return g

# COMMAND ----------

# MAGIC %md #Benchmark

# COMMAND ----------

import time
 
def benchmark(f, df, benchmarks={}, name='Default', **kwargs):
    """Benchmark the given function against the given DataFrame.
    
    Parameters
    ----------
    f: function to benchmark
    df: data frame
    benchmarks: container for benchmark results
    name: task name
    
    Returns
    -------
    Duration (in seconds) of the given operation
    """
    start_time = time.time()
    ret = f(df, **kwargs)
    benchmarks['duration'].append(time.time() - start_time)
    benchmarks['task'].append(name)
    print(f"{name} took: {benchmarks['duration'][-1]} seconds")
    return benchmarks['duration'][-1]
 
def benchmarks2df(benchmarks):
    """Return a pandas DataFrame containing benchmark results."""
    return pd.DataFrame.from_dict(benchmarks)

# COMMAND ----------

# MAGIC %md #SMOTE subspace sampling

# COMMAND ----------

try:
  from imblearn.over_sampling import SMOTE
  from imblearn.under_sampling import RandomUnderSampler
  from imblearn.pipeline import Pipeline
  from collections import Counter
  import matplotlib.pyplot as plt 
  from numpy import where


  def smote4XY(X, Y, upsampling=0.1, downsampling=0.5, seed=None, doPlot=False):
    # summarize class distribution
    counter = Counter(Y) ;  print(counter)
    if doPlot:
      # scatter plot of examples by class label
      for label, _ in counter.items():
          row_ix = where(Y == label)[0]
          plt.scatter(X.iloc[row_ix, 0], X.iloc[row_ix, 1], label=str(label))
      plt.legend()
      plt.show()

    # define pipeline
    over = SMOTE(sampling_strategy=upsampling, random_state=seed)
    under = RandomUnderSampler(sampling_strategy=downsampling, random_state=seed)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    rX, rY = pipeline.fit_resample(X, Y)
    # summarize the new class distribution
    counter = Counter(rY) ; print(counter)
    if doPlot:
      # scatter plot of examples by class label
      for label, _ in counter.items():    
          row_ix = where(rY == label)[0]
          plt.scatter(rX.iloc[row_ix, 0], rX.iloc[row_ix, 1], label=str(label))
      plt.legend()
      plt.show()
    
    return rX, rY 
except Exception as e:
  log(f"SMOTE error: {e}")
  

# COMMAND ----------

# MAGIC %md ##Tabel2Xy

# COMMAND ----------

from sklearn.model_selection import train_test_split
def tbl2Xy(tbl="pharmFeatures20k", prfx=prfx if 'prfx' in globals() else '', featureColumns=None, mkTrainValTest=True, rseed=20230725, doSmote=True, upsampling=0.1, downsampling=0.5):
  if featuresColumns is None:
    featureColumns = spark.sql(f"""select * from {prfx}{tbl} limit 1""").columns #.rdd.flatMap(lambda x:x).collect()
    featureColumns = [x for x in featureColumns if x not in ("idx", "y")]

  allXy=spark.sql(f""" select * from {prfx}{tbl} """).toPandas().astype({"idx":str})
  allXy.set_index("idx", inplace=True)
  for f in featureColumns+['y']:
    allXy.loc[:, f]=allXy.loc[:, f].apply(pd.to_numeric, downcast="float", errors='coerce') #convert "decimal" type to "float", if error than force NaN    
  #process NaN
  allXy.fillna(0, inplace=True) #since I have used Coerce to force entries to be NaN if not a float number!

  allX=allXy.loc[:,featureColumns]
  allY=allXy.loc[:, 'y']

  # from sklearn.utils.multiclass import type_of_target
  # print(type_of_target(allY))
  #
  # rseed=20230725
  if doSmote:
    rX, rY=smote4XY(allX, allY, upsampling=upsampling, downsampling=downsampling, seed=rseed) #make 2:1 for y=0/1
  else:
    rX, rY=allX, allY
  if not mkTrainValTest: return rX, rY

  allIndex=list(range(rX.shape[0]))
  trainValXi, testXi, trainValYi, testYi = train_test_split(allIndex, allIndex, random_state=rseed,  test_size=.1)
  trainValX, testX, trainValY, testY  = rX.iloc[trainValXi, :], rX.iloc[testXi, :], rY.iloc[trainValYi], rY.iloc[testYi]
  return trainValX, testX, trainValY, testY

# COMMAND ----------

# MAGIC %md #PredScores 10 parts summary

# COMMAND ----------

# import numpy as np 
# import matplotlib.pyplot as plt 
# import pandas as pd 

def predScores10partsSummary(allEp, allY, Nparts=10, topPred=-1, cumPercentPlot=0, showTopNparts=None, horizentalPopulationRate=0):
  isort=np.argsort(allEp, axis=None) #return flatten sorted indices in the ascending way, i.e., from min to max  
  idxMax2min = np.flipud(isort) #get descending indices
  allEps=allEp[idxMax2min]
  allYs=allY[idxMax2min]  
  if topPred>0: #only check the specified top 1000 predictions, i.e., zoom in to see the distribution of the top 1000
    allEps=allEps[0:topPred] 
    allYs =allYs[0:topPred]
    
  allEps10 = np.array_split(allEps, Nparts) 
  allYs10  = np.array_split(allYs,  Nparts)  
  N1s=[]; Ns=[] ; N1percents=[]
  avgScores=[]; minScores=[]; maxScores=[]; NBadNPI=[]; PBadNPI=[]; CumPBadNPI=[]
  #get summary for each part
  for idx, (predScores, y) in enumerate( list(zip(allEps10, allYs10)) ):
    muPred =np.mean(predScores)
    minPred=predScores[-1] 
    maxPred=predScores[0]
    N=len(y);     N1=np.sum(y)
    Ns.append(N); N1s.append(N1)
    N1percent   =N1/N ; N1percents.append(N1percent)
    N1CumPercent=np.cumsum(N1s)[-1]/np.cumsum(Ns)[-1]
    print(f"group={idx+1}, N={N}, cumN={np.cumsum(Ns)[-1]}, avg_score={muPred:.3f}, min_score={minPred:.3f}, max_score={maxPred:.3f}, #Bad NPI={N1}, %Bad NPI={100*N1percent:.3f}%, %Cumulative Bad NPI={100*N1CumPercent:.3f}%")
    avgScores.append(muPred); minScores.append(minPred); maxScores.append(maxPred); NBadNPI.append(N1); PBadNPI.append(N1percent); CumPBadNPI.append(N1CumPercent)
  
  df=pd.DataFrame(data={'avg_score':avgScores, 'min_score':minScores, 'max_score':maxScores, 'NBadNPI':NBadNPI, 'PBadNPI':PBadNPI, 'CumPBadNPI':CumPBadNPI})
    
  plt.figure()
  if cumPercentPlot==0:
    plt.plot(range(1, 1+Nparts), 100*np.array(N1percents), marker='o')
    plt.ylabel('Bad hit rate (%)')
  else:
    plt.plot(range(1, 1+Nparts), 100*np.array(CumPBadNPI), marker='o')
    plt.ylabel('Cumulative Bad hit rate (%)')
  plt.xlabel(f'Predicted scores sorted and evenly splitted to {Nparts} groups, {N} NPIs/group')
  
  if showTopNparts is not None: plt.xlim(0, showTopNparts)
  
  # plt.tick_params(axis='x', which='both', )
  # plt.xticks(ticks=range(1, 1+Nparts))
  plt.grid(which='both')
  
  return df 

# COMMAND ----------

# import time
# print(time.localtime())
# print(time.monotonic()) #418555.967980571
# print(time.gmtime(418555.967980571))
# print(time.time())
# print(time.gmtime(time.time()))
# print(time.strftime('%H:%M:%S', time.localtime(time.time())))

# COMMAND ----------

# MAGIC %md ##draw colorbar

# COMMAND ----------

def drawColorbar(cmap='jet', vmin=0, vmax=1, title=''):
  fig, ax=plt.subplots(figsize=(12, .2))
  gradient0 = np.linspace(vmin, vmax, 256)
  gradient = np.vstack((gradient0, gradient0)) #make an image which has at least 2-dimension
  ax.imshow(gradient, aspect='auto', cmap=cmap)
  ax.axes.get_yaxis().set_visible(False)
  name=cmap if title=='' else title
  ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10, transform=ax.transAxes)  
  xticks=[ 0, 50, 100, 150, 200, 255]
  ax.set_xticks(xticks)
  for i,v in enumerate(xticks): xticks[i]=f"{v*(vmax-vmin)/255:.2g}"
  ax.set_xticklabels(xticks)
  return ax
# drawColorbar(cmap='jet', title='testJet')  
# drawColorbar(cmap='turbo')
# ax=drawColorbar(cmap='YlOrBr')
# ax=drawColorbar(cmap='bwr')

# COMMAND ----------

# MAGIC %md ##ROC plot

# COMMAND ----------

import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score , auc, precision_recall_curve, PrecisionRecallDisplay, precision_recall_curve, average_precision_score
def plotROC(aRiskScoreDF, fpng, title='LSTM+Autoencoder', showFPR5percent=False, PositiveLabel='HHA', RiskLabel='RiskEst'):
  #aRiskScores has an index as the subject IDs, and two columns: Risk and HHA, in which Risk is the predicted scores and HHA is 0/1 labels
  # plt.cla()
  plt.figure(figsize=(8,8))
  labels      =aRiskScoreDF[PositiveLabel] #HHA
  predictions =aRiskScoreDF[RiskLabel]
  fp, tp, thresholds =roc_curve(labels, predictions)
  roc_auc = auc(fp, tp)
  
          
  #find 5% fpr threshold
  fpr5p_thrd=.0; fpr5p_tpr=.0
  if showFPR5percent:
      plt.plot(100*fp, 100*tp, linewidth=2, color="darkorange", label=f"AUC={roc_auc:0.2f}")
      plt.plot([0, 100], [0, 100],  lw=1, linestyle="--") #reference line
  
      # plt.plot([0.05, 0.05], [0, 1.05],  lw=1, linestyle="dotted") #5% fpr line
      plt.vlines(5, 0, 105, linestyles='dotted')
  
      fpr5p_idx=(np.abs(fp-.05)).argmin()
      fpr5p_tpr=tp[fpr5p_idx]
      fpr5p_thrd=thresholds[fpr5p_idx]
      msg4fpr5p=f"FPR=5%: TPR={fpr5p_tpr:.3f}, thrd={fpr5p_thrd:.3f}"
      print(msg4fpr5p)
      plt.title(f"{title}\n{msg4fpr5p}")
      plt.xlim([0,100])
      plt.ylim([0,105])
      plt.xlabel('False positives [%]')
      plt.ylabel('True positives [%]')
  else:
      plt.plot(fp, tp, linewidth=2, color="darkorange", label=f"AUC={roc_auc:0.2f}")
      # plt.plot([0, 1], [0, 1],  lw=1, linestyle="--") #reference line
  
      plt.title(f"{title}")
      plt.xlim([0,1])
      plt.ylim([0,1.05])
      plt.xlabel('False positives')
      plt.ylabel('True positives')
  
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')
  
  plt.legend(loc='lower right')
  plt.savefig(fpng)
  print(f"ROC curve was saved to {fpng}")
  # plt.close()
  return fpr5p_thrd, fpr5p_tpr
  
def plotPrecisionRecallCurve(aRiskScoreDF, fpng, title='LSTM+Autoencoder', showFPR5percent=False, PositiveLabel='HHA', RiskLabel='RiskEst'):
  #aRiskScores has an index as the subject IDs, and two columns: Risk and HHA, in which Risk is the predicted scores and HHA is 0/1 labels
  # plt.cla()
  plt.figure(figsize=(8,8))
  labels      =aRiskScoreDF[PositiveLabel]
  predictions =aRiskScoreDF[RiskLabel]
  pr, re, thresholds =precision_recall_curve(labels, predictions)
  
  sidx=np.argsort(re) 
  auprc = auc(re[sidx], pr[sidx]) #auc needs X-axis be strictly increasing
  ap=average_precision_score(labels, predictions, average='micro') #should be exactly same with auprc
  
  #find 90% recall threshold
  re90_idx=(np.abs(re-.9)).argmin()
  re90_pr=pr[re90_idx] #find corresponding precision
  re90_thrd=thresholds[re90_idx]
  msg90recall=f"Recall=90%: Precision={re90_pr:.3f}, thrd={re90_thrd:.3f}"
  print(msg90recall)
  #find 100% recall which corresponds to the ratio of HHA/(total samples) since in this situation, all samples are classified as Positive/HHA, thus FN = (all samples - TP)
  re100_idx=np.argmin(np.abs(re-1)); re100_pr=pr[re100_idx]
  msg100recall=f"Recall=100%\nPrecision={100*re100_pr:.1f}%\nThrd=0"
  # plt.text(.80, re100_pr, msg100recall)
  print(f"re100_pr={re100_pr}, {msg100recall}")
  
  if showFPR5percent:
      plt.plot(100*re, 100*pr, linewidth=2, color="darkorange", label=f"microAveragePrecision={ap:0.3f}")
      plt.plot([0, 100], [100, 0],  lw=1, linestyle="--") #reference line    
      plt.vlines(90, 0, 105, linestyles='dotted')
      
      #recall = tp/(tp+fn)
      
      plt.title(f"{title}\n{msg90recall}")
      #find 100% recall which corresponds to the ratio of HHA/(total samples) since in this situation, all samples are classified as Positive/HHA, thus FN = (all samples - TP)
      re100_idx=np.argmin(np.abs(re-1)); re100_pr=pr[re100_idx]
      msg100recall=f"Recall=100%\nPrecision={100*re100_pr:.1f}%\nThrd=0"
      plt.text(100*.80, 100*re100_pr, msg100recall)
      
      plt.xlim([0,100])
      plt.ylim([0,105])
      plt.xlabel('Recall [%]')
      plt.ylabel('Precision [%]')
  else:
      plt.plot(re, pr, linewidth=2, color="darkorange", label=f"microAveragePrecision={ap:0.3f}")
      plt.plot([0, 1], [1, 0],  lw=1, linestyle="--") #reference line    
      
      #recall = tp/(tp+fn)       
  
      plt.title(f"{title}\n")       
      
      plt.xlim([0,1.00])
      plt.ylim([0,1.05])
      plt.xlabel('Recall')
      plt.ylabel('Precision')
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')
  
  plt.legend(loc='lower left')
  plt.savefig(fpng)
  print(f"precision-recall curve was saved to {fpng}")
  # plt.close()
  return re90_thrd, re90_pr


def plotROCPrecisionRecall(aRiskScoreDF, prefix4png, showFPR5percent=False, PositiveLabel='HHA', RiskLabel='RiskEst'):
  labels      =aRiskScoreDF[PositiveLabel]
  predictions =aRiskScoreDF[RiskLabel]
      
  precision, recall, thresholds = precision_recall_curve(labels, predictions)
  average_precision = average_precision_score(labels, predictions)
  #------------------------------------------
  plt.figure(figsize=(8,8))
  plt.step(recall, precision, color='k', alpha=0.7, where='post')
  plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))
  plt.savefig(f"{prefix4png}__precision-recall.png")
  #------------------------------------------
  fpr, tpr, thresholds = roc_curve(labels, predictions)
  areaUnderROC = auc(fpr, tpr)

  plt.figure(figsize=(8,8))
  plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
  plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic: Area under the curve = {0:0.2f}'.format(areaUnderROC))
  plt.legend(loc="lower right")    
  plt.savefig(f"{prefix4png}__tpr-fpr.png")  

# COMMAND ----------

# import igraph

# COMMAND ----------

# MAGIC %md
# MAGIC #Scikit-learn Model functions

# COMMAND ----------

# DBTITLE 1,Xiaowei's masked features explaination
def num2ordinal(n):
    ''' https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement#:~:text=To%20convert%20an%20integer%20to%20%221st%22%2C%20%222nd%22%2C%20etc%2C,%27th%27%5D%5Bmin%28n%20%25%2010%2C%204%29%5D%20return%20str%28n%29%20%2B%20suffix
    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    '''
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix
  
def barplot4importances(yf, dy, prfx, style='dy', vline=0, topdy=0, figsize=(9,7), xlabel='Changes of risk score with each feature masked out', note='', explainedVarianceRatio=[]):
  nx=len(dy)
  for ic in range(nx):
    fig, ax=plt.subplots(figsize=figsize)
    if nx==1: ax.set_title(f'Risk score with full features: {yf[ic]}')
    else: ax.set_title(f'{num2ordinal(1+ic)} component explained variance ratio: {100*explainedVarianceRatio[ic]:.1f}%')

    #sort by absolute values of dy
    import numpy as np
    dySortedIdx = np.argsort(np.abs(dy.iloc[ic,:].values)) #default ascending order
    if topdy==0: topdy=len(dy.columns) #=0 means all columns           
    sdy=dy.iloc[ic, dySortedIdx].values[::-1][0:topdy]
    sdyNames=dy.columns.values[dySortedIdx][::-1][0:topdy]
    syf=[ yf[0] for i in range(len(sdy)) ]

    if style=='dy':
      ax.barh(sdyNames, sdy, align='center')      
    elif style=='stack':      
      my=sdy + syf #+ pd.DataFrame(yf[0], columns=dy.columns, index=dy.index)
      sy=-sdy
      ax.barh(sdyNames, my, align='center')
      ax.barh(sdyNames, sy, align='center', left=my)
    if vline>0: plt.axvline(x=vline, color='purple', ls='--', lw=2)
    ax.set_ylabel("masked features")
    ax.set_xlabel(f"{xlabel} {note}")
    fpng=f"{prfx}_{ic:02d}.png"
    print(f"saved to {fpng}")
    fig.savefig(fpng)


def mskPredict(skmodel, X, prefix4png='', yIdx=1, style='dy', vline=0, topdy=0):
#Author: Xiaowei Song, 20240619
#mask out each feature of the sample X (assuming only one row, or a pd.Series instead of a dataframe) and check how much change on the predict_proba()[1] risk score
  yf=skmodel.predict_proba(X)[:, yIdx] #input X is supposed to be a 2D pandas df, yf is y with full X features, i.e., no mask applied yet
  nf=len(X.columns)
  nx=len(X)
  xf=X.copy()
  dy=X.copy()
  logging.info(f"dim of X is: {nx} X {nf}")
  for i in range(nf):
    X=xf.copy()
    X.iloc[:, i]=0
    y=skmodel.predict_proba(X)[:, yIdx]
    dy.iloc[:, i]=y-yf

  if nx>1: #group samples' masked perturbation
    from sklearn.decomposition import TruncatedSVD
    ndy=dy.to_numpy()
    print(f"dy shape: {ndy.shape}")
    ndySVD=TruncatedSVD(n_components=2, algorithm='arpack', tol=1e-8).fit(ndy) #n_iter=10, not used by arpack
    dyC = pd.DataFrame(ndySVD.components_, columns=dy.columns)
    barplot4importances(yf, dyC, prefix4png, style=style, vline=vline, topdy=topdy, figsize=(9,7), explainedVarianceRatio=ndySVD.explained_variance_ratio_, note='\nCaution: sign does not matter while relative signs matter!', xlabel='Relative importance')
  else:
    barplot4importances(yf, dy,  prefix4png, style=style, vline=vline, topdy=topdy, figsize=(9,7))
  return yf, dy



# COMMAND ----------

# MAGIC %md
# MAGIC #Source Inpsection

# COMMAND ----------

# DBTITLE 1,showPythonSrc
def showPythonSrc(fcode, style="gruvbox-dark", linenos=True):
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name,get_lexer_for_filename
    from pygments.formatters import HtmlFormatter

    lexer = get_lexer_by_name("python", stripall=True) #get_lexer_for_filename(fcode) #
    formatter = HtmlFormatter(linenos=linenos, cssclass="source", style=style, full=True)
    with open(fcode, 'r') as code:
        highlighted_code = highlight(code.read(), lexer, formatter)

    displayHTML(highlighted_code)
# showPythonSrc('/local_disk0/.ephemeral_nfs/cluster_libraries/python/lib/python3.10/site-packages/fpsflow/ml_obj.py')

# from pygments.styles import get_all_styles
# styles = list(get_all_styles())
# print(styles)
# ['default', 'emacs', 'friendly', 'friendly_grayscale', 'colorful', 'autumn', 'murphy', 'manni', 'material', 'monokai', 'perldoc', 'pastie', 'borland', 'trac', 'native', 'fruity', 'bw', 'vim', 'vs', 'tango', 'rrt', 'xcode', 'igor', 'paraiso-light', 'paraiso-dark', 'lovelace', 'algol', 'algol_nu', 'arduino', 'rainbow_dash', 'abap', 'solarized-dark', 'solarized-light', 'sas', 'stata', 'stata-light', 'stata-dark', 'inkpot', 'zenburn', 'gruvbox-dark', 'gruvbox-light', 'dracula', 'one-dark', 'lilypond']

def showPythonSrc4func(func):
  import inspect
  src=inspect.getfile(func)
  print(f"src file= {src}")
  showPythonSrc(src)


# COMMAND ----------

# MAGIC %md
# MAGIC #MLFlow/FPSFlow

# COMMAND ----------

def listMLRuns(experimentName=None, sortKey='metrics.test__f1_score', descending=True):
  import fpsflow, mlflow
  from pathlib import Path
  if experimentName is None:
    nbfile=dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    nbName=Path(nbfile).name
    experimentName = f'/Users/xiaowei.song@gdit.com/MLFlow.experiments/{nbName.split("_")[0]}'
    logw(f"Guessed experiment name as: `{experimentName}`")
  else: log(f"Input experiment name as: `{experimentName}`")    
  exp=fpsflow.get_experiment_by_name(experimentName)
  if exp is None: 
    logw(f"Cannot find exp with experimentName={experimentName}")
    return None
  else:
    allRuns=fpsflow.search_runs([exp.experiment_id]) #, order_by=["metrics.test-f1_score DESC"]) #the dash of test-f1 cannot be parsed!
    allRuns=allRuns.sort_values(sortKey, ascending=not descending).set_index('run_id')
    return allRuns

# #only keep 20 best  runs sorted by test__f1_score
def rmMLRuns(allRuns, nTop2keep=20, keys2print=['metrics.test__f1_score', 'metrics.test__roc_auc_score'], doRm=False):
  for runID in allRuns.index[nTop2keep:]:
      logw(allRuns.loc[runID, keys2print ].to_string(index=False).replace('\n', '\t'))
      # fpsflow.delete_run(runID) #module 'fpsflow' has no attribute 'delete_run'
      if doRm: mlflow.delete_run(runID)
      # break 

