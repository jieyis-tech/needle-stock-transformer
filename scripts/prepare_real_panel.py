"""Read-only warehouse -> bounded, versioned course experiment cache.

No API download and no warehouse writes. Fixed historical cohort, no selection
by future survival/returns. This is a daily-input MASTER systems demonstration,
not the existing monthly PCA model or an executable stock strategy.
"""
from pathlib import Path
import hashlib
import json
import time
import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DB=Path(r'C:\AI_STOCK\dataset\duckdb\tushare.duckdb')
OUT=ROOT/'data'/'real_panel_v1'
ASOF='20140102'
FEATURES=['log_open_preclose','log_high_preclose','log_low_preclose','log_close_preclose',
          'log_volume','log_amount','log_free_turnover','log_float_cap','log_high_low','history_traded']


def main():
    if OUT.exists():
        raise FileExistsError('Panel already exists; retain v1 and choose a new version for changes')
    start=time.perf_counter()
    c=duckdb.connect(str(DB),read_only=True)
    cohort_sql="""SELECT ts_code,list_date FROM meta.stock_basic
        WHERE list_date <= '20130101' AND (delist_date IS NULL OR delist_date='' OR delist_date > '20140102')"""
    universe=c.execute(cohort_sql).fetchall()
    universe=sorted(universe,key=lambda row:hashlib.sha256(('needle-master-v1:'+row[0]).encode()).hexdigest())[:128]
    codes=np.array(sorted(x[0] for x in universe))
    c.register('cohort',pd.DataFrame({'ts_code':codes}))
    sql="""SELECT d.*,b.turnover_rate_f,b.circ_mv,a.adj_factor
        FROM market.daily d JOIN cohort u USING(ts_code)
        LEFT JOIN market.daily_basic b USING(ts_code,trade_date)
        LEFT JOIN market.adj_factor a USING(ts_code,trade_date)
        WHERE d.trade_date BETWEEN '20131001' AND '20160715'
        ORDER BY d.trade_date,d.ts_code"""
    df=c.execute(sql).df()
    if df.duplicated(['trade_date','ts_code']).any():raise ValueError('duplicate source keys')
    views=c.execute("select schema_name,view_name,sql from duckdb_views() where schema_name in ('market','meta') and view_name in ('daily','daily_basic','adj_factor','stock_basic')").fetchall()
    c.close()
    dates=np.array(sorted(df.trade_date.unique()))
    def matrix(col):return df.pivot(index='trade_date',columns='ts_code',values=col).reindex(index=dates,columns=codes).to_numpy(float)
    o,h,l,cl,pre,vol,amt,trf,cap,adj=[matrix(k) for k in ('open','high','low','close','pre_close','vol','amount','turnover_rate_f','circ_mv','adj_factor')]
    traded=(vol>0)&(cl>0)&(pre>0)&(o>0)&(h>0)&(l>0)
    with np.errstate(all='ignore'):
        raw=np.stack([np.log(o/pre),np.log(h/pre),np.log(l/pre),np.log(cl/pre),np.log(vol),np.log(amt),np.log(trf),np.log(cap),np.log(h/l)],-1)
    raw[~traded]=np.nan
    fit=(dates>='20140102')&(dates<='20141231')
    mu=np.nanmean(raw[fit],axis=(0,1));sd=np.nanstd(raw[fit],axis=(0,1))
    assert np.all(np.isfinite(mu)) and np.all(sd>1e-12)
    z=np.where(np.isfinite(raw),np.clip((raw-mu)/sd,-8,8),0).astype('float32')
    z=np.concatenate([z,traded[...,None].astype('float32')],-1)
    # Market inputs depend on dates <= t only, and never on labels.
    daylines=np.stack([np.nanmean(raw[:,:,3],1),np.nanstd(raw[:,:,3],1),
                       np.nanmean(raw[:,:,6],1),np.nanmean(raw[:,:,5],1)],1)
    market=[]
    for window in (5,20):
        frame=pd.DataFrame(daylines).rolling(window,min_periods=window)
        market.extend([frame.mean().to_numpy(),frame.std(ddof=0).to_numpy()])
    market=np.concatenate(market,1)
    mm=np.nanmean(market[fit],0);ms=np.nanstd(market[fit],0)
    market=np.nan_to_num((market-mm)/np.maximum(ms,1e-8)).astype('float32')
    # Forward 5 market-trading-day adjusted-close label. No trading claim.
    with np.errstate(all='ignore'):
        adjusted=cl*adj
        y=np.full_like(cl,np.nan)
        y[:-5]=np.log(adjusted[5:]/adjusted[:-5])
    validfuture=np.zeros_like(traded);validfuture[:-5]=traded[5:]
    y[~(traded&validfuture)]=np.nan
    frames=[];markets=[];labels=[];scoremask=[];signal_dates=[];label_dates=[];split=[]
    ranges=[('train','20140102','20141231'),('validation','20150105','20150630'),('test','20160104','20160630')]
    waterfall={}
    for name,lo,hi in ranges:
        counts=dict(candidate_days=0,purged_label_boundary_days=0,score_stock_days=0,labelled_stock_days=0,unlabelled_scored_stock_days=0,untraded_stock_days=0)
        for t in range(19,len(dates)-5):
            if not lo<=dates[t]<=hi:continue
            counts['candidate_days']+=1
            if dates[t+5]>hi:
                counts['purged_label_boundary_days']+=1
                continue
            # Keep each traded T-day cohort stock; future availability only masks loss/metrics.
            counts['score_stock_days']+=int(traded[t].sum())
            counts['untraded_stock_days']+=int((~traded[t]).sum())
            counts['labelled_stock_days']+=int((traded[t]&np.isfinite(y[t])).sum())
            counts['unlabelled_scored_stock_days']+=int((traded[t]&~np.isfinite(y[t])).sum())
            frames.append(z[t-11:t+1].transpose(1,0,2));markets.append(market[t]);labels.append(y[t]);scoremask.append(traded[t])
            signal_dates.append(dates[t]);label_dates.append(dates[t+5]);split.append(name)
        waterfall[name]=counts
    X=np.stack(frames);M=np.stack(markets);Y=np.stack(labels).astype('float32');S=np.stack(scoremask)
    splits=np.array(split);sdays=np.array(signal_dates);ldays=np.array(label_dates)
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(M))
    assert max(ldays[splits=='train'])<min(sdays[splits=='validation'])
    assert max(ldays[splits=='validation'])<min(sdays[splits=='test'])
    # Direct PIT recomputation: market at an earlier day is identical when later source rows are removed.
    cutoff=np.where(dates=='20150302')[0][0]
    check=[]
    for window in (5,20):
        frame=pd.DataFrame(daylines[:cutoff+1]).rolling(window,min_periods=window)
        check.extend([frame.mean().to_numpy()[-1],frame.std(ddof=0).to_numpy()[-1]])
    check=(np.concatenate(check)-mm)/np.maximum(ms,1e-8)
    np.testing.assert_allclose(check,market[cutoff],atol=2e-6,rtol=2e-6)
    fp=hashlib.sha256()
    for a in (X,M,Y,S,sdays,ldays,codes):fp.update(a.tobytes())
    manifest=dict(version='real_panel_v1',created_for='Needle MASTER systems demonstration only',source_db=str(DB),source_read_only=True,
                  source_db_bytes=DB.stat().st_size,source_db_mtime_ns=DB.stat().st_mtime_ns,source_views=views,
                  cohort_asof=ASOF,cohort_selection='128 smallest SHA256(needle-master-v1:code), listed by 20130101, not delisted as of 20140102; no later survival filter',
                  cohort_sql=cohort_sql,source_sql=sql,source_rows=len(df),codes=codes.tolist(),features=FEATURES,
                  input_shape=list(X.shape),signal_ranges={k:[str(min(sdays[splits==k])),str(max(sdays[splits==k])),int((splits==k).sum())] for k,_,_ in ranges},
                  normalization_fit=['20140102','20141231'],feature_mean=mu.tolist(),feature_std=sd.tolist(),market_mean=mm.tolist(),market_std=ms.tolist(),
                  label='log(close[t+5]*adj[t+5]/(close[t]*adj[t])); future untraded/missing -> loss mask only',
                  label_exit_ranges={k:[str(min(ldays[splits==k])),str(max(ldays[splits==k]))] for k,_,_ in ranges},
                  waterfall=waterfall,pit_prefix_market_check='passed at 20150302',fingerprint_sha256=fp.hexdigest(),seconds=time.perf_counter()-start,
                  limitations=['Fixed historical cohort, not full-market universe','Daily 12-step inputs, not monthly PCA production model',
                               'ST/price limits/costs not filtered: systems demo, not executable backtest','Post-delisting data not filled; no future-label universe filtering',
                               'Missing historical feature values -> train mean (zero standardized), plus traded indicator','Raw warehouse vintage can include vendor corrections'])
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'panel.npz',X=X,market=M,y=Y,score_mask=S,dates=sdays,label_dates=ldays,split=splits,codes=codes)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('input_shape','source_rows','signal_ranges','label_exit_ranges','waterfall','pit_prefix_market_check','seconds')},indent=2))


if __name__=='__main__':main()
