import duckdb, time, os
from pathlib import Path
ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SOURCE_ROOT = Path(os.environ.get("SEPSIS_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway"))
E = str(SOURCE_ROOT / "eicu")
OUT = str(ROOT / "data" / "processed" / "eicu")
os.makedirs(OUT, exist_ok=True)
con=duckdb.connect(); con.execute("PRAGMA threads=8"); con.execute("PRAGMA memory_limit='10GB'")
t0=time.time()
con.execute(f"""CREATE TEMP TABLE P AS
SELECT patientunitstayid, uniquepid, hospitalid, hospitaldischargeoffset, hospitaldischargestatus,
       hospitaldischargelocation, unitdischargeoffset, unitdischargelocation, unitvisitnumber,
       gender,
       CASE WHEN trim(age) = '> 89' THEN 90
            ELSE TRY_CAST(age AS INT) END AS age,
       unittype, unitadmitsource, hospitaladmitsource,
       admissionweight, admissionheight, hospitaldischargeyear
FROM read_csv_auto('{E}/patient.csv.gz')""")
con.execute(f"""CREATE TEMP TABLE SEP AS
SELECT DISTINCT patientunitstayid FROM read_csv_auto('{E}/diagnosis.csv.gz')
WHERE replace(icd9code,'.','') LIKE '038%' OR replace(icd9code,'.','') LIKE '9959%'
   OR replace(icd9code,'.','') LIKE '78552%'
   OR lower(diagnosisstring) LIKE '%sepsis%' OR lower(diagnosisstring) LIKE '%septic%'""")

print("status x destination (all sepsis adult first-stay):", flush=True)
print(con.execute(f"""SELECT p.hospitaldischargestatus, p.hospitaldischargelocation, COUNT(*) n
FROM P p JOIN SEP USING(patientunitstayid)
WHERE p.unitvisitnumber=1 AND p.age>=18
GROUP BY 1,2 ORDER BY n DESC LIMIT 20""").fetchdf().to_string(), flush=True)

con.execute(f"""COPY (
SELECT p.*, p.unitdischargeoffset/60.0 AS los_h,
       CASE WHEN p.hospitaldischargestatus='Expired' OR p.hospitaldischargelocation='Death' THEN 3
            WHEN p.hospitaldischargelocation='Home' THEN 0
            WHEN p.hospitaldischargelocation='Rehabilitation' THEN 1
            WHEN p.hospitaldischargelocation IN ('Skilled Nursing Facility','Nursing Home',
                 'Other Hospital','Other External','Other','Hospice') THEN 2
            ELSE NULL END AS y
FROM P p JOIN SEP USING(patientunitstayid)
WHERE p.unitvisitnumber=1 AND p.age>=18
) TO '{OUT}/cohort_v2_all.parquet' (FORMAT PARQUET)""")
con.execute(f"""COPY (SELECT * FROM read_parquet('{OUT}/cohort_v2_all.parquet') WHERE los_h>=72)
TO '{OUT}/cohort_v2.parquet' (FORMAT PARQUET)""")
print("all:", con.execute(f"SELECT COUNT(*), SUM(CASE WHEN y IS NOT NULL THEN 1 ELSE 0 END) FROM read_parquet('{OUT}/cohort_v2_all.parquet')").fetchone(), flush=True)
print("72h:", con.execute(f"SELECT COUNT(*), COUNT(DISTINCT uniquepid), SUM(CASE WHEN y IS NOT NULL THEN 1 ELSE 0 END) FROM read_parquet('{OUT}/cohort_v2.parquet')").fetchone(), flush=True)
print("y dist:", con.execute(f"SELECT y, COUNT(*), ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (),1) FROM read_parquet('{OUT}/cohort_v2.parquet') GROUP BY 1 ORDER BY 1").fetchall(), flush=True)
print(f"{time.time()-t0:.0f}s")
