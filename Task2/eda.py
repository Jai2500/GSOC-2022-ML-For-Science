import pyarrow.parquet as pq

run_0_path = "/scratch/gsoc/QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272.test.snappy.parquet"
run_1_path = "/scratch/gsoc/QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540.test.snappy.parquet"
run_2_path = "/scratch/gsoc/QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494.test.snappy.parquet"

df = pq.read_table(run_1_path)

print(df)