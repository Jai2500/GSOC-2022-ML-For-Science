from dataset import ImageDatasetFromParquet


run_0_path = "/scratch/gsoc/parquet_ds/QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272.test.snappy.parquet"
run_1_path = "/scratch/gsoc/parquet_ds/QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540.test.snappy.parquet"
run_2_path = "/scratch/gsoc/parquet_ds/QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494.test.snappy.parquet"


run_0_ds = ImageDatasetFromParquet(run_0_path)

print(run_0_ds)