"""Isolated, optional Spark MLlib demonstration using *development data only*.

No test Parquet is opened; this is not a replacement for the sklearn evaluation.
Fixed quarterly cutoff and deterministic CERT hash sample avoid random time CV.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def run(root: Path) -> dict:
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
    from pyspark.sql import SparkSession, functions as F

    result_file = root / "results" / "part5" / "spark_mllib_demo.json"
    if result_file.exists():
        raise FileExistsError("Spark demo output already exists; do not rerun automatically")
    schema = json.loads((root / "results" / "part4" / "feature_schema.json").read_text(encoding="utf-8"))
    cols = schema["model_input_columns"]
    allowed = set(schema["feature_definitions"])
    if not cols or len(cols) != len(set(cols)) or not set(cols).issubset(allowed):
        raise ValueError("Unapproved Spark features")
    fpath = root / "data_processed" / "part4" / "train_features.parquet"
    tpath = root / "data_processed" / "part4" / "train_targets.parquet"
    if not fpath.exists() or not tpath.exists():
        raise FileNotFoundError("Part 4 development inputs missing")

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (SparkSession.builder.master("local[2]").appName("FDIC-Part5-DevelopmentOnly-SparkDemo")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    try:
        features = spark.read.parquet(str(fpath.resolve())).select("CERT", "REPDTE", *cols)
        target = spark.read.parquet(str(tpath.resolve())).select("CERT", "REPDTE", "y_fail_next4q")
        df = (features.join(target, ["CERT", "REPDTE"], "inner")
              .filter(F.pmod(F.hash("CERT"), F.lit(4)) == F.lit(0)))
        # Date windows are within Part4 TRAIN, with all four 1999 quarters purged.
        train = df.filter(F.col("REPDTE") <= "19981231")
        val = df.filter((F.col("REPDTE") >= "20000331") & (F.col("REPDTE") <= "20071231"))
        counts = [train.groupBy("y_fail_next4q").count().collect(),
                  val.groupBy("y_fail_next4q").count().collect()]
        if any(len(g) != 2 for g in counts):
            raise ValueError("Spark demo split must have both labels; STOP")
        imputed = [f"imp_{i}" for i in range(len(cols))]
        imputer = Imputer(inputCols=cols, outputCols=imputed, strategy="median")
        assembler = VectorAssembler(inputCols=imputed, outputCol="raw_features", handleInvalid="error")
        scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                                withMean=True, withStd=True)
        lr = LogisticRegression(featuresCol="features", labelCol="y_fail_next4q",
                                maxIter=40, regParam=0.01)
        model = Pipeline(stages=[imputer, assembler, scaler, lr]).fit(train)
        scored = model.transform(val)
        ap = BinaryClassificationEvaluator(labelCol="y_fail_next4q",
                     rawPredictionCol="rawPrediction", metricName="areaUnderPR").evaluate(scored)
        roc = BinaryClassificationEvaluator(labelCol="y_fail_next4q",
                     rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(scored)
        output = {
            "status": "PASS_SPARK_DEVELOPMENT_ONLY", "master": "local[2]",
            "train_end": "19981231", "purged_year": "1999", "val_start": "20000331",
            "val_end": "20071231", "deterministic_sampling": "pmod(hash(CERT),4)==0",
            "model_features": cols,
            "train_counts_by_label": {str(r[0]): int(r[1]) for r in counts[0]},
            "validation_counts_by_label": {str(r[0]): int(r[1]) for r in counts[1]},
            "spark_areaUnderPR": float(ap), "spark_areaUnderROC": float(roc),
            "preprocessing_fitted_only_on_spark_demo_train": True,
            "heldout_test_read": False,
            "notes": "Separate illustrative Spark local demo; not a comparable test benchmark or a cluster speed measurement.",
        }
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return output
    finally:
        spark.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    a = ap.parse_args()
    print(json.dumps(run(a.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
