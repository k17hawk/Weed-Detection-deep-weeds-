import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

from weed_detection import logger
from weed_detection.entity.config_entity import DataValidationConfig
from weed_detection.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from weed_detection.config.configuration import ConfigurationManager
from weed_detection.utils.utility import save_json, load_json

# Required CSV columns — deep-weed format
REQUIRED_COLUMNS = {"Filename", "Label", "Species"}


# ── Load DataIngestionArtifact from JSON ──────────────────────────────────────

def load_ingestion_artifact(path: Path) -> DataIngestionArtifact:
    """
    Re-hydrate DataIngestionArtifact from data_ingestion_artifact.json.
    Called by DataValidation to find normalized dirs + split info.

    Usage:
        artifact = load_ingestion_artifact(
            Path("artifacts/data_ingestion/data_ingestion_artifact.json")
        )
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No DataIngestionArtifact at {path}\n"
            f"Run data_ingestion.py first."
        )

    d = json.loads(path.read_text())

    # Re-build the nested KafkaArtifact first
    from weed_detection.entity.artifact_entity import KafkaArtifact
    kafka = KafkaArtifact(
        kafka_data_dir    = Path(d["kafka_zip"]).parent.parent,   # kafka_data/
        version_dir       = Path(d["kafka_zip"]).parent,
        zip_file_path     = Path(d["kafka_zip"]),
        s3_bucket         = d["s3_bucket"],
        s3_key            = d["s3_key"],
        source_url        = d.get("source_url", ""),
        file_hash         = d["file_hash"],
        file_size_bytes   = d.get("file_size_bytes", 0),
        original_filename = Path(d["kafka_zip"]).name,
        kafka_topic       = d.get("kafka_topic", ""),
        kafka_partition   = d.get("kafka_partition", 0),
        kafka_offset      = d.get("kafka_offset", 0),
        received_at       = datetime.fromisoformat(d["received_at"]),
        artifact_path     = path,
    )

    return DataIngestionArtifact(
        kafka_artifact   = kafka,
        unzip_dir        = Path(d["unzip_dir"]),
        normalized_dir   = Path(d["normalized_dir"]),
        train_images_dir = Path(d["train_images_dir"]),
        train_labels_dir = Path(d["train_labels_dir"]),
        val_images_dir   = Path(d["val_images_dir"])  if d.get("val_images_dir")  else None,
        val_labels_dir   = Path(d["val_labels_dir"])  if d.get("val_labels_dir")  else None,
        test_images_dir  = Path(d["test_images_dir"]) if d.get("test_images_dir") else None,
        test_labels_dir  = Path(d["test_labels_dir"]) if d.get("test_labels_dir") else None,
        source_type      = d.get("source_type", "unknown"),
        total_images     = d.get("total_images", 0),
        total_labels     = d.get("total_labels", 0),
        splits           = d.get("splits", []),
        warnings         = d.get("warnings", []),
        artifact_path    = path,
    )


# ── DataValidation ────────────────────────────────────────────────────────────

class DataValidation:
    """
    Validates normalized deep-weed data produced by DataIngestion.
    Reads DataIngestionArtifact → runs 6 checks → writes DataValidationArtifact.
    """

    def __init__(self, config: DataValidationConfig):
        self.config        = config
        self.failed_checks : List[str] = []
        self.warnings      : List[str] = []

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> Optional[DataValidationArtifact]:
        logger.info("=" * 60)
        logger.info("🚀 Data Validation — version check")
        logger.info("=" * 60)

        # 1. load DataIngestionArtifact
        ingestion_artifact = self._load_ingestion_artifact()

        # 2. version check — skip if already validated
        version_id = ingestion_artifact.normalized_dir.name
        if self._already_validated(version_id):
            logger.info(f"⏭️  Version '{version_id}' already validated — skipping")
            return None

        logger.info(f"🆕 Validating version : {version_id}")
        logger.info(f"   Normalized dir     : {ingestion_artifact.normalized_dir}")
        logger.info(f"   Splits             : {ingestion_artifact.splits}")

        # 3. build split map  { "train": {images: Path, labels: Path, csv: Path}, ... }
        split_map = self._build_split_map(ingestion_artifact)

        # 4. run all 6 steps
        split_stats        = {}
        class_distribution = {}

        # Step 1 — schema
        self._step1_schema(split_map)

        # Only continue if schema is valid — everything else depends on it
        if self.failed_checks:
            logger.error("❌ Schema check failed — skipping remaining steps")
        else:
            # Step 2 — image ↔ CSV alignment
            split_stats = self._step2_alignment(split_map)

            # Step 3 — label integrity
            self._step3_label_integrity(split_map)

            # Step 4 — image file integrity (only if no hard failures yet)
            if not self.failed_checks:
                self._step4_image_integrity(split_map, split_stats)

            # Step 5 — class distribution
            class_distribution = self._step5_class_distribution(split_map)

            # Step 6 — cross-split leakage
            self._step6_leakage(split_map)

        # 5. verdict
        is_valid = len(self.failed_checks) == 0

        if is_valid:
            logger.info("✅ All hard checks passed — data is valid")
        else:
            logger.error(f"❌ {len(self.failed_checks)} hard failure(s) — data INVALID")
            for f in self.failed_checks:
                logger.error(f"   ✗ {f}")

        if self.warnings:
            logger.warning(f"⚠️  {len(self.warnings)} soft warning(s)")
            for w in self.warnings:
                logger.warning(f"   ⚠ {w}")

        # 6. build + save artifact
        artifact = self._build_artifact(
            ingestion_artifact, is_valid, split_stats, class_distribution
        )

        # 7. update state → next run skips this version
        self._update_state(version_id, artifact)

        self._log_summary(artifact)
        return artifact

    # ── version control ───────────────────────────────────────────────────────

    def _already_validated(self, version_id: str) -> bool:
        """True if validation_state.json records this version as already done."""
        state_path = self.config.validation_state_path
        if not state_path.exists():
            return False
        try:
            state = load_json(state_path)
            return state.get("last_version_id") == version_id
        except Exception as e:
            logger.warning(f"⚠️  Could not read validation state: {e} — will re-validate")
            return False

    def _update_state(self, version_id: str, artifact: DataValidationArtifact) -> None:
        """Write validation_state.json after run."""
        save_json(
            path = self.config.validation_state_path,
            data = {
                "last_version_id"      : version_id,
                "last_validated_at"    : artifact.validated_at.isoformat(),
                "last_is_valid"        : artifact.is_valid,
                "last_report_path"     : str(artifact.validation_report_path),
                "failed_checks_count"  : len(artifact.failed_checks),
                "warnings_count"       : len(artifact.warnings),
            }
        )
        logger.info(f"💾 Validation state saved : {self.config.validation_state_path}")

    # ── load ingestion artifact ───────────────────────────────────────────────

    def _load_ingestion_artifact(self) -> DataIngestionArtifact:
        logger.info(f"📋 Reading ingestion artifact from {self.config.ingestion_artifact_path}")
        artifact = load_ingestion_artifact(self.config.ingestion_artifact_path)
        logger.info(f"✅ Loaded version : {artifact.normalized_dir.name}")
        return artifact

    # ── build split map ───────────────────────────────────────────────────────

    def _build_split_map(
        self, artifact: DataIngestionArtifact
    ) -> Dict[str, Dict[str, Path]]:
        """
        Build a unified map from the DataIngestionArtifact:
        {
          "train": { "images": Path, "labels": Path, "csv": Path },
          "val"  : { ... },
          "test" : { ... },
        }

        deep-weed layout: all splits share the same labels/ dir.
        CSV path = labels_dir / <split>.csv
        """
        split_map = {}

        entries = [
            ("train", artifact.train_images_dir, artifact.train_labels_dir),
            ("val",   artifact.val_images_dir,   artifact.val_labels_dir),
            ("test",  artifact.test_images_dir,  artifact.test_labels_dir),
        ]

        for split, images_dir, labels_dir in entries:
            if images_dir is None or labels_dir is None:
                continue
            if not images_dir.exists() or not labels_dir.exists():
                logger.warning(f"⚠️  [{split}] directory missing — skipping: {images_dir}")
                continue

            csv_path = labels_dir / f"{split}.csv"
            if not csv_path.exists():
                logger.warning(f"⚠️  [{split}] CSV not found: {csv_path} — skipping split")
                continue

            split_map[split] = {
                "images": images_dir,
                "labels": labels_dir,
                "csv"   : csv_path,
            }
            logger.info(f"   [{split}] images={images_dir}  csv={csv_path}")

        if not split_map:
            self.failed_checks.append(
                "No valid splits found — no images/labels directories exist"
            )

        return split_map

    # ── Step 1 : Schema ───────────────────────────────────────────────────────

    def _step1_schema(self, split_map: Dict) -> None:
        """
        Every CSV must have exactly: Filename | Label | Species
        Any extra or missing columns → hard failure.
        """
        logger.info("─" * 50)
        logger.info("🔍 Step 1 — Schema check")

        for split, paths in split_map.items():
            csv_path = paths["csv"]
            try:
                with open(csv_path, "r", newline="") as f:
                    reader  = csv.DictReader(f)
                    headers = set(reader.fieldnames or [])

                missing = REQUIRED_COLUMNS - headers
                extra   = headers - REQUIRED_COLUMNS

                if missing:
                    msg = (
                        f"[{split}] CSV missing required columns: {sorted(missing)}  "
                        f"(found: {sorted(headers)})"
                    )
                    logger.error(f"   ✗ {msg}")
                    self.failed_checks.append(msg)
                elif extra:
                    # extra columns are tolerated — just warn
                    msg = f"[{split}] CSV has unexpected extra columns: {sorted(extra)}"
                    logger.warning(f"   ⚠ {msg}")
                    self.warnings.append(msg)
                    logger.info(f"   ✅ [{split}] schema OK (extra columns tolerated)")
                else:
                    logger.info(f"   ✅ [{split}] schema OK")

            except Exception as e:
                msg = f"[{split}] Could not read CSV: {e}"
                logger.error(f"   ✗ {msg}")
                self.failed_checks.append(msg)

    # ── Step 2 : Image ↔ CSV Alignment ───────────────────────────────────────

    def _step2_alignment(
        self, split_map: Dict
    ) -> Dict[str, Dict]:
        """
        For each split:
          - csv_files  = set of Filename values from CSV
          - disk_files = set of filenames in images/<split>/
          - in CSV but not on disk → missing_from_disk  (hard fail if > threshold)
          - on disk but not in CSV → orphan_on_disk     (soft warning)

        Returns split_stats dict with counts.
        """
        logger.info("─" * 50)
        logger.info("🔍 Step 2 — Image ↔ CSV alignment")

        split_stats = {}

        for split, paths in split_map.items():
            csv_path   = paths["csv"]
            images_dir = paths["images"]

            # collect CSV filenames
            with open(csv_path, "r", newline="") as f:
                csv_files = {
                    row["Filename"].strip()
                    for row in csv.DictReader(f)
                    if row.get("Filename", "").strip()
                }

            # collect disk filenames
            disk_files = {f.name for f in images_dir.iterdir() if f.is_file()}

            missing_from_disk = sorted(csv_files - disk_files)   # in CSV, not on disk
            orphan_on_disk    = sorted(disk_files - csv_files)   # on disk, not in CSV

            missing_ratio = len(missing_from_disk) / max(len(csv_files), 1)

            split_stats[split] = {
                "csv_rows"        : len(csv_files),
                "images_on_disk"  : len(disk_files),
                "missing_from_disk": len(missing_from_disk),
                "orphan_on_disk"  : len(orphan_on_disk),
                "corrupt"         : 0,   # filled in by step 4
            }

            logger.info(
                f"   [{split}] csv_rows={len(csv_files)}  "
                f"disk={len(disk_files)}  "
                f"missing={len(missing_from_disk)}  "
                f"orphan={len(orphan_on_disk)}"
            )

            # hard failure: too many files referenced in CSV but absent on disk
            if missing_ratio > self.config.missing_file_threshold:
                msg = (
                    f"[{split}] {len(missing_from_disk)}/{len(csv_files)} "
                    f"({missing_ratio:.1%}) CSV filenames missing from disk — "
                    f"threshold is {self.config.missing_file_threshold:.0%}. "
                    f"First few: {missing_from_disk[:5]}"
                )
                logger.error(f"   ✗ {msg}")
                self.failed_checks.append(msg)
            elif missing_from_disk:
                msg = (
                    f"[{split}] {len(missing_from_disk)} CSV filename(s) missing from disk "
                    f"(below threshold — soft warning). First few: {missing_from_disk[:3]}"
                )
                logger.warning(f"   ⚠ {msg}")
                self.warnings.append(msg)
            else:
                logger.info(f"   ✅ [{split}] all CSV filenames present on disk")

            # soft warning: files on disk not referenced in CSV
            if orphan_on_disk:
                msg = (
                    f"[{split}] {len(orphan_on_disk)} image(s) on disk not referenced "
                    f"in CSV. First few: {orphan_on_disk[:3]}"
                )
                logger.warning(f"   ⚠ {msg}")
                self.warnings.append(msg)

        return split_stats

    # ── Step 3 : Label Integrity ──────────────────────────────────────────────

    def _step3_label_integrity(self, split_map: Dict) -> None:
        """
        For every CSV row:
          - Label must be integer in [valid_label_min .. valid_label_max]
          - Species must be a non-empty string
          - Filename must be non-empty
          - No null / empty values in any required column
        Any violation → hard failure.
        """
        logger.info("─" * 50)
        logger.info("🔍 Step 3 — Label integrity")

        for split, paths in split_map.items():
            csv_path   = paths["csv"]
            bad_rows   : List[Tuple[int, str]] = []   # (row_number, reason)
            total_rows = 0

            with open(csv_path, "r", newline="") as f:
                for row_num, row in enumerate(csv.DictReader(f), start=2):
                    total_rows += 1
                    filename = row.get("Filename", "").strip()
                    label    = row.get("Label",    "").strip()
                    species  = row.get("Species",  "").strip()

                    # null checks
                    if not filename:
                        bad_rows.append((row_num, "Empty Filename"))
                        continue
                    if not label:
                        bad_rows.append((row_num, f"Empty Label (Filename={filename})"))
                        continue
                    if not species:
                        bad_rows.append((row_num, f"Empty Species (Filename={filename})"))
                        continue

                    # label must be integer
                    try:
                        label_int = int(label)
                    except ValueError:
                        bad_rows.append(
                            (row_num, f"Non-integer Label='{label}' (Filename={filename})")
                        )
                        continue

                    # label must be in valid range
                    if not (self.config.valid_label_min <= label_int <= self.config.valid_label_max):
                        bad_rows.append((
                            row_num,
                            f"Label={label_int} out of range "
                            f"[{self.config.valid_label_min}..{self.config.valid_label_max}] "
                            f"(Filename={filename})"
                        ))

            if bad_rows:
                msg = (
                    f"[{split}] {len(bad_rows)}/{total_rows} row(s) with label errors. "
                    f"First few: {[r[1] for r in bad_rows[:3]]}"
                )
                logger.error(f"   ✗ {msg}")
                self.failed_checks.append(msg)
            else:
                logger.info(
                    f"   ✅ [{split}] all {total_rows} rows have valid labels "
                    f"in [{self.config.valid_label_min}..{self.config.valid_label_max}]"
                )

    # ── Step 4 : Image File Integrity ─────────────────────────────────────────

    def _step4_image_integrity(
        self, split_map: Dict, split_stats: Dict
    ) -> None:
        """
        Open every image file with Pillow and call verify().
        Catches corrupt / truncated files that would crash training.
        Any corrupt image → hard failure.

        split_stats["corrupt"] is updated in-place for reporting.
        """
        logger.info("─" * 50)
        logger.info("🔍 Step 4 — Image file integrity")

        for split, paths in split_map.items():
            images_dir   = paths["images"]
            image_files  = sorted(f for f in images_dir.iterdir() if f.is_file())
            corrupt      : List[str] = []

            for img_path in image_files:
                try:
                    with Image.open(img_path) as img:
                        img.verify()   # catches truncation / corruption
                except (UnidentifiedImageError, Exception) as e:
                    corrupt.append(img_path.name)
                    logger.warning(f"   ⚠ [{split}] corrupt: {img_path.name} — {e}")

            # update split_stats in-place
            if split in split_stats:
                split_stats[split]["corrupt"] = len(corrupt)

            if corrupt:
                msg = (
                    f"[{split}] {len(corrupt)} corrupt image(s) found. "
                    f"First few: {corrupt[:5]}"
                )
                logger.error(f"   ✗ {msg}")
                self.failed_checks.append(msg)
            else:
                logger.info(
                    f"   ✅ [{split}] all {len(image_files)} images readable"
                )

    # ── Step 5 : Class Distribution ───────────────────────────────────────────

    def _step5_class_distribution(
        self, split_map: Dict
    ) -> Dict[str, Dict[str, int]]:
        """
        Count images per Label per split.
        Flags:
          - Any class < imbalance_threshold fraction of split → soft warning
          - Classes in train absent from val or test          → soft warning

        Returns:
          {
            "train": {"0": 1234, "1": 456, ...},
            "val"  : {"0": 310,  "1": 112, ...},
            "test" : {"0": 308,  "1": 110, ...},
          }
        """
        logger.info("─" * 50)
        logger.info("🔍 Step 5 — Class distribution")

        distribution: Dict[str, Dict[str, int]] = {}

        for split, paths in split_map.items():
            csv_path = paths["csv"]
            counts: Dict[str, int] = defaultdict(int)

            with open(csv_path, "r", newline="") as f:
                for row in csv.DictReader(f):
                    label = row.get("Label", "").strip()
                    if label:
                        counts[label] += 1

            distribution[split] = dict(counts)
            total = sum(counts.values())

            logger.info(f"   [{split}] total={total}  classes={len(counts)}")
            for label, count in sorted(counts.items(), key=lambda x: int(x[0])):
                pct = count / total if total else 0
                logger.info(f"      Label {label:>2} : {count:>5}  ({pct:.1%})")

            # imbalance check
            for label, count in counts.items():
                pct = count / total if total else 0
                if pct < self.config.imbalance_threshold:
                    msg = (
                        f"[{split}] Label {label} has only {count} samples "
                        f"({pct:.2%} of split) — below imbalance threshold "
                        f"{self.config.imbalance_threshold:.0%}"
                    )
                    logger.warning(f"   ⚠ {msg}")
                    self.warnings.append(msg)

        # cross-split class presence check
        if "train" in distribution:
            train_classes = set(distribution["train"].keys())
            for other_split in ("val", "test"):
                if other_split not in distribution:
                    continue
                other_classes = set(distribution[other_split].keys())
                missing_in_other = train_classes - other_classes
                if missing_in_other:
                    msg = (
                        f"Classes present in train but absent in {other_split}: "
                        f"{sorted(missing_in_other)}"
                    )
                    logger.warning(f"   ⚠ {msg}")
                    self.warnings.append(msg)

        return distribution

    # ── Step 6 : Cross-Split Leakage ──────────────────────────────────────────

    def _step6_leakage(self, split_map: Dict) -> None:
        """
        Collect Filename sets per split.
        Report any overlap between splits (data leakage risk) → soft warning.
        """
        logger.info("─" * 50)
        logger.info("🔍 Step 6 — Cross-split leakage detection")

        split_filenames: Dict[str, set] = {}

        for split, paths in split_map.items():
            csv_path = paths["csv"]
            with open(csv_path, "r", newline="") as f:
                split_filenames[split] = {
                    row["Filename"].strip()
                    for row in csv.DictReader(f)
                    if row.get("Filename", "").strip()
                }

        splits = list(split_filenames.keys())
        found_leakage = False

        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                a, b  = splits[i], splits[j]
                overlap = split_filenames[a] & split_filenames[b]
                if overlap:
                    msg = (
                        f"Data leakage: {len(overlap)} filename(s) appear in both "
                        f"'{a}' and '{b}'. First few: {sorted(overlap)[:5]}"
                    )
                    logger.warning(f"   ⚠ {msg}")
                    self.warnings.append(msg)
                    found_leakage = True

        if not found_leakage:
            logger.info("   ✅ No cross-split filename overlap detected")

    # ── Build + Save DataValidationArtifact ───────────────────────────────────

    def _build_artifact(
        self,
        ingestion_artifact : DataIngestionArtifact,
        is_valid           : bool,
        split_stats        : Dict,
        class_distribution : Dict,
    ) -> DataValidationArtifact:

        validated_at = datetime.now()

        artifact = DataValidationArtifact(
            ingestion_artifact    = ingestion_artifact,
            is_valid              = is_valid,
            failed_checks         = list(self.failed_checks),
            warnings              = list(self.warnings),
            split_stats           = split_stats,
            class_distribution    = class_distribution,
            validated_at          = validated_at,
            validation_report_path= self.config.validation_report_path,
        )

        # write full JSON report
        report = {
            "version_id"        : ingestion_artifact.normalized_dir.name,
            "is_valid"          : is_valid,
            "validated_at"      : validated_at.isoformat(),
            "failed_checks"     : list(self.failed_checks),
            "warnings"          : list(self.warnings),
            "split_stats"       : split_stats,
            "class_distribution": class_distribution,
            "ingestion_artifact": str(ingestion_artifact.artifact_path),
            "normalized_dir"    : str(ingestion_artifact.normalized_dir),
            "source_type"       : ingestion_artifact.source_type,
            "s3_bucket"         : ingestion_artifact.kafka_artifact.s3_bucket,
            "s3_key"            : ingestion_artifact.kafka_artifact.s3_key,
            "file_hash"         : ingestion_artifact.kafka_artifact.file_hash,
        }

        save_json(path=self.config.validation_report_path, data=report)
        logger.info(f"📋 Validation report saved : {self.config.validation_report_path}")

        return artifact

    # ── Summary ───────────────────────────────────────────────────────────────

    def _log_summary(self, artifact: DataValidationArtifact) -> None:
        logger.info("=" * 60)
        logger.info("📊 DATA VALIDATION COMPLETE")
        logger.info(f"   Version        : {artifact.ingestion_artifact.normalized_dir.name}")
        logger.info(f"   Is valid       : {'✅ YES' if artifact.is_valid else '❌ NO'}")
        logger.info(f"   Hard failures  : {len(artifact.failed_checks)}")
        logger.info(f"   Soft warnings  : {len(artifact.warnings)}")
        logger.info(f"   Validated at   : {artifact.validated_at}")
        for split, stats in artifact.split_stats.items():
            logger.info(
                f"   [{split}] "
                f"csv={stats.get('csv_rows', '?')}  "
                f"disk={stats.get('images_on_disk', '?')}  "
                f"corrupt={stats.get('corrupt', 0)}"
            )
        logger.info(f"   Report         : {artifact.validation_report_path}")
        if not artifact.is_valid:
            logger.error("🚫 Pipeline should STOP — data did not pass validation")
        logger.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("🚀 Starting Data Validation")
    logger.info("=" * 60)

    config_manager = ConfigurationManager()
    config         = config_manager.get_data_validation_config()

    validator = DataValidation(config)
    artifact  = validator.run()

    if artifact is None:
        logger.info("✅ Nothing to do — already validated (same version)")
    elif artifact.is_valid:
        logger.info("✅ Validation passed — safe to proceed to data transformation")
    else:
        logger.error("❌ Validation FAILED — check validation_report.json")
        logger.error(f"   Report : {config.validation_report_path}")
        exit(1)   # non-zero exit so pipeline orchestrator knows to stop

    return artifact


if __name__ == "__main__":
    main()