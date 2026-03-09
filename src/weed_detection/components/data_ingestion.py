# weed_detection/components/data_ingestion.py
# ─────────────────────────────────────────────────────────────────────────────
# Version-aware ingestion stage.
#
# Version check:
#   Reads latest_artifact.json (written by kafka_consumer)
#   Reads ingestion_state.json (written by this component after each run)
#   version_id = timestamp + file_hash
#   SKIP  → same version_id already in ingestion_state.json
#   RUN   → new version_id → full extract + normalize + artifact
#
# Supported layouts:
#   real      → deep-weed format: images/ flat + labels/subset CSVs
#   synthetic → train_image_*/train_label_* naming with split folders
#   flat      → fallback, everything dumped under train/
#
# Output per version (real/deep-weed):
#   normalized/v_<timestamp>_<hash>/
#   ├── images/
#   │   ├── train/   images from merged train_subset*.csv
#   │   ├── val/     images from merged val_subset*.csv
#   │   └── test/    images from merged test_subset*.csv
#   └── labels/
#       ├── train.csv
#       ├── val.csv
#       ├── test.csv
#       └── labels.csv  (master copy)
# ─────────────────────────────────────────────────────────────────────────────

import zipfile
import shutil
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

from weed_detection import logger
from weed_detection.entity.config_entity import DataIngestionConfig
from weed_detection.entity.artifact_entity import KafkaArtifact, DataIngestionArtifact
from weed_detection.config.configuration import ConfigurationManager
from weed_detection.components.kafka_consumer import load_kafka_artifact
from weed_detection.utils.utility import create_directories, save_json, load_json

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_LABEL_EXTS = {".txt", ".csv"}

SPLIT_PREFIXES = {
    "train": "train_subset",
    "val"  : "val_subset",
    "test" : "test_subset",
}


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> Optional[DataIngestionArtifact]:
        logger.info("=" * 60)
        logger.info("🚀 Data Ingestion — version check")
        logger.info("=" * 60)

        # 1. load KafkaArtifact → get zip path + metadata
        kafka_artifact = self._load_kafka_artifact()

        # 2. check version — skip if already processed
        version_id = self._make_version_id(kafka_artifact)
        if self._already_processed(version_id):
            logger.info(f"⏭️  Version '{version_id}' already processed — skipping")
            logger.info("   No new data since last run.")
            return None

        logger.info(f"🆕 New version  : {version_id}")
        logger.info(f"   Zip         : {kafka_artifact.zip_file_path}")
        logger.info(f"   Received    : {kafka_artifact.received_at}")
        logger.info(f"   Hash        : {kafka_artifact.file_hash}")

        # 3. extract zip into versioned unzip folder
        extract_dir = self._extract_zip(kafka_artifact.zip_file_path, version_id)

        # 4. detect layout
        layout = self._detect_layout(extract_dir)
        logger.info(f"🔍 Layout : {layout}")

        # 5. normalize into isolated versioned folder
        versioned_out = self.config.normalized_dir / version_id
        splits        = self._normalize(extract_dir, layout, versioned_out)

        # 6. validate image ↔ label consistency
        warnings = self._validate(splits, layout)

        # 7. build + save DataIngestionArtifact
        artifact = self._build_artifact(
            kafka_artifact, splits, layout, warnings, versioned_out
        )

        # 8. update ingestion state → next run skips this version
        self._update_state(version_id, artifact)

        self._log_summary(artifact)
        return artifact

    # ── version control ───────────────────────────────────────────────────────

    @staticmethod
    def _make_version_id(kafka_artifact: KafkaArtifact) -> str:
        """
        version_id = v_<timestamp>_<hash>
        e.g. v_20260307_201234_a3f1b2c4

        Same zip sent twice → identical hash + timestamp → same version_id → skipped.
        Different zip → different hash → different version_id → processed.
        """
        ts = kafka_artifact.received_at.strftime("%Y%m%d_%H%M%S")
        return f"v_{ts}_{kafka_artifact.file_hash}"

    def _already_processed(self, version_id: str) -> bool:
        """True if ingestion_state.json records this version_id as already done."""
        state_path = self.config.ingestion_state_path
        if not state_path.exists():
            return False
        try:
            state = load_json(state_path)
            return state.get("last_version_id") == version_id
        except Exception as e:
            logger.warning(f"⚠️  Could not read ingestion state: {e} — will reprocess")
            return False

    def _update_state(self, version_id: str, artifact: DataIngestionArtifact) -> None:
        """Write ingestion_state.json after successful run."""
        save_json(
            path = self.config.ingestion_state_path,
            data = {
                "last_version_id"    : version_id,
                "last_processed_at"  : datetime.now().isoformat(),
                "last_zip"           : str(artifact.kafka_artifact.zip_file_path),
                "last_normalized_dir": str(artifact.normalized_dir),
                "last_artifact_path" : str(artifact.artifact_path),
            }
        )
        logger.info(f"💾 State saved  : {self.config.ingestion_state_path}")
        logger.info(f"   Version     : {version_id}")

    # ── step 1 : load KafkaArtifact ───────────────────────────────────────────

    def _load_kafka_artifact(self) -> KafkaArtifact:
        logger.info(f"📋 Reading latest_artifact.json from {self.config.kafka_data_dir}")
        artifact = load_kafka_artifact(self.config.kafka_data_dir)
        logger.info(f"✅ Loaded : {artifact.zip_file_path.name}")
        return artifact

    # ── step 2 : extract zip ──────────────────────────────────────────────────

    def _extract_zip(self, zip_path: Path, version_id: str) -> Path:
        """Extract zip into unzip_dir/<version_id>/"""
        extract_dir = self.config.unzip_dir / version_id
        extract_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📂 Extracting → {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            zf.extractall(extract_dir)
        logger.info(f"   {len(members)} entries extracted")
        return extract_dir

    # ── step 3 : detect layout ────────────────────────────────────────────────

    def _detect_layout(self, root: Path) -> str:
        """
        synthetic  →  split folders (train/val/test) + train_image_* naming
        real       →  images/ + labels/ folders  (deep-weed format)
        flat       →  fallback — no recognizable structure
        """
        all_dirs  = {p.name.lower() for p in root.rglob("*") if p.is_dir()}
        all_files = [f for f in root.rglob("*") if f.is_file()]

        has_split_dirs       = bool(all_dirs & {"train", "val", "test"})
        has_synthetic_naming = any(
            f.stem.startswith((
                "train_image_", "val_image_",
                "train_label_", "val_label_",
            ))
            for f in all_files
        )

        if has_split_dirs and has_synthetic_naming:
            return "synthetic"
        if "images" in all_dirs and "labels" in all_dirs:
            return "real"
        return "flat"

    # ── step 4 : normalize ────────────────────────────────────────────────────

    def _normalize(
        self, root: Path, layout: str, out_root: Path
    ) -> Dict[str, Dict[str, Path]]:
        if layout == "synthetic":
            return self._normalize_synthetic(root, out_root)
        elif layout == "real":
            return self._normalize_real(root, out_root)
        else:
            return self._normalize_flat(root, out_root)

    # ── normalize: synthetic ──────────────────────────────────────────────────

    def _normalize_synthetic(
        self, root: Path, out_root: Path
    ) -> Dict[str, Dict[str, Path]]:
        """
        Input : root/train/train_image_000.png + train_label_000.txt
        Output: out_root/images/train/000.png  + out_root/labels/train/000.txt
        Strips train_image_ / train_label_ prefixes for clean filenames.
        """
        split_dirs = [
            d for d in root.rglob("*")
            if d.is_dir() and d.name.lower() in ("train", "val", "test")
        ] or [root]

        splits = {}
        for split_dir in split_dirs:
            split     = split_dir.name.lower() if split_dir != root else "train"
            img_out   = out_root / "images" / split
            label_out = out_root / "labels" / split
            img_out.mkdir(parents=True, exist_ok=True)
            label_out.mkdir(parents=True, exist_ok=True)

            for f in sorted(split_dir.iterdir()):
                if not f.is_file():
                    continue
                clean = self._strip_synthetic_prefix(f.stem, split)
                if f.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                    shutil.copy2(f, img_out / (clean + f.suffix.lower()))
                elif f.suffix.lower() in SUPPORTED_LABEL_EXTS:
                    shutil.copy2(f, label_out / (clean + f.suffix.lower()))

            splits[split] = {"images": img_out, "labels": label_out}
            logger.info(
                f"   [{split}] {len(list(img_out.iterdir()))} images  "
                f"{len(list(label_out.iterdir()))} labels"
            )
        return splits

    @staticmethod
    def _strip_synthetic_prefix(stem: str, split: str) -> str:
        for prefix in (
            f"{split}_image_", f"{split}_label_",
            "train_image_", "train_label_",
            "val_image_",   "val_label_",
            "test_image_",  "test_label_",
        ):
            if stem.startswith(prefix):
                return stem[len(prefix):]
        return stem

    # ── normalize: real (deep-weed) ───────────────────────────────────────────

    def _normalize_real(
        self, root: Path, out_root: Path
    ) -> Dict[str, Dict[str, Path]]:
        """
        deep-weed structure:
          images/  → flat directory of ALL images
          labels/  → train_subset0..4.csv  val_subset0..4.csv  test_subset0..4.csv
                     labels.csv  (master — all 17,509 images)

        Each CSV row: Filename | Label | Species

        For each split:
          1. Find all subset CSVs (train_subset*.csv)
          2. Merge them into one CSV (train.csv)
          3. Copy only the images referenced in those CSVs into images/train/
        """
        images_src = self._find_dir(root, "images")
        labels_src = self._find_dir(root, "labels")

        # index all source images by filename → O(1) lookup
        all_images = {
            f.name: f for f in images_src.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
        }
        logger.info(f"   Total source images : {len(all_images)}")

        labels_out = out_root / "labels"
        labels_out.mkdir(parents=True, exist_ok=True)

        splits = {}

        for split, prefix in SPLIT_PREFIXES.items():
            subset_files = sorted(
                f for f in labels_src.iterdir()
                if f.is_file()
                and f.stem.startswith(prefix)
                and f.suffix == ".csv"
            )
            if not subset_files:
                logger.info(f"   [{split}] no subset CSVs found — skipping")
                continue

            logger.info(
                f"   [{split}] merging {len(subset_files)} subset(s): "
                f"{[f.name for f in subset_files]}"
            )

            img_out        = out_root / "images" / split
            img_out.mkdir(parents=True, exist_ok=True)
            merged_csv     = labels_out / f"{split}.csv"
            imgs_copied    = 0
            rows_written   = 0
            imgs_missing   = []
            header_written = False

            with open(merged_csv, "w", newline="") as out_f:
                writer = None
                for subset_file in subset_files:
                    with open(subset_file, "r", newline="") as in_f:
                        reader = csv.DictReader(in_f)
                        if not header_written:
                            writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
                            writer.writeheader()
                            header_written = True
                        for row in reader:
                            filename = row.get("Filename", "").strip()
                            if filename in all_images:
                                dest = img_out / filename
                                if not dest.exists():
                                    shutil.copy2(all_images[filename], dest)
                                    imgs_copied += 1
                            else:
                                imgs_missing.append(filename)
                            writer.writerow(row)
                            rows_written += 1

            logger.info(
                f"   [{split}] {imgs_copied} images copied  "
                f"{rows_written} label rows → {merged_csv.name}"
            )
            if imgs_missing:
                logger.warning(
                    f"   [{split}] ⚠️  {len(imgs_missing)} filename(s) in CSV "
                    f"but not found in images/: {imgs_missing[:3]}"
                )

            splits[split] = {"images": img_out, "labels": labels_out}

        # copy master labels.csv as-is
        master = labels_src / "labels.csv"
        if master.exists():
            shutil.copy2(master, labels_out / "labels.csv")
            logger.info(f"   📋 Master labels.csv copied")

        return splits

    # ── normalize: flat fallback ──────────────────────────────────────────────

    def _normalize_real(
        self, root: Path, out_root: Path
    ) -> Dict[str, Dict[str, Path]]:
        images_src = self._find_dir(root, "images")
        labels_src = self._find_dir(root, "labels")

        all_images = {
            f.name: f for f in images_src.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
        }
        logger.info(f"   Total source images : {len(all_images)}")

        # ── build Species lookup from master labels.csv ───────────────────────
        master_csv = labels_src / "labels.csv"
        species_lookup: dict = {}
        if master_csv.exists():
            with open(master_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                if "Species" in (reader.fieldnames or []):
                    species_lookup = {
                        row["Filename"].strip(): row["Species"].strip()
                        for row in reader
                    }
            logger.info(f"   Species lookup built : {len(species_lookup)} entries")
        else:
            logger.warning("   ⚠️  labels.csv not found — Species column will be empty")

        labels_out = out_root / "labels"
        labels_out.mkdir(parents=True, exist_ok=True)

        splits = {}

        for split, prefix in SPLIT_PREFIXES.items():
            subset_files = sorted(
                f for f in labels_src.iterdir()
                if f.is_file()
                and f.stem.startswith(prefix)
                and f.suffix == ".csv"
            )
            if not subset_files:
                logger.info(f"   [{split}] no subset CSVs found — skipping")
                continue

            logger.info(
                f"   [{split}] merging {len(subset_files)} subset(s): "
                f"{[f.name for f in subset_files]}"
            )

            img_out        = out_root / "images" / split
            img_out.mkdir(parents=True, exist_ok=True)
            merged_csv     = labels_out / f"{split}.csv"
            imgs_copied    = 0
            rows_written   = 0
            imgs_missing   = []
            header_written = False

            with open(merged_csv, "w", newline="") as out_f:
                writer = None
                for subset_file in subset_files:
                    with open(subset_file, "r", newline="") as in_f:
                        reader = csv.DictReader(in_f)

                        # ensure output fieldnames always include Species
                        out_fieldnames = list(reader.fieldnames or [])
                        if "Species" not in out_fieldnames:
                            out_fieldnames = out_fieldnames + ["Species"]

                        if not header_written:
                            writer = csv.DictWriter(out_f, fieldnames=out_fieldnames)
                            writer.writeheader()
                            header_written = True

                        for row in reader:
                            filename = row.get("Filename", "").strip()

                            # inject Species from master lookup if missing
                            if "Species" not in row or not row.get("Species", "").strip():
                                row["Species"] = species_lookup.get(filename, "Unknown")

                            if filename in all_images:
                                dest = img_out / filename
                                if not dest.exists():
                                    shutil.copy2(all_images[filename], dest)
                                    imgs_copied += 1
                            else:
                                imgs_missing.append(filename)

                            writer.writerow(row)
                            rows_written += 1

            logger.info(
                f"   [{split}] {imgs_copied} images copied  "
                f"{rows_written} label rows → {merged_csv.name}"
            )
            if imgs_missing:
                logger.warning(
                    f"   [{split}] ⚠️  {len(imgs_missing)} filename(s) in CSV "
                    f"but not found in images/: {imgs_missing[:3]}"
                )

            splits[split] = {"images": img_out, "labels": labels_out}

        # copy master labels.csv as-is
        if master_csv.exists():
            shutil.copy2(master_csv, labels_out / "labels.csv")
            logger.info(f"   📋 Master labels.csv copied")

        return splits
    # ── step 5 : validate ─────────────────────────────────────────────────────

    def _validate(
        self, splits: Dict[str, Dict[str, Path]], layout: str
    ) -> List[str]:
        """
        real layout      → cross-check image files against Filename column in split CSV
        other layouts    → cross-check image stems against label stems
        Returns list of warning strings (empty = clean).
        """
        warnings = []

        for split, paths in splits.items():
            img_dir   = paths["images"]
            label_dir = paths["labels"]

            if layout == "real":
                merged_csv = label_dir / f"{split}.csv"
                if not merged_csv.exists():
                    continue
                with open(merged_csv, "r") as f:
                    csv_files = {row["Filename"].strip() for row in csv.DictReader(f)}
                img_files = {f.name for f in img_dir.iterdir() if f.is_file()}

                if extra := sorted(img_files - csv_files):
                    msg = f"[{split}] {len(extra)} image(s) on disk not in CSV: {extra[:3]}"
                    logger.warning(f"⚠️  {msg}"); warnings.append(msg)
                if missing := sorted(csv_files - img_files):
                    msg = f"[{split}] {len(missing)} CSV row(s) with no image on disk: {missing[:3]}"
                    logger.warning(f"⚠️  {msg}"); warnings.append(msg)
            else:
                img_stems   = {f.stem for f in img_dir.iterdir()   if f.is_file()}
                label_stems = {f.stem for f in label_dir.iterdir() if f.is_file()}
                if no_label := sorted(img_stems - label_stems):
                    msg = f"[{split}] {len(no_label)} image(s) missing label: {no_label[:3]}"
                    logger.warning(f"⚠️  {msg}"); warnings.append(msg)
                if no_image := sorted(label_stems - img_stems):
                    msg = f"[{split}] {len(no_image)} label(s) missing image: {no_image[:3]}"
                    logger.warning(f"⚠️  {msg}"); warnings.append(msg)

        return warnings

    # ── step 6 : build artifact ───────────────────────────────────────────────

    def _build_artifact(
        self,
        kafka_artifact : KafkaArtifact,
        splits         : Dict[str, Dict[str, Path]],
        layout         : str,
        warnings       : List[str],
        versioned_out  : Path,
    ) -> DataIngestionArtifact:

        total_images = sum(
            len([f for f in p["images"].iterdir() if f.is_file()])
            for p in splits.values()
        )
        total_labels = sum(
            len([f for f in p["labels"].iterdir() if f.is_file()])
            for p in splits.values()
        )

        artifact = DataIngestionArtifact(
            kafka_artifact   = kafka_artifact,
            unzip_dir        = self.config.unzip_dir,
            normalized_dir   = versioned_out,
            train_images_dir = splits["train"]["images"],
            train_labels_dir = splits["train"]["labels"],
            val_images_dir   = splits.get("val",  {}).get("images"),
            val_labels_dir   = splits.get("val",  {}).get("labels"),
            test_images_dir  = splits.get("test", {}).get("images"),
            test_labels_dir  = splits.get("test", {}).get("labels"),
            source_type      = layout,
            total_images     = total_images,
            total_labels     = total_labels,
            splits           = list(splits.keys()),
            warnings         = warnings,
            artifact_path    = self.config.artifact_path,
        )

        save_json(
            path = self.config.artifact_path,
            data = {
                "version_id"      : versioned_out.name,
                "kafka_zip"       : str(kafka_artifact.zip_file_path),
                "received_at"     : kafka_artifact.received_at.isoformat(),
                "file_hash"       : kafka_artifact.file_hash,
                "s3_bucket"       : kafka_artifact.s3_bucket,
                "s3_key"          : kafka_artifact.s3_key,
                "unzip_dir"       : str(artifact.unzip_dir),
                "normalized_dir"  : str(artifact.normalized_dir),
                "train_images_dir": str(artifact.train_images_dir),
                "train_labels_dir": str(artifact.train_labels_dir),
                "val_images_dir"  : str(artifact.val_images_dir)  if artifact.val_images_dir  else None,
                "val_labels_dir"  : str(artifact.val_labels_dir)  if artifact.val_labels_dir  else None,
                "test_images_dir" : str(artifact.test_images_dir) if artifact.test_images_dir else None,
                "test_labels_dir" : str(artifact.test_labels_dir) if artifact.test_labels_dir else None,
                "source_type"     : artifact.source_type,
                "total_images"    : artifact.total_images,
                "total_labels"    : artifact.total_labels,
                "splits"          : artifact.splits,
                "warnings"        : artifact.warnings,
                "timestamp"       : datetime.now().isoformat(),
            }
        )
        logger.info(f"📋 Artifact saved : {self.config.artifact_path}")
        return artifact

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_dir(root: Path, name: str) -> Path:
        for d in root.rglob("*"):
            if d.is_dir() and d.name.lower() == name:
                return d
        raise FileNotFoundError(f"Directory '{name}/' not found under {root}")

    def _log_summary(self, artifact: DataIngestionArtifact) -> None:
        logger.info("=" * 60)
        logger.info("📊 DATA INGESTION COMPLETE")
        logger.info(f"   Version      : {artifact.normalized_dir.name}")
        logger.info(f"   Source type  : {artifact.source_type}")
        logger.info(f"   Splits       : {artifact.splits}")
        logger.info(f"   Total images : {artifact.total_images}")
        logger.info(f"   Total labels : {artifact.total_labels}")
        logger.info(f"   Warnings     : {len(artifact.warnings)}")
        logger.info(f"   Train images : {artifact.train_images_dir}")
        if artifact.val_images_dir:
            logger.info(f"   Val images   : {artifact.val_images_dir}")
        if artifact.test_images_dir:
            logger.info(f"   Test images  : {artifact.test_images_dir}")
        logger.info(f"   Artifact     : {artifact.artifact_path}")
        logger.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("🚀 Starting Data Ingestion")
    logger.info("=" * 60)

    config_manager = ConfigurationManager()
    config         = config_manager.get_data_ingestion_config()

    ingestion = DataIngestion(config)
    artifact  = ingestion.run()

    if artifact is None:
        logger.info("✅ Nothing to do — already up to date")
    else:
        logger.info(f"✅ Ingestion complete — version: {artifact.normalized_dir.name}")

    return artifact


if __name__ == "__main__":
    main()