
import os
import heapq
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import timm
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
import mlflow

from weed_detection import logger
from weed_detection.config.configuration import ConfigurationManager
from weed_detection.constants.constant import IMAGENET_MEAN, IMAGENET_STD
from weed_detection.entity.artifact_entity import ModelExportArtifact
from weed_detection.entity.config_entity import ModelExportConfig, ModelRegistryConfig
from weed_detection.utils.utility import load_json, save_json


class ModelExport:
    """Export champion model to ONNX formats"""
    
    def __init__(
        self,
        export_config    : ModelExportConfig,
        registry_config  : ModelRegistryConfig,
    ):
        self.export_config = export_config
        self.registry_config = registry_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def run(self) -> Optional[ModelExportArtifact]:
        """Execute model export pipeline"""
        logger.info("=" * 70)
        logger.info("🚀 Model Export — champion → ONNX")
        logger.info("=" * 70)
        
        # 1. Load champion metadata
        champion_meta = self._load_champion_metadata()
        if champion_meta is None:
            raise RuntimeError(
                "No champion model found. Run model_evaluation.py first."
            )
        
        champion_run_id = champion_meta["champion_run_id"]
        architecture    = champion_meta["architecture"]
        dropout_rate    = champion_meta.get("dropout_rate", 0.3)
        num_classes     = champion_meta.get("num_classes", self.export_config.num_classes)
        input_size      = champion_meta.get("input_size", self.export_config.input_size)
        weighted_f1     = champion_meta.get("weighted_f1", 0.0)
        
        logger.info(f"   Champion run  : {champion_run_id}")
        logger.info(f"   Architecture  : {architecture}")
        logger.info(f"   Weighted F1   : {weighted_f1:.4f}")
        logger.info(f"   Input size    : {input_size}")
        
        # 2. Version check - skip if already exported
        if self._already_exported(champion_run_id):
            logger.info(f"⏭️  Champion '{champion_run_id}' already exported — skipping")
            return None
        
        # 3. Reconstruct and load champion model
        model = self._load_champion_model(architecture, dropout_rate, num_classes)
        
        # 4. Warm-up forward pass
        self._warmup(model, input_size)
        
        # 5. Export FP32 ONNX
        self._export_onnx_fp32(model, input_size)
        
        # 6. Export FP16 ONNX (optional)
        fp16_path = None
        if self.export_config.export_fp16 and float16 is not None:
            fp16_path = self._export_onnx_fp16()
        
        # 7. Validate ONNX
        validated = False
        if self.export_config.validate_onnx:
            validated = self._validate_onnx(model, input_size)
        
        # 8. Log to MLflow
        self._log_mlflow(champion_meta, validated, fp16_path is not None)
        
        # 9. Build and save artifact
        artifact = self._build_artifact(
            champion_run_id=champion_run_id,
            architecture=architecture,
            input_size=input_size,
            num_classes=num_classes,
            fp16_path=fp16_path,
            validated=validated,
        )
        
        # 10. Update state
        self._update_state(champion_run_id, artifact)
        self._log_summary(artifact)
        
        return artifact
    
    def _load_champion_metadata(self) -> Optional[Dict]:
        """Load champion metadata from registry"""
        meta_path = self.registry_config.champion_metadata_path
        if not meta_path.exists():
            logger.warning(f"⚠️  No champion metadata at {meta_path}")
            return None
        
        meta = load_json(meta_path)
        logger.info(f"✅ Champion metadata loaded")
        return meta
    
    def _load_champion_model(
        self,
        architecture: str,
        dropout_rate: float,
        num_classes: int,
    ) -> nn.Module:
        """Reconstruct and load champion model weights"""
        logger.info("─" * 50)
        logger.info("🏗️  Reconstructing champion model")
        
        # Build exact same architecture as training
        model = timm.create_model(
            architecture,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        num_features = model.num_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes),
        )
        
        # Load weights
        champion_path = self.registry_config.champion_model_path
        if not champion_path.exists():
            raise FileNotFoundError(f"Champion weights not found at {champion_path}")
        
        state_dict = torch.load(champion_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   ✅ Weights loaded from : {champion_path}")
        logger.info(f"   Total params : {total_params:,}")
        
        return model
    
    def _warmup(self, model: nn.Module, input_size: int) -> None:
        """Run dummy forward pass to verify model"""
        logger.info("─" * 50)
        logger.info("🔥 Warm-up forward pass")
        
        dummy = torch.randn(1, 3, input_size, input_size).to(self.device)
        with torch.no_grad():
            output = model(dummy)
        
        assert output.shape == (1, self.export_config.num_classes), \
            f"Unexpected output shape: {output.shape}"
        assert not torch.isnan(output).any(), "NaN detected in warm-up"
        
        logger.info(f"   ✅ Forward pass OK : {dummy.shape} → {output.shape}")
    
    def _export_onnx_fp32(self, model: nn.Module, input_size: int) -> None:
        """Export FP32 ONNX model"""
        logger.info("─" * 50)
        logger.info("📦 Exporting FP32 ONNX")
        
        dummy = torch.randn(1, 3, input_size, input_size).to(self.device)
        
        dynamic_axes = None
        if self.export_config.dynamic_batch:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy,
            str(self.export_config.onnx_model_path),
            export_params=True,
            opset_version=self.export_config.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(str(self.export_config.onnx_model_path))
        onnx.checker.check_model(onnx_model)
        
        size_mb = os.path.getsize(self.export_config.onnx_model_path) / (1024 ** 2)
        logger.info(f"   ✅ FP32 ONNX saved : {self.export_config.onnx_model_path}")
        logger.info(f"   Size : {size_mb:.2f} MB")
    
    def _export_onnx_fp16(self) -> Path:
        """Convert FP32 ONNX to FP16"""
        logger.info("─" * 50)
        logger.info("📦 Converting FP32 → FP16 ONNX")
        
        if float16 is None:
            logger.warning("⚠️  onnxconverter-common not installed — skipping FP16")
            return None
        
        try:
            onnx_model = onnx.load(str(self.export_config.onnx_model_path))
            fp16_model = float16.convert_float_to_float16(
                onnx_model,
                keep_io_types=True,  # Keep input/output as FP32 for compatibility
            )
            onnx.save(fp16_model, str(self.export_config.onnx_fp16_model_path))
            
            size_mb = os.path.getsize(self.export_config.onnx_fp16_model_path) / (1024 ** 2)
            logger.info(f"   ✅ FP16 ONNX saved : {self.export_config.onnx_fp16_model_path}")
            logger.info(f"   Size : {size_mb:.2f} MB")
            
            return self.export_config.onnx_fp16_model_path
            
        except Exception as e:
            logger.warning(f"⚠️  FP16 export failed — {e}")
            return None
    
    def _validate_onnx(self, model: nn.Module, input_size: int) -> bool:
        """Validate ONNX against PyTorch model"""
        logger.info("─" * 50)
        logger.info("🔍 Validating ONNX with onnxruntime")
        
        try:
            # Create dummy input
            dummy_np = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
            dummy_pt = torch.from_numpy(dummy_np).to(self.device)
            
            # PyTorch inference
            with torch.no_grad():
                pt_output = model(dummy_pt).cpu().numpy()
            
            # ONNX Runtime inference
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device.type == "cuda"
                else ["CPUExecutionProvider"]
            )
            session = ort.InferenceSession(
                str(self.export_config.onnx_model_path),
                providers=providers,
            )
            ort_output = session.run(None, {"input": dummy_np})[0]
            
            # Compare outputs
            max_diff = float(np.abs(pt_output - ort_output).max())
            mean_diff = float(np.abs(pt_output - ort_output).mean())
            
            pt_class = int(np.argmax(pt_output))
            ort_class = int(np.argmax(ort_output))
            class_match = pt_class == ort_class
            
            logger.info(f"   Max diff  : {max_diff:.6f}")
            logger.info(f"   Mean diff : {mean_diff:.6f}")
            logger.info(f"   Class match : {class_match} ({pt_class} == {ort_class})")
            
            tolerance = 1e-4
            if max_diff > tolerance:
                logger.warning(f"⚠️  Max diff {max_diff:.6f} exceeds tolerance")
                return False
            
            logger.info("   ✅ ONNX validation PASSED")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  ONNX validation failed — {e}")
            return False
    
    def _log_mlflow(self, champion_meta: Dict, validated: bool, has_fp16: bool) -> None:
        """Log export artifacts to MLflow"""
        if mlflow is None:
            return
        
        try:
            mlflow.set_tracking_uri(champion_meta.get("mlflow_tracking_uri", "sqlite:///mlflow.db"))
            mlflow.set_experiment(champion_meta.get("mlflow_experiment_name", "weed-detection"))
            
            champion_run_id = champion_meta["champion_run_id"]
            
            with mlflow.start_run(run_name=f"export_{champion_run_id[:12]}"):
                fp32_mb = os.path.getsize(self.export_config.onnx_model_path) / (1024 ** 2)
                fp16_mb = (
                    os.path.getsize(self.export_config.onnx_fp16_model_path) / (1024 ** 2)
                    if self.export_config.onnx_fp16_model_path.exists() else 0.0
                )
                
                mlflow.log_params({
                    "export_source": "champion",
                    "champion_run_id": champion_run_id,
                    "architecture": champion_meta.get("architecture"),
                    "opset_version": self.export_config.opset_version,
                    "dynamic_batch": self.export_config.dynamic_batch,
                    "export_fp16": self.export_config.export_fp16,
                })
                
                mlflow.log_metrics({
                    "fp32_size_mb": fp32_mb,
                    "fp16_size_mb": fp16_mb,
                    "onnx_validated": int(validated),
                    "champion_f1": champion_meta.get("weighted_f1", 0),
                })
                
                mlflow.log_artifact(str(self.export_config.onnx_model_path))
                if has_fp16:
                    mlflow.log_artifact(str(self.export_config.onnx_fp16_model_path))
                mlflow.log_artifact(str(self.export_config.model_info_path))
                
                mlflow.set_tags({
                    "stage": "export",
                    "export_format": "onnx",
                    "champion_run": champion_run_id,
                })
                
            logger.info("   ✅ MLflow export logged")
            
        except Exception as e:
            logger.warning(f"MLflow logging failed — {e}")
    
    def _build_artifact(
        self,
        champion_run_id: str,
        architecture: str,
        input_size: int,
        num_classes: int,
        fp16_path: Optional[Path],
        validated: bool,
    ) -> ModelExportArtifact:
        """Build and save export artifact"""
        exported_at = datetime.now()
        
        fp32_mb = os.path.getsize(self.export_config.onnx_model_path) / (1024 ** 2)
        fp16_mb = (
            os.path.getsize(fp16_path) / (1024 ** 2)
            if fp16_path and fp16_path.exists() else None
        )
        
        # Create model_info.json for next stage
        model_info = {
            "champion_run_id": champion_run_id,
            "architecture": architecture,
            "input_size": input_size,
            "num_classes": num_classes,
            "opset_version": self.export_config.opset_version,
            "onnx_model_path": str(self.export_config.onnx_model_path),
            "onnx_fp16_path": str(fp16_path) if fp16_path else None,
            "fp32_size_mb": round(fp32_mb, 2),
            "fp16_size_mb": round(fp16_mb, 2) if fp16_mb else None,
            "onnx_validated": validated,
            "dynamic_batch": self.export_config.dynamic_batch,
            "input_names": ["input"],
            "output_names": ["output"],
            "imagenet_mean": IMAGENET_MEAN,
            "imagenet_std": IMAGENET_STD,
            "exported_at": exported_at.isoformat(),
            "champion_model_path": str(self.registry_config.champion_model_path),
        }
        
        save_json(path=self.export_config.model_info_path, data=model_info)
        
        artifact = ModelExportArtifact(
            onnx_model_path=self.export_config.onnx_model_path,
            onnx_fp16_path=fp16_path,
            model_info_path=self.export_config.model_info_path,
            champion_run_id=champion_run_id,
            architecture=architecture,
            input_size=input_size,
            num_classes=num_classes,
            opset_version=self.export_config.opset_version,
            fp32_size_mb=fp32_mb,
            fp16_size_mb=fp16_mb,
            onnx_validated=validated,
            dynamic_batch=self.export_config.dynamic_batch,
            exported_at=exported_at,
            artifact_path=self.export_config.artifact_path,
        )
        
        # Save artifact JSON
        artifact_dict = {**model_info, "artifact_path": str(self.export_config.artifact_path)}
        save_json(path=self.export_config.artifact_path, data=artifact_dict)
        
        logger.info(f"📋 model_info.json saved : {self.export_config.model_info_path}")
        
        return artifact
    
    def _already_exported(self, champion_run_id: str) -> bool:
        """Check if this champion has already been exported"""
        state_path = self.export_config.export_state_path
        if not state_path.exists():
            return False
        
        try:
            state = load_json(state_path)
            return state.get("last_champion_run_id") == champion_run_id
        except Exception:
            return False
    
    def _update_state(self, champion_run_id: str, artifact: ModelExportArtifact) -> None:
        """Update export state tracking"""
        save_json(path=self.export_config.export_state_path, data={
            "last_champion_run_id": champion_run_id,
            "last_exported_at": artifact.exported_at.isoformat(),
            "onnx_model_path": str(artifact.onnx_model_path),
            "onnx_fp16_path": str(artifact.onnx_fp16_path) if artifact.onnx_fp16_path else None,
            "fp32_size_mb": artifact.fp32_size_mb,
            "onnx_validated": artifact.onnx_validated,
        })
        logger.info(f"💾 Export state saved : {self.export_config.export_state_path}")
    
    def _log_summary(self, artifact: ModelExportArtifact) -> None:
        """Log export summary"""
        logger.info("=" * 70)
        logger.info("📦 MODEL EXPORT COMPLETE")
        logger.info(f"   Champion run  : {artifact.champion_run_id}")
        logger.info(f"   Architecture  : {artifact.architecture}")
        logger.info(f"   FP32 ONNX     : {artifact.onnx_model_path} ({artifact.fp32_size_mb:.1f} MB)")
        if artifact.onnx_fp16_path:
            logger.info(f"   FP16 ONNX     : {artifact.onnx_fp16_path} ({artifact.fp16_size_mb:.1f} MB)")
        logger.info(f"   Validated     : {artifact.onnx_validated}")
        logger.info(f"   model_info    : {artifact.model_info_path}")
        logger.info("=" * 70)
        logger.info("   ➡️  Next stage : model_quantization.py")


def main():
    """Entry point for model export"""
    logger.info("=" * 70)
    logger.info("🚀 Starting Model Export")
    logger.info("=" * 70)
    
    config_manager = ConfigurationManager()
    export_config = config_manager.get_model_export_config()
    registry_config = config_manager.get_model_registry_config()
    
    exporter = ModelExport(export_config, registry_config)
    artifact = exporter.run()
    
    if artifact is None:
        logger.info("✅ Nothing to do — champion already exported")
    else:
        logger.info(f"✅ Export complete")
        logger.info(f"   FP32 ONNX : {artifact.onnx_model_path}")
        if artifact.onnx_fp16_path:
            logger.info(f"   FP16 ONNX : {artifact.onnx_fp16_path}")
    
    return artifact


if __name__ == "__main__":
    main()