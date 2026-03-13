# weed_detection/components/model_quantization.py

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from torch.utils.data import DataLoader
from weed_detection.components.data_transformation import DeepWeedDataset
from torchvision import transforms


from weed_detection import logger
from weed_detection.config.configuration import ConfigurationManager
from weed_detection.constants.constant import IMAGENET_MEAN, IMAGENET_STD
from weed_detection.entity.artifact_entity import (
    ModelExportArtifact,
    ModelQuantizationArtifact,
)
from weed_detection.entity.config_entity import ModelQuantizationConfig
from weed_detection.utils.utility import load_json, save_json


class TensorRTLogger(trt.ILogger):
    """TensorRT logger wrapper"""
    
    def __init__(self):
        trt.ILogger.__init__(self)
    
    def log(self, severity, msg):
        severity_map = {
            trt.ILogger.INTERNAL_ERROR: "ERROR",
            trt.ILogger.ERROR: "ERROR",
            trt.ILogger.WARNING: "WARNING",
            trt.ILogger.INFO: "INFO",
            trt.ILogger.VERBOSE: "DEBUG",
        }
        log_level = severity_map.get(severity, "INFO")
        
        if severity in [trt.ILogger.ERROR, trt.ILogger.INTERNAL_ERROR]:
            logger.error(f"[TensorRT] {msg}")
        elif severity == trt.ILogger.WARNING:
            logger.warning(f"[TensorRT] {msg}")
        else:
            logger.debug(f"[TensorRT] {msg}")


class ModelQuantization:
    """Convert ONNX to TensorRT engine for edge deployment"""
    
    def __init__(
        self,
        config: ModelQuantizationConfig,
    ):
        self.config = config
        self.logger = TensorRTLogger()
        
        # Check TensorRT availability
        if trt is None:
            raise ImportError(
                "TensorRT not installed. Please install on Jetson device:\n"
                "pip install tensorrt pycuda"
            )
        
        # Check if running on Jetson
        self.is_jetson = self._check_jetson()
        logger.info(f"🚀 Running on Jetson: {self.is_jetson}")
    
    def _check_jetson(self) -> bool:
        """Detect if running on Jetson platform"""
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
                return "jetson" in model
        except:
            return False
    
    def run(self, export_artifact: ModelExportArtifact) -> Optional[ModelQuantizationArtifact]:
        """Execute model quantization pipeline"""
        logger.info("=" * 70)
        logger.info("🚀 Model Quantization — ONNX → TensorRT")
        logger.info("=" * 70)
        
        # 1. Version check
        champion_run_id = export_artifact.champion_run_id
        if self._already_quantized(champion_run_id):
            logger.info(f"⏭️  Champion '{champion_run_id}' already quantized — skipping")
            return None
        
        # 2. Select ONNX model (prefer FP16 if available)
        onnx_path = export_artifact.onnx_fp16_path or export_artifact.onnx_model_path
        logger.info(f"   Using ONNX: {onnx_path}")
        
        # 3. Build TensorRT engine
        start_time = time.time()
        
        if self.config.quant_precision == "int8":
            # INT8 requires calibration data
            calibration_loader = self._create_calibration_loader()
            engine_path, latency, throughput = self._build_int8_engine(
                onnx_path, calibration_loader
            )
            calibration_used = True
            int8_path = engine_path
            fp16_path = None
        else:
            # FP16 (default for Jetson)
            engine_path, latency, throughput = self._build_fp16_engine(onnx_path)
            calibration_used = False
            fp16_path = engine_path
            int8_path = None
        
        build_time = time.time() - start_time
        
        # 4. Get engine sizes
        fp16_size = os.path.getsize(fp16_path) / (1024 ** 2) if fp16_path else None
        int8_size = os.path.getsize(int8_path) / (1024 ** 2) if int8_path else None
        
        # 5. Build artifact
        artifact = self._build_artifact(
            export_artifact=export_artifact,
            champion_run_id=champion_run_id,
            fp16_path=fp16_path,
            int8_path=int8_path,
            calibration_used=calibration_used,
            build_time=build_time,
            latency=latency,
            throughput=throughput,
            fp16_size=fp16_size,
            int8_size=int8_size,
        )
        
        # 6. Update state
        self._update_state(champion_run_id, artifact)
        self._log_summary(artifact)
        
        return artifact
    
    def _build_fp16_engine(self, onnx_path: Path) -> Tuple[Path, float, float]:
        """Build FP16 TensorRT engine from ONNX"""
        logger.info("─" * 50)
        logger.info("🔧 Building FP16 TensorRT engine")
        
        engine_path = self.config.trt_engine_fp16_path
        
        # Create builder
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX file")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.config.quant_max_workspace_size * 1024 * 1024
        )
        
        # Set FP16 mode
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("   ✅ FP16 mode enabled")
        else:
            logger.warning("   ⚠️  Platform does not support fast FP16")
        
        # Set profiling
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.min_timing_iterations = self.config.quant_min_timing_iters
        config.avg_timing_iterations = self.config.quant_avg_timing_iters
        
        # Build engine
        logger.info("   Building engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        # Profile inference latency
        latency, throughput = self._profile_engine(engine_path)
        
        logger.info(f"   ✅ Engine saved: {engine_path}")
        logger.info(f"   Size: {os.path.getsize(engine_path) / (1024**2):.2f} MB")
        logger.info(f"   Latency: {latency:.2f} ms")
        logger.info(f"   Throughput: {throughput:.1f} FPS")
        
        return engine_path, latency, throughput
    
    def _build_int8_engine(
        self, 
        onnx_path: Path,
        calibration_loader: DataLoader,
    ) -> Tuple[Path, float, float]:
        """Build INT8 TensorRT engine with calibration"""
        logger.info("─" * 50)
        logger.info("🔧 Building INT8 TensorRT engine with calibration")
        
        engine_path = self.config.trt_engine_int8_path
        cache_path = self.config.calibration_cache_path
        
        # Create builder
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX file")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.config.quant_max_workspace_size * 1024 * 1024
        )
        
        # Set INT8 mode
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            
            # Create calibrator
            calibrator = EntropyCalibrator(
                loader=calibration_loader,
                cache_path=cache_path,
                input_size=self.config.input_size,
                batch_size=1,
            )
            config.int8_calibrator = calibrator
            logger.info("   ✅ INT8 mode enabled with calibration")
        else:
            logger.warning("   ⚠️  Platform does not support fast INT8, falling back to FP16")
            return self._build_fp16_engine(onnx_path)
        
        # Build engine
        logger.info(f"   Calibrating with {self.config.quant_calibration_batches} batches...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build INT8 TensorRT engine")
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        # Profile inference latency
        latency, throughput = self._profile_engine(engine_path)
        
        logger.info(f"   ✅ Engine saved: {engine_path}")
        logger.info(f"   Size: {os.path.getsize(engine_path) / (1024**2):.2f} MB")
        logger.info(f"   Latency: {latency:.2f} ms")
        logger.info(f"   Throughput: {throughput:.1f} FPS")
        
        return engine_path, latency, throughput
    
    def _profile_engine(self, engine_path: Path) -> Tuple[float, float]:
        """Profile engine inference latency"""
        import pycuda.driver as cuda
        
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        # Create context
        context = engine.create_execution_context()
        
        # Allocate buffers
        inputs = []
        outputs = []
        bindings = []
        
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
            
            size = trt.volume(shape)
            size_in_bytes = size * dtype.itemsize
            
            # Allocate device memory
            allocation = cuda.mem_alloc(size_in_bytes)
            bindings.append(int(allocation))
            
            if engine.binding_is_input(i):
                inputs.append({
                    'name': name,
                    'dtype': dtype,
                    'shape': shape,
                    'allocation': allocation,
                })
            else:
                outputs.append({
                    'name': name,
                    'dtype': dtype,
                    'shape': shape,
                    'allocation': allocation,
                })
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, self.config.input_size, self.config.input_size).astype(np.float32)
        
        # Warm-up
        for _ in range(10):
            cuda.memcpy_htod(inputs[0]['allocation'], dummy_input)
            context.execute_v2(bindings)
            cuda.Context.synchronize()
        
        # Profile
        num_iterations = 100
        start = time.time()
        
        for _ in range(num_iterations):
            cuda.memcpy_htod(inputs[0]['allocation'], dummy_input)
            context.execute_v2(bindings)
            cuda.Context.synchronize()
        
        end = time.time()
        
        # Calculate metrics
        total_time = (end - start) * 1000  # ms
        avg_latency = total_time / num_iterations
        throughput = 1000 / avg_latency  # FPS
        
        # Cleanup
        for binding in bindings:
            cuda.mem_free(binding)
        
        return avg_latency, throughput
    
    def _create_calibration_loader(self) -> Optional[DataLoader]:
        """Create calibration dataloader for INT8 quantization"""
        if DataLoader is None:
            logger.warning("   ⚠️  PyTorch not available for calibration")
            return None
        
        # Use validation samples for calibration
        from weed_detection.components.data_transformation import DataTransformation
        config_manager = ConfigurationManager()
        dt_config = config_manager.get_data_transformation_config()
        transformation = DataTransformation(dt_config)
        ta = transformation.load_artifact()
        
        # Get calibration samples (use validation set)
        if not ta.val_csv_path or not ta.val_images_dir:
            logger.warning("   ⚠️  No validation set for calibration")
            return None
        
        # Load samples
        samples = []
        with open(ta.val_csv_path, 'r') as f:
            import csv
            for row in csv.DictReader(f):
                if len(samples) >= self.config.quant_calibration_batches:
                    break
                samples.append((str(ta.val_images_dir / row['Filename']), int(row['Label'])))
        
        # Create transform (no augmentation, just normalization)
        transform = transforms.Compose([
            transforms.Resize((self.config.input_size, self.config.input_size)),
            transforms.CenterCrop(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        dataset = DeepWeedDataset(samples, transform)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        logger.info(f"   Created calibration loader with {len(samples)} samples")
        return loader
    
    def _build_artifact(
        self,
        export_artifact: ModelExportArtifact,
        champion_run_id: str,
        fp16_path: Optional[Path],
        int8_path: Optional[Path],
        calibration_used: bool,
        build_time: float,
        latency: float,
        throughput: float,
        fp16_size: Optional[float],
        int8_size: Optional[float],
    ) -> ModelQuantizationArtifact:
        """Build and save quantization artifact"""
        
        artifact = ModelQuantizationArtifact(
            export_artifact=export_artifact,
            champion_run_id=champion_run_id,
            architecture=export_artifact.architecture,
            trt_engine_fp16_path=fp16_path,
            trt_engine_int8_path=int8_path,
            calibration_cache_path=self.config.calibration_cache_path if calibration_used else None,
            quant_precision=self.config.quant_precision,
            input_size=self.config.input_size,
            num_classes=self.config.num_classes,
            engine_fp16_size_mb=fp16_size,
            engine_int8_size_mb=int8_size,
            calibration_used=calibration_used,
            build_time_s=build_time,
            inference_latency_ms=latency,
            inference_throughput=throughput,
            quantized_at=datetime.now(),
            artifact_path=self.config.artifact_path,
        )
        
        # Save artifact JSON
        artifact_dict = {
            "champion_run_id": champion_run_id,
            "architecture": export_artifact.architecture,
            "quant_precision": self.config.quant_precision,
            "trt_engine_fp16_path": str(fp16_path) if fp16_path else None,
            "trt_engine_int8_path": str(int8_path) if int8_path else None,
            "calibration_cache_path": str(self.config.calibration_cache_path) if calibration_used else None,
            "engine_fp16_size_mb": fp16_size,
            "engine_int8_size_mb": int8_size,
            "inference_latency_ms": latency,
            "inference_throughput": throughput,
            "quantized_at": artifact.quantized_at.isoformat(),
            "artifact_path": str(self.config.artifact_path),
        }
        
        save_json(path=self.config.artifact_path, data=artifact_dict)
        logger.info(f"📋 Quantization artifact saved: {self.config.artifact_path}")
        
        return artifact
    
    def _already_quantized(self, champion_run_id: str) -> bool:
        """Check if this champion has already been quantized"""
        state_path = self.config.quantization_state_path
        if not state_path.exists():
            return False
        
        try:
            state = load_json(state_path)
            return state.get("last_champion_run_id") == champion_run_id
        except Exception:
            return False
    
    def _update_state(self, champion_run_id: str, artifact: ModelQuantizationArtifact) -> None:
        """Update quantization state tracking"""
        save_json(path=self.config.quantization_state_path, data={
            "last_champion_run_id": champion_run_id,
            "last_quantized_at": artifact.quantized_at.isoformat(),
            "quant_precision": artifact.quant_precision,
            "inference_latency_ms": artifact.inference_latency_ms,
            "inference_throughput": artifact.inference_throughput,
            "trt_engine_fp16_path": str(artifact.trt_engine_fp16_path) if artifact.trt_engine_fp16_path else None,
        })
        logger.info(f"💾 Quantization state saved: {self.config.quantization_state_path}")
    
    def _log_summary(self, artifact: ModelQuantizationArtifact) -> None:
        """Log quantization summary"""
        logger.info("=" * 70)
        logger.info("📦 MODEL QUANTIZATION COMPLETE")
        logger.info(f"   Champion run  : {artifact.champion_run_id}")
        logger.info(f"   Architecture  : {artifact.architecture}")
        logger.info(f"   Precision     : {artifact.quant_precision}")
        
        if artifact.trt_engine_fp16_path:
            logger.info(f"   FP16 Engine   : {artifact.trt_engine_fp16_path} ({artifact.engine_fp16_size_mb:.1f} MB)")
        
        if artifact.trt_engine_int8_path:
            logger.info(f"   INT8 Engine   : {artifact.trt_engine_int8_path} ({artifact.engine_int8_size_mb:.1f} MB)")
        
        logger.info(f"   Latency       : {artifact.inference_latency_ms:.2f} ms")
        logger.info(f"   Throughput    : {artifact.inference_throughput:.1f} FPS")
        logger.info(f"   Build time    : {artifact.build_time_s / 60:.1f} min")
        logger.info("=" * 70)
        logger.info("   ➡️  Next stage : model_deployment.py (deploy to edge)")


# Custom INT8 calibrator
class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """Entropy calibrator for INT8 quantization"""
    
    def __init__(self, loader, cache_path, input_size, batch_size=1):
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.loader = loader
        self.cache_path = Path(cache_path)
        self.input_size = input_size
        self.batch_size = batch_size
        self.batch_idx = 0
        
        # Allocate device memory
        self.device_input = cuda.mem_alloc(batch_size * 3 * input_size * input_size * 4)
        
        # Create iterator
        self.iterator = iter(loader)
    
    def get_batch_size(self) -> int:
        return self.batch_size
    
    def get_batch(self, names) -> Optional[List[int]]:
        """Get next batch for calibration"""
        try:
            batch = next(self.iterator)
            
            # Handle tuple or dict
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            # Copy to device
            cuda.memcpy_htod(self.device_input, images.numpy())
            self.batch_idx += 1
            
            return [int(self.device_input)]
            
        except StopIteration:
            return None
    
    def read_calibration_cache(self) -> Optional[bytes]:
        """Read calibration cache if exists"""
        if self.cache_path.exists():
            with open(self.cache_path, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache: bytes) -> None:
        """Write calibration cache"""
        with open(self.cache_path, 'wb') as f:
            f.write(cache)


def main():
    """Entry point for model quantization"""
    logger.info("=" * 70)
    logger.info("🚀 Starting Model Quantization")
    logger.info("=" * 70)
    
    config_manager = ConfigurationManager()
    quant_config = config_manager.get_model_quantization_config()
    
    # Load export artifact
    from weed_detection.components.model_export import ModelExport
    export_config = config_manager.get_model_export_config()
    
    if not export_config.artifact_path.exists():
        raise FileNotFoundError(
            f"No ModelExportArtifact at {export_config.artifact_path}\n"
            "Run model_export.py first."
        )
    
    export_artifact_dict = load_json(export_config.artifact_path)
    
    # Reconstruct export artifact
    export_artifact = ModelExportArtifact(
        onnx_model_path=Path(export_artifact_dict["onnx_model_path"]),
        onnx_fp16_path=Path(export_artifact_dict["onnx_fp16_path"]) if export_artifact_dict.get("onnx_fp16_path") else None,
        model_info_path=Path(export_artifact_dict["model_info_path"]),
        champion_run_id=export_artifact_dict["champion_run_id"],
        architecture=export_artifact_dict["architecture"],
        input_size=export_artifact_dict["input_size"],
        num_classes=export_artifact_dict["num_classes"],
        opset_version=export_artifact_dict["opset_version"],
        fp32_size_mb=export_artifact_dict["fp32_size_mb"],
        fp16_size_mb=export_artifact_dict.get("fp16_size_mb"),
        onnx_validated=export_artifact_dict["onnx_validated"],
        dynamic_batch=export_artifact_dict["dynamic_batch"],
        exported_at=datetime.fromisoformat(export_artifact_dict["exported_at"]),
        artifact_path=export_config.artifact_path,
    )
    
    # Run quantization
    quantizer = ModelQuantization(quant_config)
    artifact = quantizer.run(export_artifact)
    
    if artifact is None:
        logger.info("✅ Nothing to do — champion already quantized")
    else:
        logger.info(f"✅ Quantization complete")
        logger.info(f"   TensorRT engine: {artifact.trt_engine_fp16_path}")
        logger.info(f"   Latency: {artifact.inference_latency_ms:.2f} ms")
    
    return artifact


if __name__ == "__main__":
    main()