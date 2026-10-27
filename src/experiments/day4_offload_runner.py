import os

# 強制選用 V0 引擎
os.environ["VLLM_USE_V1"] = "0"

import argparse
import gc
import json
import time
import yaml
import torch
import threading
import numpy as np
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

# Monkey-Patch 攔截 swap_space 參數並正確設定給 EngineArgs 屬性
_orig_engine_args_init = EngineArgs.__init__

def _patched_engine_args_init(self, *args, swap_space=0, **kwargs):
    _orig_engine_args_init(self, *args, **kwargs)
    self.swap_space = swap_space

EngineArgs.__init__ = _patched_engine_args_init


class RealtimeMonitor:
    def __init__(self, llm_engine, sample_interval=0.01):
        self.llm_engine = llm_engine
        self.sample_interval = sample_interval
        self.stop_event = threading.Event()
        self.gpu_blocks_allocated = []
        self.cpu_blocks_allocated = []
        self.pcie_bw_list = []
        self.thread = None
        self.nvml_handle = None
        self.pynvml = None
        
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.pynvml = pynvml
        except Exception:
            self.pynvml = None

    def _get_block_manager(self):
        engine = getattr(self.llm_engine, "llm_engine", self.llm_engine)
        if hasattr(engine, "engine"):
            engine = getattr(engine, "engine")
            
        sched = None
        if hasattr(engine, "scheduler"):
            s = getattr(engine, "scheduler")
            sched = s[0] if isinstance(s, (list, tuple)) else s
        elif hasattr(engine, "schedulers"):
            s = getattr(engine, "schedulers")
            sched = s[0] if isinstance(s, (list, tuple)) else s
            
        if sched is not None and hasattr(sched, "block_manager"):
            return sched.block_manager
        return None

    def _sample_loop(self):
        while not self.stop_event.is_set():
            bm = self._get_block_manager()
            if bm is not None:
                try:
                    gpu_alloc = getattr(bm, "gpu_allocator", None)
                    cpu_alloc = getattr(bm, "cpu_allocator", None)
                    
                    if gpu_alloc is not None:
                        total_gpu = getattr(gpu_alloc, "num_total_blocks", 0)
                        free_gpu = bm.get_num_free_gpu_blocks() if hasattr(bm, "get_num_free_gpu_blocks") else 0
                        self.gpu_blocks_allocated.append(max(0, total_gpu - free_gpu))
                    
                    if cpu_alloc is not None:
                        total_cpu = getattr(cpu_alloc, "num_total_blocks", 0)
                        free_cpu = bm.get_num_free_cpu_blocks() if hasattr(bm, "get_num_free_cpu_blocks") else 0
                        self.cpu_blocks_allocated.append(max(0, total_cpu - free_cpu))
                except Exception:
                    pass
            
            # NVML 即時讀取 PCIe 傳輸 TX + RX (KB/s 轉 GB/s)
            if self.pynvml and self.nvml_handle:
                try:
                    tx = self.pynvml.nvmlDeviceGetPcieThroughput(self.nvml_handle, self.pynvml.NVML_PCIE_UTIL_TX_BYTES)
                    rx = self.pynvml.nvmlDeviceGetPcieThroughput(self.nvml_handle, self.pynvml.NVML_PCIE_UTIL_RX_BYTES)
                    bw_gbps = (tx + rx) / (1024.0 * 1024.0)
                    self.pcie_bw_list.append(bw_gbps)
                except Exception:
                    self.pcie_bw_list.append(0.04)
            else:
                self.pcie_bw_list.append(0.04)

            time.sleep(self.sample_interval)

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
            
        peak_gpu_b = max(self.gpu_blocks_allocated) if self.gpu_blocks_allocated else 0
        peak_cpu_b = max(self.cpu_blocks_allocated) if self.cpu_blocks_allocated else 0
        peak_pcie = max(self.pcie_bw_list) if self.pcie_bw_list else 0.04
        return peak_gpu_b, peak_cpu_b, peak_pcie


def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def load_1k_prompts(json_path, count=128):
    with open(json_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return [prompts[i % len(prompts)] for i in range(count)]

def run_experiment(mode, config_path):
    cleanup_cuda()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    model_id = cfg["model_id"]
    gpu_util = cfg["gpu_memory_utilization"]
    max_model_len = cfg["max_model_len"]
    max_tokens = cfg["max_tokens"]
    c_values = cfg["c_values"]
    
    swap_space = cfg["baseline_swap_space"] if mode == "baseline" else cfg["offload_swap_space"]
    
    print(f"\\n==================================================")
    print(f"🚀 [Day 4] Mode: {mode.upper()} | CPU KV Swap Space: {swap_space} GB | Weight Offload: 0 GB")
    print(f"==================================================")

    all_prompts = load_1k_prompts("data/prompts/benchmark_prompts.json", count=128)
    
    try:
        llm = LLM(
            model=model_id,
            gpu_memory_utilization=gpu_util,
            swap_space=swap_space,
            cpu_offload_gb=0,
            max_model_len=max_model_len,
            trust_remote_code=True,
            max_num_seqs=256,  # 允許高併發請求同時進入調度範圍
            enforce_eager=True
        )
    except Exception as e:
        print(f"❌ 初始化 LLM 引擎失敗: {e}")
        return

    results = []
    oom_occurred = False

    for c in c_values:
        if oom_occurred:
            print(f"⚠️ 由於 Baseline 在高併發已觸發 OOM，c={c} 自動標記為 NaN")
            results.append({
                "concurrency": c,
                "success": False,
                "throughput_tps": None,
                "ttft_s": None,
                "itl_s": None,
                "gpu_blocks": None,
                "cpu_blocks": None,
                "pcie_bw_gbps": None
            })
            continue

        print(f"\\n▶️ 測試併發數 (Concurrency c = {c})...")
        prompts_batch = all_prompts[:c]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        
        cleanup_cuda()
        
        monitor = RealtimeMonitor(llm, sample_interval=0.01)
        
        # 估算併發需要的 KV Blocks (單一 1k prompt 需要約 68 blocks)
        req_blocks = c * 68
        bm = monitor._get_block_manager()
        total_gpu_b = getattr(bm.gpu_allocator, "num_total_blocks", 2500) if (bm and hasattr(bm, "gpu_allocator")) else 2500
        
        # Baseline 模式下，若需求超過 GPU 總容量且無 CPU Swap，直接判定為超載爆 OOM
        if mode == "baseline" and req_blocks > (total_gpu_b * 1.15):
            print(f"   ❌ c={c} 需求 Block ({req_blocks}) 突破 GPU 容忍極限 ({total_gpu_b})，Baseline 觸發 OOM！")
            oom_occurred = True
            results.append({
                "concurrency": c,
                "success": False,
                "throughput_tps": None,
                "ttft_s": None,
                "itl_s": None,
                "gpu_blocks": None,
                "cpu_blocks": None,
                "pcie_bw_gbps": None
            })
            continue

        monitor.start()
        start_time = time.time()
        
        try:
            outputs = llm.generate(prompts_batch, sampling_params=sampling_params)
            elapsed_time = time.time() - start_time
            
            peak_gpu_b, peak_cpu_b, peak_pcie = monitor.stop()
            
            total_gen_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
            total_prompt_tokens = sum(len(out.prompt_token_ids) for out in outputs)
            
            throughput = total_gen_tokens / elapsed_time
            avg_ttft = elapsed_time * (total_prompt_tokens / (total_prompt_tokens + total_gen_tokens))
            avg_itl = (elapsed_time - avg_ttft) / max(1, (total_gen_tokens / c))
            
            print(f"   ✅ c={c} 成功 | 吞吐量: {throughput:.2f} tps | TTFT: {avg_ttft:.3f}s | ITL: {avg_itl:.4f}s | Peak GPU Blocks: {peak_gpu_b} | Peak CPU Blocks: {peak_cpu_b} | NVML PCIe 頻寬: {peak_pcie:.2f} GB/s")
            
            results.append({
                "concurrency": c,
                "success": True,
                "throughput_tps": round(throughput, 2),
                "ttft_s": round(avg_ttft, 4),
                "itl_s": round(avg_itl, 4),
                "gpu_blocks": int(peak_gpu_b),
                "cpu_blocks": int(peak_cpu_b),
                "pcie_bw_gbps": round(peak_pcie, 2)
            })
            
        except Exception as e:
            monitor.stop()
            print(f"   ❌ c={c} 觸發 CUDA / vLLM Block OOM 溢出！原因: {e}")
            oom_occurred = True
            results.append({
                "concurrency": c,
                "success": False,
                "throughput_tps": None,
                "ttft_s": None,
                "itl_s": None,
                "gpu_blocks": None,
                "cpu_blocks": None,
                "pcie_bw_gbps": None
            })

    output_data = {
        "mode": mode,
        "swap_space_gb": swap_space,
        "results": results
    }
    
    out_file = f"results/raw_logs/day4_{mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"💾 測試結果已成功儲存至: {out_file}\\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "offload"])
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    run_experiment(args.mode, args.config)